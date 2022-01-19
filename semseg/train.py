"""
Training script for Sequential RedNet

Based off https://github.com/JindongJiang/RedNet
"""
import os
import argparse
import time
import datetime

import numpy as np
import skimage.transform
import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from tensorboardX import SummaryWriter

from habitat_baselines.rl.models.rednet_rnn import SeqRedNet, RedNetRNNModule
from semseg.dataset import TrajectoryDataset


def build_parser():
    parser = argparse.ArgumentParser(description='RedNet+RNN')
    parser.add_argument('--data-dir', default='rednet_data/', metavar='DIR',
                        help='path to trajectory dataset')
    parser.add_argument('--freeze-rednet', action='store_true', default=True,
                        help='freeze rednet')
    parser.add_argument('--seq-len', default=500, type=int,
                        help='length of sequences')
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 10)')
    parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                        help='number of total epochs to run (default: 1500)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--rednet-ckpt', default='weights/rednet_semmap_mp3d_tuned.pth',
                        help='rednet starting checkpoint')
    
    parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    
    parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                        help='decay rate of learning rate (default: 0.8)')
    parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                        help='epoch of per decay of learning rate (default: 150)')
    
    parser.add_argument('--summary-dir', default='./semseg/summary/', metavar='DIR',
                        help='path to save summary')
    parser.add_argument('--ckpt-dir', default='./semseg/checkpoints/', metavar='DIR',
                        help='path to save checkpoints')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print batch frequency (default: 50)')
    parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                        metavar='N', help='save epoch frequency (default: 5)')
    return parser


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']
        rgb = rgb / 255
        rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(rgb)
        depth = torchvision.transforms.Normalize(mean=[0.213],
                                                 std=[0.285])(depth)
        sample['rgb'] = rgb
        sample['depth'] = depth

        return sample


class Interpolate:
    """Interpolate up 240x320 for now"""

    def __call__(self, sample):
        return {'rgb': F.interpolate(sample['rgb'], (480, 640), mode='bilinear'),
                'depth': F.interpolate(sample['depth'], (480, 640), mode='nearest'),
                'semantic': F.interpolate(sample['semantic'].unsqueeze(1), (480, 640), mode='nearest').squeeze(1),
                'semantic2': F.interpolate(sample['semantic2'].unsqueeze(1), (240, 320), mode='nearest').squeeze(1),
                'semantic3': F.interpolate(sample['semantic3'].unsqueeze(1), (120, 160), mode='nearest').squeeze(1),
                'semantic4': F.interpolate(sample['semantic4'].unsqueeze(1), (60, 80), mode='nearest').squeeze(1),
                'semantic5': F.interpolate(sample['semantic5'].unsqueeze(1), (30, 40), mode='nearest').squeeze(1),
                'actions': sample['actions']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, depth, semantic, actions = sample['rgb'], sample['depth'], sample['semantic'], sample['actions']

        # Generate different semantic scales
        l, h, w = semantic.shape
        semantic = semantic.transpose((1, 2, 0))
        semantic2 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 2, semantic.shape[1] // 2),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic3 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 4, semantic.shape[1] // 4),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic4 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 8, semantic.shape[1] // 8),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic5 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 16, semantic.shape[1] // 16),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic = semantic.transpose((2, 0, 1))

        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C X H X W
        rgb = rgb.transpose(0, 3, 1, 2)
        depth = depth.transpose(0, 3, 1, 2)
        return {'rgb': torch.from_numpy(rgb).float(),
                'depth': torch.from_numpy(depth).float(),
                'semantic': torch.from_numpy(semantic).float(),
                'semantic2': torch.from_numpy(semantic2).float(),
                'semantic3': torch.from_numpy(semantic3).float(),
                'semantic4': torch.from_numpy(semantic4).float(),
                'semantic5': torch.from_numpy(semantic5).float(),
                'actions': torch.from_numpy(actions).long()}


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            _, _, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)
            targets = targets.view(-1, h, w)

            mask = targets > 0
            mask_neg = targets < 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            targets_m[mask_neg] += c
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

print('Start time:', datetime.datetime.now())
parser = build_parser()
args = parser.parse_args()

writer = SummaryWriter(args.summary_dir)

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

# total_gb = torch.cuda.get_device_properties(0).total_memory / 1000000000
# print('Total CUDA memory:', '{0:.2f}'.format(total_gb), 'GB')

transform = torchvision.transforms.Compose([ToTensor(),  Normalize()])
train_data = TrajectoryDataset(args.data_dir, seq_len=args.seq_len, phase_train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=False)
num_train = len(train_data)

# print('Dataset made')
# print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

model = SeqRedNet(RedNetRNNModule(), freeze_encoder=args.freeze_rednet, pretrained=False)

# Load RedNet checkpoint
# if device.type == 'cuda':
#     checkpoint = torch.load(args.rednet_ckpt, map_location='cpu')
# else:
#     checkpoint = torch.load(args.rednet_ckpt, map_location=lambda storage, loc: storage)
checkpoint = torch.load(args.rednet_ckpt)
state_dict = checkpoint['model_state']
prefix = 'module.'
state_dict = {
    (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
}
model.rednet.load_state_dict(state_dict)
print("=> loaded checkpoint '{}' (epoch {})"
        .format(args.rednet_ckpt, checkpoint['epoch']))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.train()
model.to(device)

# print('Model loaded')
# print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

loss_fn = CrossEntropyLoss2d()
loss_fn.to(device)

if args.freeze_rednet:
    optimizer = torch.optim.SGD(model.module.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

global_step = 0

if args.last_ckpt:
    global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

# print('Before training')
# print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

for epoch in range(int(args.start_epoch), args.epochs):
    local_count = 0
    last_count = 0
    end_time = time.time()
    if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
        save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                    local_count, num_train)

    for batch_idx, sample in enumerate(train_loader):
        # print('Epoch', epoch, 'Batch', batch_idx)
        optimizer.zero_grad()
               
        image = sample['rgb'].to(device)
        depth = sample['depth'].to(device)
        prev_actions = sample['actions'].to(device)
        # print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))
        target_scales = [sample[s].to(device) for s in ['semantic', 'semantic2', 'semantic3', 'semantic4', 'semantic5']]
        # print('Target sizes:', [t.size() for t in target_scales])

        # print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

        *pred_scales, _ = model(image, depth, prev_actions)  # drops the hidden states
        loss = loss_fn(pred_scales, target_scales)
        loss.backward()
        optimizer.step()
        scheduler.step()

        local_count += image.detach().shape[0]
        global_step += 1
        if global_step % args.print_freq == 0 or global_step == 1:
            time_inter = time.time() - end_time
            count_inter = local_count - last_count
            print_log(global_step, epoch, local_count, count_inter,
                        num_train, loss.detach(), time_inter)
            end_time = time.time()

            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
            # grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            # writer.add_image('image', grid_image, global_step)
            # grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
            # writer.add_image('depth', grid_image, global_step)
            # grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3, normalize=False,
            #                         range=(0, 255))
            # writer.add_image('Predicted label', grid_image, global_step)
            # grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
            # writer.add_image('Groundtruth label', grid_image, global_step)
            writer.add_scalar('CrossEntropyLoss', loss.detach(), global_step=global_step)
            writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], global_step=global_step)
            last_count = local_count

save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs, 0, num_train)

print("Training completed.")
print(datetime.datetime.now())

