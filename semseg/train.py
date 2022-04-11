"""
Training script for Sequential RedNet

Based off https://github.com/JindongJiang/RedNet
"""
import os
import argparse
import time
import datetime

import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms
from tensorboardX import SummaryWriter

from habitat_baselines.rl.models.rednet_rnn import SeqRedNet, RedNetRNNModule
from semseg.dataset import TrajectoryDataset
from semseg.loss import CrossEntropyLoss2d
from semseg.transforms_rednet import ToTensor, Normalize
from semseg.utils import print_log, save_ckpt, load_ckpt


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

val_data = TrajectoryDataset(args.data_dir, seq_len=args.seq_len, phase_train=False, transform=transform)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=False)
num_val = len(val_data)

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
    model.train()

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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            image = sample['rgb'].to(device)
            depth = sample['depth'].to(device)
            prev_actions = sample['actions'].to(device)
            
            target = sample['semantic'].to(device)

            pred, _ = model(image, depth, prev_actions)  # drops the hidden states
            loss = loss_fn([pred], [target])
            val_loss = loss.detach() * image.size(0) / num_val # i.e. * batch_size 

    if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
        print('Epoch: {:>3} Val Loss: {:.6f}'.format(epoch, val_loss))
        writer.add_scalar('Val CrossEntropyLoss', val_loss, global_step=epoch)

save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs, 0, num_train)

print("Training completed.")
print(datetime.datetime.now())
