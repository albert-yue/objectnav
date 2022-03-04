import os
import argparse
import time
import datetime

import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import MNIST
import torchvision.transforms
from tensorboardX import SummaryWriter

from habitat_baselines.rl.models.rednet import RedNet
from semseg.utils import print_log, save_ckpt, load_ckpt


class RedNetEnsemble(nn.Module):
    def __init__(self, ensemble_size, num_classes=10):
        super().__init__()
        self.rednet = RedNet()
        self.ensemble = nn.ModuleList([
            # nn.ConvTranspose2d(self.rednet.inplanes, num_classes, kernel_size=2,
            #                    stride=2, padding=0, bias=True)
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, num_classes)
            )
            for _ in range(ensemble_size)
        ])
    
    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         nn.init.kaiming_normal_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()

    def load_rednet(self, ckpt):
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['model_state']
        prefix = 'module.'
        state_dict = {
            (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
        }
        self.rednet.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch']))
    
    def forward(self, rgb, depth):
        fuses = self.rednet.forward_downsample(rgb, depth)
        features = fuses[-1]

        outs = [fc(features) for fc in self.ensemble]
        return outs


def build_parser():
    parser = argparse.ArgumentParser(description='RedNet Ensemble')
    parser.add_argument('--data-dir', default='data/', metavar='DIR',
                        help='path to MNIST dataset - should be directory that contains MNIST/') 
    parser.add_argument('--ensemble-size', type=int, default=10,
                        help='ensemble size')
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
    
    parser.add_argument('--summary-dir', default='./semseg/summary/mnist-ensemble/', metavar='DIR',
                        help='path to save summary')
    parser.add_argument('--ckpt-dir', default='./semseg/checkpoints/mnist-ensemble/', metavar='DIR',
                        help='path to save checkpoints')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print batch frequency (default: 50)')
    parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                        metavar='N', help='save epoch frequency (default: 5)')
    return parser


if __name__ == '__main__':
    print('Start time:', datetime.datetime.now())
    parser = build_parser()
    args = parser.parse_args()

    writer = SummaryWriter(args.summary_dir)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    # total_gb = torch.cuda.get_device_properties(0).total_memory / 1000000000
    # print('Total CUDA memory:', '{0:.2f}'.format(total_gb), 'GB')

    print('Building dataset...')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
    ])
    train_data = MNIST(args.data_dir, train=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=False)
    num_train = len(train_data)

    print('Dataset made')
    # print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

    model = RedNetEnsemble(ensemble_size=args.ensemble_size)
    model.load_rednet(args.rednet_ckpt)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.train()
    model.to(device)

    # TODO(ayue): DistributedDataParallel

    print('Model loaded')
    # print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

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

            image = sample[0].repeat(1, 3, 1, 1).to(device)
            fake_depth = torch.zeros(sample[0].size()).to(device)
            # print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))
            target = sample[1].to(device)
            # print('Range:', target[0].min(), target[0].max())
            # print('Target size:', target[0].size())

            # print('Image size:', image.size())
            # print('Target size:', target.size())

            # print('reserved | allocated:', torch.cuda.memory_reserved(0), '|', torch.cuda.memory_allocated(0))

            preds = model(image, fake_depth)
            losses = []
            for pred in preds:
                losses.append(loss_fn(pred, target))
            loss = sum(losses)
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

                writer.add_scalar('CrossEntropyLoss', loss.detach(), global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], global_step=global_step)
                last_count = local_count

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs, 0, num_train)

    print("Training completed.")
    print(datetime.datetime.now())
