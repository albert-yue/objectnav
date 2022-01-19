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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms
from torchmetrics import JaccardIndex
from tensorboardX import SummaryWriter

from habitat_baselines.rl.models.rednet_rnn import SeqRedNet, RedNetRNNModule
from semseg.dataset import TrajectoryDataset
from semseg.loss import CrossEntropyLoss2d
from semseg.transforms import ToTensor, Normalize
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

    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='path to checkpoint to load')
    
    parser.add_argument('--summary-dir', default='./semseg/summary/', metavar='DIR',
                        help='path to save summary')
    parser.add_argument('--ckpt-dir', default='./semseg/checkpoints/', metavar='DIR',
                        help='path to save checkpoints')
    parser.add_argument('--print-freq', '-p', default=200, type=int,
                        metavar='N', help='print batch frequency (default: 50)')
    parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                        metavar='N', help='save epoch frequency (default: 5)')
    return parser


parser = build_parser()
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([ToTensor(),  Normalize()])
val_data = TrajectoryDataset(args.data_dir, seq_len=args.seq_len, phase_train=False, transform=transform)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=False)
num_val = len(val_data)


model = SeqRedNet(RedNetRNNModule(), freeze_encoder=args.freeze_rednet, pretrained=False)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.eval()
model.to(device)

if args.last_ckpt:
    global_step, args.start_epoch = load_ckpt(model, None, args.last_ckpt, device)


loss_fn = CrossEntropyLoss2d()
loss_fn.to(device)
val_loss = 0.0

iou_fn = JaccardIndex(num_classes=40, reduction='sum')
miou = 0.0

flag = True

with torch.no_grad():
    for batch_idx, sample in enumerate(val_loader):
        image = sample['rgb'].to(device)
        depth = sample['depth'].to(device)
        prev_actions = sample['actions'].to(device)
        target = sample['semantic'].to(device)
        
        if flag:
            print('Target size:', target.size())
            flag = False

        pred, _ = model(image, depth, prev_actions)  # drops the hidden states
        loss = loss_fn([pred], [target])
        val_loss += loss.detach() * image.size(0) / num_val

        _, c, h, w = pred.size()
        miou += iou_fn(pred.view(-1, c, h, w), target.view(-1, h, w)) / num_val

print('Val CrossEntropyLoss:', val_loss)
print('mIoU', miou)
