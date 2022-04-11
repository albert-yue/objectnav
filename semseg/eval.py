"""
Eval script for Sequential RedNet
"""
import os
import argparse
import time
import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchmetrics import JaccardIndex
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
    parser.add_argument('--seq-len', default=500, type=int,
                        help='length of sequences')
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='path to checkpoint to load')
    return parser


parser = build_parser()
args = parser.parse_args()

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([ToTensor(),  Normalize()])
val_data = TrajectoryDataset(args.data_dir, seq_len=args.seq_len, phase_train=False, transform=transform)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=False)
num_val = len(val_data)


model = SeqRedNet(RedNetRNNModule())

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.eval()
model.to(device)

if args.ckpt:
    global_step, args.start_epoch = load_ckpt(model, None, args.ckpt, device, prefix='module.')
    print('Loading ckpt from', args.ckpt)

loss_fn = CrossEntropyLoss2d()
loss_fn.to(device)
val_loss = 0.0

iou_fn = JaccardIndex(num_classes=40, reduction='sum')
iou_fn.to(device)
miou = 0.0

with torch.no_grad():
    for batch_idx, sample in enumerate(val_loader):
        image = sample['rgb'].to(device)
        depth = sample['depth'].to(device)
        prev_actions = sample['actions'].to(device)
        target = sample['semantic'].to(device)
        
        pred, _ = model(image, depth, prev_actions)  # drops the hidden states
        loss = loss_fn([pred], [target])
        val_loss += loss.detach() * image.size(0) / num_val

        _, _, c, h, w = pred.size()
        mask = target > 0
        mask_neg = target < 0
        targets_m = target.clone().to(device)
        targets_m[mask] -= 1
        targets_m[mask_neg] += c
        miou += iou_fn(pred.view(-1, c, h, w), targets_m.long().view(-1, h, w)) / (args.seq_len * num_val)

print('Val CrossEntropyLoss:', val_loss)
print('mIoU', miou)

