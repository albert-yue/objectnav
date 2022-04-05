import sys
import os
import pickle
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

from habitat_baselines.rl.models.ensemble import load_rednet_ensemble
from habitat_baselines.rl.models.rednet import load_rednet
from semseg.mpcat40 import mpcat40
from semseg.resnet import resnet18


category_to_index = {obj[1]: obj[0] for obj in mpcat40}

def make_loader():
    def loader(fp):
        with open(fp, 'rb+') as f:
            sample = pickle.load(f)
    
        ### Semantic image
        sem = sample['semantic']
        # merge void, misc, and unlabeled labels
        sem[sem == 0] = 40
        sem[sem == 41] = 40
        sem = sem-1
        sample['semantic'] = sem

        sample['semantics_rednet'] = sample['semantics_rednet'] - 1
        
        ### Object category
        if 'object_category' not in sample:
            sample['object_category'] = fp.split('/')[-3]
        sample['object_category'] = category_to_index[sample['object_category']]
    
        for k in ['metadata', 'scene', 'position', 'rotation', 'rednet_path', 'ensemble_path']:
            if k in sample:
                del sample[k]
    
        return sample
    return loader


class SuccessGTClassifier(nn.Module):
    def __init__(self, n_categories=40):
        super().__init__()
        sem_embed_size = 4
        goal_embed_size = 4

        self.semantic_embedder = nn.Embedding(n_categories, sem_embed_size)
        self.goal_embedder = nn.Embedding(n_categories, goal_embed_size)
#         self.backbone = resnet18(pretrained=True)
        self.backbone = resnet18(sem_embed_size)
        self.fc = nn.Sequential(
            nn.Linear(1000+4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, category):
        '''
        x: (BxHxW)
        category: target object category: (B)
        '''
        x = self.semantic_embedder(x)
        x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)

        goal_embed = self.goal_embedder(category)
        x = torch.cat([x, goal_embed], dim=1)
        x = self.fc(x)

        return x


class SuccessEnsembleClassifier(nn.Module):
    def __init__(self, n_categories=40):
        super().__init__()
        sem_embed_size = 4
        goal_embed_size = 4

        self.semantic_embedder = nn.Linear(2 * n_categories, sem_embed_size)
        self.goal_embedder = nn.Embedding(n_categories, goal_embed_size)
#         self.backbone = resnet18(pretrained=True)
        self.backbone = resnet18(sem_embed_size)
        self.fc = nn.Sequential(
            nn.Linear(1000+4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, category):
        '''
        x: (BxHxW)
        category: target object category: (B)
        '''
        x = self.semantic_embedder(x)
        x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)

        goal_embed = self.goal_embedder(category)
        x = torch.cat([x, goal_embed], dim=1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    num_epochs = 200
    mode = sys.argv[1] # 'gt', 'ngt', 'ensemble'(TODO)
    if mode == 'gt':
        sem_key = 'semantic'
    elif mode == 'ngt':
        sem_key = 'semantics_rednet'
    elif mode == 'ensemble':
        sem_key = 'semantics_ensemble'
    
    print('Mode:', mode)

    device = torch.device('cuda:0')
    
    dataset = DatasetFolder('data/semseg/stopping_small/', loader=make_loader(), extensions=('.pkl',))
    rng = torch.Generator().manual_seed(42)
    num_train = int(0.8 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=rng)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_per_class = {0: 0, 1: 0}
    for idx in train_dataset.indices:
        y = dataset.samples[idx][1]
        num_per_class[y] += 1
    num_negative = num_per_class[0]
    num_positive = num_per_class[1]
    print('Num negative and positive:', num_negative, 'and', num_positive)

    class_occurrence = [num_negative, num_positive]
    class_weights = [1 - (num / len(dataset)) for num in class_occurrence]
    class_weights

    if mode == 'ensemble':
        model = SuccessEnsembleClassifier()
    else:
        model = SuccessGTClassifier()
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
    loss_fn.to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)

    os.makedirs(os.path.join('semseg/checkpoints/stopping/', mode), exist_ok=True)
    train_loss = []
    train_epoch_loss = []
    val_loss = []
    val_epoch_loss = []
    for t in range(num_epochs):
        epoch_loss = 0
        model.train()
        for i, (batch, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()

            logits = model(batch[sem_key].to(device), batch['object_category'].to(device))

            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().item())
            epoch_loss += loss.detach().cpu().item() * batch['object_category'].size(0)
        
        train_epoch_loss.append(epoch_loss / len(train_dataset))
        
        epoch_loss = 0
        model.eval()
        for i, (batch, y) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            with torch.no_grad():
                logits = model(batch[sem_key].to(device), batch['object_category'].to(device))
                loss = loss_fn(logits, y.to(device))

            val_loss.append(loss.detach().cpu().item())
            epoch_loss += loss.detach().cpu().item() * batch['object_category'].size(0)
        val_epoch_loss.append(epoch_loss / len(val_dataset))

        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_epoch_loss[-1],
            'val_loss': val_epoch_loss[-1],
        }, 'semseg/checkpoints/stopping/{}/{}.pth'.format(mode, t))

    with open('semseg/checkpoints/stopping/{}/train_loss_gt.pkl'.format(mode), 'wb+') as f:
        pickle.dump(train_loss, f)
    with open('semseg/checkpoints/stopping/{}/train_epoch_loss_gt.pkl'.format(mode), 'wb+') as f:
        pickle.dump(train_epoch_loss, f)
    with open('semseg/checkpoints/stopping/{}/val_loss_gt.pkl'.format(mode), 'wb+') as f:
        pickle.dump(val_loss, f)
    with open('semseg/checkpoints/stopping/{}/val_epoch_loss_gt.pkl'.format(mode), 'wb+') as f:
        pickle.dump(val_epoch_loss, f)

