from typing import List, Dict
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semseg.resnet import resnet18


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


class SuccessClassifier(nn.Module):
    def __init__(self, n_categories=40):
        super().__init__()
        sem_embed_size = 4
        goal_embed_size = 4

        self.semantic_embedder = nn.Linear(n_categories, sem_embed_size)
        self.goal_embedder = nn.Embedding(n_categories, goal_embed_size)
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
        x = x.permute(0, 2, 3, 1)  # move channel dim to end for "embedder"
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
        x: (batch (B) x 2*num_categories (2*C) x H x W): mean and std of ensemble predictions across all semantic categories
        category: target object category: (B)
        '''
        x = x.permute(0, 2, 3, 1)  # move channel dim to end for "embedder"
        x = self.semantic_embedder(x)
        x = x.permute(0, 3, 1, 2)  # move channel dim back up for ResNet backbone
        x = self.backbone(x)

        goal_embed = self.goal_embedder(category)
        x = torch.cat([x, goal_embed], dim=1)
        x = self.fc(x)

        return x
