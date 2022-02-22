import torch
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            *_, c, h, w = inputs.size()
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
