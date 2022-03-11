import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.models.rednet import RedNet, BatchNormalize

class RedNetEnsemble(nn.Module):
    '''
    A ensemble version of RedNet, replacing the last layer with E parallel convtranspose layers.
    '''
    def __init__(self, ensemble_size, num_classes=40):
        super().__init__()
        self.rednet = RedNet(num_classes=num_classes)
        self.ensemble = nn.ModuleList([
            nn.ConvTranspose2d(self.rednet.inplanes, num_classes, kernel_size=2,
                               stride=2, padding=0, bias=True)
            for i in range(ensemble_size)
        ])
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

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
        _, last_layer, *_ = self.rednet.forward_upsample(*fuses)

        outs = [fc(last_layer) for fc in self.ensemble]
        return outs

class RedNetEnsembleResizeWrapper(nn.Module):
    def __init__(self, ensemble_size, device, resize=True, stabilize=False, **kwargs):
        super().__init__()

        if stabilize:
            # Just checking if stabilize is used. Seems like no, but want to double check
            raise ValueError('Uh oh, looks like Joel used stabilize')

        self.rednet_ensemble = RedNetEnsemble(ensemble_size, **kwargs)
        self.semmap_rgb_norm = BatchNormalize(
            mean=[0.493, 0.468, 0.438],
            std=[0.544, 0.521, 0.499],
            device=device
        )
        self.semmap_depth_norm = BatchNormalize(
            mean=[0.213],
            std=[0.285],
            device=device
        )
        self.pretrained_size = (480, 640)
        self.resize = resize
        self.stabilize = stabilize

    def forward(self, rgb, depth, return_scores=False):
        r"""
            Args:
                Raw sensor inputs.
                rgb: B x H x W x 3
                depth: B x H x W x 1
            Returns:
                semantic: drop-in replacement for default semantic sensor. B x H x W  (no channel, for some reason)
        """
        if self.resize:
            _, og_h, og_w, _ = rgb.size() # b h w c
        # rgb = rgb.permute(0, 3, 1, 2)
        # depth = depth.permute(0, 3, 1, 2)

        rgb = rgb.float() / 255
        if self.resize:
            rgb = F.interpolate(rgb, self.pretrained_size, mode='bilinear')
            depth = F.interpolate(depth, self.pretrained_size, mode='nearest')

        # rgb = self.semmap_rgb_norm(rgb)

        depth_clip = (depth < 1.0).squeeze(1)
        # depth_clip = ((depth < 1.0) & (depth > 0.0)).squeeze(1)
        # depth = self.semmap_depth_norm(depth)

        with torch.no_grad():
            scores = self.rednet_ensemble(rgb, depth)
            scores = torch.stack(scores, dim=1)  # batch x emsemble x n_classes x height x width
            
        # if self.stabilize: # downsample tiny
        #     # Mask out out of depth samples
        #     pred[~depth_clip] = -1 # 41 is UNK in MP3D, but hab-sim renders with -1
        #     # pred = F.interpolate(pred.unsqueeze(1), (15, 20), mode='nearest').squeeze(1)
        if self.resize:
            b, m, c, h, w = scores.size()
            scores = F.interpolate(scores.view(-1, c, h, w), (og_h, og_w), mode='bilinear').view(b, m, c, og_h, og_w)

        # Calculate prediction probabilities. Mean and std of prob to avoid issues from arbitrary scaling in logit space
        probs = F.softmax(scores, dim=2)
        mean_probs = torch.mean(probs, dim=1)  # batch x n_classes x height x width
        std_probs = torch.std(probs, dim=1)
        
        # pred_mean, pred_indices = torch.max(mean_probs, dim=1, keepdim=True)
        # pred_std = torch.gather(std_probs, dim=1, index=pred_indices)
        # pred = torch.cat([pred_mean, pred_std], dim=1)  # batch x 2 x height x width

        pred = torch.cat([mean_probs, std_probs], dim=1)  # batch x 2*n_classes x height x width
        pred = pred.permute(0, 2, 3, 1)  # to shape -> batch x height x width x 2*n_classes

        if return_scores:
            return pred, scores
        return pred

def load_rednet_ensemble(ensemble_size, device, ckpt="", resize=True, stabilize=False, **kwargs):
    if not os.path.isfile(ckpt):
        raise Exception(f"invalid path {ckpt} provided for rednet weights")

    model = RedNetEnsembleResizeWrapper(ensemble_size, device, resize=resize, stabilize=stabilize, **kwargs).to(device)

    print("=> loading RedNetEnsemble checkpoint '{}'".format(ckpt))
    if device.type == 'cuda':
        checkpoint = torch.load(ckpt, map_location='cpu')
    else:
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

    state_dict = checkpoint['state_dict']
    prefix = 'module.'
    state_dict = {
        (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
    }
    model.rednet_ensemble.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch']))

    return model
