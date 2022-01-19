import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.models.rednet import RedNet, BatchNormalize


class RedNetRNNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat2rnn_conv = nn.Conv2d(512, 8, kernel_size=1)
        self.lat2rnn_lin = nn.Linear(8*15*20, 1024)
        self.rnn = nn.LSTM(1024, 1024, batch_first=True)
        self.rnn2lat_lin = nn.Linear(1024, 8*15*20)
        self.rnn2lat_conv = nn.ConvTranspose2d(8, 512, kernel_size=1)

        # initialize final layer weights to 0, to work with skip connection in SeqRedNet
        nn.init.zeros_(self.rnn2lat_lin.weight)
        nn.init.zeros_(self.rnn2lat_lin.bias)

    def forward(self, features, hidden=None):
        '''
        features (Tensor): size (batch x seq_len x channels x height x width)
        '''
        batch_size, seq_len, c_lat, h_lat, w_lat = features.size()
        
        rnn_input = self.lat2rnn_conv(features.view(-1, c_lat, h_lat, w_lat))
        rnn_input = self.lat2rnn_lin(rnn_input.view(batch_size, seq_len, -1))
        
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        
        rnn_out = self.rnn2lat_lin(rnn_out)
        rnn_out = self.rnn2lat_conv(rnn_out.view(-1, 8, h_lat, w_lat))
        return rnn_out.view(batch_size, seq_len, c_lat, h_lat, w_lat)

class SeqRedNet(nn.Module):
    def __init__(self, module, num_classes=40, pretrained=False):
        super().__init__()
        self.rednet = RedNet(num_classes, pretrained=pretrained)
        self.module = module

    def forward(self, rgb, depth, hidden=None):
        """
        image dims are batch_size x length x channels x height x width
        """
        batch_size, seq_len, _, h, w = rgb.size()
        fuses = self.rednet.forward_downsample(rgb.view(-1, 3, h, w), depth.view(-1, 1, h, w))

        features_encoder = fuses[-1]  # latent size is batch_size*seq_len x c_lat x h_lat x w_lat
        fuses = fuses[:-1]

        _, c_lat, h_lat, w_lat = features_encoder.size()
        module_out = self.module(features_encoder.view(batch_size, seq_len, c_lat, h_lat, w_lat), hidden=hidden)
        module_out = module_out.view(-1, c_lat, h_lat, w_lat)

        # Skip connection around intermediate module
        final_latent = module_out + features_encoder

        # We only need predictions.
        # features_encoder = fuses[-1]
        # scores, features_lastlayer = self.forward_upsample(*fuses)
        # debug_tensor('scores', scores)
        # return features_encoder, features_lastlayer, scores
        
        if self.training:
            scores, _, out2, out3, out4, out5 = self.rednet.forward_upsample(*fuses, final_latent)
            return scores.view(batch_size, seq_len, -1, h, w), \
                   out2.view(batch_size, seq_len, -1, out2.size(2), out2.size(3)), \
                   out3.view(batch_size, seq_len, -1, out3.size(2), out3.size(3)), \
                   out4.view(batch_size, seq_len, -1, out4.size(2), out4.size(3)), \
                   out5.view(batch_size, seq_len, -1, out5.size(2), out5.size(3)), \
                   hidden

        scores, *_ = self.rednet.forward_upsample(*fuses, final_latent)
        return scores.view(batch_size, seq_len, -1, h, w), hidden

class SeqRedNetResizeWrapper(nn.Module):
    def __init__(self, device, resize=True, stabilize=False):
        super().__init__()
        self.rednet_rnn = SeqRedNet(RedNetRNNModule())
        self.rednet_rnn.eval()
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

    def forward(self, rgb, depth, hidden=None):
        r"""
            Args:
                Raw sensor inputs.
                rgb: B x seq_len x H=256 x W=256 x 3
                depth: B x seq_len x H x W x 1
            Returns:
                semantic: drop-in replacement for default semantic sensor. B x H x W  (no channel, for some reason)
        """
        # Not quite sure what depth is produced here. Let's just check
        rgb = rgb.permute(0, 1, 4, 2, 3)
        depth = depth.permute(0, 1, 4, 2, 3)

        rgb = rgb.float() / 255
        if self.resize:
            _, l, _, og_h, og_w = rgb.size() # b l c h w
            rgb = F.interpolate(rgb.view(-1, 3, og_h, og_w), self.pretrained_size, mode='bilinear')
            depth = F.interpolate(depth.view(-1, 1, og_h, og_w), self.pretrained_size, mode='nearest')

            _, _, new_h, new_w = rgb.size()
            rgb = rgb.view(-1, l, 3, new_h, new_w)
            depth = depth.view(-1, l, 1, new_h, new_w)

        rgb = self.semmap_rgb_norm(rgb)

        depth_clip = (depth < 1.0).squeeze(1)
        # depth_clip = ((depth < 1.0) & (depth > 0.0)).squeeze(1)
        depth = self.semmap_depth_norm(depth)
        with torch.no_grad():
            scores, hidden = self.rednet_rnn(rgb, depth, hidden=hidden)
            pred = (torch.max(scores, 2)[1] + 1).float() # B x L x 480 x 640
        if self.stabilize: # downsample tiny
            # Mask out out of depth samples
            _, l, pred_h, pred_w = pred.size()
            pred = pred.view(-1, pred_h, pred_w)
            pred[~depth_clip] = -1 # 41 is UNK in MP3D, but hab-sim renders with -1
            pred = pred.view(-1, l, pred_h, pred_w)
            # pred = F.interpolate(pred.unsqueeze(1), (15, 20), mode='nearest').squeeze(1)
        if self.resize:
            _, l, pred_h, pred_w = pred.size()
            pred = F.interpolate(pred.unsqueeze(2).view(-1, 1, pred_h, pred_w), (og_h, og_w), mode='nearest')
            pred = pred.view(-1, l, 1, og_h, og_w)

        return pred.long().squeeze(2), hidden

def load_rednet_seq(device, ckpt="", resize=True, stabilize=False):
    if not os.path.isfile(ckpt):
        raise Exception(f"invalid path {ckpt} provided for rednet weights")

    model = SeqRedNetResizeWrapper(device, resize=resize, stabilize=stabilize).to(device)

    print("=> loading RedNet checkpoint '{}'".format(ckpt))
    if device.type == 'cuda':
        checkpoint = torch.load(ckpt, map_location='cpu')
    else:
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

    state_dict = checkpoint['model_state']
    prefix = 'module.'
    state_dict = {
        (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
    }
    model.rednet_rnn.rednet.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(ckpt, checkpoint['epoch']))

    return model

