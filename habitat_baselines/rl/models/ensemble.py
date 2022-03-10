import torch
import torch.nn as nn

from habitat_baselines.rl.models.rednet import RedNet

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
