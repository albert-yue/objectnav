import torch
import torch.nn.functional as F
import torchvision
import skimage.transform

# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']
        rgb = rgb / 255
        rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(rgb)
        depth = torchvision.transforms.Normalize(mean=[0.213],
                                                 std=[0.285])(depth)
        sample['rgb'] = rgb
        sample['depth'] = depth

        return sample


class Interpolate:
    """Interpolate up 240x320 for now"""

    def __call__(self, sample):
        return {'rgb': F.interpolate(sample['rgb'], (480, 640), mode='bilinear'),
                'depth': F.interpolate(sample['depth'], (480, 640), mode='nearest'),
                'semantic': F.interpolate(sample['semantic'].unsqueeze(1), (480, 640), mode='nearest').squeeze(1),
                'semantic2': F.interpolate(sample['semantic2'].unsqueeze(1), (240, 320), mode='nearest').squeeze(1),
                'semantic3': F.interpolate(sample['semantic3'].unsqueeze(1), (120, 160), mode='nearest').squeeze(1),
                'semantic4': F.interpolate(sample['semantic4'].unsqueeze(1), (60, 80), mode='nearest').squeeze(1),
                'semantic5': F.interpolate(sample['semantic5'].unsqueeze(1), (30, 40), mode='nearest').squeeze(1),
                'actions': sample['actions']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, depth, semantic, actions = sample['rgb'], sample['depth'], sample['semantic'], sample['actions']

        # Generate different semantic scales
        l, h, w = semantic.shape
        semantic = semantic.transpose((1, 2, 0))
        semantic2 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 2, semantic.shape[1] // 2),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic3 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 4, semantic.shape[1] // 4),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic4 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 8, semantic.shape[1] // 8),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic5 = skimage.transform.resize(
            semantic, (semantic.shape[0] // 16, semantic.shape[1] // 16),
            order=0, mode='reflect', preserve_range=True
        ).transpose((2, 0, 1))
        semantic = semantic.transpose((2, 0, 1))

        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C X H X W
        rgb = rgb.transpose(0, 3, 1, 2)
        depth = depth.transpose(0, 3, 1, 2)
        return {'rgb': torch.from_numpy(rgb).float(),
                'depth': torch.from_numpy(depth).float(),
                'semantic': torch.from_numpy(semantic).float(),
                'semantic2': torch.from_numpy(semantic2).float(),
                'semantic3': torch.from_numpy(semantic3).float(),
                'semantic4': torch.from_numpy(semantic4).float(),
                'semantic5': torch.from_numpy(semantic5).float(),
                'actions': torch.from_numpy(actions).long()}
