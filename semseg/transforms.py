import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as TF


IMAGE_KEYS = ['rgb', 'depth', 'semantic']

class Compose:
    def __init__(self, transforms, image_keys=None):
        self.transforms = transforms
    
    def __call__(self, sample):
        '''
        sample (dict): data sample, str-keyed. Some of the values
            should be arrays representing images
        '''
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:
    '''
    Converts images (mainly in our case HxWxC np.ndarrays) to tensors
    Also moves the channel dim ahead of the HxW dims -> CxHxW
    '''
    def __init__(self, image_keys=None):
        self.image_keys = image_keys
        if image_keys is None:
            self.image_keys = IMAGE_KEYS

    def __call__(self, sample):
        for k in self.image_keys:
            sample[k] = TF.to_tensor(sample[k])
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5, image_keys=None):
        self.p = p
        self.image_keys = image_keys
        if image_keys is None:
            self.image_keys = IMAGE_KEYS
    
    def __call__(self, sample):
        if torch.rand(1) < self.p:
            for k in self.image_keys:
                sample[k] = TF.hflip(sample[k])
        return sample


class RandomResizedCrop:
    def __init__(self, size, scale=(0.2, 1.0), ratio=(3./4., 4./3.), image_keys=None):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.image_keys = image_keys
        if image_keys is None:
            self.image_keys = IMAGE_KEYS

    def __call__(self, sample):
        if len(self.image_keys) > 0:
            params = torchvision.transforms.RandomResizedCrop.get_params(sample[self.image_keys[0]], self.scale, self.ratio)
            for k in self.image_keys:
                sample[k] = TF.resized_crop(sample[k], *params, size=self.size)
        return sample


class GaussianNoise:
    def __init__(self, mean=0., std=1., image_keys=None):
        self.mean = mean
        self.std = std        
        self.image_keys = image_keys
        if image_keys is None:
            self.image_keys = IMAGE_KEYS
        
    def __call__(self, sample): 
        for k in self.image_keys:
            sample[k].add_( torch.randn(sample[k].size()) * self.std + self.mean )
        return sample
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip:
    def __init__(self, low=0., high=1., image_keys=None):
        self.low = low
        self.high = high
        self.image_keys = image_keys
        if image_keys is None:
            self.image_keys = IMAGE_KEYS

    def __call__(self, sample):
        for k in self.image_keys:
            sample[k] = torch.clamp(sample[k], self.low, self.high)
        return sample

