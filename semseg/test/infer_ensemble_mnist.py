from collections import Counter
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms

from semseg.test.ensemble_mnist import RedNetEnsemble


data_dir = 'data/'
cuda = False #True
ensemble_size = 10
batch_size = 64

device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor(),
])
test_data = MNIST(data_dir, train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=False)

model = RedNetEnsemble(ensemble_size=ensemble_size)

ckpt = 'semseg/checkpoints/mnist-ensemble/ckpt_epoch_20.00.pth'
checkpoint = torch.load(ckpt)
state_dict = checkpoint['state_dict']
prefix = 'module.'
state_dict = {
    (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
}
model.load_state_dict(state_dict)
print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch']))

model.eval()
model.to(device)

num_correct = 0
num_correct_for_correct = 0
num_correct_for_incorrect = 0
incorrect_ensemble_preds = []

for batch_idx, sample in enumerate(test_loader):
    image = sample[0].repeat(1, 3, 1, 1).to(device)
    fake_depth = torch.zeros(sample[0].size()).to(device)
    target = sample[1].to(device)

    # print('Target class:', target.detach().cpu().numpy())

    scores = model(image, fake_depth)
    scores = torch.stack(scores, dim=1)  # batch x emsemble x n_classes

    preds = torch.max(scores, 2)[1]
    final_pred = preds.mode(dim=1)[0]

    num_correct += (final_pred == target).sum().detach().cpu().item()

    expanded_target = target.unsqueeze(1).expand(-1, ensemble_size)
    ensemble_correct = (preds == expanded_target).sum(dim=1)

    num_correct_for_correct += ensemble_correct[final_pred == target].sum().item()
    num_correct_for_incorrect += ensemble_correct[final_pred != target].sum().item()

    if len(preds[final_pred != target]) > 0:
        for p, t in zip(preds[final_pred != target], target[final_pred != target]):
            incorrect_ensemble_preds.append((t.item(), dict(Counter(p.tolist()))))

print('Accuracy:', num_correct / len(test_data))
print('Num incorrect:', len(test_data) - num_correct)

print('Avg correct when correct:', num_correct_for_correct / num_correct)
print('Avg correct when incorrect:', num_correct_for_incorrect / (len(test_data) - num_correct))

pprint(incorrect_ensemble_preds)
