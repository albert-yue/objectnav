import os
import pickle

import torch

from habitat_baselines.rl.models.ensemble import load_rednet_ensemble
from habitat_baselines.rl.models.rednet import load_rednet


if __name__ == '__main__':
    print('start')
    data_dir = 'data/semseg/stopping_small/0/'
    device = torch.device('cuda:0')

    ckpt_path = 'weights/rednet_semmap_mp3d_tuned.pth'
    semantic_predictor = load_rednet(device, ckpt_path, resize=True)
    semantic_predictor.eval()

    ensemble_ckpt_path = 'semseg/checkpoints/ensemble/ckpt_epoch_100.00.pth'
    ensemble_semantic_predictor = load_rednet_ensemble(10, device, ensemble_ckpt_path, resize=True)
    ensemble_semantic_predictor.eval()

    for (root, dirs, files) in os.walk(data_dir):
        if len(files) == 0:
            print(root)

        for fname in files:
            fp = os.path.join(root, fname)
            with open(fp, 'rb') as f:
                sample = pickle.load(f)

            rgb, depth = torch.Tensor([sample['rgb']]), torch.Tensor([sample['depth']]).unsqueeze(-1)
            with torch.no_grad():
                sem = semantic_predictor(rgb.to(device), depth.to(device)).detach().cpu().squeeze(0).numpy()
            sample['semantics_rednet'] = sem
            sample['rednet_path'] = ckpt_path

            with torch.no_grad():
                sem = ensemble_semantic_predictor(
                    rgb.to(device),
                    depth.to(device)
                ).detach().cpu().squeeze(0).numpy()
            sample['semantics_ensemble'] = sem
            sample['ensemble_path'] = ensemble_ckpt_path

            with open(fp, 'wb') as f:
                pickle.dump(sample, f)

