import os
import pickle

import numpy as np
import torch

from habitat import make_dataset
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector
)
from habitat_baselines.config.default import get_config

from semseg.viewer import Viewer


def get_sim_position_and_rotation(gps, compass, start_position, start_rotation):
    if isinstance(compass, np.ndarray):
        compass = compass.item()
    if not isinstance(start_rotation, np.ndarray):
        start_rotation = np.array(start_rotation)

    origin = np.array(start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(start_rotation)
    
    # we know that rho = 1
    x, y = np.cos(compass), np.sin(compass)
    rot_q = np.quaternion(0, 0, 0, 0)
    rot_q.imag = np.array([y, 0, -x])

    # Solved using the rotation matrix from of q*p*q_inv
    # with p.imag = [0, 0, -1]
    # see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    # also assuming that the given quat to quaternion_rotate_vector is (r, 0, j, 0)
    j = np.sqrt((1 - x)/ 2)  # sign doesn't seem to change result
    r = y / (-2*j + 1e-8)
    q = np.quaternion(r, 0, j, 0)
    if j == 0:
        return None, None # temp fail out
    rotation = (q * rotation_world_start.inverse()).inverse()

    # reverse the rotation from sim position - origin -> gps
    position = quaternion_rotate_vector(rotation_world_start, gps) + origin
    return position, rotation


if __name__ == '__main__':
    DATA_DIR = 'data/semseg/stopping/'

    viewer = Viewer(
        img_size=(480, 640),
        output=['rgb', 'depth', 'semantic'],
    )

    trajs = torch.load('objectnav_eval/split_clamp/31/eval_gt_False.pth')['payload']
    failures = [tr for tr in trajs if tr['stats']['success'] == 0]

    i = 0
    indetermined = 0
    for ep in failures:
        info = ep['info']
        scene_fp = info['episode_info']['scene_id']

        scene = os.path.splitext(os.path.basename(scene_fp))[0]
        object_category = info['episode_info']['object_category']
        os.makedirs(os.path.join(DATA_DIR, '0', object_category, scene), exist_ok=True)

        gps = info['observations']['gps'][-2].numpy()
        compass = info['observations']['compass'][-2].numpy()

        position, rotation = get_sim_position_and_rotation(gps, compass, info['episode_info']['start_position'], info['episode_info']['start_rotation'])
        if position is None:
            indetermined += 1
            continue
        sample = viewer.observe(scene_fp, position, rotation)

        sample = {
            'metadata': info['episode_info'],
            'position': position,
            'rotation': rotation,
            'rgb': sample['rgb'],
            'depth': sample['depth'],
            'semantic': sample['semantic'],
        }
        with open(os.path.join(DATA_DIR, '0', object_category, scene, '{:05d}.pkl'.format(i)), 'wb+') as output:
            pickle.dump(sample, output)
        i += 1
    print('could not reverse:', indetermined, 'out of', len(failures))

