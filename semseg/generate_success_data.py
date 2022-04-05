import os
import pickle
import random

import numpy as np
import torch

from habitat import make_dataset
from habitat_baselines.config.default import get_config

from semseg.viewer import Viewer


if __name__ == '__main__':
    SEED = 17290381
    rng = random.Random(SEED)
    
    MAX_VIEW_POINTS = 5  # max view points for a single goal object instance
    DATA_DIR = 'data/semseg/stopping/'

    config = get_config([
        'habitat_baselines/config/objectnav/obj_base.on.yaml',
        'habitat_baselines/config/objectnav/full/split_clamp.on.yaml'
    ])
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)

    viewer = Viewer(
        img_size=(480, 640),
        output=['rgb', 'depth', 'semantic'],
    )

    for cat, goals in dataset.goals_by_category.items():
        scene, object_category = cat.split('_', maxsplit=1)
        scene = os.path.splitext(scene)[0]
        scene_fp = '../habitat-lab/data/scene_datasets/mp3d/{s}/{s}.glb'.format(s=scene)
            
        os.makedirs(os.path.join(DATA_DIR, '1', object_category, scene), exist_ok=True)
        
        i = 0   # increment for unique samples
        viewer.init_sim(scene_fp)
        for instance in goals:
            # Get MAX_VIEW_POINTS (5) random view points of the goal object
            view_points_indices = list(range(len(instance.view_points)))
            if len(instance.view_points) > MAX_VIEW_POINTS:
                rng.shuffle(view_points_indices)
                view_points_indices = view_points_indices[:MAX_VIEW_POINTS]

            for view_idx in view_points_indices:
                view = instance.view_points[view_idx]
                state = view.agent_state
                position = np.array(state.position)
                rotation = state.rotation
                
                sample = viewer._observe(position, rotation)
                sample['scene'] = scene
                sample['object_category'] = object_category
                sample['position'] = state.position
                sample['rotation'] = state.rotation
                
                with open(os.path.join(DATA_DIR, '1', object_category, scene, '{:05d}.pkl'.format(i)), 'wb+') as output:
                    pickle.dump(sample, output)
                i += 1
        viewer.close_sim()

