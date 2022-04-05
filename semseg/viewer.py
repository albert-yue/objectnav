'''
Visualization tool to view semantics as a given place in a given mesh
'''
from re import A
from typing import List, Tuple, Union, Optional
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import imageio
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector
)
import habitat_sim
from habitat_sim.agent import AgentConfiguration, AgentState

from semseg.mpcat40 import mpcat40


# Aligns with Habitat Challenge's min and max
# MIN_DEPTH = 0.5
# MAX_DEPTH = 5.0

# Aligns with Habitat Simulator's defaults
MIN_DEPTH = 0.0
MAX_DEPTH = 10.0

SEMANTIC_PALETTE = np.array([cat[2] for cat in mpcat40])

class Viewer:
    def __init__(
        self,
        img_size: Tuple[int] = (480, 640),
        output: Optional[List[str]] = None,
    ):        
        self.img_size = img_size

        if output is None:
            output = ['rgb']
        
        self.out_name_to_sensor_name = {
            'rgb': 'color_sensor',
            'depth': 'depth_sensor',
            'semantic': 'semantic_sensor',
        }
        self.output = output

    def init_sim(self, scene_filepath):
        # Initialize sim
        self.cfg = self._config_sim(scene_filepath, self.img_size)
        self.sim = habitat_sim.Simulator(self.cfg)
    
    def close_sim(self):
        self.sim.close()
        self.sim = None
    
    def _observe(self, position, rotation):
        '''
        Observe a single position/rotation. Useful to separate for when you plan to call _observe()
        multiple times in the same sim scene.
        '''
        # Initialize label map
        instance_id_to_name = self._generate_label_map(self.sim.semantic_scene)
        labels = {name: index for index, name, *_ in mpcat40}
        map_to_class_labels = np.vectorize(
            lambda x: labels.get(instance_id_to_name.get(x, 'unlabeled'), 41)
        )
        
        # Set agent location
        agent = self.sim.initialize_agent(self.cfg.sim_cfg.default_agent_id)
        state = agent.get_state()
        new_state = AgentState(position, rotation, state.sensor_states)
        agent.set_state(new_state)

        # Generate sample
        obs = self.sim.get_sensor_observations()
        sample = {
            out_name: obs[self.out_name_to_sensor_name[out_name]]
            for out_name in self.output
        }
        if 'rgb' in sample:
            sample['rgb'] = sample['rgb'][:, :, :3]  # Get rid of A channel from RGBA
        if 'depth' in sample:
            sample['depth'] = np.clip(sample['depth'], MIN_DEPTH, MAX_DEPTH)
            # normalize to [0, 1]
            sample['depth'] = (sample['depth'] - MIN_DEPTH) / ( MAX_DEPTH - MIN_DEPTH )
        if 'semantic' in sample:
            sample['semantic'] = map_to_class_labels(sample['semantic'])
        
        return sample

    def observe(self, scene_filepath, position, rotation):
        self.init_sim(scene_filepath)

        sample = self._observe(position, rotation)

        self.close_sim()

        return sample

    def view(self, scene_filepath, position, rotation):
        sample = self.observe(scene_filepath, position, rotation)

        if 'semantic' in sample:
            sample['semantic'] = SEMANTIC_PALETTE[sample['semantic']].astype(np.uint8)

        plt.figure(figsize=(12, 5))
        for i, (title, data) in enumerate(sample.items()):
            ax = plt.subplot(1, 3, i + 1)
            ax.axis("off")
            ax.set_title(title)
            plt.imshow(data)

        plt.show()

    @staticmethod
    def _generate_label_map(scene, verbose=False):
        if verbose:
            print(
                f'House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects'
            )
            print(f'House center:{scene.aabb.center} dims:{scene.aabb.sizes}')

        instance_id_to_name = {}
        for obj in scene.objects:
            if obj and obj.category:
                obj_id = int(obj.id.split('_')[-1])
                instance_id_to_name[obj_id] = obj.category.name()

        return instance_id_to_name

    @staticmethod
    def _config_sim(scene_filepath, img_size):
        settings = {
            'width': img_size[1],  # Spatial resolution of the observations
            'height': img_size[0],
            'scene': scene_filepath,  # Scene path
            'default_agent': 0,
            'sensor_height': 1.5,  # Height of sensors in meters
            'color_sensor': True,  # RGBA sensor
            'semantic_sensor': True,  # Semantic sensor
            'depth_sensor': True,  # Depth sensor
            'silent': True,
        }

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings['scene']

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensors = {
            'color_sensor': {  # active if sim_settings['color_sensor']
                'sensor_type': habitat_sim.SensorType.COLOR,
                'resolution': [settings['height'], settings['width']],
                'position': [0.0, settings['sensor_height'], 0.0],
            },
            'depth_sensor': {  # active if sim_settings['depth_sensor']
                'sensor_type': habitat_sim.SensorType.DEPTH,
                'resolution': [settings['height'], settings['width']],
                'position': [0.0, settings['sensor_height'], 0.0],
            },
            'semantic_sensor': {  # active if sim_settings['semantic_sensor']
                'sensor_type': habitat_sim.SensorType.SEMANTIC,
                'resolution': [settings['height'], settings['width']],
                'position': [0.0, settings['sensor_height'], 0.0],
            },
        }

        # create sensor specifications
        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params['sensor_type']
                sensor_spec.resolution = sensor_params['resolution']
                sensor_spec.position = sensor_params['position']
                sensor_spec.gpu2gpu_transfer = False
                sensor_specs.append(sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_sim_position_and_rotation(gps, compass, start_position, start_rotation):
    if not isinstance(start_rotation, np.ndarray):
        start_rotation = np.array(start_rotation)

    origin = np.array(start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(start_rotation)
    
    # we know that rho = 1
    x, y = np.cos(compass), np.sin(compass)
    rot_q = np.quaternion(0, 0, 0, 0)
    rot_q.imag = np.array([y, 0, -x])

    # Using the rotation matrix from of q*p*q_inv
    # see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    # also assuming that the given quat to quaternion_rotate_vector is (r, 0, j, 0)
    j = np.sqrt((1 - x)/ 2)  # sign doesn't seem to change result
    r = y / (-2*j)
    q = np.quaternion(r, 0, j, 0)
    rotation = (q * rotation_world_start.inverse()).inverse()

    # reverse the rotation from sim position - origin -> gps
    position = quaternion_rotate_vector(rotation_world_start, gps) + origin
    return position, rotation


if __name__ == '__main__':
    # Agent position [-1.99132,0.0451003,-0.931545] Agent orientation         0 -0.923884         0 -0.382672
    # gps = np.array([ 7.637e+00, -7.153e-07,  5.246e+00], dtype=np.float16)
    # compass = np.array([-2.0938], dtype=np.float16)
    # position, rotation = get_sim_position_and_rotation(gps, compass, [7.32451, 0.0244, -4.7948], np.array([0, 0.86426, 0, -0.50304]))
    scene = 'TbHJrupSAjP'
    position = np.array([7.32451, 0.0244, -4.7948])
    rotation = [0.583916553009705, -0.00596639206195039, -0.811581240284739, -0.018486527659681]
    # position = [-1.00883628,  0.02439928, -0.74556395]
    # rotation = [0, -0.980788, 0, -0.195078] #[0, -0.881927, 0, -0.471386]

    # position = [-9.02773,-3.18813,-11.4508]
    # rotation = [0, -0.980788, 0, -0.195078]

    position = np.array(position)
    scene_fp = '../habitat-lab/data/scene_datasets/mp3d/{s}/{s}.glb'.format(s=scene)
    viewer = Viewer(
        img_size=(480, 640),
        output=['rgb', 'depth', 'semantic'],
    )
    viewer.view(scene_fp, position, rotation)

