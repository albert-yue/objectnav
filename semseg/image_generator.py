'''
Collects frames from scenes using Habitat-Sim

Some of this code is adapted from habitat-sim's ImageExtractor,
as the application is similar but ImageExtractor is limited in speed.
'''

from typing import List, Tuple, Union, Optional
import os

import numpy as np
import imageio
import habitat_sim
from habitat_sim.agent import AgentConfiguration

from semseg.mpcat40 import mpcat40


# Aligns with Habitat Challenge's min and max
# MIN_DEPTH = 0.5
# MAX_DEPTH = 5.0

# Aligns with Habitat Simulator's defaults
MIN_DEPTH = 0.0
MAX_DEPTH = 10.0

SEMANTIC_PALETTE = np.array([cat[2] for cat in mpcat40])

class ImageGenerator:
    def __init__(
        self,
        scene_filepath: Union[str, List[str]],
        img_size: Tuple[int] = (480, 640),
        output: Optional[List[str]]=None,
        seed: Optional[int] = None,
        save_dir: str = '',
    ):
        if isinstance(scene_filepath, str):
            self.scene_filepaths = [scene_filepath]
        else:
            self.scene_filepaths = scene_filepath
        
        self.img_size = img_size

        if output is None:
            output = ['rgb']
        
        self.out_name_to_sensor_name = {
            'rgb': 'color_sensor',
            'depth': 'depth_sensor',
            'semantic': 'semantic_sensor',
        }
        self.output = output

        self.seed = seed
        self.set_seed(seed)

        self.save_dir = save_dir

    def generate_images(self, num_images_per_scene, save=True, save_image=False):
        samples_by_scene = {}
        for fp in self.scene_filepaths:
            samples_by_scene[fp] = self._generate_images_for_scene(fp, num_images_per_scene, save=save, save_image=save_image)
        return samples_by_scene

    def _generate_images_for_scene(self, scene_filepath, num_images, save=True, save_image=False):
        scene_name = os.path.splitext(os.path.basename(scene_filepath))[0]

        # Initialize sim
        cfg = self._config_sim(scene_filepath, self.img_size)
        sim = habitat_sim.Simulator(cfg)

        if self.seed is not None:
            sim.seed(self.seed)

        # Initialize label map
        instance_id_to_name = self._generate_label_map(sim.semantic_scene)
        labels = {name: index for index, name, *_ in mpcat40}
        map_to_class_labels = np.vectorize(
            lambda x: labels.get(instance_id_to_name.get(x, 'unlabeled'), 41)
        )
        
        # Generate images
        samples = []
        for i in range(num_images):
            # Set agent location
            sim.initialize_agent(cfg.sim_cfg.default_agent_id)
            print(cfg.sim_cfg.default_agent_id)

            # Generate sample
            obs = sim.get_sensor_observations()
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

            # Save sample
            if save:
                os.makedirs(os.path.join(self.save_dir, scene_name), exist_ok=True)
                for out_name in self.output:
                    save_fp = os.path.join(self.save_dir, scene_name, '{}_{:04d}.npy'.format(out_name, i))
                    np.save(save_fp, sample[out_name])

            if save_image:
                if 'rgb' in sample:
                    save_fp = os.path.join(self.save_dir, scene_name, 'rgb_{:04d}.png'.format(i))
                    imageio.imwrite(save_fp, sample['rgb'])
                if 'depth' in sample:
                    save_fp = os.path.join(self.save_dir, scene_name, 'dep_{:04d}.png'.format(i))
                    imageio.imwrite(save_fp, (sample['depth'] * 255.0).astype(np.uint8))
                if 'semantic' in sample:
                    save_fp = os.path.join(self.save_dir, scene_name, 'sem_{:04d}.png'.format(i))
                    mp_semantic_colors = SEMANTIC_PALETTE[sample['semantic']].astype(np.uint8)
                    imageio.imwrite(save_fp, mp_semantic_colors)

            samples.append(sample)
        
        return samples
    
    def set_seed(self, seed=None):
        # not sure if needed rn
        pass

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


if __name__ == '__main__':
    from semseg.dataset import train_envs

    scene_filepaths = ['../habitat-lab/data/scene_datasets/mp3d/{s}/{s}.glb'.format(s=scene) for scene in train_envs[:1]]
    # scene_filepaths = ['../habitat-lab/data/scene_datasets/mp3d/{s}/{s}.glb'.format(s=scene) for scene in train_envs]
    
    generator = ImageGenerator(
        scene_filepaths,
        img_size=(480, 640),
        output=['rgb', 'depth', 'semantic'],
        seed=0,
        save_dir='data/semseg/frames',
    )
    generator.generate_images(1, save=True, save_image=True)
