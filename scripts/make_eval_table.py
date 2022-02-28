import os
import csv

FIELDNAMES = [
    'episode',
    'scene_id',
    'object',
    'ckpt',
    'success',
    'distance_to_goal',
    'spl',
    'coverage.reached',
    'coverage.mini_reached',
    'coverage.step',
    'goal_vis',
    'region_level.room_cat',
]
USE_INT = ['episode', 'ckpt', 'success']


def extract_info(filename):
    info = filename[:-4].split('-')
    target_obj, rem = info[0].split('_eval_gt_')
    use_gt, scene_name = rem.split('_')
    scene_name = scene_name.split('.')[0]  # Remove the ending .glb

    stats = {}
    for s in info[1:]:
        k, v = s.split('=')
        v = float(v)
        if k in USE_INT:
            v = int(v)
        stats[k] = v

    return target_obj, scene_name, stats



if __name__ == '__main__':
    sem_type = 'pred'
    dirpath = f'vis/videos/objectnav/split_clamp/val_300/{sem_type}_sem/'
    video_names = os.listdir(dirpath)
    with open(f'eval_summary_{sem_type}.csv', 'w+') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for vname in video_names:
            target_obj, scene_name, stats = extract_info(vname) 
            writer.writerow({'object': target_obj, 'scene_id': scene_name, **stats})
