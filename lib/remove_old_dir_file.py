import os
from os.path import join
import yaml
from lib.utils import re_mkdir_dir, remove_dir

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_dir = join(os.path.dirname(project_dir), 'indoor_data')
all_sequences = cfg['radar']['all_sequences']

for sequence in all_sequences:
    sequence_path = join(data_dir, sequence)
    if not os.path.exists(sequence_path):
        continue

    enzo_depth = join(sequence_path, 'enzo_depth')
    enzo_depth_gt_dst = join(sequence_path, 'enzo_depth_gt_dst')
    enzo_depth_gt_src = join(sequence_path, 'enzo_depth_gt_src')
    enzo_pixel_location = join(sequence_path, 'enzo_pixel_location')
    enzo_world_location = join(sequence_path, 'enzo_world_location')
    enzo_ts_match_gap4 = join(sequence_path, 'enzo_ts_match_gap4.txt ')


    remove_dir(enzo_depth)
    remove_dir(enzo_depth_gt_dst)
    remove_dir(enzo_depth_gt_src)
    remove_dir(enzo_pixel_location)
    remove_dir(enzo_world_location)

    if os.path.exists(enzo_ts_match_gap4):
        os.remove(enzo_ts_match_gap4)


