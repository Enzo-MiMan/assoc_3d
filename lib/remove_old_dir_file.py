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

        depth_gt_src

    depth_gt_src = join(sequence_path, 'depth_gt_src')
    enzo_ts_match_gap4 = join(sequence_path, 'enzo_ts_match_gap4.txt')

    remove_dir(depth_gt_src)

    if os.path.exists(enzo_ts_match_gap4):
        os.remove(enzo_ts_match_gap4)


