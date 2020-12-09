import numpy as np
import os
from os.path import join
import yaml
import glob




if __name__ == "__main__":

    # --------------------- load config ------------------------

    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']


    for sequence in train_sequences:

        sequence_path = join(data_dir, sequence)
        if not os.path.exists(sequence_path):
            continue

        # --------------------- get timestamp matches ------------------------
        depth_gap4_list_src = sorted(glob.glob(join(sequence_path, 'enzo_depth_gt_src/*.txt')))
        depth_gap4_list_dst = sorted(glob.glob(join(sequence_path, 'enzo_depth_gt_dst/*.txt')))

        for index in range(len(depth_gap4_list_src)):
            _, file_name_dst = os.path.split(depth_gap4_list_dst[index])
            _, file_name_sec = os.path.split(depth_gap4_list_src[index])
            timestamp_dst, _ = os.path.splitext(file_name_dst)
            timestamp_src, _ = os.path.splitext(file_name_sec)