
import numpy as np
import time
import shutil
import os
from os.path import join
import collections
import yaml
import cv2
import tqdm
import math
import glob
from mmwave_bag import make_frames_from_csv
from pcl2depth import velo_points_2_pano
from pyquaternion import Quaternion


def re_mkdir_dir(sequence_name, header_name):
    dir = join(data_dir, str(sequence_name), header_name)
    if os.path.exists(dir):
        shutil.rmtree(dir)
        time.sleep(5)
        os.makedirs(dir)
    else:
        os.makedirs(dir)
    return dir


def ransac_to_depth(cfg, frame, timestamp, depth_image_dir, pixel_coord_dir, world_coord_dir):
    # only select those points with the certain range (in meters) - 5.12 meter for this TI board
    eff_rows_idx = (frame[:, 0] ** 2 + frame[:, 1] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
    pano_img, pixel_coord, world_coord = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'],
                                                            cfg['pcl2depth']['h_res'],
                                                            v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    img_path = join(depth_image_dir, '{}.png'.format(timestamp))
    cv2.imwrite(img_path, pano_img)

    pixel_coord_file = join(pixel_coord_dir, '{}.txt'.format(timestamp))
    with open(pixel_coord_file, 'a+') as myfile:
        for x, y in pixel_coord:
            myfile.write(str(x) + " " + str(y) + '\n')

    pixel_coord_file = join(world_coord_dir, '{}.txt'.format(timestamp))
    with open(pixel_coord_file, 'a+') as myfile:
        for x, y, z in world_coord:
            myfile.write(str(x) + " " + str(y) + ' ' + str(z) + '\n')




# get config
project_dir = os.path.dirname(os.getcwd())    # '/Users/manmi/Documents/GitHub/assoc_3d'
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_dir = join(os.path.dirname(project_dir), 'indoor_data')
exp_names = cfg['radar']['exp_name']
all_sequences = cfg['radar']['all_sequences']
train_sequences = cfg['radar']['training']
valid_sequences = cfg['radar']['validating']
test_sequences = cfg['radar']['testing']

middle_transform = np.array(cfg['radar']['translation_matrix']['middle'])
left_transform = np.array(cfg['radar']['translation_matrix']['left'])
right_transform = np.array(cfg['radar']['translation_matrix']['right'])

v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))

topic_middle = '_slash_mmWaveDataHdl_slash_RScan_middle'
topic_left = '_slash_mmWaveDataHdl_slash_RScan_left'
topic_right = '_slash_mmWaveDataHdl_slash_RScan_right'

left_quaternion = Quaternion(axis=[0, 0, 1], angle=math.pi / 2)
right_quaternion = Quaternion(axis=[0, 0, 1], angle=-math.pi / 2)
align_interval = 5e7

for sequence_name in test_sequences:

    # ---------------------------- read ransac point -----------------------

    ransac_dst_path = join(data_dir, str(sequence_name), 'enzo_ransac_dst')
    ransac_src_path = join(data_dir, str(sequence_name), 'enzo_ransac_src')
    if (not os.path.exists(ransac_dst_path)) or (not os.path.exists(ransac_src_path)):
        continue

    print("sequence: ", sequence_name)

    dst_files = sorted(glob.glob(join(ransac_dst_path, '*.txt')))
    src_files = sorted(glob.glob(join(ransac_src_path, '*.txt')))

    # ------------------------- pcl to depth -------------------------

    # output file rebuild
    depth_image_src_dir = re_mkdir_dir(sequence_name, 'enzo_ransac_depth_src')
    pixel_coord_src_dir = re_mkdir_dir(sequence_name, 'enzo_ransac_pixel_location_src')
    world_coord_src_dir = re_mkdir_dir(sequence_name, 'enzo_ransac_world_location_src')

    depth_image_dst_dir = re_mkdir_dir(sequence_name, 'enzo_ransac_depth_dst')
    pixel_coord_dst_dir = re_mkdir_dir(sequence_name, 'enzo_ransac_pixel_location_dst')
    world_coord_dst_dir = re_mkdir_dir(sequence_name, 'enzo_ransac_world_location_dst')

    frame_idx = 0
    corrd_dict = dict()
    for src_file, dst_file in tqdm.tqdm(zip(src_files, dst_files), total=len(src_files)):

        _, file_name_dst = os.path.split(dst_file)
        _, file_name_sec = os.path.split(src_file)
        timestamp_dst, _ = os.path.splitext(file_name_dst)
        timestamp_src, _ = os.path.splitext(file_name_sec)

        frame_src = np.loadtxt(src_file, delimiter=' ', dtype=np.float)
        frame_dst = np.loadtxt(dst_file, delimiter=' ', dtype=np.float)

        ransac_to_depth(cfg, frame_src, timestamp_src, depth_image_src_dir, pixel_coord_src_dir, world_coord_src_dir)
        ransac_to_depth(cfg, frame_dst, timestamp_dst, depth_image_dst_dir, pixel_coord_dst_dir, world_coord_dst_dir)

        frame_idx += 1
    print('finished process {}'.format(sequence_name))
    print('In total {} images'.format(frame_idx))
