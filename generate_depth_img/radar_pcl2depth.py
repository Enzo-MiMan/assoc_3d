'''
input:
    data file:
        exp_names = cfg['radar']['exp_name'] = '2019-11-28-15-43-32'
        train_name = cfg['radar']['training']
        test_name = cfg['radar']['testing']

                '../../indoor_data/2019-11-28-15-43-32/ _slash_mmWaveDataHdl_slash_RScan_left.csv'
                '../../indoor_data/2019-11-28-15-43-32/_slash_mmWaveDataHdl_slash_RScan_middle.csv'
                '../../indoor_data/2019-11-28-15-43-32/_slash_mmWaveDataHdl_slash_RScan_right.csv'

output:  depth image   #  save in '../../indoor_data/2019-11-28-15-43-32/depth_enzo'
'''


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


# get config
project_dir = os.path.dirname(os.getcwd())    # '/Users/manmi/Documents/GitHub/assoc_3d'
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_dir = join(os.path.dirname(project_dir), 'indoor_data')
exp_names = cfg['radar']['exp_name']
all_sequences = cfg['radar']['all_sequences']

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


for sequence_name in all_sequences:

    # --------------------------------- process middle ---------------------------------

    csv_path = join(data_dir, str(sequence_name), topic_middle + '.csv')
    if not os.path.exists(csv_path):
        continue

    print("sequence: ", sequence_name)

    readings_dict = make_frames_from_csv(csv_path)
    data_dict = collections.OrderedDict(sorted(readings_dict.items()))  # !!! sort the dict before using

    frames = list()
    timestamps = list()
    for timestamp, pts in data_dict.items():
        frame = list()
        for pt in pts:
            pt = np.array(pt)
            translated_pt = pt + middle_transform[0:3]
            frame.append(translated_pt)

        # do not add empty frames
        if not frame:
            continue
        frames.append(np.array(frame))
        timestamps.append(timestamp)


    # --------------------------------- process left and right ---------------------------------

    # process left and right
    for topic in [topic_left, topic_right]:
        csv_path = join(data_dir, str(sequence_name), topic + '.csv')

        if not os.path.exists(csv_path):
            continue

        readings_dict = make_frames_from_csv(csv_path)
        data_dict = collections.OrderedDict(sorted(readings_dict.items()))  # !!! sort the dict before using

        for timestamp, pts in data_dict.items():
            frame = list()
            for pt in pts:
                pt = np.array(pt)

                if topic == topic_left:
                    translated_pt = left_quaternion.rotate(pt) + left_transform[0:3]
                else:
                    translated_pt = right_quaternion.rotate(pt) + right_transform[0:3]
                assert (pt[0] != translated_pt[0])

                frame.append(translated_pt)
            # do not add empty frames
            if not frame:
                continue

            # ----------------- concatenate -------------------------
            # align frames
            for i in range(len(timestamps)):
                if abs(int(timestamp) - int(timestamps[i])) <= align_interval:
                    frames[i] = np.concatenate((frames[i], np.array(frame)))

    # ------------------------- pcl to depth -------------------------

    # output file rebuild
    depth_image_dir = re_mkdir_dir(sequence_name, 'enzo_depth')
    pixel_coord_dir = re_mkdir_dir(sequence_name, 'enzo_pixel_location')
    world_coord_dir = re_mkdir_dir(sequence_name, 'enzo_world_location')


    frame_idx = 0
    corrd_dict = dict()
    for timestamp, frame in tqdm.tqdm(zip(timestamps, frames), total=len(timestamps)):

        # only select those points with the certain range (in meters) - 5.12 meter for this TI board
        eff_rows_idx = (frame[:, 0] ** 2 + frame[:, 1] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
        pano_img, pixel_coord, world_coord = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                      v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

        if pano_img.size == 0:
            print('{} frame skipped as all pts are out of fov!'.format(frame_idx))
            frame_idx = frame_idx + 1
            continue

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


        frame_idx += 1
    print('finished process {}'.format(sequence_name))
    print('In total {} images'.format(frame_idx))
