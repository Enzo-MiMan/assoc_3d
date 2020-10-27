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
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
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


def plot_depth(frame_idx, timestamp, frame, map_dir, cfg):
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))
    # only select those points with the certain range (in meters) - 5 meter for this TI board
    eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['base_conf']['img_width']
    pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                  v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))

    fig_path = join(map_dir, '{}_{}.png'.format(frame_idx, timestamp))

    cv2.imshow("grid", pano_img)
    cv2.waitKey(1)

    cv2.imwrite(fig_path, pano_img)


def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix


def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return np.array([qx, qy, qz, qw])


# get config
project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_dir = cfg['base_conf']['data_base']
exp_names = cfg['radar']['exp_name']
sequence_names = cfg['radar']['all_sequences']

middle_transform = np.array(cfg['radar']['translation_matrix']['middle'])
left_transform = np.array(cfg['radar']['translation_matrix']['left'])
right_transform = np.array(cfg['radar']['translation_matrix']['right'])
mt_matrix = quaternion_to_rotation_matrix(euler_to_quaternion(middle_transform[3], middle_transform[4], middle_transform[5]))
lt_matrix = quaternion_to_rotation_matrix(left_transform[3:7])
rt_matrix = quaternion_to_rotation_matrix(right_transform[3:7])
left_quaternion = Quaternion(axis=[0, 0, 1], angle=math.pi / 2)
right_quaternion = Quaternion(axis=[0, 0, 1], angle=-math.pi / 2)

align_interval = 5e7

for sequence_name in sequence_names:

    # --------------------------------- process middle ---------------------------------

    print(sequence_name)

    topic = '_slash_mmWaveDataHdl_slash_RScan_middle'
    csv_path = join(data_dir, str(sequence_name), topic + '.csv')

    if not os.path.exists(csv_path):
        continue

    readings_dict = make_frames_from_csv(csv_path)
    # !!! sort the dict before using
    data_dict = collections.OrderedDict(sorted(readings_dict.items()))

    frames = list()
    timestamps = list()

    for timestamp, pts in data_dict.items():
        # iterate each pt
        heatmap_per_frame = list()
        for pt in pts:
            tmp = np.array(pt)
            tmp_loc = tmp[0:3]

            if topic == '_slash_mmWaveDataHdl_slash_RScan_middle':
                translated_tmp = tmp_loc + middle_transform[0:3]
            elif topic == '_slash_mmWaveDataHdl_slash_RScan_left':
                translated_tmp = left_quaternion.rotate(tmp_loc) + left_transform[0:3]
            elif topic == '_slash_mmWaveDataHdl_slash_RScan_right':
                translated_tmp = right_quaternion.rotate(tmp_loc) + right_transform[0:3]
            tmp[0] = translated_tmp[0]
            tmp[1] = translated_tmp[1]
            tmp[2] = translated_tmp[2]
            heatmap_per_frame.append(tmp[[0, 1, 2, 3, 5]])

        # do not add empty frames
        if not heatmap_per_frame:
            continue
        frames.append(np.array(heatmap_per_frame))
        timestamps.append(timestamp)


    # --------------------------------- process left and right ---------------------------------

    # process left and right
    for topic in ['_slash_mmWaveDataHdl_slash_RScan_left', '_slash_mmWaveDataHdl_slash_RScan_right']:
        csv_path = join(data_dir, str(sequence_name), topic + '.csv')

        if not os.path.exists(csv_path):
            continue

        readings_dict = make_frames_from_csv(csv_path)
        # !!! sort the dict before using
        data_dict = collections.OrderedDict(sorted(readings_dict.items()))


        for timestamp, pts in data_dict.items():

            # iterate each pt
            heatmap_per_frame = list()

            for pt in pts:
                tmp = np.array(pt)
                tmp_loc = tmp[0:3]

                if topic == '_slash_mmWaveDataHdl_slash_RScan_middle':
                    translated_tmp = tmp_loc + middle_transform[0:3]
                elif topic == '_slash_mmWaveDataHdl_slash_RScan_left':
                    translated_tmp = left_quaternion.rotate(tmp_loc) + left_transform[0:3]
                elif topic == '_slash_mmWaveDataHdl_slash_RScan_right':
                    translated_tmp = right_quaternion.rotate(tmp_loc) + right_transform[0:3]
                assert (tmp[0] != translated_tmp[0])
                tmp[0] = translated_tmp[0]
                tmp[1] = translated_tmp[1]
                tmp[2] = translated_tmp[2]
                heatmap_per_frame.append(tmp[[0, 1, 2, 3, 5]])

            # do not add empty frames
            if not heatmap_per_frame:
                continue

            # ----------------- concatenate -------------------------
            # align frames
            for i in range(0, len(timestamps)):
                if abs(int(timestamp) - int(timestamps[i])) <= align_interval:
                    frames[i] = np.concatenate((frames[i], np.array(heatmap_per_frame)))

    # ------------------------- pcl to depth -------------------------

    radar_map_dir = join(data_dir, str(sequence_name), 'depth_enzo')
    if os.path.exists(radar_map_dir):
        shutil.rmtree(radar_map_dir)
        time.sleep(5)
        os.makedirs(radar_map_dir)
    else:
        os.makedirs(radar_map_dir)

    point_world_location_dir = join(data_dir, str(sequence_name), 'point_world_location')
    if os.path.exists(point_world_location_dir):
        shutil.rmtree(point_world_location_dir)
        time.sleep(5)
        os.makedirs(point_world_location_dir)
    else:
        os.makedirs(point_world_location_dir)

    point_pixel_location_dir = join(data_dir, str(sequence_name), 'point_pixel_location_dir')
    if os.path.exists(point_pixel_location_dir):
        shutil.rmtree(point_pixel_location_dir)
        time.sleep(5)
        os.makedirs(point_pixel_location_dir)
    else:
        os.makedirs(point_pixel_location_dir)

    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))

    frame_idx = 0
    corrd_dict = dict()
    for timestamp, frame in tqdm.tqdm(zip(timestamps, frames), total=len(timestamps)):

        # only select those points with the certain range (in meters) - 5.12 meter for this TI board
        eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
        pano_img, pixel_coord, point_world_location = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                      v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

        if pano_img.size == 0:
            print('{} frame skipped as all pts are out of fov!'.format(frame_idx))
            frame_idx = frame_idx + 1
            continue

        # cv2.imshow("grid", pano_img)
        # cv2.waitKey(1)

        img_path = join(radar_map_dir, '{}.png'.format(timestamp))
        cv2.imwrite(img_path, pano_img)

        pixel_coord_file = join(point_pixel_location_dir, '{}.txt'.format(timestamp))
        with open(pixel_coord_file, 'a+') as myfile:
            for x, y, dist in pixel_coord:
                myfile.write(str(x) + " " + str(y) + ' ' + str(dist) + '\n')

        pixel_coord_file = join(point_world_location_dir, '{}.txt'.format(timestamp))
        with open(pixel_coord_file, 'a+') as myfile:
            for x, y, z in point_world_location:
                myfile.write(str(x) + " " + str(y) + ' ' + str(z) + '\n')


        frame_idx += 1
    print('In total {} images'.format(frame_idx))
