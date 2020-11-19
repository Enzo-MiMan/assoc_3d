"""
input: (adjust sequences "file name" in the cofig.yaml)

    2019-11-28-15-43-32
        _slash_mmWaveDataHdl_slash_RScan_middle.csv
        _slash_mmWaveDataHdl_slash_RScan_left
        _slash_mmWaveDataHdl_slash_RScan_right

    2019-10-27-15-24-29
        _slash_mmWaveDataHdl_slash_RScan_middle.csv
        _slash_mmWaveDataHdl_slash_RScan_left
        _slash_mmWaveDataHdl_slash_RScan_right

    ....

output:
    2019-11-28-15-43-32
        _slash_mmWaveDataHdl_slash_RScan_middle.csv
        _slash_mmWaveDataHdl_slash_RScan_left
        _slash_mmWaveDataHdl_slash_RScan_right
        LMR_xyz
            1574955812871341906.xyz
            1574955812921587941.xyz
            .....

    2019-10-27-15-24-29
        _slash_mmWaveDataHdl_slash_RScan_middle.csv
        _slash_mmWaveDataHdl_slash_RScan_left
        _slash_mmWaveDataHdl_slash_RScan_right
        LMR_xyz
            1574955812871341906.xyz
            1574955812921587941.xyz
            .....
    .....

"""


import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import os
import yaml
from os.path import join
import collections
import math
from pyquaternion import Quaternion
from mmwave_bag import make_frames_from_csv


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
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])


def stitch_3_boards_data(data_dir, sequence):

    # get config
    project_dir = os.path.dirname(os.getcwd())
    with open(os.path.join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    middle_transform = np.array(cfg['radar']['translation_matrix']['middle'])
    left_transform = np.array(cfg['radar']['translation_matrix']['left'])
    right_transform = np.array(cfg['radar']['translation_matrix']['right'])

    left_quaternion = Quaternion(axis=[0, 0, 1], angle=math.pi/2)
    right_quaternion = Quaternion(axis=[0, 0, 1], angle=-math.pi/2)

    topic_middle = '_slash_mmWaveDataHdl_slash_RScan_middle'
    topic_left = '_slash_mmWaveDataHdl_slash_RScan_left'
    topic_right = '_slash_mmWaveDataHdl_slash_RScan_right'

    # hyper-parameters
    align_interval = 5e7


    # ------------------------ process middle ------------------------

    csv_path = join(data_dir, str(sequence), topic_middle + '.csv')

    readings_dict = make_frames_from_csv(csv_path)
    data_dict = collections.OrderedDict(sorted(readings_dict.items()))  # !!! sort the dict before using

    dict = {}
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
        csv_path = join(data_dir, str(sequence), topic + '.csv')

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

        for i, timestamp in enumerate(timestamps):
            dict[timestamp] = frames[i]

    return dict




    #
    # file = os.path.join(data_dir, str(sequence), 'LMR_xyz')
    # if os.path.exists(file):
    #     shutil.rmtree(file)
    #     time.sleep(5)
    #     os.makedirs(file)
    # else:
    #     os.makedirs(file)
    #
    # for i in range(len(timestamps)):
    #     frame = frames[i]
    #     for point in frame:
    #         with open(os.path.join(data_dir, str(sequence), 'LMR_xyz', str(timestamps[i])+'.xyz'), 'a+') as file:
    #             file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')


