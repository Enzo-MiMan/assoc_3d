import numpy as np
import os
import yaml
from os.path import join
import collections
import math
from pyquaternion import Quaternion
from lib.mmwave_bag import make_frames_from_csv
from lib.utils import re_mkdir_dir, remove_dir


if __name__ == "__main__":

    # ------------------------ get config ------------------------
    project_dir = os.getcwd()
    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    with open(os.path.join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']

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

    sequence_num = 0
    all_sequence_num = len(train_sequences)
    for sequence in all_sequences:

        sequence_dir = join(data_dir, sequence)
        if not os.path.exists(sequence_dir):
            continue
        print('processing sequence: {}/{}  {}'.format(sequence_num, all_sequence_num, sequence))


        # ------------------------ process middle ------------------------

        csv_path = join(sequence_dir, topic_middle + '.csv')

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

            csv_path = join(sequence_dir, topic + '.csv')
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

                for i in range(len(timestamps)):
                    if abs(int(timestamp) - int(timestamps[i])) <= align_interval:
                        frames[i] = np.concatenate((frames[i], np.array(frame)))

        # for i, timestamp in enumerate(timestamps):
        #     dict[timestamp] = frames[i]

        save_dir = os.path.join(sequence_dir, 'enzo_LMR_xyz')
        save_dir = re_mkdir_dir(save_dir)

        old_save_dir = os.path.join(sequence_dir, 'LMR_xyz')
        remove_dir(old_save_dir)

        for i in range(len(timestamps)):
            frame = frames[i]
            for point in frame:
                with open(os.path.join(save_dir, str(timestamps[i])+'.xyz'), 'a+') as file:
                    file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')

        sequence_num += sequence_num


