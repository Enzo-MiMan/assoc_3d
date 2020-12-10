"""
input:
    true_delta_gmapping.csv  # include fields: timestamp, x, y, z, qx, qy, qz, qw

output:
    translation, rotation    # array (frame_num, 4),  array (frame_num, 10)

"""

import numpy as np
import os
import yaml
import csv


def quaternion_to_matrix(l):
    _EPS = np.finfo(float).eps * 4.0

    q = [l[1], l[2], l[3], l[4]]
    q = np.array(q, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1]),), dtype=np.float64)



def gmapping_TR(sequence_dir):

    # ------------------------ read in csv ------------------------

    csv_path = os.path.join(sequence_dir, 'true_delta_gmapping.csv')
    with open(csv_path, 'r') as input_file:
        reader = csv.reader(input_file)
        next(reader)

        translation = list()
        rotation = list()
        for row in reader:

            # --------------------- read (x,y,z) -------------------------

            translation.append((int(row[0]), float(row[1]), float(row[2]), float(row[3])))

            # --------- read (qx,qy,qz,qw) and convert to matrix----------

            l = np.array([row[0], float(row[5]), float(row[6]), float(row[7]), float(row[4])])
            M = quaternion_to_matrix(l)       # quaternion to matix
            rotation.append((int(row[0]), M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2], M[2, 0], M[2, 1], M[2, 2]))


    # !!! sort the dict before using
    translation.sort()
    rotation.sort()
    translation = np.array(translation)
    rotation = np.array(rotation)

    return translation, rotation



    # ---------------- save translation and rotation ------------------------
    #
    # translation_path = os.path.join(data_dir, str(sequence), 'gt_translation.txt')
    # for timestamp, translation in translation_dict.items():
    #     with open(translation_path, 'a+') as file:
    #         file.write(str(timestamp) + ' ' + str(translation[0]) + ' ' + str(translation[1]) + ' ' + str(translation[2]) + '\n')
    #
    #
    # rotation_path = os.path.join(data_dir, str(sequence), 'gt_rotation.txt')
    # for timestamp, rotation in rotation_dict.items():
    #     with open(rotation_path,'a+') as file:
    #         file.write(str(timestamp)+' ' + str(rotation[0]) + ' ' + str(rotation[1]) + ' ' + str(rotation[2]) + ' '
    #                                       + str(rotation[3]) + ' ' + str(rotation[4]) + ' ' + str(rotation[5]) + ' '
    #                                       + str(rotation[6]) + ' ' + str(rotation[7]) + ' ' + str(rotation[8]) + '\n')
