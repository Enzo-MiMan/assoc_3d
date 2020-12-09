import open3d as o3d
import numpy as np
import re
import copy
from os.path import join
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def draw_registration_result(source, target):
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source, target])


if __name__ == '__main__':

    threshold = 0.00001
    A = []
    B = []
    timestamps = []

    # read frames
    with open('/Users/manmi/Documents/data/2-2_circle/data/mm/timestamp_middle.txt') as f:
        content = f.readlines()
        for line in content:
            timestamp = int(line)
            timestamps.append(timestamp)
    timestamps.sort()

    # every 2 framesqtimestampsq
    for k in range(0, len(timestamps) - 2, 2):
        target_timestamp = timestamps[k]
        source_timestamp = timestamps[k + 2]

        source_path = join('/Users/manmi/Documents/data/2-2_circle/data/mm/LMR_xyz', str(source_timestamp) + '.xyz')
        target_path = join('/Users/manmi/Documents/data/2-2_circle/data/mm/LMR_xyz', str(target_timestamp) + '.xyz')

        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)

        # processed_source, outlier_index_src = source.remove_radius_outlier(nb_points=10, radius=0. 03)
        # processed_target, outlier_index_tgt = target.remove_radius_outlier(nb_points=10, radius=0.03)

        threshold = 1
        trans_init = np.array([[1, 0., 0, 0.],
                               [0, 1, 0., 0.],
                               [0., 0, 1, 0.],
                               [0., 0., 0., 1.]])

        # draw_registration_result(processed_source, processed_target)

        reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(max_iteration=20))

        # write rotation, translation
        with open('/Users/manmi/Documents/data/2-2_circle/data/mm/RT/mm_T_gap1.txt', 'a+') as myfile:
            myfile.write(str(target_timestamp) + ' ' + str(T[0, 3]) + ' ' + str(T[1, 3]) + ' ' + str(T[2, 3]) + '\n')

        with open('/Users/manmi/Documents/data/2-2_circle/data/mm/RT/mm_R_gap1.txt', 'a+') as myfile:
            myfile.write(str(target_timestamp) +
                         ' ' + str(T[0, 0]) + ' ' + str(T[0, 1]) + ' ' + str(T[0, 2]) +
                         ' ' + str(T[1, 0]) + ' ' + str(T[1, 1]) + ' ' + str(T[1, 2]) +
                         ' ' + str(T[2, 0]) + ' ' + str(T[2, 1]) + ' ' + str(T[2, 2]) + '\n')







