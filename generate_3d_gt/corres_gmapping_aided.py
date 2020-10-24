"""
input:
    1. timestamp matches between mm and gmapping
    2. gmapping:  gmapping_T,  gmapping_R_matrix
    3. LMR point cloud: overlay left, middle and right mm-wave point clouds

parameter:
    gap = 4   # read from config.yaml

output:
    source point cloud:  mm_src_gt_3d.txt
    matched destination point cloud:  mm_dst_gt_3d.txt

"""

import open3d as o3d
import numpy as np
import shutil
from os.path import join
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import yaml
import os
from timestamp_match_mm_gmapping import timestamp_match
from gmapping_R_T_from_csv import gmapping_TR


def draw_matplotlib(src, dst):
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(src[:, 0], src[:, 1], src[:, 2], s=10, c='r', marker='o')
    ax.scatter(dst[:, 0], dst[:, 1], dst[:, 2], s=10, c='b', marker='o')
    plt.show()


def draw_o3d(src, dst):
    pcd_src = o3d.geometry.PointCloud()
    pcd_dst = o3d.geometry.PointCloud()

    pcd_src.points = o3d.utility.Vector3dVector(src)
    pcd_dst.points = o3d.utility.Vector3dVector(dst)

    pcd_src.paint_uniform_color([1, 0.706, 0])
    pcd_dst.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd_src, pcd_dst])



def read_mm_pcl(mm_ts):

    # read the mm-wave radar point clouds from two frames which have been matched with gmapping
    mm_path = join(data_dir, sequence, 'LMR_xyz', str(mm_ts) + '.xyz')
    mm_collect = o3d.io.read_point_cloud(mm_path)
    mm_collect = np.array(mm_collect.points)

    return mm_collect


def predict_next_pose(src_mm_ts, dst_mm_ts, src_gmap_ts, dst_gmap_ts, gmap_T, gmap_R):

    mm_src_collect = read_mm_pcl(src_mm_ts)
    mm_dst_collect = read_mm_pcl(dst_mm_ts)

    # find the gap between two matched frames within gmapping timestamps
    first_index = np.array(np.where(gmap_T[:, 0] == dst_gmap_ts))[0, 0]
    second_index = np.array(np.where(gmap_T[:, 0] == src_gmap_ts))[0, 0]

    # predict the pose of mm-wave destination point cloud
    for k in range(second_index - first_index):
        t = np.reshape(gmap_T[first_index + k, 1:4], (3, 1))
        R = np.reshape(gmap_R[first_index + k, 1:10], (3, 3))

        mm_src_pred = (np.matmul(R, mm_dst_collect.T) + t).T
        mm_dst_collect = mm_src_pred

    return mm_src_pred, mm_src_collect




def nearest_neighbor(pc1, pc2):
    '''
    Find the nearest (Euclidean) neighbor in pc2 for each point in pc1
    Input:
        pc1: Nxm array of points
        pc2: Kxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: pc2 indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pc2)
    distances, indices = neigh.kneighbors(pc1, return_distance=True)
    return distances.ravel(), indices.ravel()


def save_assoc_mmpcl(sample_src_indices, sample_dst_indices, src_mm_ts, dst_mm_ts):

        assert len(sample_src_indices) == len(sample_dst_indices)

        mm_src_collect = read_mm_pcl(src_mm_ts)
        mm_dst_collect = read_mm_pcl(dst_mm_ts)

        sample_src = mm_src_collect[sample_src_indices, :]
        sample_dst = mm_dst_collect[sample_dst_indices, :]

        # draw_matplotlib(sample_src_collect, sample_dst_collect)
        # draw_o3d(mm_dst_calcul, mm_dst_collect)

        l1 = ''
        l2 = ''
        for point in sample_src:
            l1 = (l1 + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ')

        for point in sample_dst:
            l2 = (l2 + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ')

        with open(join(data_dir, str(sequence), 'mm_src_gt_3d.txt'), 'a+') as myfile:
            myfile.write(str(src_mm_ts) + ' ' + l1 + '\n')

        with open(join(data_dir, str(sequence), 'mm_dts_gt_3d.txt'), 'a+') as myfile:
            myfile.write(str(dst_mm_ts) + ' ' + l2 + '\n')




if __name__ == '__main__':

    # ------------------------ get config ------------------------

    project_dir = os.path.dirname(os.getcwd())
    with open(os.path.join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = cfg['base_conf']['data_base']
    exp_names = cfg['radar']['exp_name']
    sequence_names = cfg['radar']['all_sequences']
    gap = cfg['radar']['gap']
    DISTANCE_THRESHOLD = cfg['radar']['DISTANCE_THRESHOLD']


    for sequence in sequence_names:

        file_src = join(data_dir, str(sequence), 'mm_src_gt_3d.txt')
        if os.path.exists(file_src):
            os.remove(file_src)

        file_dst = join(data_dir, str(sequence), 'mm_dts_gt_3d.txt')
        if os.path.exists(file_dst):
            os.remove(file_dst)

        ts_matches = timestamp_match(data_dir, sequence, gap)
        gmap_T, gmap_R = gmapping_TR(data_dir, sequence)

        num_match_pc = []
        for i in range(1, len(ts_matches)):

            src_mm_ts, src_gmap_ts = ts_matches[i, :]
            dst_mm_ts, dst_gmap_ts = ts_matches[i-1, :]

            """
            compose R, t within consecutive gap frames from gmapping 
            then apply R, t on source point cloud of mm-wave to predict the pose of point cloud in next frame
            """
            mm_src_pred, mm_src_collect = predict_next_pose(src_mm_ts, dst_mm_ts, src_gmap_ts, dst_gmap_ts, gmap_T, gmap_R)

            # find the intersection between predicted point cloud and collected point cloud
            distances, indices = nearest_neighbor(mm_src_pred, mm_src_collect)
            pc_match = np.array([(i, v) for (i, v) in enumerate(indices)])

            # intersection: sample the points by the DISTANCE_THRESHOLD between two frames of mm-wave point cloud
            sample_dst_indices = np.reshape(np.where(distances < DISTANCE_THRESHOLD), (-1))
            sample_src_indices = pc_match[sample_dst_indices, 1]

            if len(sample_dst_indices) < 3 or len(sample_src_indices) < 3:
                print('in sequence:', sequence, ', src_mm_ts:', src_mm_ts, 'match with dst_mm_ts:', dst_mm_ts, ', the intersections are less than 3.')
                continue

            # save the intersection points separately (correspondence point cloud)
            save_assoc_mmpcl(sample_src_indices, sample_dst_indices, src_mm_ts, dst_mm_ts)

        print('finished the', sequence, 'processing')

