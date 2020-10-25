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
import time
from os.path import join
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import yaml
import os
from pcl2depth import velo_points_2_pano
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
    for k in range(1, second_index-first_index+1):
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



def depth_gt(frame, timestamp, dir_path, cfg):

    # only select those points with the certain range (in meters) - 5.12 meter for this TI board
    eff_rows_idx = (frame[:, 0] ** 2 + frame[:, 1] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
    pano_img, point_info = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                  v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    pixel_coord_file = join(dir_path, '{}.txt'.format(timestamp))
    with open(pixel_coord_file, 'a+') as myfile:
        for row, col, dist in point_info:
            myfile.write(str(row) + " " + str(col) + ' ' + str(dist) + '\n')




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
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))


    for sequence in sequence_names:

        ts_matches = timestamp_match(data_dir, sequence, gap)
        gmap_T, gmap_R = gmapping_TR(data_dir, sequence)

        src_gt_indices = join(data_dir, str(sequence), 'depth_gt_src')
        dst_gt_indices = join(data_dir, str(sequence), 'depth_gt_dst')

        if os.path.exists(src_gt_indices):
            shutil.rmtree(src_gt_indices)
            time.sleep(5)
            os.makedirs(src_gt_indices)
        else:
            os.makedirs(src_gt_indices)

        if os.path.exists(dst_gt_indices):
            shutil.rmtree(dst_gt_indices)
            time.sleep(5)
            os.makedirs(dst_gt_indices)
        else:
            os.makedirs(dst_gt_indices)

        # ------------------------- pcl to depth -------------------------

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
                continue

            mm_src_collect = read_mm_pcl(src_mm_ts)
            mm_dst_collect = read_mm_pcl(dst_mm_ts)

            sample_src = mm_src_collect[sample_src_indices, :]
            sample_dst = mm_dst_collect[sample_dst_indices, :]

            depth_gt(sample_src, src_mm_ts, src_gt_indices, cfg)
            depth_gt(sample_dst, dst_mm_ts, dst_gt_indices, cfg)





















