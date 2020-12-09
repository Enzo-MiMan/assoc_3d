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
import yaml
import os
from pcl2depth import filter_point, velo_points_2_pano
from timestamp_match_mm_gmapping import timestamp_match
from gmapping_R_T_from_csv import gmapping_TR
from lib.mm_3_to_1 import stitch_3_boards_data
from lib.utils import re_mkdir_dir


def read_mm_pcl(frame, dst_mm_ts):

    mm_path = join(data_dir, sequence, 'LMR_xyz', str(mm_ts) + '.xyz')
    mm_collect = o3d.io.read_point_cloud(mm_path)
    mm_collect = np.array(mm_collect.points)

    return mm_collect


def predict_last_pose(mm_collect_src, mm_collect_dst, gmap_ts_src, gmap_ts_dst, gmap_T, gmap_R):

    # find the gap between two matched frames within gmapping timestamps
    first_index = np.array(np.where(gmap_T[:, 0] == gmap_ts_dst))[0, 0]
    second_index = np.array(np.where(gmap_T[:, 0] == gmap_ts_src))[0, 0]

    # predict the pose of mm-wave destination point cloud
    for k in range(1, second_index-first_index+1):

        t = np.reshape(gmap_T[first_index + k, 1:4], (3, 1))
        R = np.reshape(gmap_R[first_index + k, 1:10], (3, 3))

        mm_pred_src = (np.matmul(R, mm_collect_dst.T) + t).T
        mm_collect_dst = mm_pred_src

    return mm_pred_src, mm_collect_src


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
    neigh = NearestNeighbors(n_neighbors=1).fit(pc2)
    distances, indices = neigh.kneighbors(pc1, return_distance=True)
    return distances.ravel(), indices.ravel()


if __name__ == '__main__':

    # ------------------------ get config ------------------------

    project_dir = os.path.dirname(os.getcwd())
    with open(os.path.join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    gap = cfg['radar']['gap']
    DISTANCE_THRESHOLD = cfg['radar']['DISTANCE_THRESHOLD']
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))

    sequeence_num = 0
    all_sequences_num = len(all_sequences)
    for sequence in all_sequences:

        if not os.path.exists(join(data_dir, str(sequence))):
            continue
        print("sequence: {}/{},   {}", sequeence_num, all_sequences_num, sequence)
        sequence_dir = join(data_dir, str(sequence))

        """stitch 3 boards point clouds"""
        frames = stitch_3_boards_data(sequence_dir)

        """ 
        find timestamp matches between gmapping and mm-wave with gap=4
        the first column is mm-wave timestamps,  the second column is gmapping timestamps
        """
        timestamp_path = join(sequence_dir, 'enzo_ts_match_gap4.txt')
        ts_matches = np.loadtxt(timestamp_path, delimiter=' ', dtype=np.int64)

        gmap_T, gmap_R = gmapping_TR(data_dir, sequence)
        src_gt_file = re_mkdir_dir(join(sequence_dir, 'enzo_depth_gt_src'))
        dst_gt_file = re_mkdir_dir(join(sequence_dir, 'enzo_depth_gt_dst'))

        # ------------------------- pcl to depth -------------------------

        for i in range(1, len(ts_matches)):

            src_mm_ts, src_gmap_ts = ts_matches[i, :]
            dst_mm_ts, dst_gmap_ts = ts_matches[i-1, :]

            # then apply R, t on source points of mm-wave to predict coordinate of destination points (last next frame)
            mm_pred_src, mm_collect_src = predict_last_pose(frames[str(src_mm_ts)], frames[str(dst_mm_ts)],
                                                            src_gmap_ts, dst_gmap_ts, gmap_T, gmap_R)

            # find the intersection between predicted point cloud and collected point cloud
            distances, indices = nearest_neighbor(mm_collect_src, mm_pred_src)
            pc_match = np.array([(i, v) for (i, v) in enumerate(indices)])

            # intersection: sample the points by the DISTANCE_THRESHOLD between two frames of mm-wave point cloud
            sample_src_indices = np.reshape(np.where(distances < DISTANCE_THRESHOLD), (-1))
            sample_dst_indices = pc_match[sample_src_indices, 1]

            if len(sample_dst_indices) < 4 or len(sample_src_indices) < 4:
                continue

            mm_collect_dst = frames[str(dst_mm_ts)]
            mm_collect_src = frames[str(src_mm_ts)]

            sample_dst = mm_collect_dst[sample_dst_indices, :]
            sample_src = mm_collect_src[sample_src_indices, :]

            # only select those points with the certain range (in meters) - 5.12 meter for this TI board
            eff_rows_idx_dst = (sample_dst[:, 0] ** 2 + sample_dst[:, 1] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']
            eff_rows_idx_src = (sample_src[:, 0] ** 2 + sample_src[:, 1] ** 2) ** 0.5 < cfg['pcl2depth']['mmwave_dist_thre']

            eff_points_dst = sample_dst[eff_rows_idx_dst & eff_rows_idx_src, :]
            eff_points_src = sample_src[eff_rows_idx_dst & eff_rows_idx_src, :]

            """ filter points based on v_fov, h_fov  """
            valid_index_dst = filter_point(eff_points_dst, v_fov, h_fov)
            valid_index_src = filter_point(eff_points_src, v_fov, h_fov)

            """ project 3D point cloud into 2D depth image """
            frame_dst = eff_points_dst[valid_index_dst & valid_index_src, :]
            frame_src = eff_points_src[valid_index_dst & valid_index_src, :]

            pano_img_dst, pixel_coor_dst = velo_points_2_pano(frame_dst, cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                                      v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)
            pano_img_src, pixel_coor_src = velo_points_2_pano(frame_src, cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                                      v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

            correspondence = np.hstack((pixel_coor_src, pixel_coor_dst))
            new_correspondence = []

            # """ When several points project on a same pixel, choose the closest one (biggest color value)"""
            for row in np.unique(correspondence[:, 0]):
                point_index = np.where(correspondence[:, 0] == row)
                for col in np.unique(correspondence[point_index, 1]):
                    indices = np.where(np.logical_and((correspondence[:, 0] == row), (correspondence[:, 1] == col)))[0]
                    if len(indices) > 1:
                        closest_point = indices[np.argmax(correspondence[indices, 2])]
                        new_correspondence.append(correspondence[closest_point].reshape(-1))
                    else:
                        new_correspondence.append(correspondence[indices].reshape(-1))

            new_correspondence = np.array(new_correspondence)

            """ save ground truth: pixel coordination """
            pixel_coord_src = join(src_gt_file, '{}.txt'.format(src_mm_ts))
            with open(pixel_coord_src, 'a+') as myfile:
                for row_src, col_src, dist_src, row_dst, col_dst, dist_dst in new_correspondence:
                    myfile.write(str(row_src) + ' ' + str(col_src) + '\n')

            pixel_coord_dst = join(dst_gt_file, '{}.txt'.format(dst_mm_ts))
            with open(pixel_coord_dst, 'a+') as myfile:
                for row_src, col_src, dist_src, row_dst, col_dst, dist_dst in new_correspondence:
                    myfile.write(str(row_dst) + ' ' + str(col_dst) + '\n')
