import numpy as np
import os
from os.path import join
from sklearn.neighbors import NearestNeighbors
import yaml
from lib.utils import re_mkdir_dir, extract_frames, data_prepare
from lib.pcl2depth import filter_point, velo_points_2_pano


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

    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    gap = cfg['radar']['gap']
    DISTANCE_THRESHOLD = cfg['radar']['DISTANCE_THRESHOLD']
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))
    v_res = cfg['pcl2depth']['v_res']
    h_res = cfg['pcl2depth']['h_res']
    max_v = cfg['pcl2depth']['max_v']
    mmwave_dist_thre = cfg['pcl2depth']['mmwave_dist_thre']

    sequeence_num = 0
    all_sequences_num = len(all_sequences)
    for sequence in all_sequences:

        if not os.path.exists(join(data_dir, str(sequence))):
            continue
        print("sequence: {}/{},   {}".format(sequeence_num, all_sequences_num, sequence))
        sequence_dir = join(data_dir, str(sequence))

        frames, ts_matches, gmap_T, gmap_R = data_prepare(sequence_dir, gap=4, save_matches=True)

        enzo_gt_pixel = re_mkdir_dir(join(sequence_dir, 'enzo_gt_pixel'))
        enzo_gt_world = re_mkdir_dir(join(sequence_dir, 'enzo_gt_world'))

        for i in range(1, len(ts_matches)):

            src_mm_ts, src_gmap_ts = ts_matches[i, :]
            dst_mm_ts, dst_gmap_ts = ts_matches[i-1, :]

            mm_collect_dst = frames[str(dst_mm_ts)]
            mm_collect_src = frames[str(src_mm_ts)]

            # --------------------- find intersections -------------------
            """apply R, t on source points of mm-wave to predict coordinate of destination points (last next frame)"""
            mm_pred_src, mm_collect_src = predict_last_pose(mm_collect_src, mm_collect_dst,
                                                            src_gmap_ts, dst_gmap_ts, gmap_T, gmap_R)

            # find the intersection between predicted point cloud and collected point cloud
            distances, indices = nearest_neighbor(mm_collect_src, mm_pred_src)
            pc_match = np.array([(i, v) for (i, v) in enumerate(indices)])

            """sample the points by the DISTANCE_THRESHOLD between two frames of mm-wave point cloud"""
            sampled_src_indices = np.reshape(np.where(distances < DISTANCE_THRESHOLD), (-1))
            sampled_dst_indices = pc_match[sampled_src_indices, 1]

            if len(sampled_dst_indices) < 4 or len(sampled_src_indices) < 4:
                continue

            sampled_dst = mm_collect_dst[sampled_dst_indices, :]
            sampled_src = mm_collect_src[sampled_src_indices, :]

            # --------------------- filter effective points -------------------
            """only select those points with the certain range (in meters) - 5.12 meter for this TI board"""
            eff_rows_idx_dst = (sampled_dst[:, 0] ** 2 + sampled_dst[:, 1] ** 2) ** 0.5 < mmwave_dist_thre
            eff_rows_idx_src = (sampled_src[:, 0] ** 2 + sampled_src[:, 1] ** 2) ** 0.5 < mmwave_dist_thre
            eff_points_dst = sampled_dst[eff_rows_idx_dst & eff_rows_idx_src, :]
            eff_points_src = sampled_src[eff_rows_idx_dst & eff_rows_idx_src, :]

            """ filter points based on v_fov, h_fov  """
            valid_index_dst = filter_point(eff_points_dst, v_fov, h_fov)
            valid_index_src = filter_point(eff_points_src, v_fov, h_fov)
            frame_dst = eff_points_dst[valid_index_dst & valid_index_src, :]
            frame_src = eff_points_src[valid_index_dst & valid_index_src, :]

            # ------------------------- pcl to depth -------------------------
            pano_img_dst, pixel_coor_dst, world_coor_dst = velo_points_2_pano(frame_dst, v_res, h_res,
                                                                              v_fov, h_fov, max_v, depth=True)
            pano_img_src, pixel_coor_src, world_coor_src = velo_points_2_pano(frame_src, v_res, h_res,
                                                                              v_fov, h_fov, max_v, depth=True)

            # ------------------ save ground truth: pixel and world coordination --------------------

            pixel_coord_src = join(enzo_gt_pixel, '{}.txt'.format(src_mm_ts))
            with open(pixel_coord_src, 'a+') as myfile:
                for src_pixel, dst_pixel in zip(pixel_coor_src, pixel_coor_dst):
                    myfile.write(str(src_pixel[0]) + ' ' + str(src_pixel[1]) + ' ' +
                                 str(dst_pixel[0]) + ' ' + str(dst_pixel[1]) + '\n')

            pixel_coord_dst = join(enzo_gt_world, '{}.txt'.format(dst_mm_ts))
            with open(pixel_coord_dst, 'a+') as myfile:
                for src_world, dst_world in zip(world_coor_src, world_coor_dst):
                    myfile.write(str(src_world[0]) + ' ' + str(src_world[1]) + ' ' + str(src_world[2]) + ' ' +
                                 str(dst_world[0]) + ' ' + str(dst_world[1]) + ' ' + str(dst_world[1]) + '\n')

        sequeence_num += 1