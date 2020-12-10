import open3d as o3d
from os.path import join
import numpy as np
import copy
import os
import yaml
from scipy import spatial
import random
import math
import cv2
from lib.utils import re_mkdir_dir
from lib.pcl2depth import filter_point, velo_points_2_pano

THRESHOLD = 3.5
MAX_ITERATION = 100


# Kabsch Algorithm
def compute_transformation(source, target):
    # Normalization
    number = len(source)
    # the centroid of source points
    cs = np.zeros((3, 1))
    # the centroid of target points
    ct = copy.deepcopy(cs)
    cs[0] = np.mean(source[:][0]);
    cs[1] = np.mean(source[:][1]);
    cs[2] = np.mean(source[:][2])
    ct[0] = np.mean(target[:][0]);
    cs[1] = np.mean(target[:][1]);
    cs[2] = np.mean(target[:][2])
    # covariance matrix
    cov = np.zeros((3, 3))
    # translate the centroids of both models to the origin of the coordinate system (0,0,0)
    # subtract from each point coordinates the coordinates of its corresponding centroid
    for i in range(number):
        sources = source[i].reshape(-1, 1) - cs
        targets = target[i].reshape(-1, 1) - ct
        cov = cov + np.dot(sources, np.transpose(targets))
    # SVD (singular values decomposition)
    u, w, v = np.linalg.svd(cov)
    # rotation matrix
    R = np.dot(u, np.transpose(v))
    # Transformation vector
    T = ct - np.dot(R, cs)
    return R, T


# compute the transformed points from source to target based on the R/T found in Kabsch Algorithm
def _transform(source, R, T):
    points = []
    for point in source:
        points.append(np.dot(R, point.reshape(-1, 1) + T))
    return points


# compute the root mean square error between source and target
def compute_rmse(source, target, R, T):
    rmse = 0
    sampled_src = []
    sampled_dst = []
    number = len(target)
    points = _transform(source, R, T)
    for i in range(number):
        error = target[i].reshape(-1, 1) - points[i]
        if math.sqrt(error[0] ** 2 + error[1] ** 2 + error[2] ** 2) < THRESHOLD:
            sampled_src.append(source[i])
            sampled_dst.append(target[i])
        rmse = rmse + math.sqrt(error[0] ** 2 + error[1] ** 2 + error[2] ** 2)
    return sampled_src, sampled_dst, rmse


def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if recolor:  # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:  # transforma source to targets
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def registration_RANSAC(source, target, source_feature, target_feature, ransac_n=4, max_iteration=1000):
    # the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    s = np.asarray(source.points)  # 26
    t = np.asarray(target.points)  # 34

    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    # create a KD tree
    tree = spatial.KDTree(tf)
    corres_stock = tree.query(sf)[1]  # dis, nearest_loc
    for i in range(max_iteration):
        # take ransac_n points randomly
        idx = [random.randint(0, s.shape[0] - 1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[idx, ...]
        target_point = t[corres_idx, ...]
        # estimate transformation
        # use Kabsch Algorithm
        R, T = compute_transformation(source_point, target_point)
        # calculate rmse for all points
        source_point = s
        target_point = t[corres_stock, ...]
        sampled_src, sampled_dst, rmse = compute_rmse(source_point, target_point, R, T)
        # compare rmse and optimal rmse and then store the smaller one as optimal values
        if not i:
            opt_rmse = rmse
            opt_R = R
            opt_T = T
        else:
            if rmse < opt_rmse:
                opt_rmse = rmse
                sampled_src = np.array(sampled_src)
                sampled_dst = np.array(sampled_dst)
                opt_R = R
                opt_T = T
    return opt_R, opt_T, np.array(sampled_src), np.array(sampled_dst)


# this is to get the fpfh features, just call the library
def get_fpfh(cp):
    # cp = cp.voxel_down_sample(voxel_size=0.05)
    cp.estimate_normals()
    return cp, o3d.registration.compute_fpfh_feature(cp, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))


if __name__ == "__main__":

    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']
    gap = cfg['radar']['gap']

    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))
    v_res = cfg['pcl2depth']['v_res']
    h_res = cfg['pcl2depth']['h_res']
    max_v = cfg['pcl2depth']['max_v']
    mmwave_dist_thre = cfg['pcl2depth']['mmwave_dist_thre']

    sequence_num = 0
    all_sequence_num = len(test_sequences)
    for sequence in test_sequences:

        sequence_dir = join(data_dir, sequence)
        xyz_folder = join(data_dir, sequence, 'enzo_LMR_xyz')
        if not os.path.exists(xyz_folder):
            continue
        print('sequence: {}/{},  {}'.format(sequence_num, all_sequence_num, sequence))

        # rebuild the save dir
        ransac_depth_image_src = re_mkdir_dir(join(sequence_dir, 'enzo_ransac_depth_image_src'))
        ransac_depth_image_dst = re_mkdir_dir(join(sequence_dir, 'enzo_ransac_depth_image_dst'))
        ransac_pixel_coor_pair = re_mkdir_dir(join(sequence_dir, 'enzo_ransac_pixel_coor_pair'))

        timestamps_path = join(sequence_dir, 'enzo_timestamp_matches.txt')
        mm_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

        for i in range(len(mm_timestamps)-1):

            ts_src = mm_timestamps[i+1]
            ts_dst = mm_timestamps[i]

            # read point cloud from .xyz file
            src_xyz_file = join(sequence_dir, 'enzo_LMR_xyz', str(ts_src) + '.xyz')
            dst_xyz_file = join(sequence_dir, 'enzo_LMR_xyz', str(ts_dst) + '.xyz')

            src_xyz = np.loadtxt(src_xyz_file, delimiter=' ', dtype=np.float)
            dst_xyz = np.loadtxt(dst_xyz_file, delimiter=' ', dtype=np.float)

            r1 = o3d.geometry.PointCloud()
            r2 = o3d.geometry.PointCloud()
            r1.points = o3d.utility.Vector3dVector(src_xyz)
            r2.points = o3d.utility.Vector3dVector(dst_xyz)

            # if we want to use RANSAC registration, get_fpfh features should be acquired firstly
            r1, f1 = get_fpfh(r1)
            r2, f2 = get_fpfh(r2)
            R, T, sampled_src, sampled_dst = registration_RANSAC(r1, r2, f1, f2)

            if len(sampled_src) < 4:
                print("less than 4 points")

            # # save 3D point cloud after processing by RANSAC
            # with open(os.path.join(ransac_save_src, str(ts_src) + '.txt'), 'a+') as file:
            #     for point in sampled_src:
            #         file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
            #
            # with open(os.path.join(ransac_save_dst, str(ts_dst) + '.txt'), 'a+') as file:
            #     for point in sampled_dst:
            #         file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')


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

            img_path = join(ransac_depth_image_src, '{}.png'.format(ts_src))
            cv2.imwrite(img_path, pano_img_src)

            img_path = join(ransac_depth_image_dst, '{}.png'.format(ts_dst))
            cv2.imwrite(img_path, pano_img_dst)

            pixel_coord_src = join(ransac_pixel_coor_pair, '{}.txt'.format(ts_src))
            with open(pixel_coord_src, 'a+') as myfile:
                for src_pixel, dst_pixel in zip(pixel_coor_src, pixel_coor_dst):
                    myfile.write(str(src_pixel[0]) + ' ' + str(src_pixel[1]) + ' ' +
                                 str(dst_pixel[0]) + ' ' + str(dst_pixel[1]) + '\n')

            # pixel_coord_dst = join(enzo_ransac_world, '{}.txt'.format(dst_mm_ts))
            # with open(pixel_coord_dst, 'a+') as myfile:
            #     for src_world, dst_world in zip(world_coor_src, world_coor_dst):
            #         myfile.write(str(src_world[0]) + ' ' + str(src_world[1]) + ' ' + str(src_world[2]) + ' ' +
            #                      str(dst_world[0]) + ' ' + str(dst_world[1]) + ' ' + str(dst_world[1]) + '\n')




            # ----------- draw the correspondence -----------

        #     # transformation matrix is formed by R, T based on np.hstack and np.vstack(corporate two matrices by rows)
        #     # Notice we need add the last row [0 0 0 1] to make it homogeneous
        #     transformation = np.vstack((np.hstack((np.float64(R), np.float64(T))), np.array([0,0,0,1])))
        #
        #     # draw_registrations(r1, r2, transformation, True)
        #
        #     # draw sampled point cloud
        #     pcd_src = o3d.geometry.PointCloud()
        #     pcd_dst = o3d.geometry.PointCloud()
        #     pcd_src.points = o3d.utility.Vector3dVector(sampled_src)
        #     pcd_dst.points = o3d.utility.Vector3dVector(sampled_dst)
        #     draw_registrations(pcd_src, pcd_dst, transformation, True)

        sequence_num += 1

