import cv2
import open3d as o3d
from os.path import join
from matplotlib import pyplot as pyplot
import numpy as np
import copy
import os
import yaml
import scipy
from scipy import spatial
import random
import shutil
import time
import glob
import sys
import math


THRESHOLD = 3.5
MAX_ITERATION = 100


# Kabsch Algorithm
def compute_transformation(source, target):
    # Normalization
    number = len(source)
    # the centroid of source points
    cs = np.zeros((3,1))
    # the centroid of target points
    ct = copy.deepcopy(cs)
    cs[0] = np.mean(source[:][0]); cs[1]=np.mean(source[:][1]); cs[2]=np.mean(source[:][2])
    ct[0] = np.mean(target[:][0]); cs[1]=np.mean(target[:][1]); cs[2]=np.mean(target[:][2])
    # covariance matrix
    cov = np.zeros((3, 3))
    # translate the centroids of both models to the origin of the coordinate system (0,0,0)
    # subtract from each point coordinates the coordinates of its corresponding centroid
    for i in range(number):
        sources = source[i].reshape(-1, 1)-cs
        targets = target[i].reshape(-1, 1)-ct
        cov = cov + np.dot(sources,np.transpose(targets))
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
        points.append(np.dot(R, point.reshape(-1, 1)+T))
    return points


# compute the root mean square error between source and target
def compute_rmse(source, target, R, T):
    rmse = 0
    sampled_src = []
    sampled_dst = []
    number = len(target)
    points = _transform(source, R, T)
    for i in range(number):
        error = target[i].reshape(-1, 1)-points[i]
        if math.sqrt(error[0]**2+error[1]**2+error[2]**2) < THRESHOLD:
            sampled_src.append(source[i])
            sampled_dst.append(target[i])
        rmse = rmse + math.sqrt(error[0]**2+error[1]**2+error[2]**2)
    return sampled_src, sampled_dst, rmse


def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if recolor: # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:  # transforma source to targets
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pc2array(pointcloud):
    return np.asarray(pointcloud.points)


def registration_RANSAC(source,target,source_feature,target_feature,ransac_n=4,max_iteration=100,max_validation=100):
    # the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    s = pc2array(source)  # 26
    t = pc2array(target)  # 34

    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    # create a KD tree
    tree = spatial.KDTree(tf)
    corres_stock = tree.query(sf)[1]  # dis, nearest_loc
    for i in range(max_iteration):
        # take ransac_n points randomly
        idx = [random.randint(0, s.shape[0]-1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[idx,...]
        target_point = t[corres_idx,...]
        # estimate transformation
        # use Kabsch Algorithm
        R, T = compute_transformation(source_point,target_point)
        # calculate rmse for all points
        source_point = s
        target_point = t[corres_stock,...]
        sampled_src, sampled_dst, rmse = compute_rmse(source_point,target_point,R,T)
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
    return opt_R, opt_T, sampled_src, sampled_dst



# this is to get the fpfh features, just call the library
def get_fpfh(cp):
    # cp = cp.voxel_down_sample(voxel_size=0.05)
    cp.estimate_normals()
    return cp, o3d.registration.compute_fpfh_feature(cp, o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))


if __name__ == "__main__":

    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']
    gap = cfg['radar']['gap']

    m = 0
    for sequence in all_sequences:
        print('No.{}/36,  processing sequence: {}'.format(m, sequence))

        xyz_folder = join(data_dir, sequence, 'enzo_LMR_xyz')
        if not os.path.exists(xyz_folder):
            continue

        # create the save dir
        src_ransac_dir = join(data_dir, str(sequence), 'enzo_ransac_src')
        dst_ransac_dir = join(data_dir, str(sequence), 'enzo_ransac_dst')

        if os.path.exists(src_ransac_dir):
            shutil.rmtree(src_ransac_dir)
            time.sleep(5)
            os.makedirs(src_ransac_dir)
        else:
            os.makedirs(src_ransac_dir)

        if os.path.exists(dst_ransac_dir):
            shutil.rmtree(dst_ransac_dir)
            time.sleep(5)
            os.makedirs(dst_ransac_dir)
        else:
            os.makedirs(dst_ransac_dir)

        depth_gap4_list_dst = sorted(glob.glob(join(data_dir, str(sequence), 'enzo_depth_gt_dst/*.txt')))
        depth_gap4_list_src = sorted(glob.glob(join(data_dir, str(sequence), 'enzo_depth_gt_src/*.txt')))

        for i in range(len(depth_gap4_list_dst)):

            _, file_name_dst = os.path.split(depth_gap4_list_dst[i])
            _, file_name_src = os.path.split(depth_gap4_list_src[i])
            dst_ts, _ = os.path.splitext(file_name_dst)
            src_ts, _ = os.path.splitext(file_name_src)

            # read point cloud from .xyz file
            src_xyz_file = join(data_dir, str(sequence), 'enzo_LMR_xyz', str(src_ts) +'.xyz')
            dst_xyz_file = join(data_dir, str(sequence), 'enzo_LMR_xyz', str(dst_ts) +'.xyz')

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

            with open(os.path.join(src_ransac_dir, str(src_ts) + '.txt'), 'a+') as file:
                for point in sampled_src:
                    file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')

            with open(os.path.join(dst_ransac_dir, str(dst_ts) + '.txt'), 'a+') as file:
                for point in sampled_dst:
                    file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')


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
        #
        # m = m + 1

