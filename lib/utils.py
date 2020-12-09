import numpy as np
import cv2
import torch
from collections import OrderedDict
from os.path import join
import matplotlib.pyplot as plt
import os
import shutil
import time

def pred_matches(dst_descriptors, src_descriptors, pixel_location_src):
    """
    input: the src and dst feature map(description map)
    input: the recorded src frame pixel locations
    --->> predict dst frame pixel locations
    """

    location_src = []
    location_dst = []
    similarity = []

    for i in range(len(pixel_location_src)):
        row, col = pixel_location_src[i, :]
        src_descriptor = src_descriptors[0, :, row, col]
        simi = torch.cosine_similarity(src_descriptor.unsqueeze(1).unsqueeze(1), dst_descriptors[0, :, :, :], dim=0)
        pred_dst_x = torch.argmax(simi.view(1, -1)) / dst_descriptors.size()[-1]
        pred_dst_y = torch.argmax(simi.view(1, -1)) % dst_descriptors.size()[-1]

        location_src.append([int(row), int(col)])
        location_dst.append([int(pred_dst_x), int(pred_dst_y)])
        similarity.append(torch.max(simi.view(1, -1)))

    return np.array(location_dst), np.array(location_src), similarity



def draw_predicted_matches(timestamp_dst, image_dst, image_src, location_dst, location_src, gt_locations_dst,
                           gt_locations_src, similarity, test_data_dir):
    (hA, wA) = image_src.shape[:2]
    (hB, wB) = image_dst.shape[:2]

    # sort_similarity = OrderedDict({i: similarity[i].item() for i in range(len(similarity))})
    # sort_similarity = sorted(sort_similarity.items(), key=lambda item: item[1], reverse=True)

    cr = 0
    for k, gt_point in enumerate(gt_locations_src[:, :2]):
        for index, point in enumerate(location_src):
            if point[0] != gt_point[0] or point[1] != gt_point[1]:
                continue

            elif point[1] in (location_src[np.where(point[0]==location_src[:index, 0]), 1]):
                continue

            else:
                A = location_src[index]
                B = location_dst[index]
                vis = np.ones((max(hA, hB), wA + wB + 5, 3), dtype='uint8') * 255
                vis[0:hA, 0:wA] = image_src
                vis[0:hB, wA + 5:] = image_dst

                pixel_A = (int(A[1]), int(A[0]))
                pixel_B = (int(B[1]) + wA + 5, int(B[0]))

                gt_dst_x, gt_dst_y = gt_point

                if gt_dst_x - 0 <= location_dst[index, 0] <= gt_dst_x + 0 and gt_dst_y - 0 <= location_dst[index, 1] <= gt_dst_y + 0:
                    cv2.line(vis, pixel_A, pixel_B, (0, 255, 0), 1)
                    cr += 1
                else:
                    cv2.line(vis, pixel_A, pixel_B, (0, 0, 255), 1)
                save_file = join(test_data_dir, 'predicted_correspondence', timestamp_dst + '-' + str(k) + '.png')
                cv2.imwrite(save_file, vis)
    print('correct rate: ', cr/len(gt_locations_src))
    return cr/len(gt_locations_src)



def draw_gt_matches(timestamp_dst, image_dst, image_src, gt_sampled_locations_dst, gt_sampled_locations_src, similarity):
    (hA, wA) = image_src.shape[:2]
    (hB, wB) = image_dst.shape[:2]

    sort_similarity = OrderedDict({i: similarity[i].item() for i in range(len(similarity))})
    sort_similarity = sorted(sort_similarity.items(), key=lambda item: item[1], reverse=True)

    for element in sort_similarity:
        index = element[0]
        A = gt_sampled_locations_src[index]
        B = gt_sampled_locations_dst[index]
        vis = np.ones((max(hA, hB), wA + wB + 5, 3), dtype='uint8') * 255
        vis[0:hA, 0:wA] = image_src
        vis[0:hB, wA + 5:] = image_dst

        pixel_A = (int(A[1]), int(A[0]))
        pixel_B = (int(B[1]) + wA + 5, int(B[0]))

        cv2.line(vis, pixel_A, pixel_B, (0, 255, 0), 1)

        save_file = join('/Users/manmi/Documents/GitHub/indoor_data/2019-11-28-15-43-32/correspondence_gt_sampled', timestamp_dst + '-' + str(index) + '.png')
        cv2.imwrite(save_file, vis)


def correct_rate(test_data_dir, location_src, location_dst, gt_sampled_locations_dst, gt_sampled_locations_src):

    strict_count = 0
    for i in range(len(gt_sampled_locations_src)):
        if gt_sampled_locations_dst[i, 0] == location_dst[i, 0] and gt_sampled_locations_dst[i, 1] == location_dst[i, 1]:
            strict_count += 1

    tolerant_count = 0
    for i in range(len(gt_sampled_locations_src)):
        gt_dst_x, gt_dst_y = gt_sampled_locations_dst[i, :2]
        if gt_dst_x-1 <= location_dst[i, 0] <= gt_dst_x+1 and gt_dst_y-1 <= location_dst[i, 1] <= gt_dst_y+1:
            tolerant_count += 1

    return strict_count/len(gt_sampled_locations_src), tolerant_count/len(gt_sampled_locations_src)


def read_locations(file):
    with open(file) as file:
        pixel_locations = []
        for point in file:
            row = int(point.split()[0])
            col = int(point.split()[1])
            pixel_locations.append([row, col])
    return np.array(pixel_locations)

def read_world_locations(file):
    with open(file) as file:
        world_locations = []
        for point in file:
            x = point.split()[0]
            y = point.split()[1]
            z = point.split()[2]
            world_locations.append([x, y, z])
    return np.array(world_locations)


def re_mkdir_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        time.sleep(5)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)
    return dir_path


def remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        time.sleep(5)



def compose_trajectory(transformations, plot_trajectory=True):
    full_traj = []

    # initialize the origin
    pred_transform_t_1 = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    for pred_transform_t in transformations:

        abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)

        full_traj.append(
            [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
             abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
             abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2], abs_pred_transform[2, 3]])

        pred_transform_t_1 = abs_pred_transform

    full_traj = np.array(full_traj)
    if plot_trajectory:
        show_trajectory(full_traj[:, 3], full_traj[:, 7], 'predicted_trajectory')

    # saved_filename = '/Users/manmi/Documents/data/square_data/lidar_data/trajectory_lidar.txt'
    # np.savetxt(saved_filename, full_traj, delimiter=",")


def show_trajectory(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=20)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    # traj.set_xlim(1, 5)
    # traj.set_ylim([10, 40])
    # traj.set_xticks(range(1, 5))
    # traj.set_yticks([(i*10) for i in range(1, 5)])

    plt.scatter(x, y, s=12, c='r', marker='o')
    plt.show()