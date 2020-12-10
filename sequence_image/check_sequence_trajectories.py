"""
input:
Iterate sequence names from config.yaml:  sequence_names = cfg['radar']['all_sequences']

function:
Read ['timestamp', 'x', 'y', 'z', 'w', 'x_l', 'y_l', 'z_l'] from "true_delta_gmapping.csv" for each sequence
Then compose poses to create trajectory

output:
Show / save gmapping trajectories in directory "../project_dir/check_sequence/sequence_trajectories"
"""


import os
from os.path import join
import yaml
import numpy as np
import pandas
import matplotlib.pyplot as plt
import shutil
import time

lw = 3
linestyle_ls = [':', '-', '-.', '--', ':', '-', '-.', '--']


def show_trajectory(x, y, point_nub, title):
    annotation_nb = ('number of points = ' + str(point_nub))
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=15)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)

    plt.scatter(x, y, s=1, c='g', marker='o')
    plt.show()


def plot_slash_odom(output_gt, title, fig_path):
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'
    plt.rcParams['grid.color'] = '#C0C0C0'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 18
    gt_x, gt_y = output_gt[:, 0], output_gt[:, 1]
    plt.title(title, fontsize=15)
    plt.plot(gt_x, gt_y, linestyle=linestyle_ls[1], linewidth=lw, label='Ground_truth')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def transform44(l):
    _EPS = np.finfo(float).eps * 4.0
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array([[1.0, 0.0, 0.0, t[0]],
                         [0.0, 1.0, 0.0, t[1]],
                         [0.0, 0.0, 1.0, t[2]],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array([[1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]],
                     [q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]],
                     [q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def composing_delta_pose(gmapping_file, save_traj=True, show_traj=False):
    full_traj = []
    delta_pose = pandas.read_csv(gmapping_file)

    delta_pose = delta_pose[['timestamp', 'x', 'y', 'z', 'w', 'x_l', 'y_l', 'z_l']]
    delta_pose = np.array(delta_pose)

    # initialize the origin
    pred_transform_t_1 = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    for i in range(len(delta_pose)):
        # use the first 3 vectors to rotate the global pose if you like, otherwise just put 0,0,0
        pred_transform_t = transform44([delta_pose[i][0], delta_pose[i][1], delta_pose[i][2], delta_pose[i][3],
                                        delta_pose[i][5], delta_pose[i][6], delta_pose[i][7], delta_pose[i][4]])
        abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)
        full_traj.append(
            [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
             abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
             abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2],
             abs_pred_transform[2, 3]])
        pred_transform_t_1 = abs_pred_transform

    full_trajectory = np.array(full_traj)

    return full_trajectory


if __name__ == "__main__":

    # get config
    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']

    saved_dir = join(project_dir, 'check_sequence/sequence_trajectories')
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
        time.sleep(3)
        os.makedirs(saved_dir)
    else:
        os.makedirs(saved_dir)

    # iterate sequences
    for sequence_name in all_sequences:
        gmapping_file = join(data_dir, sequence_name, 'true_delta_gmapping.csv')
        if not os.path.exists(gmapping_file):
            continue
        full_traj = composing_delta_pose(gmapping_file, save_traj=False, show_traj=True)

        # --------------- show trajectory ------------------
        # show_trajectory(full_traj[:, 3], full_traj[:, 7], len(full_traj), sequence_name+'   Gmapping')


        # --------------- save trajectory image ------------

        saved_filename = join(saved_dir, sequence_name+'.png')
        image_title = sequence_name+'   Gmapping'
        plot_slash_odom(np.array([full_traj[:, 3], full_traj[:, 7]]).T, image_title, saved_filename)
        print('The trajectory' + sequence_name + 'has been saved')

    print('The trajectories are saved in ', join(project_dir, 'check_sequence/sequence_trajectories'))

