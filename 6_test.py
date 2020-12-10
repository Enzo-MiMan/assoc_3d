import os
from os.path import join
import torch
import yaml
import re
import open3d as o3d
import numpy as np
import copy
from lib.data_loader import Pair_Loader
from lib.model import U_Net
from lib.utils import pred_matches, draw_gt_matches, correct_rate, read_locations, \
    draw_predicted_matches, read_world_locations, compose_trajectory




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


def read_world_coordination(test_data_dir, timestamp, ):
    world_locations = list()
    file = join(test_data_dir, 'enzo_world_location', timestamp + '.txt')
    with open(file) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])
            world_locations.append((x, y, z))
    return np.array(world_locations)


def test(test_loader, model, test_data_dir):
    model.eval()
    strict_sum = 0
    tolerant_sum = 0
    sum_cr = 0
    transformations = []
    for i, (timestamp_dst, timestamp_src, image_dst, image_src) in enumerate(test_loader):

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)
        dst_descriptors = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptors = model(image_src)

        # obtain all projected pixel point
        all_locations_dst_file = join(test_data_dir, 'enzo_pixel_location', timestamp_dst[0] + '.txt')
        all_locations_src_file = join(test_data_dir, 'enzo_pixel_location', timestamp_src[0] + '.txt')
        all_locations_dst = read_locations(all_locations_dst_file)
        all_locations_src = read_locations(all_locations_src_file)

        # predict correspondence, represent by pixel location
        location_dst, location_src, similarity = pred_matches(dst_descriptors, src_descriptors, all_locations_src)

        # world location
        world_locations_dst = read_world_coordination(test_data_dir, timestamp_dst[0])
        world_locations_src = read_world_coordination(test_data_dir, timestamp_src[0])

        # sample validate predicted correspondence
        valid_index_dst = []
        valid_index_src = []
        for i in range(len(location_dst)):
            j = np.where(all_locations_dst[:, 0]==location_dst[i, 0])[0]
            for k in j:
                if all_locations_dst[k, 1] == location_dst[i, 1]:
                    valid_index_dst.append(k)
                    valid_index_src.append(i)

        valid_dst = world_locations_dst[list(set(valid_index_dst))]
        valid_src = world_locations_src[valid_index_src]

        valid_dst = np.array(valid_dst)
        valid_src = np.array(valid_src)

        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(valid_src)
        target.points = o3d.utility.Vector3dVector(valid_dst)

        threshold = 1
        trans_init = np.array([[1, 0., 0, 0.],
                               [0, 1, 0., 0.],
                               [0., 0, 1, 0.],
                               [0., 0., 0., 1.]])

        reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(max_iteration=20))

        transformation = reg_p2p.transformation
        transformations.append(transformation)

    transformations = np.array(transformations)
    compose_trajectory(transformations)



    #     # draw
    #     cr = draw_predicted_matches(timestamp_dst[0], image_dst_org.squeeze(), image_src_org.squeeze(), location_dst,
    #                            location_src, gt_locations_dst, gt_locations_src, similarity, test_data_dir)
    #     sum_cr = sum_cr + cr
    # print('average correct rate: ', sum_cr/i)
    #
    #     draw_gt_matches(timestamp_dst[0], image_dst_org.squeeze(), image_src_org.squeeze(), gt_sampled_locations_dst, gt_sampled_locations_src, similarity)
    #
    #     calculate correct rate
    #     strict_cr, tolerant_cr = correct_rate(test_data_dir, location_src, location_dst, gt_sampled_locations_dst, gt_sampled_locations_src)
    #     strict_sum += strict_cr
    #     tolerant_sum += tolerant_cr
    #
    # average_strict_cr = strict_sum / i
    # average_tolerant_cr = tolerant_sum / i
    # print("average strict correct rate = ", average_strict_cr)
    # print("average tolerant correct rate = ", average_tolerant_cr)


if __name__ == "__main__":

    # --------------------- load config ------------------------

    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']

    # ----------------------------------- set up --------------------------------

    data_root = join(os.path.dirname(project_dir), 'indoor_data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # ----------------------------------- test --------------------------------

    for test_sequence in exp_names:

        test_data_dir = join(data_root, test_sequence)
        test_data = Pair_Loader(test_data_dir)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=True)

        test(test_loader, model, test_data_dir)
        print('finished test on sequence {}'.format(test_sequence))
        break




