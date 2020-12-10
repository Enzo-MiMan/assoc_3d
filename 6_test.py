import os
import open3d as o3d
from os.path import join
import torch
import numpy as np
from lib.data_loader import Pair_Loader, Ransac_Data_Loader
import yaml
from lib.model import U_Net
from lib.utils import pred_matches, draw_gt_matches, correct_rate, read_pixel_coordination, read_world_coordination, \
        draw_predicted_matches, read_world_locations, compose_trajectory, re_mkdir_dir, read_ransac_pixel_coordination


def test(test_loader, model, sequence_path, sequence, save_file, patten):
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

        if patten == "RANSAC":
            pass
            # obtain pixel coordination of sampled points by RANSAC
            ransac_pixel_coor_dst = join(sequence_path, 'enzo_ransac_pixel_coor_pair', str(timestamp_dst.item()) + '.txt')
            recorded_pixel_src, recorded_pixel_dst = read_ransac_pixel_coordination(ransac_pixel_coor_dst)
        else:
            # obtain pixel coordination of all points
            all_pixel_coor_dst = join(sequence_path, 'enzo_all_pixel', str(timestamp_dst.item()) + '.txt')
            all_pixel_coor_src = join(sequence_path, 'enzo_all_pixel', str(timestamp_src.item()) + '.txt')
            recorded_pixel_dst = read_pixel_coordination(all_pixel_coor_dst)
            recorded_pixel_src = read_pixel_coordination(all_pixel_coor_src)

        # predict correspondence, represent by pixel location
        location_dst, location_src, similarity = pred_matches(dst_descriptors, src_descriptors, recorded_pixel_src)

        # world location
        all_world_coor_dst = join(sequence_path, 'enzo_all_world', str(timestamp_dst.item()) + '.txt')
        all_world_coor_src = join(sequence_path, 'enzo_all_world', str(timestamp_src.item()) + '.txt')
        world_locations_dst = read_world_coordination(all_world_coor_dst)
        world_locations_src = read_world_coordination(all_world_coor_src)

        # sample validate predicted correspondence
        valid_index_dst = []
        valid_index_src = []
        for i in range(len(location_dst)):
            j = np.where(recorded_pixel_dst[:, 0] == location_dst[i, 0])[0]
            for k in j:
                if recorded_pixel_dst[k, 1] == location_dst[i, 1]:
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
    compose_trajectory(transformations, sequence, save_file, plot_trajectory=False, save_trajectory=True)



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
    patten = "RANSAC"

    # ----------------------------------- set up --------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # ----------------------------------- test --------------------------------

    save_dir = re_mkdir_dir(join(data_dir, 'predicted_trajectory'))

    sequence_num = 0
    all_sequence_num = len(test_sequences)
    for sequence in test_sequences:

        sequence_path = join(data_dir, sequence)
        save_file = join(save_dir, '{}.png'.format(str(sequence)))

        if not os.path.exists(sequence_path):
            continue
        print('sequence: {}/{},  {}'.format(sequence_num, all_sequence_num, sequence))

        if patten == "RANSAC":
            test_data = Ransac_Data_Loader(sequence_path)
        else:  # all points without sampling in advance
            test_data = Pair_Loader(sequence_path)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        test(test_loader, model, sequence_path, sequence, save_file, patten)
        print('finished test on sequence {}'.format(sequence))

        sequence_num += 1


