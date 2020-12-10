import os
from os.path import join
import torch
import yaml
import shutil
import time

from lib.dataset import Single_Loader, Pair_Loader
from lib.model import U_Net
from lib.utils import pred_matches, draw_gt_matches, correct_rate, read_locations, draw_predicted_matches


def test(test_loader, model, test_data_dir):
    model.eval()
    strict_sum = 0
    tolerant_sum = 0
    sum_cr = 0
    data = {}
    for i, (timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org) in enumerate(test_loader):
        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)
        dst_score, dst_descriptors = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_score, src_descriptors = model(image_src)

        # obtain all projected pixel point
        all_locations_dst_file = join(test_data_dir, 'enzo_pixel_location', timestamp_dst[0] + '.txt')
        all_locations_src_file = join(test_data_dir, 'enzo_pixel_location', timestamp_src[0] + '.txt')
        all_locations_dst = read_locations(all_locations_dst_file)
        all_locations_src = read_locations(all_locations_src_file)

        dst_score = dst_score.squeeze()[all_locations_dst[:, 0], all_locations_dst[:, 1]]
        src_score = src_score.squeeze()[all_locations_src[:, 0], all_locations_src[:, 1]]
        dst_descriptors = dst_descriptors.squeeze()[:, all_locations_dst[:, 0], all_locations_dst[:, 1]]
        src_descriptors = src_descriptors.squeeze()[:, all_locations_src[:, 0], all_locations_src[:, 1]]

        data['keypoints0'] = all_locations_dst
        data['scores0'] = dst_score
        data['descriptors0'] = dst_descriptors
        data['keypoints1'] = all_locations_src
        data['scores1'] = src_score
        data['descriptors1'] = src_descriptors







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

    root_dir = os.path.dirname(os.getcwd())
    data_root = join(root_dir, 'indoor_data')

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




