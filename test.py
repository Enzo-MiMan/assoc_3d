import os
from os.path import join
import torch
from lib.dataset import Scan_Loader, Scan_Loader_NoLabel
from lib.model import U_Net
from lib.utils import pred_matches, draw_matches, draw_gt_matches, correct_rate, read_locations


def test(test_loader, model, test_data_dir):
    model.eval()
    strict_sum = 0
    tolerant_sum = 0
    for i, (timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org) in enumerate(test_loader):

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)
        dst_descriptors = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptors = model(image_src)

        # obtain all projected pixel point
        all_locations_dst_file = join(test_data_dir, 'enzo_pixel_location', timestamp_dst[0] + '.txt')
        all_locations_src_file = join(test_data_dir, 'enzo_pixel_location', timestamp_src[0] + '.txt')
        gt_all_locations_dst = read_locations(all_locations_dst_file)
        gt_all_locations_src = read_locations(all_locations_src_file)

        # obtain sampled projected pixel point (intersections)
        # sampled_locations_dst_file = join(test_data_dir, 'depth_gt_dst', timestamp_dst[0] + '.txt')
        # sampled_locations_src_file = join(test_data_dir, 'depth_gt_src', timestamp_src[0] + '.txt')
        # gt_sampled_locations_dst = read_locations(sampled_locations_dst_file)
        # gt_sampled_locations_src = read_locations(sampled_locations_src_file)

        gt_sampled_locations_dst = torch.tensor(gt_all_locations_dst).to(device=device, dtype=torch.int)
        gt_sampled_locations_src = torch.tensor(gt_all_locations_src).to(device=device, dtype=torch.int)

        # predict pixel correspondence
        location_dst, location_src, similarity = pred_matches(dst_descriptors, src_descriptors, gt_sampled_locations_src)

        # draw
        # draw_predicted_matches(timestamp_dst[0], image_dst_org.squeeze(), image_src_org.squeeze(), location_dst, location_src, gt_sampled_locations_dst, similarity)
        draw_gt_matches(timestamp_dst[0], image_dst_org.squeeze(), image_src_org.squeeze(), gt_sampled_locations_dst, gt_sampled_locations_src, similarity)

        # calculate correct rate
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

    root_dir = os.path.dirname(os.getcwd())
    data_path = join(root_dir, 'indoor_data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U_Net()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # ----------------------------------- test --------------------------------

    for test_sequence in test_sequences:

        test_data_dir = join(data_path, test_sequence)
        test_data = Scan_Loader_NoLabel(test_data_dir)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=True)

        test(test_loader, model, test_data_dir)




