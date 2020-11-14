from lib.model_test import UNet
import numpy as np
import os
from os.path import join
import torch
from lib.dataset_test import Scan_Loader
import yaml


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_dir = os.path.dirname(os.getcwd())  # /data/greyostrich/not-backed-up/aims/aimsre/xxlu/assoc/workspace


if __name__ == "__main__":

    # ----------------------------------- load dataset --------------------------------

    # get config
    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    test_data_dir = os.path.join(os.path.dirname(project_dir), 'indoor_data/2019-11-28-15-43-32')
    test_data = Scan_Loader(test_data_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=True)


    # ------------------------------------ define network ------------------------------------

    model = UNet()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # ------------------------------------ testing -----------------------------------------

    model.train()
    for i, (image_dst, image_src, dst_timestamp, src_timestamp) in enumerate(test_loader):

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)

        dst_descriptor, dst_scores = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptor, src_scores = model(image_src)

        # dst_descriptor = src_descriptor.detach().numpy()
        # src_descriptor = src_descriptor.detach().numpy()
        # dst_scores = dst_descriptor.detach().numpy()
        # src_scores = src_scores.detach().numpy()

        point_dst_file = os.path.join(test_data_dir, 'point_pixel_location', dst_timestamp[0]+'.txt')
        point_src_file = os.path.join(test_data_dir, 'point_pixel_location', src_timestamp[0]+'.txt')

        dst_world_coord_file = os.path.join(test_data_dir, 'point_world_location', dst_timestamp[0] + '.txt')
        src_world_coord_file = os.path.join(test_data_dir, 'point_world_location', src_timestamp[0] + '.txt')

        point_dst = np.loadtxt(point_dst_file, delimiter=' ', usecols=[0, 1], dtype=np.int64)
        point_src = np.loadtxt(point_src_file, delimiter=' ', usecols=[0, 1], dtype=np.int64)
        dst_world_coord = np.loadtxt(dst_world_coord_file, delimiter=' ', dtype=np.float)
        src_world_coord = np.loadtxt(src_world_coord_file, delimiter=' ', dtype=np.float)

        dst_matched_point_coord = torch.zeros((point_src.shape[0]),3)
        weight = torch.zeros(point_src.shape[0])
        det = torch.eye(3).to(device)

        for point_index_src in range(point_src.shape[0]):
            scr_pixel_x, src_pixel_y = point_src[point_index_src]
            d_src = src_descriptor[0, :, scr_pixel_x, src_pixel_y]
            score_src = src_scores[0, :, scr_pixel_x, src_pixel_y]

            simi_max = torch.zeros(1)
            for point_index_dst in range(point_dst.shape[0]):
                dst_x, dst_y = point_dst[point_index_dst]
                simi = torch.cosine_similarity(d_src, dst_descriptor[0, :, dst_x, dst_y], dim=0)
                if simi > simi_max:
                    simi_max = simi
                    k = point_index_dst

            dst_pixel_x, dst_pixel_y = torch.tensor(point_dst[k])
            dst_matched_point_coord[point_index_src, :] = torch.tensor(dst_world_coord[k])

            d_dst = dst_descriptor[0, :, dst_pixel_x, dst_pixel_y]
            score_dst = dst_scores[0, :, dst_pixel_x, dst_pixel_y]

            weight[point_index_src] = (torch.cosine_similarity(d_dst, d_src, dim=0) + 1) * score_dst * score_src / 2

        q_mean_src = torch.sum(weight * torch.tensor(src_world_coord).permute(1, 0), dim=1) / torch.sum(weight)  # torch.Size([2, N_1])
        q_mean_dst = torch.sum(weight * dst_matched_point_coord.permute(1, 0), dim=1) / torch.sum(weight)  # torch.Size([2, N_1])

        AA = (torch.tensor(src_world_coord) - q_mean_src).float()  # torch.Size([N_1, 400, 2])
        BB = (dst_matched_point_coord - q_mean_dst).float()   # torch.Size([N_1, 400, 2])

        H = torch.matmul(AA.transpose(1, 0), BB)  # torch.Size([4, 2, 2])
        U, S, V = torch.svd(H)  # U = Vt = torch.Size([N_1, 400, 400])

        det[1, -1] = torch.det(torch.matmul(V, U.transpose(0, 1)))
        temp = torch.matmul(V, det)
        rotation = torch.matmul(temp, U.transpose(0, 1))  # torch.Size([4, 2, 2])
        translation = q_mean_dst.unsqueeze(1) - torch.matmul(rotation, q_mean_src.unsqueeze(1).float())

        rotation_file = os.path.join(test_data_dir, 'predicted_rotation.txt')
        with open(rotation_file, 'a+') as myfile:
                myfile.write(str(rotation[0,0].item()) + " " + str(rotation[0,1].item()) + str(rotation[0,2].item()) + ' ' +
                             str(rotation[1,0].item()) + " " + str(rotation[1,1].item()) + str(rotation[1,2].item()) + ' ' +
                             str(rotation[2,0].item()) + " " + str(rotation[2,1].item()) + str(rotation[2,2].item()) + ' ' + '\n')

        translation_file = os.path.join(test_data_dir, 'predicted_translation.txt')
        with open(translation_file, 'a+') as myfile:
                myfile.write(str(translation[0, 0].item()) + " " + str(translation[1, 0].item()) + ' ' + str(translation[2, 0].item()) + '\n')












