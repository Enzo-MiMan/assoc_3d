import os
from os.path import join
import torch
import numpy as np
from torch import optim
import cv2
from lib.dataset import Scan_Loader, Scan_Loader_NoLabel
from lib.model import U_Net
from lib.loss import triplet_loss, d2_loss
from tensorboardX import SummaryWriter
import shutil



def pred_matches(dst_descriptors, src_descriptors, pixel_location_src):
    """
    input: the src and dst feature map(description map)
    input: the recorded src frame pixel locations
    --->> predict dst frame pixel locations
    """

    location_src = []
    location_dst = []
    for i in range(len(pixel_location_src)):
        row, col, dep = pixel_location_src[i, :]
        src_descriptor = src_descriptors[0, :, row, col]
        simi = torch.cosine_similarity(src_descriptor.unsqueeze(1).unsqueeze(1), dst_descriptors[0, :, :, :], dim=0)
        pred_dst_x = torch.argmax(simi.view(1, -1)) / dst_descriptors.size()[-1]
        pred_dst_y = torch.argmax(simi.view(1, -1)) % dst_descriptors.size()[-1]

        location_src.append([int(row), int(col)])
        location_dst.append([int(pred_dst_x), int(pred_dst_y)])

    return np.array(location_dst), np.array(location_src)


def draw_matches(timestamp_dst, image_dst, image_src, location_dst, location_src):

    (hA, wA) = image_src.shape[:2]
    (hB, wB) = image_dst.shape[:2]

    for j, (A, B) in enumerate(zip(location_src, location_dst)):
        vis = np.ones((max(hA, hB), wA + wB + 5, 3), dtype='uint8') * 255
        vis[0:hA, 0:wA] = image_src
        vis[0:hB, wA+5:] = image_dst

        pixel_A = (int(A[1]), int(A[0]))
        pixel_B = (int(B[1])+wA+5, int(B[0]))
        cv2.line(vis, pixel_A, pixel_B, (0, 255, 0), 1)

        save_file = join('/Users/manmi/Documents/GitHub/indoor_data/2019-11-28-15-43-32/gt_matched_imgs_line', timestamp_dst + '-' + str(j) + '.png')
        cv2.imwrite(save_file, vis)
        # cv2.imshow('img', vis)
        # cv2.waitKey()


def correct_rate(train_data_dir, location_src, location_dst, gt_sampled_locations_dst, gt_sampled_locations_src):
    gt_dst = gt_sampled_locations_dst[:, :2]
    count = 0
    for i in range(len(gt_sampled_locations_src)):
        if gt_sampled_locations_dst[i, 0] == location_dst[i, 0] and gt_sampled_locations_dst[i, 1] == location_dst[i, 1]:
            count += 1

    return count/len(gt_sampled_locations_src)


def read_locations(file):
    with open(file) as file:
        pixel_locations = []
        for point in file:
            row = int(point.split()[0])
            col = int(point.split()[1])
            dep = int(point.split()[2])
            pixel_locations.append([row, col, dep])
    return np.array(pixel_locations)


def train(train_loader, model, optimizer, epoch, train_data_dir, writer):


    model.train()
    for i, (timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org) in enumerate(train_loader):
        # print('epoch:', epoch, 'step:', i)

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)

        dst_descriptors = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptors = model(image_src)

        # # obtain all projected pixel point
        # all_locations_dst_file = join(train_data_dir, 'enzo_pixel_location', timestamp_dst[0] + '.txt')
        # all_locations_src_file = join(train_data_dir, 'enzo_pixel_location', timestamp_src[0] + '.txt')
        # gt_all_locations_dst = read_locations(all_locations_dst_file)
        # gt_all_locations_src = read_locations(all_locations_src_file)

        # obtain sampled projected pixel point (intersections)
        sampled_locations_dst_file = join(train_data_dir, 'depth_gt_dst', timestamp_dst[0] + '.txt')
        sampled_locations_src_file = join(train_data_dir, 'depth_gt_src', timestamp_src[0] + '.txt')
        gt_sampled_locations_dst = read_locations(sampled_locations_dst_file)
        gt_sampled_locations_src = read_locations(sampled_locations_src_file)

        gt_sampled_locations_dst = torch.tensor(gt_sampled_locations_dst).to(device=device, dtype=torch.int)
        gt_sampled_locations_src = torch.tensor(gt_sampled_locations_src).to(device=device, dtype=torch.int)

        # loss = d2_loss(dst_descriptors, src_descriptors, gt_sampled_locations_dst, gt_sampled_locations_src)
        loss = d2_loss(dst_descriptors, src_descriptors, gt_sampled_locations_dst, gt_sampled_locations_src)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


        writer.add_scalar('loss', loss)
        if i % 100 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoint.pth')

    writer.add_scalar('train/epoch_loss', loss, global_step=epoch)



if __name__ == "__main__":

    # ----------------------------------- define path --------------------------------

    root_dir = os.path.dirname(os.getcwd())
    data_path = join(root_dir, 'indoor_data')

    train_sequence = '2019-11-28-15-43-32'
    valid_sequence = '2019-11-28-15-43-32'

    train_data_dir = join(data_path, train_sequence)
    valid_data_dir = join(data_path, valid_sequence)

    # ------------------------------------ set up ------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = Scan_Loader_NoLabel(train_data_dir)
    valid_data = Scan_Loader(valid_data_dir)

    batch_size = 1
    epochs = 20
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, drop_last=True)

    model = U_Net()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location = torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])

    # --------------------------- define loss function and optimizer -------------------------

    lr_init = 0.001
    # lr_stepsize = 500
    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.8, last_epoch=3)
    writer = SummaryWriter('runs')

    # ------------------------------------ training -----------------------------------------


    for epoch in range(epochs):
        # scheduler.step()
        train(train_loader, model, optimizer, epoch, train_data_dir, writer)
        # valid(valid_loader, model, optimizer, epoch, writer, valid_sequence, valid_data_dir)
    writer.close()


