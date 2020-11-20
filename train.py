import os
from os.path import join
import torch
import yaml
from torch import optim
from tensorboardX import SummaryWriter

from lib.dataset import Scan_Loader, Data_Loader
from lib.model import U_Net
from lib.loss import triplet_loss
from lib.utils import read_locations


def train(train_loader, model, optimizer, epoch, train_data_dir, writer):
    model.train()
    for i, (timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org) in enumerate(train_loader):
        # print('epoch:', epoch, 'step:', i)

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)

        dst_descriptors = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptors = model(image_src)

        # obtain ground truth with pixel point matching (intersections)
        gt_locations_dst_file = join(train_data_dir, 'enzo_depth_gt_dst', timestamp_dst[0] + '.txt')
        gt_locations_src_file = join(train_data_dir, 'enzo_depth_gt_src', timestamp_src[0] + '.txt')
        gt_locations_dst = read_locations(gt_locations_dst_file)
        gt_locations_src = read_locations(gt_locations_src_file)
        gt_sampled_locations_dst = torch.tensor(gt_locations_dst).to(device=device, dtype=torch.int)
        gt_sampled_locations_src = torch.tensor(gt_locations_src).to(device=device, dtype=torch.int)

        loss = triplet_loss(dst_descriptors, src_descriptors, gt_sampled_locations_dst, gt_sampled_locations_src)

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

    # writer.add_scalar('train/epoch_loss', loss)



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

    # --------------------- load model ------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = U_Net()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # --------------------- set up ------------------------

    root_dir = os.path.dirname(os.getcwd())
    data_path = join(root_dir, 'indoor_data')

    batch_size = 1
    epochs = 10

    lr_init = 0.001
    # lr_stepsize = 500
    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.8, last_epoch=3)
    writer = SummaryWriter('runs')


    # ------------------ training and validate --------------

    for epoch in range(epochs):
        i = 0
        # scheduler.step()

        # --------------------- train ------------------------
        for train_sequence in test_sequences:

            train_data_dir = join(data_path, train_sequence)
            if not os.path.exists(train_data_dir):
                continue

            print('training on dataset: No.{},  sequence:{}'.format(i, train_sequence))
            train_data = Data_Loader(train_data_dir)
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=True)

            train(train_loader, model, optimizer, epoch, train_data_dir, writer)
            i += 1


        # --------------------- validate ------------------------

        # for valid_sequence in valid_sequences:
        #     valid_data_dir = join(data_path, valid_sequence)
        #     valid_data = Scan_Loader(valid_data_dir)
        #     valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, drop_last=True)
        #     valid(valid_loader, model, optimizer, epoch, writer, valid_sequence, valid_data_dir)



    writer.close()




