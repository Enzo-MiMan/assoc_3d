import os
from os.path import join
import torch
import yaml
from torch import optim
from tensorboardX import SummaryWriter
from lib.data_loader import Pair_Loader
from lib.model import U_Net
from lib.loss import triplet_loss
from lib.utils import read_locations


def train(train_loader, model, optimizer, epoch, train_data_dir, writer):
    model.train()
    for i, (timestamp_dst, timestamp_src, image_dst, image_src) in enumerate(train_loader):
        # print('epoch:', epoch, 'step:', i)

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)

        dst_descriptors = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptors = model(image_src)

        # obtain ground truth with pixel point matching (intersections)
        gt_locations_dst_file = join(train_data_dir, 'enzo_depth_gt_dst', str(timestamp_dst[0]) + '.txt')
        gt_locations_src_file = join(train_data_dir, 'enzo_depth_gt_src', str(timestamp_src[0])+ '.txt')
        gt_locations_dst = read_locations(gt_locations_dst_file)
        gt_locations_src = read_locations(gt_locations_src_file)
        gt_sampled_locations_dst = torch.tensor(gt_locations_dst).to(device=device, dtype=torch.long)
        gt_sampled_locations_src = torch.tensor(gt_locations_src).to(device=device, dtype=torch.long)

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
    print('Device is {}'.format(device))
    model = U_Net()
    model.to(device=device)
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])

    # --------------------- set up ------------------------

    data_path = join(os.path.dirname(project_dir), 'indoor_data')
    batch_size = 1
    epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
    writer = SummaryWriter('runs')

    # ------------------ train  --------------------

    for epoch in range(epochs):
        scheduler_step.step()

        sequence_num = 0
        all_sequence_num = len(train_sequences)
        for sequence in train_sequences:

            sequence_path = join(data_path, sequence)
            if not os.path.exists(sequence_path):
                continue

            print('\n\n\nepoch: {}/{}'.format(epoch, epochs))
            print('sequence: {}/{},  {}'.format(sequence_num, all_sequence_num, sequence))

            train_data = Pair_Loader(sequence_path, patten='intersection')
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)

            train(train_loader, model, optimizer, epoch, sequence_path, writer)
            sequence_num += 1

    writer.close()




