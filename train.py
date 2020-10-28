from lib.model import UNet
import os
from os.path import join
import torch
from torch import optim
from lib.dataset import Scan_Loader
from lib.loss import loss_function
import shutil
import yaml
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_dir = os.path.dirname(os.getcwd())  # /data/greyostrich/not-backed-up/aims/aimsre/xxlu/assoc/workspace


def save_checkpoint(state, filename):
    torch.save(state, filename)
    shutil.copyfile(filename, 'model_best_' + filename)


def train(train_loader, model, optimizer, epoch, writer):

    model.train()
    for i, (image_dst, image_src, dst_timestamp, src_timestamp) in enumerate(train_loader):
        print('epoch:', epoch, 'step:', i)

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)

        dst_descriptors, dst_scores = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptors, src_scores = model(image_src)

        batch_loss = loss_function(dst_descriptors, dst_scores, src_descriptors, src_scores, dst_timestamp, src_timestamp)
        print('batch_loss :', batch_loss)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # write tensorboardX
        writer.add_scalar('batch_loss', batch_loss)

        if i % 20 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, filename='checkpoint.pth')


if __name__ == "__main__":

    # ----------------------------------- load dataset --------------------------------

    # get config
    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_data_dir = os.path.join(os.path.dirname(project_dir), 'indoor_data/2019-10-27-14-28-21')
    valid_data_dir = os.path.join(os.path.dirname(project_dir), 'indoor_data/2019-11-28-15-43-32')
    train_data = Scan_Loader(train_data_dir)
    valid_data = Scan_Loader(valid_data_dir)

    batch_size = 10
    epochs = 20
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, drop_last=True)


    # ------------------------------------ define network ------------------------------------

    model = UNet()
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

    best_trans_error = float('inf')
    best_rotat_error = float('inf')
    for epoch in range(epochs):
        # scheduler.step()
        train(train_loader, model, optimizer, epoch, writer)
    writer.close()


