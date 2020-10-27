from lib.model import UNet
# from lib.rpe import *
import numpy as np
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


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_' + filename)


def train(train_loader, model, optimizer, epoch, writer):
    losses = AverageMeter()
    trans = AverageMeter()
    rotat = AverageMeter()

    model.train()
    for i, (image_dst, image_src, gt_dst_files, gt_src_files) in enumerate(train_loader):
        print('epoch:', epoch, 'step:', i)

        image_dst = image_dst.to(device=device, dtype=torch.float32)
        image_src = image_src.to(device=device, dtype=torch.float32)

        dst_descriptor, dst_scores = model(image_dst)  # torch.Size([9, 3, 1])  torch.Size([9, 3, 3])
        src_descriptor, src_scores = model(image_src)

        batch_loss = loss_function(dst_descriptor, dst_scores, src_descriptor, src_scores, gt_dst_files, gt_src_files)

        # transformation error
        # t_error_mean, r_error_mean = transformation_error(pred_rotat, pred_trans, label_rotat, label_trans)
        print('batch_loss :', batch_loss)
        # print('translation_error_mean:', t_error_mean)
        # print('rotation_error_mean:', r_error_mean, '\n')

        # losses.update(batch_loss.item(), label_rotat.size(0))
        # trans.update(t_error_mean, label_rotat.size(0))
        # rotat.update(r_error_mean, label_rotat.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # save and plot
        # writer.add_scalar('translation_error_mean', t_error_mean)
        # writer.add_scalar('rotation_error_mean', r_error_mean)
        writer.add_scalar('batch_loss', batch_loss)


        if i % 20 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_trans_error': best_trans_error,
                'best_rotat_error': best_rotat_error,
                'optimizer': optimizer.state_dict(),
            }, False, filename='checkpoint.pth')
    writer.add_scalar('train/train_loss', losses.val, global_step=epoch)


def validate(valid_loader, model, epoch, writer):

    losses = AverageMeter()
    trans = AverageMeter()
    rotat = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (image, label_rotat, label_trans) in enumerate(valid_loader):
            print(i)
            image = image.to(device=device, dtype=torch.float32)
            label_rotat = label_rotat[1:,:,:].to(device=device, dtype=torch.float32)
            label_trans = label_trans[1:,:,:].to(device=device, dtype=torch.float32)

            # compute output
            pred_rotat, pred_trans = model(image)
            batch_loss = loss_function(pred_rotat, pred_trans, label_rotat, label_trans)

            # transformation error
            t_error_mean, r_error_mean = transformation_error(pred_rotat, pred_trans, label_rotat, label_trans)

            print('batch_loss :', batch_loss)
            print('translation_error_mean:', t_error_mean)
            print('rotation_error_mean:', r_error_mean, '\n')

            losses.update(batch_loss.item(), label_rotat.size(0))
            trans.update(t_error_mean, label_rotat.size(0))
            rotat.update(r_error_mean, label_rotat.size(0))

            writer.add_scalar('translation_error_mean', t_error_mean)
            writer.add_scalar('rotation_error_mean', r_error_mean)
            writer.add_scalar('batch_loss', batch_loss)
    writer.add_scalar('loss/valid_loss', losses.val, global_step=epoch)
    return t_error_mean, r_error_mean


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":

    # ----------------------------------- load dataset --------------------------------

    # get config
    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_data_dir = os.path.join(os.path.dirname(project_dir), 'indoor_data/2019-11-28-15-43-32')
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

        # # evaluate on validate dataset
        # trans_error, rotat_error = validate(valid_loader, model, epoch, writer)
        # is_best = (best_trans_error > trans_error) and (best_rotat_error > rotat_error)
        # best_trans_error = min(trans_error, best_trans_error)
        # best_rotat_error = min(rotat_error, best_rotat_error)
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_trans_error': best_trans_error,
        #     'best_rotat_error': best_rotat_error,
        #     'optimizer' : optimizer.state_dict(),
        #     }, is_best, filename='checkpoint.pth')

    writer.close()


