import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
matplotlib.use('Agg')

from lib.model import UNet
# from lib.rpe import *
import numpy as np
import os
from os.path import join
import torch
from torch import optim
from lib.dataset import Scan_Loader

import shutil
import yaml
from tensorboardX import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_function(dst_descriptor, dst_scores, src_descriptor, src_scores, gt_dst_files, gt_src_files):
    loss = torch.zeros(dst_descriptor.size(0))
    for idx_in_batch in range(dst_descriptor.size(0)):

        gt_dst = np.loadtxt(gt_dst_files[idx_in_batch], delimiter=' ', usecols=[0, 1], dtype=np.int64)
        gt_src = np.loadtxt(gt_src_files[idx_in_batch], delimiter=' ', usecols=[0, 1], dtype=np.int64)

        gt_dst = torch.tensor(gt_dst)
        gt_src = torch.tensor(gt_src)

        gt_dst = gt_dst.to(device=device)
        gt_src = gt_src.to(device=device)

        temp = torch.zeros(gt_dst.size(0)).to(device)
        score_sum = torch.zeros(1).to(device)
        for point_index in range(gt_dst.size(0)):

            dst_x, dst_y = gt_dst[point_index]
            src_x, src_y = gt_src[point_index]

            d_dst = dst_descriptor[idx_in_batch, :, dst_x, dst_y]
            d_src = src_descriptor[idx_in_batch, :, src_x, src_y]

            p = torch.cosine_similarity(d_dst, d_src, dim=0)

            n1 = torch.max(torch.cosine_similarity(d_src.unsqueeze(1).unsqueeze(1), dst_descriptor[idx_in_batch, :, :, :], dim=0))
            n2 = torch.max(torch.cosine_similarity(d_dst.unsqueeze(1).unsqueeze(1), src_descriptor[idx_in_batch, :, :, :], dim=0))
            n = torch.max(n1, n2)

            m = torch.max(torch.zeros(1).to(device), n-p)

            score_dst = dst_scores[idx_in_batch, :, dst_x, dst_y]
            score_src = src_scores[idx_in_batch, :, dst_x, dst_y]

            score_sum += score_dst * score_src
            temp[point_index] = score_dst * score_src * m

        loss[idx_in_batch] = torch.sum(temp / score_sum)
    loss_ave = torch.mean(loss)

    return loss_ave



