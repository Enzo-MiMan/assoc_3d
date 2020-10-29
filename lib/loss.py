import numpy as np
import os
from os.path import join
import torch
import yaml



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_gt(depth_gt, timestamp, batch_i, sequence):
    gt_file = os.path.join("../indoor_data", sequence, depth_gt, timestamp[batch_i] + '.txt')
    gt = np.loadtxt(gt_file, delimiter=' ', usecols=[0, 1], dtype=np.int64)
    gt = torch.tensor(gt)
    return gt


def loss_function(dst_descriptors, dst_scores, src_descriptors, src_scores, dst_timestamp, src_timestamp, sequence):
    loss = torch.zeros(dst_descriptors.size(0))
    for batch_i in range(dst_descriptors.size(0)):

        gt_dst = read_gt('depth_gt_dst', dst_timestamp, batch_i, sequence)
        gt_src = read_gt('depth_gt_src', src_timestamp, batch_i, sequence)

        temp = torch.zeros(gt_dst.size(0)).to(device)
        score_sum = torch.zeros(1).to(device)

        for point_i in range(gt_dst.size(0)):

            d_dst = dst_descriptors[batch_i, :, gt_dst[point_i, 0], gt_dst[point_i, 1]]
            d_src = src_descriptors[batch_i, :, gt_src[point_i, 0], gt_src[point_i, 1]]

            simi_gt = torch.cosine_similarity(d_dst, d_src, dim=0)

            n1 = torch.max(torch.cosine_similarity(d_src.unsqueeze(1).unsqueeze(1), dst_descriptors[batch_i, :, :, :]))
            n2 = torch.max(torch.cosine_similarity(d_dst.unsqueeze(1).unsqueeze(1), src_descriptors[batch_i, :, :, :]))
            simi_rest = torch.max(torch.tensor([n1, n2]))

            m = torch.max(torch.tensor([torch.zeros(1), simi_rest-simi_gt]))

            score_dst = dst_scores[batch_i, :, gt_dst[point_i, 0], gt_dst[point_i, 1]]
            score_src = src_scores[batch_i, :, gt_src[point_i, 0], gt_src[point_i, 1]]

            score_sum += score_dst * score_src
            temp[point_i] = score_dst * score_src * m

        loss[batch_i] = torch.sum(temp / score_sum)
    loss_mean = torch.mean(loss)

    return loss_mean



