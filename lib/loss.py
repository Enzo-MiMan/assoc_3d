import torch.nn.functional as F
import torch

def triplet_loss(dst_descriptors, src_descriptors, gt_sampled_locations_dst, gt_sampled_locations_src):

    distance = torch.zeros(1).cuda()

    for i in range(gt_sampled_locations_src.size(0)):

        src_x, src_y = gt_sampled_locations_src[i, :2]
        d_src = F.normalize(src_descriptors[0, :, src_x, src_y], dim=0)

        n_cosine_distance = torch.zeros(gt_sampled_locations_src.size(0))
        for j in range(gt_sampled_locations_dst.size(0)):

            if i == j:
                dst_x, dst_y = gt_sampled_locations_dst[j, :2]
                d_dst = F.normalize(dst_descriptors[0, :, dst_x, dst_y], dim=0)
                p_cosine_distance = torch.cosine_similarity(d_dst, d_src, dim=0)

            else:
                dst_x, dst_y = gt_sampled_locations_dst[j, :2]
                d_dst = F.normalize(dst_descriptors[0, :, dst_x, dst_y], dim=0)
                n_cosine_distance[j] = torch.cosine_similarity(d_dst, d_src, dim=0)

        max_n_distance = torch.max(n_cosine_distance, dim=0)[0]
        distance = distance + (max_n_distance - p_cosine_distance)

    return torch.log(1 + torch.exp(5 * distance))


def loss(dst_descriptors, src_descriptors, gt_sampled_locations_dst, gt_sampled_locations_src):
    loss = torch.zeros(1)

    for i in range(gt_sampled_locations_src.size(0)):

        src_x, src_y = gt_sampled_locations_src[i, :2]
        d_src = src_descriptors[0, :, src_x, src_y]

        cosine_distance_negative = 0
        for j in range(gt_sampled_locations_dst.size(0)):

            if i == j:
                dst_x, dst_y = gt_sampled_locations_dst[j, :2]

                d_dst = dst_descriptors[0, :, dst_x, dst_y]
                cosine_distance_positive = torch.cosine_similarity(d_dst, d_src, dim=0)
                # eucl_distance = torch.sqrt(torch.sum((d_dst - d_src) ** 2))

            else:
                dst_x, dst_y = gt_sampled_locations_dst[j, :2]
                d_dst = dst_descriptors[0, :, dst_x, dst_y]
                cosine_distance_negative += torch.cosine_similarity(d_dst, d_src, dim=0)

        distance_negative = torch.sum(cosine_distance_negative) / i
        distance= cosine_distance_positive - distance_negative

        alpha = 1
        loss += torch.log(1 + torch.exp(alpha * distance))

    loss_mean = loss / gt_sampled_locations_src.size(0)
    return loss_mean