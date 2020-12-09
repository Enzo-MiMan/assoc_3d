import torch.nn.functional as F
import torch.nn as nn
import torch


def conv3x3_bn_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


def upsample(in_features, out_features):
    shape = out_features.shape[2:]  # h w
    # return F.upsample(in_features, size=shape, mode='bilinear', align_corners=True)
    return nn.functional.interpolate(in_features, size=shape, mode='bilinear', align_corners=True)


def concat(in_features1, in_features2):
    return torch.cat([in_features1, in_features2], dim=1)


class U_Net(nn.Module):
    def __init__(self, class_number=1, in_channels=1):
        super().__init__()
        # encoder
        self.conv1_1 = conv3x3_bn_relu(in_channels, 8)
        self.conv1_2 = conv3x3_bn_relu(8, 8)

        self.conv2_1 = conv3x3_bn_relu(8, 16)
        self.conv2_2 = conv3x3_bn_relu(16, 16)

        self.conv3_1 = conv3x3_bn_relu(16, 32)
        self.conv3_2 = conv3x3_bn_relu(32, 32)

        self.maxpool = nn.MaxPool2d(2, 2)  # only one for all

        # decoder
        self.conv4 = conv3x3_bn_relu(32, 16)
        self.conv4_1 = conv3x3_bn_relu(32, 16)
        self.conv4_2 = conv3x3_bn_relu(16, 16)

        self.conv5 = conv3x3_bn_relu(16, 8)
        self.conv5_1 = conv3x3_bn_relu(16, 8)
        self.conv5_2 = conv3x3_bn_relu(8, 8)

        self.score = nn.Conv2d(8, class_number, 1, 1)

    def forward(self, x):
        # encoder
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)

        # decoder
        up4 = upsample(conv3_2, conv2_2)
        conv4 = self.conv4(up4)
        merge4 = concat(conv4, conv2_2)
        conv4_1 = self.conv4_1(merge4)
        conv4_2 = self.conv4_2(conv4_1)

        up5 = upsample(conv4_2, conv1_2)
        conv5 = self.conv5(up5)
        merge5 = concat(conv5, conv1_2)
        conv5_1 = self.conv5_1(merge5)
        conv5_2 = self.conv5_2(conv5_1)

        score = self.score(conv5_2)

        # calculate descriptors
        up2_1 = upsample(conv2_2, conv1_2)
        merge6 = concat(up2_1, conv1_2)
        up3_1 = upsample(conv3_2, conv1_2)
        descriptors = concat(up3_1, merge6)

        return descriptors