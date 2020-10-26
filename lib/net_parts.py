import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.single_conv(x1)
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Descrip(nn.Module):
    def __init__(self, factor):
        super(Descrip, self).__init__()
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return x


class SpatialSoftmax(nn.Module):
    """
    N, C, W, H => C*2 # for each channel, get good point [x,y]
    """

    def __init__(self, n_rows, n_cols, temperature):
        super(SpatialSoftmax, self).__init__()

        x_map = np.zeros((n_rows, n_cols))
        y_map = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                x_map[i, j] = i
                y_map[i, j] = j
        x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).to(device)
        y_map = torch.from_numpy(np.array(y_map.reshape((-1)), np.float32)).to(device)
        self.temperature = temperature
        self.x_map = x_map
        self.y_map = y_map

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # batch, C, W*H
        s = F.softmax(x / self.temperature, dim=2)  # batch, C, W*H
        fp_x = torch.matmul(s, self.x_map)  # batch, C
        fp_y = torch.matmul(s, self.y_map)  # batch, C
        s = torch.cat((fp_x, fp_y), 1)

        return s  # batch, C*2