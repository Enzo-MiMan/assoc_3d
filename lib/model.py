import sys
sys.path.append("..")
from lib.net_parts import *
import torch.nn.functional as F
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(1, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.up1 = Up(32, 16)
        self.up2 = Up(16, 8)
        self.outc = OutConv(8, 1)
        self.descrip1 = Descrip(2)
        self.descrip2 = Descrip(4)
        self.cell_ss = SpatialSoftmax(4, 16, temperature=0.1)
        self.dst_ss = SpatialSoftmax(16, 128, temperature=0.1)


    def forward(self, img):    # torch.Size([10, 1, 16, 128])
        x1 = self.inc(img)     # torch.Size([10, 8, 16, 128])
        x2 = self.down1(x1)    # torch.Size([10, 16, 8, 64])
        x3 = self.down2(x2)    # torch.Size([10, 32, 4, 32])
        x = self.up1(x3, x2)    # torch.Size([10, 16, 8, 64])
        x = self.up2(x, x1)    # torch.Size([10, 8, 16, 128])
        logits = self.outc(x)  # torch.Size([10, 1, 16, 128])

        # calculate descriptors
        x1 = self.descrip1(x2, x1)
        descriptor = self.descrip2(x3, x1)  # torch.Size([N, 56, 16, 128])
        descriptor = F.normalize(descriptor, p=2, dim=1)  # torch.Size([248])


        # calculate socres
        scores = torch.sigmoid(logits)  # torch.Size([N, 1, 16, 128])

        return descriptor, scores

