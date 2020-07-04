import torch
# from .quaternion_layers import *
from .qunet_parts import *

import torch.nn.functional as F

class QUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, para=1):
        super(QUNet, self).__init__()
        self.name = "QUNet"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.para = para

        self.rgb2q = torch.nn.Conv2d(n_channels, 4, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(4, 64*self.para)
        self.down1 = Down(64*self.para, 128*self.para)
        self.down2 = Down(128*self.para, 256*self.para)
        self.down3 = Down(256*self.para, 512*self.para)
        factor = 2 if bilinear else 1
        self.down4 = Down(512*self.para, 1024*self.para // factor)
        self.up1 = Up(1024*self.para, 512*self.para // factor, bilinear)
        self.up2 = Up(512*self.para, 256*self.para // factor, bilinear)
        self.up3 = Up(256*self.para, 128*self.para // factor, bilinear)
        self.up4 = Up(128*self.para, 64*self.para, bilinear)
        self.outc = OutConv(64*self.para, n_classes)

    def forward(self, x):
        x = self.rgb2q(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits