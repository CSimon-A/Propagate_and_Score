# src/models/Unet3D_smaller.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class Unet3D_smaller(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # reduced feature-map widths: 16, 32, 64
        self.down1 = DoubleConv(in_ch, 16)
        self.pool  = nn.MaxPool3d(2)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)

        # use transpose conv without output_padding
        self.up2   = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)

        self.up1   = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(32, 16)

        self.outc  = nn.Conv3d(16, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [B, in_ch, 193, 229, 193]
        x1 = self.down1(x)                        # x1: [B, 16, 193, 229, 193]
        x2 = self.down2(self.pool(x1))           # x2: [B, 32, 96, 114, 96]
        x3 = self.down3(self.pool(x2))           # x3: [B, 64, 48, 57, 48]

        x = self.up2(x3)                         # x: [B, 32, 96, 114, 96]
        x = self.conv2(torch.cat([x, x2], dim=1))# x: [B, 32, 96, 114, 96]

        x = self.up1(x)                          # x: [B, 16, 192, 228, 192]
        # pad to match x1 spatial dims (193,229,193)
        diffD = x1.size(2) - x.size(2)           # diffD = 1
        diffH = x1.size(3) - x.size(3)           # diffH = 1
        diffW = x1.size(4) - x.size(4)           # diffW = 1
        x = F.pad(x, (0, diffW, 0, diffH, 0, diffD))
        # now x: [B, 16, 193, 229, 193]

        x = self.conv1(torch.cat([x, x1], dim=1))# x: [B, 16, 193, 229, 193]
        out = self.outc(x)                       # out: [B, out_ch, 193, 229, 193]
        return out
