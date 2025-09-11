# src/models/Unet3D_larger.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class Unet3D_larger(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 32)
        self.pool  = nn.MaxPool3d(2)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)

        # use transpose conv without output_padding
        self.up4   = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(512, 256)

        self.up3   = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2   = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1   = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.outc  = nn.Conv3d(32, out_ch, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)                # → [in_ch, D, H, W]
        x2 = self.down2(self.pool(x1))    # → [x, D/2, H/2, W/2]
        x3 = self.down3(self.pool(x2))    # → [x, D/4, H/4, W/4]
        x4 = self.down4(self.pool(x3))    # → [x, D/8, H/8, W/8]
        x5 = self.down5(self.pool(x4))    # → [x, D/16, H/16, W/16]

        # decoder
        x = self.up4(x5)                  # → [x, D/8, H/8, W/8]
        x = self.conv4(torch.cat([x, x4], dim=1))

        x = self.up3(x)                   # → [x, D/4, H/4, W/4]
        x = self.conv3(torch.cat([x, x3], dim=1))

        x = self.up2(x)                   # → [x, D/2, H/2, W/2]
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up1(x)                   # → [x, D, H, W]
        x = self.conv1(torch.cat([x, x1], dim=1))

        out = self.outc(x)                # → [out_ch, D, H, W]
        return out

