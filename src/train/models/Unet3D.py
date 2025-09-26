# train/models/Unet3D.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


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

class Unet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Encoder
        self.pool  = nn.MaxPool3d(2)
        self.down1 = DoubleConv(in_ch, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)
        self.down6 = DoubleConv(1024, 2048)

        # Decoder
        self.up5   = nn.ConvTranspose3d(2048, 1024, kernel_size=2, stride=2)
        self.skip5 = nn.Conv3d(1024, 1024, kernel_size=1)
        self.conv5 = DoubleConv(1024 + 1024, 1024)

        self.up4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.skip4 = nn.Conv3d(512, 512, kernel_size=1)
        self.conv4 = DoubleConv(512 + 512, 512)

        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.skip3 = nn.Conv3d(256, 256, kernel_size=1)
        self.conv3 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.skip2 = nn.Conv3d(128, 128, kernel_size=1)
        self.conv2 = DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.skip1 = nn.Conv3d(64, 64, kernel_size=1)
        self.conv1 = DoubleConv(64 + 64, 64)

        self.outc = nn.Conv3d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = cp.checkpoint(self.down1, x, use_reentrant=False)
        # x1 = self.down1(x)
        x2 = cp.checkpoint(self.down2, self.pool(x1), use_reentrant=False)
        # x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))
        x6 = self.down6(self.pool(x5))

        # Decoder
        x = self.up5(x6)
        x = self.conv5(torch.cat([x, self.skip5(x5)], dim=1))

        x = self.up4(x)
        x = self.conv4(torch.cat([x, self.skip4(x4)], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, self.skip3(x3)], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, self.skip2(x2)], dim=1))

        x = self.up1(x)
        x = self.conv1(torch.cat([x, self.skip1(x1)], dim=1))

        out = self.outc(x)
        out = F.elu(out, alpha=1.0)
        return out
