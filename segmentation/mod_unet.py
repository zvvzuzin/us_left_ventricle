""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.double_conv(x)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=None):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_size, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        
        # if dropout:
        #    layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        layers = [        
            nn.Conv2d(out_size, out_size, 4, 2, 1, bias=True),
            nn.BatchNorm2d(out_size, momentum=0.8),
            layers.append(nn.ReLU(inplace=True))
        ]
        self.down = nn.Sequential(*layers)
        for m in self.down:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        skip = self.block(x)
        return self.down(skip), skip



class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, 4 * out_size, 3, 1, 1),
            nn.BatchNorm2d(4 * out_size, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        for m in self.model:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, skip_input):
        x = torch.cat((x, skip_input), 1)
        x = self.model(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = UNetDown(1, 32)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)  # , dropout=0.5)
        self.down4 = UNetDown(256, 512)  # , dropout=0.5)
        self.down5 = UNetDown(512, 1024)  # , dropout=0.5)
        self.down6 = UNetDown(1024, 1024)  # , dropout=0.5)
#         self.down8 = UNetDown(1024, 1024, dropout=0.5)
        #
        # self.up1 = UNetUp(1024, 1024, dropout=0.5)
        self.up6 = UNetUp(1024, 1024)  # , dropout=0.5)
        self.up5 = UNetUp(2*1024, 1024)  # , dropout=0.5)
        self.up4 = UNetUp(2*1024, 1024)  # , dropout=0.5)
        self.up3 = UNetUp(2*1024, 256)
        self.up2 = UNetUp(512, 128)
        self.up1 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Conv2d(128, 4 * out_channels, 3, 1, 1),
            nn.PixelShuffle(2),
            # nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            # nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u2, d5)
        u3 = self.up3(u3, d4)
        u4 = self.up4(u4, d3)
        u5 = self.up5(u5, d2)
        u6 = self.up6(u6, d1)

        return self.final(u6)