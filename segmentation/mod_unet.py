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
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size, momentum=0.8),
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
            nn.ReLU(inplace=True),
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
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        layers = [
#             nn.Conv2d(in_size, 2 * in_size, 3, 1, 1),
#             nn.BatchNorm2d(4 * out_size, momentum=0.8),
#             nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        ]
        self.up = nn.Sequential(*layers)
        for m in self.up:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, skip_input):
        x = self.up(x)
        x = torch.cat((x, skip_input), 1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = UNetDown(1, 32)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)  # , dropout=0.5)
        self.down4 = UNetDown(128, 256)  # , dropout=0.5)
        self.down5 = UNetDown(256, 512)  # , dropout=0.5)
        self.down6 = UNetDown(512, 1024)  # , dropout=0.5)
#         self.down8 = UNetDown(1024, 1024, dropout=0.5)
        #
        # self.up1 = UNetUp(1024, 1024, dropout=0.5)
#         self.up6 = UNetUp(1024, 512)  # , dropout=0.5)
        self.up5 = UNetUp(512 + 1024 // 4, 512)  # , dropout=0.5)
        self.up4 = UNetUp(256 + 512 // 4, 256)  # , dropout=0.5)
        self.up3 = UNetUp(128 + 256 // 4, 128)
        self.up2 = UNetUp(64 + 128 // 4, 64)
        self.up1 = UNetUp(32 + 64 // 4, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            # nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1, skip_1 = self.down1(x)
        d2, skip_2 = self.down2(d1)
        d3, skip_3 = self.down3(d2)
        d4, skip_4 = self.down4(d3)
        d5, skip_5 = self.down5(d4)
        _, d6 = self.down6(d5)
#         d7, _ = self.down7(d6)

        u5 = self.up5(d6, skip_5)
        u4 = self.up4(u5, skip_4)
        u3 = self.up3(u4, skip_3)
        u2 = self.up2(u3, skip_2)
        u1 = self.up1(u2, skip_1)

        return self.final(u1)