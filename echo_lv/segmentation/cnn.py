import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batchnorm = True, padding=0):
        super().__init__()
        layers = []
        if not mid_channels:
            mid_channels = out_channels
            layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding))
            if batchnorm:
                layers.append(nn.BatchNorm2d(mid_channels, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding))
            if batchnorm:
                layers.append(nn.BatchNorm2d(mid_channels, momentum=0.8))
            layers.append(nn.ReLU(inplace=True))
        self.double_conv = nn.Sequential(*layers)
        
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batchnorm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, padding=0, batchnorm=batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, batchnorm=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, padding=0)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, padding=0, batchnorm=batchnorm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2[:,:,diffY//2:x2.size()[2] - diffY//2, diffX//2:x2.size()[3] - diffX//2], x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
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
            nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        
        layers += [
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        
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
            nn.ReLU(inplace=True),]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
            
        layers += [
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        layers = [
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


class mUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=False):
        super(mUNet, self).__init__()
        
        self.down1 = UNetDown(1, 32, dropout=dropout)
        self.down2 = UNetDown(32, 64, dropout=dropout)
        self.down3 = UNetDown(64, 128, dropout=dropout)  # , dropout=0.5)
        self.down4 = UNetDown(128, 256, dropout=dropout)  # , dropout=0.5)
        self.down5 = UNetDown(256, 512, dropout=dropout)  # , dropout=0.5)
        self.down6 = UNetDown(512, 1024, dropout=dropout)  # , dropout=0.5)
#         self.down8 = UNetDown(1024, 1024, dropout=0.5)
        #
        # self.up1 = UNetUp(1024, 1024, dropout=0.5)
#         self.up6 = UNetUp(1024, 512)  # , dropout=0.5)
        self.up5 = UNetUp(512 + 1024 // 4, 512, dropout=dropout)  # , dropout=0.5)
        self.up4 = UNetUp(256 + 512 // 4, 256, dropout=dropout)  # , dropout=0.5)
        self.up3 = UNetUp(128 + 256 // 4, 128, dropout=dropout)
        self.up2 = UNetUp(64 + 128 // 4, 64, dropout=dropout)
        self.up1 = UNetUp(32 + 64 // 4, 32, dropout=dropout)

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
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, batchnorm=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, padding=0, batchnorm=batchnorm)
        self.down1 = Down(64, 128, batchnorm=batchnorm)
        self.down2 = Down(128, 256, batchnorm=batchnorm)
        self.down3 = Down(256, 512, batchnorm=batchnorm)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, batchnorm=batchnorm)
        self.up1 = Up(1024, 512 // factor, bilinear, batchnorm=batchnorm)
        self.up2 = Up(512, 256 // factor, bilinear, batchnorm=batchnorm)
        self.up3 = Up(256, 128 // factor, bilinear, batchnorm=batchnorm)
        self.up4 = Up(128, 64, bilinear, batchnorm=batchnorm)
        self.outc = OutConv(64, n_classes)
        

    def forward(self, x):
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