import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv to N to ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            #reduce channels for concatenation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            #transposed conv to upsample
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        #x1 is the decoder feature, to be upsampled and x2 is the skip connection
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return 0.1 * self.conv(x)
    
class UNet(nn.Module):
    """
    Classic U-Net. Output is residual, the same number of channels as input.
    Args:
        n_channels: input image channels
        n_classes: output channels
        bilinear: whether to use bilinear upsampling (fewer params) or transposed conv
        base_c: number of channels in first layer
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True, base_c=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        c = base_c
        self.inc = DoubleConv(n_channels, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*8)

        self.up1 = Up(c*16, c*4, bilinear)
        self.up2 = Up(c*8, c*2, bilinear)
        self.up3 = Up(c*4, c, bilinear)
        self.up4 = Up(c*2, c, bilinear)
        self.outc = OutConv(c, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x) #[B, c, H, W]
        x2 = self.down1(x1) #[B, 2c, H/2, W/2]
        x3 = self.down2(x2) #[B, 4c, H/4, W/4]
        x4 = self.down3(x3) #[B, 8c, H/8, W/8]
        x5 = self.down4(x4) #[B, 8c, H/16, W/16]

        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        out = self.outc(u4)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
