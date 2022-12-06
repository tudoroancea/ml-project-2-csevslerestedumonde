# The aim of this script is to define the model used for the segmentation task

# ================================================================================================
# IMPORTS
# ================================================================================================
import torch
import torch.nn as nn


# ================================================================================================
# MODULES
# ================================================================================================

# ==========================================
# DOUBLE CONVOLUTION
# ==========================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


# ==========================================
# GO DOWN IN U-NET BY 1 STAGE
# ==========================================
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# ==========================================
# GO UP IN U-NET BY 1 STAGE
# ==========================================
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        del x1
        del x2
        torch.cuda.empty_cache()
        return self.conv(x)


# ==========================================
# OUTPUT CONVOLUTION
# ==========================================
class OutConv(nn.Module):
    """Output convolution"""

    def __init__(self, in_channels, out_channels, sigmoid: bool = True):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid() if sigmoid else None

    def forward(self, x):
        if self.sigmoid:
            return self.sigmoid(self.conv(x))
        else:
            return self.conv(x)
        # return self.sigmoid(self.conv(x))


# ================================================================================================
# MODELS
# ================================================================================================

# ==========================================
# U-NET MODEL
# ==========================================
class UNet(nn.Module):
    """Create a U-Net model"""

    def __init__(self, n_channels: int = 3, n_classes: int = 1, sigmoid: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes, sigmoid=sigmoid)

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
        probas = self.outc(x)
        torch.cuda.empty_cache()
        return probas
