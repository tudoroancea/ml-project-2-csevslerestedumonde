import torch
from torchvision.transforms.functional import resize


# ================================================================================================
# UNET
# ================================================================================================


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
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


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inconv = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outconv = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outconv(x)
        torch.cuda.empty_cache()
        return logits


# ================================================================================================
# LINKNET
# ================================================================================================


class LinkNetDecoderBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinkNetDecoderBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel // 4, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channel // 4)
        self.up = torch.nn.ConvTranspose2d(
            in_channel // 4,
            in_channel // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn2 = torch.nn.BatchNorm2d(in_channel // 4)
        self.conv2 = torch.nn.Conv2d(in_channel // 4, out_channel, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.up(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        return x


class LinkNet(torch.nn.Module):
    def __init__(
        self,
        encoder,
        channels=(64, 128, 256, 512),
    ):
        super().__init__()
        assert len(channels) == 4
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.maxpool1 = encoder.maxpool
        self.encoders = torch.nn.ModuleList()
        self.encoders.append(encoder.layer1)
        self.encoders.append(encoder.layer2)
        self.encoders.append(encoder.layer3)
        self.encoders.append(encoder.layer4)

        self.decoders = torch.nn.ModuleList()
        channels = channels[::-1]
        for i in range(len(channels) - 1):
            self.decoders.append(LinkNetDecoderBlock(channels[i], channels[i + 1]))
        self.decoders.append(LinkNetDecoderBlock(channels[-1], channels[-1]))
        self.up = torch.nn.ConvTranspose2d(channels[-1], 32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        xs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool1(x)
        for enc in self.encoders:
            x = enc(x)
            xs.append(x)
        xs = xs[::-1]
        for i in range(3):
            x = self.decoders[i](x)
            if x.shape[2:] != xs[i + 1].shape[2:]:
                x = resize(x, xs[i + 1].shape[2:])
            x = x + xs[i + 1]
        x = self.decoders[3](x)

        x = self.up(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        return x
