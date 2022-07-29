import torch
import torch.nn as nn


class GConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoder=True, dropout=False, leaky_relu=False):
        super(GConvBlock, self).__init__()
        self.seq1 = nn.Sequential(
            (nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect",
                       bias=False)
             if encoder else
             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(out_channels),
        )

        self.seq2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2) if leaky_relu else nn.ReLU()
        ) if dropout else (nn.LeakyReLU(0.2) if leaky_relu else nn.ReLU())

    def forward(self, x):
        return self.seq2(self.seq1(x))


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, feature=64):
        super(Generator, self).__init__()
        self.first_down = nn.Sequential(
            nn.Conv2d(in_channels, feature, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down1 = GConvBlock(feature * 1, feature * 2, leaky_relu=True)
        self.down2 = GConvBlock(feature * 2, feature * 4, leaky_relu=True)
        self.down3 = GConvBlock(feature * 4, feature * 8, leaky_relu=True)
        self.down4 = GConvBlock(feature * 8, feature * 8, leaky_relu=True)
        self.down5 = GConvBlock(feature * 8, feature * 8, leaky_relu=True)
        self.down6 = GConvBlock(feature * 8, feature * 8, leaky_relu=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature * 8, feature * 8, 4, 2, 1),
            nn.ReLU()
        )

        self.up1 = GConvBlock(feature * 8, feature * 8,
                              encoder=False, leaky_relu=False, dropout=True)
        self.up2 = GConvBlock(feature * 8 * 2, feature * 8,
                              encoder=False, leaky_relu=False, dropout=True)
        self.up3 = GConvBlock(feature * 8 * 2, feature * 8,
                              encoder=False, leaky_relu=False, dropout=True)
        self.up4 = GConvBlock(feature * 8 * 2, feature * 8,
                              encoder=False, leaky_relu=False, dropout=False)
        self.up5 = GConvBlock(feature * 8 * 2, feature * 4,
                              encoder=False, leaky_relu=False, dropout=False)
        self.up6 = GConvBlock(feature * 4 * 2, feature * 2,
                              encoder=False, leaky_relu=False, dropout=False)
        self.up7 = GConvBlock(feature * 2 * 2, feature * 1,
                              encoder=False, leaky_relu=False, dropout=False)

        self.last_up = nn.Sequential(
            nn.ConvTranspose2d(feature * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.first_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        last = self.last_up(torch.cat([u7, d1], dim=1))

        return last
