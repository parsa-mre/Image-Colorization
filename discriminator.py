import torch
import torch.nn as nn


class DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dropout=False):
        super(DConvBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2)
        ) if dropout else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.seq(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 4, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            DConvBlock(64, 128, stride=1),
            DConvBlock(128, 256, stride=2),
            DConvBlock(256, 512, stride=2),
            nn.Conv2d(512, 1, 4, stride=2)
        )
        
    def forward(self, x, y):
        print(x.shape)
        print(y.shape)
        tmp = torch.cat([x, y], dim=1)
        return self.seq(tmp)


