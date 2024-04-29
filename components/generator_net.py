import torch
import torch.nn as nn
from components.down_block import DownBlock
from components.up_block import UpBlock
from torch.nn import functional as F


class GeneratorNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeneratorNet, self).__init__()
        self.down_1 = DownBlock(in_channels, 64, use_batch_norm=False)
        self.down_2 = DownBlock(64, 128)
        self.down_3 = DownBlock(128, 256)
        self.down_4 = DownBlock(256, 512)
        self.down_5 = DownBlock(512, 512)
        self.down_6 = DownBlock(512, 512)
        self.down_7 = DownBlock(512, 512)
        self.down_8 = DownBlock(512, 512, use_batch_norm=False)

        self.up_1 = UpBlock(512, 512, dropout=True)
        self.up_2 = UpBlock(1024, 512, dropout=True)
        self.up_3 = UpBlock(1024, 512, dropout=True)
        self.up_4 = UpBlock(1024, 512, dropout=False)
        self.up_5 = UpBlock(1024, 256, dropout=False)
        self.up_6 = UpBlock(512, 128, dropout=False)
        self.up_7 = UpBlock(256, 64, dropout=False)

        self.output_layer = nn.ConvTranspose2d(
            128, out_channels, kernel_size=4, stride=2, padding=1
        )
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)

    def forward(self, x):
        d1 = self.down_1(x)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        d5 = self.down_5(d4)
        d6 = self.down_6(d5)
        d7 = self.down_7(d6)
        d8 = self.down_8(d7)

        u1 = self.up_1(d8)
        u2 = self.up_2(torch.cat([u1, d7], dim=1))
        u3 = self.up_3(torch.cat([u2, d6], dim=1))
        u4 = self.up_4(torch.cat([u3, d5], dim=1))
        u5 = self.up_5(torch.cat([u4, d4], dim=1))
        u6 = self.up_6(torch.cat([u5, d3], dim=1))
        u7 = self.up_7(torch.cat([u6, d2], dim=1))

        output = F.tanh(self.output_layer(torch.cat([u7, d1], dim=1)))
        return output
