import torch.nn as nn
from torch.nn import functional as F


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_batch_norm=True):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(output_channels) if use_batch_norm else None
        self.conv2d = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        nn.init.normal_(self.conv_layer.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv2d.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.2)
        x = F.leaky_relu(self.conv2d(x), 0.2)
        return x
