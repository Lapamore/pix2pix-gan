import torch.nn as nn
from torch.nn import functional as F


class UpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=False):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.conv2d = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout2d(0.5) if dropout else None
        nn.init.normal_(self.up_conv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv2d.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.batch_norm(self.up_conv(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(x)
        x = F.relu(self.conv2d(x))
        return x
