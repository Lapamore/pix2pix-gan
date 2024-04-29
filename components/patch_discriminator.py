import torch.nn as nn
from components.down_block import DownBlock

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(PatchDiscriminator, self).__init__()
        self.conv_1 = DownBlock(3, 64, use_batch_norm=False)
        self.conv_2 = DownBlock(64, 128)
        self.conv_3 = DownBlock(128, 256)
        self.conv_4 = DownBlock(256, 512)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.final_layer = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.zero_pad(x)
        x = self.final_layer(x)
        return x
    