# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # depthwise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pointwize
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),      # 300x300x3     -> 150x150x32
            conv_dw(32, 64, 1),     # 150x150x32    -> 150x150x64
            conv_dw(64, 128, 2),    # 150x150x64    -> 75x75x128
            conv_dw(128, 128, 1),   # 75x75x128     -> 75x75x128
            conv_dw(128, 256, 2),   # 75x75x128     -> 38x38x256
            conv_dw(256, 256, 1),   # 38x38x256     -> 38x38x256
            conv_dw(256, 512, 2),   # 38x38x256     -> 19x19x512
            conv_dw(512, 512, 1),   # 19x19x512     -> 19x19x512
            conv_dw(512, 512, 1),   # 19x19x512     -> 19x19x512
            conv_dw(512, 512, 1),   # 19x19x512     -> 19x19x512
            conv_dw(512, 512, 1),   # 19x19x512     -> 19x19x512
            conv_dw(512, 512, 1),   # 19x19x512     -> 19x19x512
            conv_dw(512, 1024, 2),  # 19x19x512     -> 10x10x1024
            conv_dw(1024, 1024, 1), # 10x10x1024    -> 10x10x1024
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x