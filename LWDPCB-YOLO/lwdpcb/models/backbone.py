"""YOLO26-Nano-style backbone with wavelet-pool downsampling.

The reference architecture follows the nano configuration (width multiplier
0.25, depth multiplier 0.34) and replaces every strided CBS downsampling by a
:class:`WaveletPool` layer.  Stage outputs at strides 8, 16, 32 are returned
for the feature pyramid.
"""
import torch
import torch.nn as nn

from .common import ConvBNSiLU, CSPStage, WaveletPool


class Backbone(nn.Module):
    def __init__(self, wavelet=True):
        super().__init__()
        self.wavelet = wavelet
        # channel plan (nano)
        c0, c1, c2, c3, c4 = 16, 32, 64, 128, 256
        self.stem = ConvBNSiLU(3, c0, k=3, s=1)

        def down(c_in, c_out):
            return WaveletPool(c_in, c_out) if wavelet else ConvBNSiLU(c_in, c_out, k=3, s=2)

        # stride 2
        self.d1 = down(c0, c1)
        self.b1 = CSPStage(c1, c1, n=1)
        # stride 4
        self.d2 = down(c1, c2)
        self.b2 = CSPStage(c2, c2, n=2)
        # stride 8
        self.d3 = down(c2, c3)
        self.b3 = CSPStage(c3, c3, n=2)
        # stride 16
        self.d4 = down(c3, c3)
        self.b4 = CSPStage(c3, c3, n=1)
        # stride 32
        self.d5 = down(c3, c4)
        self.b5 = CSPStage(c4, c4, n=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(self.d1(x))      # /2
        x = self.b2(self.d2(x))      # /4
        p3 = self.b3(self.d3(x))     # /8
        p4 = self.b4(self.d4(p3))    # /16
        p5 = self.b5(self.d5(p4))    # /32
        return p3, p4, p5
