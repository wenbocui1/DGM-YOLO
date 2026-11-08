"""Lightweight PAN neck."""
import torch
import torch.nn as nn

from .common import ConvBNSiLU, CSPStage


class Neck(nn.Module):
    def __init__(self, in_channels=(64, 128, 256), out_ch=128):
        super().__init__()
        c3, c4, c5 = in_channels
        # lateral 1x1
        self.lat5 = ConvBNSiLU(c5, out_ch, 1)
        self.lat4 = ConvBNSiLU(c4, out_ch, 1)
        self.lat3 = ConvBNSiLU(c3, out_ch, 1)
        # top-down
        self.up4 = CSPStage(out_ch, out_ch, n=1)
        self.up3 = CSPStage(out_ch, out_ch, n=1)
        # bottom-up
        self.dn4 = CSPStage(out_ch, out_ch, n=1)
        self.dn5 = CSPStage(out_ch, out_ch, n=1)

    def forward(self, p3, p4, p5):
        n5 = self.lat5(p5)
        n4 = self.up4(self.lat4(p4) + nn.functional.interpolate(n5, size=p4.shape[-2:], mode="nearest"))
        n3 = self.up3(self.lat3(p3) + nn.functional.interpolate(n4, size=p3.shape[-2:], mode="nearest"))
        m4 = self.dn4(n4 + nn.functional.avg_pool2d(n3, 2))
        m5 = self.dn5(n5 + nn.functional.avg_pool2d(m4, 2))
        return n3, m4, m5
