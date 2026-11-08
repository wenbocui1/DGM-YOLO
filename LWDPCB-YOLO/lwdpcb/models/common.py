"""Common building blocks: CBS conv, Haar wavelet-pool downsampling."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNSiLU(nn.Module):
    """Standard CBS block: Conv - BatchNorm - SiLU."""

    def __init__(self, c_in, c_out, k=1, s=1, g=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Residual(nn.Module):
    """Bottleneck-style residual used in the nano backbone."""

    def __init__(self, c):
        super().__init__()
        self.cv1 = ConvBNSiLU(c, c, 1, 1)
        self.cv2 = ConvBNSiLU(c, c, 3, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CSPStage(nn.Module):
    """Lightweight CSP stage: split channels, one branch with n residual blocks."""

    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        c_half = c_out // 2
        self.cv1 = ConvBNSiLU(c_in, c_half, 1, 1)
        self.cv2 = ConvBNSiLU(c_in, c_half, 1, 1)
        self.blocks = nn.Sequential(*(Residual(c_half) for _ in range(n)))
        self.cv3 = ConvBNSiLU(c_out, c_out, 1, 1)

    def forward(self, x):
        a = self.cv1(x)
        b = self.blocks(self.cv2(x))
        return self.cv3(torch.cat([a, b], dim=1))


# ---------------------------------------------------------------------------
# Haar wavelet pooling (replaces CBS strided downsampling)
# ---------------------------------------------------------------------------
class WaveletPool(nn.Module):
    r"""2D Haar analysis with factor-2 subsampling, followed by a learnable 1x1 fusion.

    For input ``X`` (B,C,H,W) the fixed analysis filters
    l = (1,1)/sqrt(2), h = (1,-1)/sqrt(2) are applied separably along rows
    and columns with stride 2, producing four subbands LL, LH, HL, HH of size
    C x H/2 x W/2.  The four subbands are concatenated on the channel axis
    (4C) and fused back to ``c_out`` channels with one 1x1 convolution, so all
    three detail streams are retained in the output while spatial resolution
    is halved.  Parameters of the wavelet filters are *not* learned.
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        # fixed Haar filters, registered as buffers (non-learnable)
        lo = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        hi = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)
        # out_channels=1, in_channels=1, kernel=2 -> weight (1,1,1,2)
        self.register_buffer("lo1d", lo.view(1, 1, 1, 2))
        self.register_buffer("hi1d", hi.view(1, 1, 1, 2))
        self.fuse = nn.Conv2d(c_in * 4, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def _filt(self, x, ker, dim):
        # separable 1D conv along `dim` (dim=3 -> along width, dim=2 -> along height)
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        k = ker
        if dim == 3:
            x = F.conv2d(x, k, stride=(1, 2))           # horizontal
        else:
            x = x.permute(0, 1, 3, 2)
            x = F.conv2d(x, k, stride=(1, 2))
            x = x.permute(0, 1, 3, 2)
        return x.reshape(b, c, x.shape[2], x.shape[3])

    def forward(self, x):
        ll = self._filt(self._filt(x, self.lo1d, 3), self.lo1d, 2)
        lh = self._filt(self._filt(x, self.lo1d, 3), self.hi1d, 2)   # horizontal detail
        hl = self._filt(self._filt(x, self.hi1d, 3), self.lo1d, 2)   # vertical detail
        hh = self._filt(self._filt(x, self.hi1d, 3), self.hi1d, 2)    # diagonal
        out = torch.cat([ll, lh, hl, hh], dim=1)
        return self.act(self.bn(self.fuse(out)))
