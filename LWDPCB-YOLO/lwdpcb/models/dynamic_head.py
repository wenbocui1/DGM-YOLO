"""DCNv3-DH: cascaded scale / spatial / channel dynamic detection head.

Implements the three attention axes of Dynamic Head on top of a DCNv3-style
modulated deformable conv for the spatial axis.  If torchvision's deformable
ops are unavailable the spatial branch falls back to a regular 3x3 conv with
the same interface so the project still runs.
"""
import torch
import torch.nn as nn

try:
    from torchvision.ops import deform_conv2d
    _HAS_DCN = True
except Exception:  # pragma: no cover
    _HAS_DCN = False


class ScaleAttention(nn.Module):
    """L: channel attention over the pyramid (one weight per level)."""

    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c, c)

    def forward(self, feats):
        # feats: list of (B,C,H,W) at L levels
        w = []
        for f in feats:
            w.append(self.fc(f.mean(dim=(2, 3))))          # (B,C)
        w = torch.softmax(torch.stack(w, dim=1), dim=1)      # (B,L,C)
        out = []
        for i, f in enumerate(feats):
            out.append(f * w[:, i].view(-1, f.shape[1], 1, 1).sigmoid())
        return out


class SpatialDCN(nn.Module):
    """S axis: modulated deformable convolution (DCNv3-style)."""

    def __init__(self, c, k=3):
        super().__init__()
        self.k = k
        self.offset = nn.Conv2d(c, 3 * k * k, 3, padding=1)
        self.weight = nn.Parameter(torch.empty(c, c, k, k))
        nn.init.kaiming_normal_(self.weight)
        self.bias = None

    def forward(self, f):
        B, C, H, W = f.shape
        off_mask = self.offset(f)
        o1, o2, m = off_mask.split(self.k * self.k, dim=1)
        m = torch.sigmoid(m)
        if _HAS_DCN:
            return deform_conv2d(f, torch.cat([o1, o2], dim=1), self.weight,
                                 bias=self.bias, stride=1, padding=1, mask=m)
        # fallback: plain conv (keeps the project runnable)
        return nn.functional.conv2d(f, self.weight, padding=1)


class ChannelAttention(nn.Module):
    """C axis: dynamic ReLU-style gate conditioned on global features."""

    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)

    def forward(self, f):
        g = f.mean(dim=(2, 3))
        a1 = self.fc1(g).sigmoid().view(-1, f.shape[1], 1, 1)
        a2 = self.fc2(g).sigmoid().view(-1, f.shape[1], 1, 1)
        return torch.maximum(a1 * f, a2 * f)


class DynamicHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = ScaleAttention(c)
        self.spatial = SpatialDCN(c)
        self.channel = ChannelAttention(c)

    def forward(self, feats):
        feats = self.scale(feats)
        out = [self.channel(self.spatial(f)) for f in feats]
        return out
