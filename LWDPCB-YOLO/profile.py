"""Reproduce the deployment numbers of Tables 10 and 11.

Reports parameter count and FLOPs at 640x640 batch 1 for:
  * the wavelet-disabled baseline (YOLO26-Nano stand-in)
  * the full LWDPCB-YOLO
  * the DCBG sub-network alone
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lwdpcb.models import LWDPCB


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def flops_estimate(model, device):
    # fvcore if available
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, 3, 640, 640).to(device)
        f = FlopCountAnalysis(model.forward_features, x)
        f.unsupported_ops_warnings(False)
        f.uncalled_modules_warnings(False)
        return f.total() / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = args.device

    base = LWDPCB(num_classes=6, wavelet=False).to(device).eval()
    full = LWDPCB(num_classes=6, wavelet=True).to(device).eval()

    print(f"baseline (CBS downsample): {count_params(base):.2f} M params")
    print(f"full    (Waveletpool+DCBG+DH): {count_params(full):.2f} M params")
    print(f"DCBG sub-network:           {count_params(full.dcbg):.2f} M params")

    gf_base = flops_estimate(base, device)
    gf_full = flops_estimate(full, device)
    print(f"baseline FLOPs: {gf_base:.1f} GFLOPs (paper: 4.4)")
    print(f"full    FLOPs: {gf_full:.1f} GFLOPs (paper: 6.3)")


if __name__ == "__main__":
    main()
