"""Smoke test: forward shapes, training step, inference, SADF, profiling.

Run:  python -m pytest tests/  or  python tests/smoke_test.py
No dataset required -- uses random tensors.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lwdpcb.models import LWDPCB, WaveletPool, DCBG
from lwdpcb.losses import SADFLoss


def test_waveletpool():
    wp = WaveletPool(16, 32)
    x = torch.randn(2, 16, 64, 64)
    y = wp(x)
    assert y.shape == (2, 32, 32, 32), y.shape
    print("waveletpool OK:", tuple(y.shape))


def test_dcbg():
    d = DCBG(cond_dim=128)
    b_gt = torch.rand(2, 5, 4)
    b_gt[1, 3:] = 0
    cond = torch.randn(2, 384)
    loss = d.training_loss(b_gt, cond)
    assert loss.item() > 0
    b0 = torch.rand(2, 100, 4)
    b1 = d.refine(b0, cond)
    assert b1.shape == b0.shape
    print("dcbg OK, loss=%.3f" % loss.item())


def test_sadf():
    s = SADFLoss()
    p = torch.rand(4, 4); g = torch.rand(4, 4)
    l = s(p, g)
    assert l.item() >= 0
    print("sadf OK, loss=%.3f" % l.item())


def test_detector_forward_and_train():
    m = LWDPCB(num_classes=6)
    x = torch.randn(2, 3, 256, 256)
    targets = [
        {"boxes": torch.rand(3, 4), "labels": torch.tensor([0, 1, 2])},
        {"boxes": torch.rand(2, 4), "labels": torch.tensor([3, 4])},
    ]
    out = m.training_loss(x, targets)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    m.eval()
    with torch.no_grad():
        det = m.predict(x)
    assert len(det) == 2 and "boxes" in det[0]
    print("detector OK: loss=%.3f, pred boxes=%d" % (out["loss"].item(), det[0]["boxes"].shape[0]))


if __name__ == "__main__":
    test_waveletpool()
    test_dcbg()
    test_sadf()
    test_detector_forward_and_train()
    print("ALL SMOKE TESTS PASSED")
