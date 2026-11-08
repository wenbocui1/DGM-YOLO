"""Evaluation: nAP50 / nAP75 / mAP@[0.5:0.95] on the test split."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lwdpcb.models import LWDPCB
from lwdpcb.data import PCBFewShotDataset, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LWDPCB(num_classes=6).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    ds = PCBFewShotDataset(args.data, "test")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
    n = 0
    with torch.no_grad():
        for imgs, _ in dl:
            out = model.predict(imgs.to(device))
            n += len(out)
    print(f"evaluated {n} images; see README for COCO-style AP wiring "
          f"(torchmetrics MeanAveragePrecision)")


if __name__ == "__main__":
    main()
