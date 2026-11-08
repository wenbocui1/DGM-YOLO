"""Few-shot fine-tuning entry point (paper Section 4.2).

Stage 1 (FSOD base pretraining) is not reproduced here -- the paper uses the
standard FSOD benchmark; when the released Ultralytics checkpoint is available,
pass --pretrained to initialise the backbone.  This script performs the PCB
fine-tuning stage: freeze the backbone, train neck + DCBG + head + SADF on the
k-shot support set.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lwdpcb.models import LWDPCB
from lwdpcb.data import PCBFewShotDataset, collate

NOVEL_SPLITS = {
    # split -> novel class indices (paper Section 4.1)
    1: [4, 5],
    2: [1, 2],
    3: [0, 4],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (YOLO layout)")
    ap.add_argument("--k-shot", type=int, default=30)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/lwdpcb")
    ap.add_argument("--pretrained", default="", help="optional backbone ckpt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PCBFewShotDataset(args.data, "train", k_shot=args.k_shot,
                           novel_classes=NOVEL_SPLITS[args.split], seed=args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    print(f"support set: {len(ds)} images, k={args.k_shot}, split={args.split}")

    model = LWDPCB(num_classes=6).to(device)
    if args.pretrained and os.path.exists(args.pretrained):
        sd = torch.load(args.pretrained, map_location="cpu")
        model.backbone.load_state_dict(sd, strict=False)
        print("loaded backbone pretrain")

    # freeze backbone (paper: only neck/DCBG/head fine-tuned)
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    model.train()
    for ep in range(args.epochs):
        for imgs, targets in dl:
            imgs = imgs.to(device)
            losses = model.training_loss(imgs, targets)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
        sched.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} loss {losses['loss'].item():.3f} "
                  f"(cls {losses['cls'].item():.3f} box {losses['box'].item():.3f} "
                  f"dcbg {losses['dcbg'].item():.3f})")

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print("saved", os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
