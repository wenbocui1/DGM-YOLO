"""PCB few-shot detection dataset.

Expects a YOCO-format directory::

    root/
      images/train/xxx.jpg
      images/test/xxx.jpg
      labels/train/xxx.txt
      labels/test/xxx.txt

Label lines are ``class cx cy w h`` with normalised coordinates.
The few-shot sampler keeps exactly ``k`` annotated instances per (novel)
class, matching the paper's shot definition (k annotated boxes, not images).
"""
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# class names (shared across the two benchmarks in the paper)
CLASSES = ["missing-hole", "mouse-bite", "open", "short", "spur", "spurious-copper"]


def _read_label(path):
    boxes, labels = [], []
    if not os.path.exists(path):
        return boxes, labels
    for line in open(path):
        p = line.strip().split()
        if len(p) < 5:
            continue
        labels.append(int(p[0]))
        boxes.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return boxes, labels


class PCBFewShotDataset(Dataset):
    def __init__(self, root, split="train", k_shot=None, novel_classes=None,
                 seed=0, img_size=640):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        img_dir = self.root / "images" / split
        self.samples = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        self.labels_dir = self.root / "labels" / split

        if k_shot is not None and novel_classes is not None:
            self.samples = self._sample_support(k_shot, novel_classes, seed)

    def _sample_support(self, k, novel_classes, seed):
        rng = random.Random(seed)
        picked_ids = set()
        per_class = {c: 0 for c in novel_classes}
        # shuffle image order so support draw is reproducible across seeds
        order = list(range(len(self.samples)))
        rng.shuffle(order)
        for i in order:
            lab = self.labels_dir / (self.samples[i].stem + ".txt")
            _, cls = _read_label(lab)
            keep = False
            for c in cls:
                if c in per_class and per_class[c] < k:
                    per_class[c] += 1
                    keep = True
            if keep:
                picked_ids.add(i)
            if all(v >= k for v in per_class.values()):
                break
        return [self.samples[i] for i in sorted(picked_ids)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        lab_path = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = _read_label(lab_path)
        import numpy as np
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"image": img, "boxes": boxes, "labels": labels, "path": str(img_path)}


def collate(batch):
    imgs = torch.stack([b["image"] for b in batch])
    targets = [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch]
    return imgs, targets
