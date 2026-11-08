"""Full LWDPCB-YOLO detector: backbone + neck + dynamic head + one-to-one head + DCBG."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .neck import Neck
from .dynamic_head import DynamicHead
from .dcbg import DCBG
from ..losses.sadf import SADFLoss


class DetectHead(nn.Module):
    def __init__(self, c, num_classes):
        super().__init__()
        self.cls = nn.Conv2d(c, num_classes, 1)
        self.box = nn.Conv2d(c, 4, 1)

    def forward(self, feats):
        cls_out, box_out = [], []
        for f in feats:
            cls_out.append(self.cls(f).flatten(2).transpose(1, 2))   # (B,N,C)
            box_out.append(self.box(f).flatten(2).transpose(1, 2))
        return torch.cat(cls_out, 1), torch.cat(box_out, 1)


class LWDPCB(nn.Module):
    def __init__(self, num_classes=6, wavelet=True, num_boxes=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.backbone = Backbone(wavelet=wavelet)
        self.neck = Neck(in_channels=(64, 128, 256), out_ch=128)
        self.head_dyn = DynamicHead(128)
        self.head = DetectHead(128, num_classes)
        self.dcbg = DCBG(cond_dim=128, num_boxes=num_boxes)
        self.sadf = SADFLoss()

    # ------------------------------------------------------------------
    def _pool_cond(self, feats):
        return torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)

    def forward_features(self, x):
        p3, p4, p5 = self.backbone(x)
        feats = self.neck(p3, p4, p5)
        feats = self.head_dyn(feats)
        return feats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x, score_thresh=0.05):
        feats = self.forward_features(x)
        cls, box = self.head(feats)                       # box: (B,N,4) raw offsets
        box = torch.sigmoid(box)                          # normalised (cx,cy,w,h)
        score = cls.sigmoid().max(dim=-1).values          # (B,N)
        B = x.shape[0]
        outs = []
        for b in range(B):
            s = score[b]
            k = min(self.num_boxes, s.shape[0])
            topk = torch.topk(s, k).indices
            b_init = box[b, topk].unsqueeze(0)           # (1,K,4)
            cond = self._pool_cond(feats).narrow(0, b, 1)
            b_ref = self.dcbg.refine(b_init, cond).squeeze(0)
            cls_ref = cls[b, topk]
            outs.append({"boxes": b_ref, "scores": s[topk], "labels": cls_ref.argmax(-1)})
        return outs

    # ------------------------------------------------------------------
    def training_loss(self, x, targets):
        """targets: list of dicts {boxes: (M,4) normalised cxcywh, labels: (M,)}.
        Simplified one-to-one assignment: for each GT pick the nearest cell."""
        feats = self.forward_features(x)
        cls, box = self.head(feats)                       # (B,N,C), (B,N,4)
        box = torch.sigmoid(box)
        B, N, _ = box.shape

        # build cell grid (assume equal stride coverage across the 3 levels;
        # for training we only need per-cell centre coordinates)
        device = x.device
        idx = torch.arange(N, device=device)
        cell_cx = (idx.float() / N).view(1, N).expand(B, N)
        cell_cy = (idx.float() % N).view(1, N).expand(B, N) / N

        total_cls = box.new_zeros(())
        total_box = box.new_zeros(())
        n_pos = 0
        for b in range(B):
            gt = targets[b]["boxes"].to(device)          # (M,4)
            if gt.numel() == 0:
                continue
            # nearest cell per GT
            d = (cell_cx[b] - gt[:, 0:1]) ** 2 + (cell_cy[b] - gt[:, 1:2]) ** 2
            assign = d.argmin(dim=1)                       # (M,)
            pred_box = box[b, assign]
            total_box = total_box + self.sadf(pred_box, gt)
            # focal-style cls target: assigned cells positive
            target = torch.full((N,), 0, dtype=torch.long, device=device)
            target[assign] = targets[b]["labels"].to(device)
            logits = cls[b]
            total_cls = total_cls + F.cross_entropy(logits, target)
            n_pos += 1

        # DCBG noise loss on GT boxes (padded)
        maxM = max(t["boxes"].shape[0] for t in targets)
        pad = torch.zeros(B, maxM, 4, device=device)
        for b in range(B):
            m = targets[b]["boxes"].shape[0]
            pad[b, :m] = targets[b]["boxes"].to(device)
        cond = self._pool_cond(feats)
        l_dcbg = self.dcbg.training_loss(pad, cond)

        loss = total_cls / max(n_pos, 1) + total_box / max(n_pos, 1) + 0.5 * l_dcbg
        return {"loss": loss, "cls": total_cls / max(n_pos, 1),
                "box": total_box / max(n_pos, 1), "dcbg": l_dcbg}
