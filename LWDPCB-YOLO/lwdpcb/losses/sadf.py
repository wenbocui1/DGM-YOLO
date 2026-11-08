"""SADF: Shape-Aware Dynamic Focus loss (paper Section 3.4).

L = gamma * R_pos * L_IoU, with
  h_w = 2 h_gt / (h_gt + w_gt),  w_w = 2 w_gt / (h_gt + w_gt)
  beta = L_IoU_stopgrad / EMA(L_IoU)
  gamma = beta^2 / (lambda * mu^(beta - lambda))
  R_pos = exp( (h_w (cx-cxg)^2 + w_w (cy-cyg)^2) / (wc^2 + hc^2)^* )
where (.)^* means the normaliser is detached from the graph.
"""
import torch
import torch.nn as nn


class SADFLoss(nn.Module):
    def __init__(self, lam=1.0, mu=3.0, ema_decay=0.99):
        super().__init__()
        self.lam = lam
        self.mu = mu
        self.ema_decay = ema_decay
        self.register_buffer("ema_iou", torch.tensor(1.0))

    @staticmethod
    def _box_iou_wh(b1, b2):
        # b: (..., 4) in (cx, cy, w, h)
        b1x1, b1y1 = b1[..., 0] - b1[..., 2] / 2, b1[..., 1] - b1[..., 3] / 2
        b1x2, b1y2 = b1[..., 0] + b1[..., 2] / 2, b1[..., 1] + b1[..., 3] / 2
        b2x1, b2y1 = b2[..., 0] - b2[..., 2] / 2, b2[..., 1] - b2[..., 3] / 2
        b2x2, b2y2 = b2[..., 0] + b2[..., 2] / 2, b2[..., 1] + b2[..., 3] / 2
        ix1 = torch.maximum(b1x1, b2x1); iy1 = torch.maximum(b1y1, b2y1)
        ix2 = torch.minimum(b1x2, b2x2); iy2 = torch.minimum(b1y2, b2y2)
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        union = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - inter
        return inter / union.clamp(min=1e-6)

    def forward(self, pred, gt, update_ema=True):
        # pred, gt: (N, 4) normalised centre form
        iou = self._box_iou_wh(pred, gt)
        l_iou = 1.0 - iou                                   # live IoU loss
        if update_ema:
            with torch.no_grad():
                self.ema_iou.mul_(self.ema_decay).add_(l_iou.mean().detach() * (1 - self.ema_decay))
        beta = l_iou.detach() / self.ema_iou.clamp(min=1e-6)
        gamma = (beta ** 2) / (self.lam * self.mu ** (beta - self.lam))

        ww = gt[..., 3]; hh = gt[..., 2]
        h_w = 2 * hh / (hh + ww).clamp(min=1e-6)
        w_w = 2 * ww / (hh + ww).clamp(min=1e-6)
        cx = pred[..., 0]; cy = pred[..., 1]
        cxg = gt[..., 0]; cyg = gt[..., 1]
        # enclosing box side lengths (detached)
        with torch.no_grad():
            wc = (torch.maximum(pred[..., 0] + pred[..., 2] / 2, gt[..., 0] + gt[..., 2] / 2)
                  - torch.minimum(pred[..., 0] - pred[..., 2] / 2, gt[..., 0] - gt[..., 2] / 2))
            hc = (torch.maximum(pred[..., 1] + pred[..., 3] / 2, gt[..., 1] + gt[..., 3] / 2)
                  - torch.minimum(pred[..., 1] - pred[..., 3] / 2, gt[..., 1] - gt[..., 3] / 2))
            denom = (wc ** 2 + hc ** 2).clamp(min=1e-6)
        r_pos = torch.exp((h_w * (cx - cxg) ** 2 + w_w * (cy - cyg) ** 2) / denom)
        loss = gamma * r_pos * l_iou
        return loss.mean()
