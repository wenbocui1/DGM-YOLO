"""DCBG: Diffusion Candidate Box-Guided refinement (paper Section 3.3).

Boxes live in image-normalised centre form ``(cx, cy, w, h)``.  Training
corrupts ground-truth boxes with the standard linear-beta DDPM schedule
(T=1000, beta 1e-4..0.02) and a shared 3-layer MLP predicts the injected
Gaussian noise.  Inference runs the deterministic DDIM subsequence
tau = (100, 75, 50, 25), eta = 0, starting from the detector's initial
boxes at tau_0=100 rather than from pure noise.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DCBG(nn.Module):
    def __init__(self, cond_dim=128, t_dim=64, hidden=256, T=1000,
                 beta_start=1e-4, beta_end=0.02, num_boxes=100,
                 inference_taus=(100, 75, 50, 25)):
        super().__init__()
        self.T = T
        self.num_boxes = num_boxes
        self.inference_taus = list(inference_taus)

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))
        self.register_buffer("sqrt_one_minus_abar", torch.sqrt(1.0 - abar))

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.cond_proj = nn.Linear(cond_dim * 3, hidden)
        self.net = nn.Sequential(
            nn.Linear(4 + hidden + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    # ------------------------------------------------------------------
    def _extract(self, abar_buf, t):
        return abar_buf[t.view(-1)].view(-1, 1, 1)

    def q_sample(self, b0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(b0)
        sa = self._extract(self.sqrt_abar, t)
        sn = self._extract(self.sqrt_one_minus_abar, t)
        return sa * b0 + sn * noise, noise

    def predict_eps(self, b_t, cond, t):
        # b_t: (B, N, 4); cond: (B, hidden); t: (B,) long
        temb = self.t_mlp(timestep_embedding(t, 64))
        c = self.cond_proj(cond)
        n = b_t.shape[1]
        inp = torch.cat([b_t, c.unsqueeze(1).repeat(1, n, 1),
                         temb.unsqueeze(1).repeat(1, n, 1)], dim=-1)
        return self.net(inp)

    # ------------------------------------------------------------------
    def training_loss(self, b_gt, cond):
        """b_gt: (B, M, 4) normalised centre boxes; padding boxes have w=h=0.
        Returns smooth-L1 noise loss averaged over valid boxes."""
        b = b_gt
        valid = (b[..., 2:3] > 0).float()          # (B, M, 1)
        B, M, _ = b.shape
        t = torch.randint(0, self.T, (B,), device=b.device).long()
        # per-sample t broadcast to its boxes
        b_noisy, noise = self.q_sample(b, t)
        eps_pred = self.predict_eps(b_noisy, cond, t)
        loss = F.smooth_l1_loss(eps_pred * valid, noise * valid, reduction="sum")
        return loss / valid.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def refine(self, b_init, cond):
        """DDIM deterministic refinement.

        b_init: (B, K, 4) detector initial boxes, normalised centre form.
        Returns refined (B, K, 4).
        """
        taus = self.inference_taus
        b = b_init
        for i, tau in enumerate(taus):
            t = torch.full((b.shape[0],), tau, device=b.device, dtype=torch.long)
            eps = self.predict_eps(b, cond, t)
            sa = self._extract(self.sqrt_abar, t)
            sn = self._extract(self.sqrt_one_minus_abar, t)
            b0 = (b - sn * eps) / sa
            if i == len(taus) - 1:
                b = b0
            else:
                tau_next = taus[i + 1]
                t_next = torch.full((b.shape[0],), tau_next, device=b.device, dtype=torch.long)
                sa_n = self._extract(self.sqrt_abar, t_next)
                sn_n = self._extract(self.sqrt_one_minus_abar, t_next)
                b = sa_n * b0 + sn_n * eps
        return b
