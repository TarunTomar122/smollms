"""Latent Mixture-of-Experts — Kimi-K3-inspired channel mixer (toy).

Dense FFN (what we had)
-----------------------
Every token runs through the *same* big SwiGLU. Cost ∝ d_model * hidden * 3.

LatentMoE (this file)
---------------------
1. Compress each token:  d_model → d_latent
2. Router scores experts from the latent vector
3. Each token activates only top-k experts (plus optional shared expert)
4. Experts are tiny MLPs in latent space
5. Weighted mix → expand back to d_model

Only a few experts fire per token → **sparse capacity** (K3 theme).
Routing in a smaller space → **cheaper router** (latent theme).

Load-balance aux loss
---------------------
Without pressure, the router may collapse to 1–2 experts. We add the classic
Switch-Transformer style loss:

    aux = α * n_experts * sum_i (freq_i * prob_i)

so that both traffic (freq) and probability mass (prob) stay even.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.norm import RMSNorm


class ExpertMLP(nn.Module):
    """Tiny 2-layer MLP in latent space (SiLU hidden)."""

    def __init__(self, d_latent: int, hidden_mult: int = 2, bias: bool = False) -> None:
        super().__init__()
        hidden = max(d_latent * hidden_mult, d_latent)
        self.fc1 = nn.Linear(d_latent, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_latent, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class LatentMoE(nn.Module):
    """Latent-routed MoE replacing a dense FFN.

    Args:
        d_model: residual stream width.
        d_latent: compressed width for routing + experts.
        n_experts: total routed experts.
        n_active: top-k routed experts per token.
        n_shared: always-on shared experts (0 or 1 typical).
        aux_loss_weight: multiplier for load-balance term (0 disables).
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int | None = None,
        n_experts: int = 4,
        n_active: int = 2,
        n_shared: int = 1,
        expert_hidden_mult: int = 2,
        aux_loss_weight: float = 0.01,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_active < 1 or n_active > n_experts:
            raise ValueError(f"need 1 <= n_active <= n_experts, got {n_active}, {n_experts}")
        if n_shared < 0:
            raise ValueError("n_shared must be >= 0")

        d_latent = d_model // 4 if d_latent is None else d_latent
        if d_latent < 1:
            raise ValueError("d_latent must be >= 1")

        self.d_model = d_model
        self.d_latent = d_latent
        self.n_experts = n_experts
        self.n_active = n_active
        self.n_shared = n_shared
        self.aux_loss_weight = aux_loss_weight

        self.in_norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_latent, bias=bias)
        self.out_proj = nn.Linear(d_latent, d_model, bias=bias)

        # Router: latent → logits over experts
        self.router = nn.Linear(d_latent, n_experts, bias=False)

        self.experts = nn.ModuleList(
            [ExpertMLP(d_latent, hidden_mult=expert_hidden_mult, bias=bias) for _ in range(n_experts)]
        )
        self.shared = nn.ModuleList(
            [ExpertMLP(d_latent, hidden_mult=expert_hidden_mult, bias=bias) for _ in range(n_shared)]
        )

        # last forward's aux loss (read by the model/trainer)
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) update for residual_add.
        """
        b, t, _ = x.shape
        n_tok = b * t

        h = self.in_proj(self.in_norm(x))  # (B, T, d_latent)
        flat = h.reshape(n_tok, self.d_latent)

        # ----- route -----
        logits = self.router(flat)  # (N, E)
        probs = F.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(probs, k=self.n_active, dim=-1)
        # renormalize over selected experts
        top_w = top_val / top_val.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ----- load balance aux -----
        if self.aux_loss_weight > 0 and self.training:
            # soft density: mean probability mass per expert
            density = probs.mean(dim=0)  # (E,)
            # hard assignment frequency (fraction of tokens that picked expert)
            ones = torch.zeros_like(probs)
            ones.scatter_(1, top_idx, 1.0)
            freq = ones.mean(dim=0)  # (E,)
            aux = self.n_experts * (freq * density).sum()
            self.last_aux_loss = self.aux_loss_weight * aux
        else:
            self.last_aux_loss = flat.new_zeros(())

        # ----- routed experts (loop is fine at toy scale) -----
        out = torch.zeros_like(flat)
        for k in range(self.n_active):
            expert_ids = top_idx[:, k]  # (N,)
            weights = top_w[:, k]  # (N,)
            for e in range(self.n_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                tok = flat[mask]
                y = self.experts[e](tok)
                out[mask] = out[mask] + weights[mask].unsqueeze(-1) * y

        # ----- shared experts (always on, uniform blend if several) -----
        if self.n_shared > 0:
            shared_out = 0.0
            for shared in self.shared:
                shared_out = shared_out + shared(flat)
            out = out + shared_out / self.n_shared

        out = out.view(b, t, self.d_latent)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_latent={self.d_latent}, "
            f"n_experts={self.n_experts}, n_active={self.n_active}, "
            f"n_shared={self.n_shared}"
        )
