"""Small sigmoid-routed, auxiliary-loss-free MoE for the GLM learning variant."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.norm import RMSNorm


class GLMExpert(nn.Module):
    """One small SiLU expert operating directly on the residual width."""

    def __init__(self, d_model: int, bias: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.fc2 = nn.Linear(2 * d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class GLMMoE(nn.Module):
    """Top-k sigmoid-routed experts plus optional always-on shared experts."""

    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        n_active: int = 2,
        n_shared: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_active < 1 or n_active > n_experts:
            raise ValueError(f"need 1 <= n_active <= n_experts, got {n_active}, {n_experts}")
        if n_shared < 0:
            raise ValueError("n_shared must be >= 0")

        self.n_experts = n_experts
        self.n_active = n_active
        self.n_shared = n_shared
        self.norm = RMSNorm(d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([GLMExpert(d_model, bias=bias) for _ in range(n_experts)])
        self.shared = nn.ModuleList([GLMExpert(d_model, bias=bias) for _ in range(n_shared)])
        self.register_buffer("expert_correction_bias", torch.zeros(n_experts))
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = self.norm(x).reshape(b * t, d)
        scores = torch.sigmoid(self.router(flat))
        top_idx = torch.topk(scores + self.expert_correction_bias, k=self.n_active, dim=-1).indices
        top_score = torch.gather(scores, 1, top_idx)
        top_weight = top_score / top_score.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        self.last_aux_loss = flat.new_zeros(())

        if self.training:
            assignments = torch.zeros_like(scores).scatter_(1, top_idx, 1.0)
            load = assignments.mean(dim=0)
            target = self.n_active / self.n_experts
            # ponytail: batch-local correction is a readable noaux analogue; use production load accounting only at distributed scale.
            self.expert_correction_bias.add_(0.01 * (target - load))
            self.expert_correction_bias.sub_(self.expert_correction_bias.mean())

        out = torch.zeros_like(flat)
        for slot in range(self.n_active):
            expert_ids = top_idx[:, slot]
            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id
                if mask.any():
                    out[mask] = out[mask] + top_weight[mask, slot, None] * expert(flat[mask])

        if self.shared:
            shared_out = sum((expert(flat) for expert in self.shared), torch.zeros_like(flat))
            out = out + shared_out / self.n_shared
        return out.view(b, t, d)
