"""Minimal V4-style hash bootstrap and sqrt-softplus routed MoE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.ffn import SwiGLUFFN
from smollms.atoms.norm import RMSNorm


class V4RoutedExpert(nn.Module):
    """Clamped SwiGLU routed expert, matching the V4 distinction from shared MLPs."""

    def __init__(self, d_model: int, hidden_dim: int, swiglu_limit: float, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.swiglu_limit)
        up = self.up_proj(x).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class V4MoE(nn.Module):
    """Hash-selected or learned top-k MoE with sqrt-softplus scores and noaux bias."""

    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        n_active: int = 2,
        n_shared: int = 1,
        expert_hidden_dim: int | None = None,
        swiglu_limit: float = 10.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= n_active <= n_experts:
            raise ValueError("need 1 <= n_active <= n_experts")
        if n_shared < 0:
            raise ValueError("n_shared must be >= 0")
        self.n_experts = n_experts
        self.n_active = n_active
        self.norm = RMSNorm(d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        hidden = d_model if expert_hidden_dim is None else expert_hidden_dim
        self.experts = nn.ModuleList(
            [V4RoutedExpert(d_model, hidden, swiglu_limit, bias) for _ in range(n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUFFN(d_model, hidden, bias=bias) for _ in range(n_shared)]
        )
        self.register_buffer("expert_correction_bias", torch.zeros(n_experts))
        self.last_selected_experts: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        flat = self.norm(x).reshape(b * t, d)
        scores = torch.sqrt(F.softplus(self.router(flat)))
        if token_ids is None:
            top_idx = torch.topk(scores + self.expert_correction_bias, self.n_active, dim=-1).indices
            if self.training:
                load = torch.zeros_like(scores).scatter_(1, top_idx, 1.0).mean(dim=0)
                self.expert_correction_bias.add_(0.01 * (self.n_active / self.n_experts - load))
                self.expert_correction_bias.sub_(self.expert_correction_bias.mean())
        else:
            if token_ids.shape != (b, t):
                raise ValueError("token_ids must have shape (batch, sequence)")
            primary = token_ids.reshape(-1).remainder(self.n_experts)
            offsets = torch.arange(self.n_active, device=x.device)
            top_idx = (primary[:, None] + offsets[None, :]).remainder(self.n_experts)
        self.last_selected_experts = top_idx.detach()
        top_score = torch.gather(scores, 1, top_idx)
        top_weight = top_score / top_score.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for slot in range(self.n_active):
            expert_ids = top_idx[:, slot]
            for expert_id, expert in enumerate(self.experts):
                chosen = expert_ids == expert_id
                if chosen.any():
                    out[chosen] += top_weight[chosen, slot, None] * expert(flat[chosen])
        if self.shared:
            shared_out = sum((expert(flat) for expert in self.shared), torch.zeros_like(flat))
            out += shared_out / len(self.shared)
        return out.view(b, t, d)
