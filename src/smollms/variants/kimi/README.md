# Kimi-style toy

Read the [Kimi section of the architecture chapter](../../../../docs/architecture-lab.md#3-kimi-style-route-channels-then-alter-the-token-mixer)
before treating a loss difference as meaningful.

This branch has two separable changes:

1. **LatentMoE:** replace dense SwiGLU with latent-space top-k experts plus a
   shared expert. Full causal attention remains the control.
2. **Kimi step 2:** replace selected attention layers with KDA recurrence and
   Gated MLA, then optionally add attention over earlier depth states (AttnRes).

| Schedule | Meaning |
|---|---|
| `all_full` | full causal attention in every Kimi block; use as the control |
| `3L1F` | KDA, KDA, KDA, Gated MLA; default for `--arch kimi2` |
| `1L1F` | alternate KDA and Gated MLA |

```bash
# Full-attention Kimi control, then one changed token-mixer schedule.
python -m smollms.train --arch kimi2 --hybrid-pattern all_full --attn-res false \
  --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name kimi-full
python -m smollms.train --arch kimi2 --hybrid-pattern 1L1F --attn-res false \
  --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name kimi-1l1f
```

| File | Role |
|---|---|
| `latent_moe.py` | latent routing, experts, and balance loss |
| `linear_attn.py` | readable KDA delta-rule recurrence |
| `mla.py` | compact K/V latent plus output gate |
| `attn_res.py` | content-dependent reads across depth |
| `block.py` | one token mixer plus LatentMoE |
| `model.py` | schedule, block stack, and optional AttnRes |

The recurrence is intentionally unoptimized for short contexts. See
[results.md](../../../../docs/results.md#kimi-schedule-comparison) for the
recorded schedule comparison and its limits.
