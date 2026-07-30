# Architecture branches

Every branch changes one ordinary decoder assumption. Read the full narrative
in [the architecture lab chapter](../../../docs/architecture-lab.md); this page
only tells you where each implementation begins.

| Branch | Toy idea | Source | Read next |
|---|---|---|---|
| Dense / Qwen-style | full causal attention + SwiGLU control | [`smollms/model.py`](../model.py) | [dense control](../../../docs/architecture-lab.md#1-the-control-a-small-dense-decoder) |
| Kimi-style | LatentMoE; KDA/Gated-MLA schedules; AttnRes | [`kimi/`](kimi/README.md) | [Kimi branch](../../../docs/architecture-lab.md#3-kimi-style-route-channels-then-alter-the-token-mixer) |
| GLM-style | DSA-like learned top-k selection + routed MoE | [`glm/`](glm/README.md) | [GLM branch](../../../docs/architecture-lab.md#4-glm-style-select-tokens-then-read-them) |
| DeepSeek V4-style | local raw attention + compressed memory; optional MoE | [`deepseek_v4/`](deepseek_v4/README.md) | [V4 branch](../../../docs/architecture-lab.md#5-deepseek-v4-style-preserve-nearby-detail-compress-the-prefix) |

The code is pedagogical: no claim here is a bit-exact reproduction or a
production-throughput result. Use [results.md](../../../docs/results.md) for
the saved evidence and its constraints.
