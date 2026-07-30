# DeepSeek V4-style toy

Read the [V4 section of the architecture chapter](../../../../docs/architecture-lab.md#5-deepseek-v4-style-preserve-nearby-detail-compress-the-prefix).

The attention layout makes three disjoint causal regions:

```text
completed old chunks  -> learned compressed memory
partial boundary     -> raw K/V, so no position disappears
recent local window  -> raw K/V
```

This is the local-plus-compressed-memory data flow. It reads all compressed
entries and materializes dense masks, so it is not CSA or a speed benchmark.

The channel-mixer ablation keeps that attention fixed:

```bash
python -m smollms.train --arch deepseekv4 --v4-moe dense \
  --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name v4-dense
python -m smollms.train --arch deepseekv4 --v4-moe mixed --v4-hash-moe-layers 1 \
  --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name v4-moe
```

`mixed` means one token-ID hash-MoE bootstrap layer, followed by learned
sqrt-softplus routed MoE layers with correction-bias balancing. `dense` is the
SwiGLU control. See the [recorded V4 runs](../../../../docs/results.md#same-2000-step-cpu--64-character-setup)
before inferring that the MoE helps.

| File | Role |
|---|---|
| `attention.py` | local, boundary, and compressed-memory reads |
| `moe.py` | hash and learned routed experts |
| `model.py` | configuration, block stack, and generation |
