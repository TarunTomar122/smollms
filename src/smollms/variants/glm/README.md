# GLM-style toy

Read the [GLM section of the architecture chapter](../../../../docs/architecture-lab.md#4-glm-style-select-tokens-then-read-them).

```text
layer 1:  token mixer -> residual -> dense SwiGLU -> residual
layers 2+: token mixer -> residual -> sigmoid-routed experts + shared expert -> residual
```

The branch has exactly one useful ablation: hold the dense-prefix routed MoE
fixed and change `--glm-attention`.

```bash
python -m smollms.train --arch glm --glm-attention sparse --glm-dense-layers 1 \
  --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name glm-dsa
python -m smollms.train --arch glm --glm-attention dense --glm-dense-layers 1 \
  --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name glm-dense-attn
```

| File | Role |
|---|---|
| `block.py` | sparse or dense token mixer plus channel-mixer residuals |
| `moe.py` | sigmoid routing, shared expert, correction-bias balancing |
| `model.py` | language-model stack |
| `../../atoms/sparse_attention.py` | learned causal top-k selector and reader |

The selector still builds a dense index-score matrix, so this branch is not a
sparse-speed benchmark. There is no IndexShare, production DSA kernel, or
distributed noaux balancing. The saved
[DSA ablation](../../../../docs/results.md#same-2000-step-cpu--64-character-setup)
is a model-quality observation at toy scale only.
