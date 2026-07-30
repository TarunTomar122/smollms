"""Residual stream helpers.

The residual stream is the main "highway" of information through a transformer.

    x_{l+1} = x_l + f_l(x_l)

Every block *adds* an update; it does not replace the representation. That means:

1. Gradients can flow straight back through the identity path (skip connection).
2. Early information is still present later unless later blocks overwrite it via
   cancellation (they rarely fully cancel — they refine).
3. Depth becomes "stacking refinements" rather than a chain of irreversible maps.

Almost every modern LLM is residual-stream-centric. Attention and FFN only
*write into* the stream.
"""

from __future__ import annotations

import torch


def residual_add(stream: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Add a sublayer output into the residual stream.

    Kept as a named function so call sites read like the math:

        stream = residual_add(stream, attn_out)

    rather than a bare ``+`` that hides intent.

    Shapes must match: both (batch, seq, dim) or broadcastable.
    """
    if stream.shape != update.shape:
        raise ValueError(
            f"residual_add expects matching shapes, got stream={tuple(stream.shape)} "
            f"update={tuple(update.shape)}"
        )
    return stream + update
