"""DeepSeek V4-inspired learning variant: local attention + compressed memory."""

from smollms.variants.deepseek_v4.attention import LocalCompressedAttention
from smollms.variants.deepseek_v4.model import V4ToyLM, V4ToyLMConfig
from smollms.variants.deepseek_v4.moe import V4MoE

__all__ = ["LocalCompressedAttention", "V4MoE", "V4ToyLM", "V4ToyLMConfig"]
