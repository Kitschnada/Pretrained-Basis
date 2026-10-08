"""SigLIP eight-task rank-16 LoRA; baselines come from matching checkpoints."""

from configs.common import vision_config

config = vision_config("google/siglip-base-patch16-224", "SigLIP-base-patch16-224")
