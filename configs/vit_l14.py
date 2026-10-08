"""ViT-L/14, the eight paper rank-16 LoRA adapters and their original heads."""

from configs.common import vision_config

config = vision_config(
    "openai/clip-vit-large-patch14", "ViT-L-14-CLIP",
    "hoffman-lab/KnOTS-ViT-L-14_lora_R16_",
)
