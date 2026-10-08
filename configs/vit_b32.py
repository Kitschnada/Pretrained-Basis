"""ViT-B/32, the eight paper rank-16 LoRA adapters and their original heads."""

from configs.common import vision_config

config = vision_config(
    "openai/clip-vit-base-patch32", "ViT-B-32-CLIP",
    "hoffman-lab/KnOTS-ViT-B-32_lora_R16_",
)
