from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import fields

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, SiglipModel


def _prefer_local_hf_files() -> bool:
    return any(
        str(value).lower() in {"1", "true", "yes"}
        for value in (
            os.getenv("HF_HUB_OFFLINE"),
            os.getenv("TRANSFORMERS_OFFLINE"),
            os.getenv("CORE_SPACE_MERGING_LOCAL_ONLY"),
        )
    )


def _is_hf_connection_error(exc: Exception) -> bool:
    message = str(exc)
    markers = (
        "HTTPSConnectionPool(",
        "MaxRetryError(",
        "SSLError",
        "ConnectionError",
        "ProxyError",
        "ReadTimeout",
        "ConnectTimeout",
        "Temporary failure in name resolution",
        "NameResolutionError",
        "TLS/SSL connection has been closed",
    )
    return any(marker in message for marker in markers)


def _from_pretrained_with_local_fallback(
    loader,
    resource_name: str,
    **kwargs,
):
    """Load SigLIP assets without depending on CLIP module internals."""
    if _prefer_local_hf_files():
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        return loader(resource_name, **local_kwargs)
    try:
        return loader(resource_name, **kwargs)
    except Exception as exc:
        if not _is_hf_connection_error(exc):
            raise
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        print(
            f"Falling back to local Hugging Face cache for {resource_name} "
            f"after connection error: {exc}"
        )
        return loader(resource_name, **local_kwargs)


def _ensure_rgb_image(image):
    """Convert grayscale inputs to the three channels expected by SigLIP."""
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            return image.unsqueeze(0).repeat(3, 1, 1)
        if image.ndim == 3 and image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        if image.ndim == 3 and image.shape[-1] == 1:
            return image.repeat(1, 1, 3)
    return image


def _preprocess_siglip_image(processor, image):
    return processor.image_processor(
        _ensure_rgb_image(image),
        return_tensors="pt",
    )


class HFSiglipVisionModel(nn.Module):
    """SigLIP vision tower with the interface expected by this repository."""

    def __init__(self, model_name, cache_dir="", device="cpu"):
        super().__init__()
        self.device = device
        model = _from_pretrained_with_local_fallback(
            SiglipModel.from_pretrained,
            model_name,
            cache_dir=cache_dir,
        )
        processor = _from_pretrained_with_local_fallback(
            AutoProcessor.from_pretrained,
            model_name,
            cache_dir=cache_dir,
        )
        self.vision_model = deepcopy(model.vision_model)
        self.vision_head = nn.Identity()
        self.config = model.config
        self.train_preprocess = (
            lambda image: _preprocess_siglip_image(processor, image)
        )
        self.val_preprocess = (
            lambda image: _preprocess_siglip_image(processor, image)
        )

    @staticmethod
    def _inputs(value):
        if isinstance(value, torch.Tensor):
            value = {"pixel_values": value}
        if value["pixel_values"].ndim == 5:
            value = dict(value)
            value["pixel_values"] = value["pixel_values"].squeeze(1)
        return value

    def forward(self, inputs):
        outputs = self.vision_model(**self._inputs(inputs))
        return outputs.pooler_output

    def get_base_model(self):
        return self


class HFLoRASiglipVisionModel(nn.Module):
    """PEFT-wrapped SigLIP vision tower used as the merger's target model."""

    def __init__(self, model_name, cache_dir="", lora_config=None, device="cpu"):
        super().__init__()
        self.device = device
        model = _from_pretrained_with_local_fallback(
            SiglipModel.from_pretrained,
            model_name,
            cache_dir=cache_dir,
        )
        valid_arguments = {field.name for field in fields(LoraConfig)}
        provided = lora_config or {}
        peft_config = LoraConfig(
            **{
                key: value
                for key, value in provided.items()
                if key in valid_arguments
            }
        )
        self.vision_model = get_peft_model(model.vision_model, peft_config)
        self.vision_head = nn.Identity()
        self.config = model.config
        processor = _from_pretrained_with_local_fallback(
            AutoProcessor.from_pretrained,
            model_name,
            cache_dir=cache_dir,
        )
        self.train_preprocess = (
            lambda image: _preprocess_siglip_image(processor, image)
        )
        self.val_preprocess = (
            lambda image: _preprocess_siglip_image(processor, image)
        )
        self.disable_adapters = False

    @staticmethod
    def _inputs(value):
        if isinstance(value, torch.Tensor):
            value = {"pixel_values": value}
        if value["pixel_values"].ndim == 5:
            value = dict(value)
            value["pixel_values"] = value["pixel_values"].squeeze(1)
        return value

    def forward(self, inputs):
        inputs = self._inputs(inputs)
        if self.disable_adapters:
            with self.vision_model.disable_adapter():
                outputs = self.vision_model(**inputs)
        else:
            outputs = self.vision_model(**inputs)
        return outputs.pooler_output

    def get_base_model(self):
        return self


def get_model_from_config(config, device):
    finetuning = config.get("ft_config", {})
    if finetuning.get("type") == "lora":
        return HFLoRASiglipVisionModel(
            model_name=config["base_type"],
            cache_dir=config["cachedir"],
            lora_config=finetuning,
            device=device,
        )
    return HFSiglipVisionModel(
        model_name=config["base_type"],
        cache_dir=config["cachedir"],
        device=device,
    )
