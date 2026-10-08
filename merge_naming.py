"""Canonical public configuration for the PGSB main experiments."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


MERGE_SPACES = ("full", "knots", "core", "pgsb")
MERGE_METHODS = ("tv", "ties", "dare-ties", "tsv", "cart")
PGSB_OPTIONS = {"pgsb_rank_mode", "pgsb_energy", "pgsb_eps"}


def canonicalize_merge_space(value: Any) -> Any:
    """Validate the four supported layerwise merge spaces."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in MERGE_SPACES:
        raise ValueError(
            f"Unknown merge space {value!r}; choose from {MERGE_SPACES}."
        )
    return normalized


def canonicalize_merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate merge options; Iso-C alone means TV with isotropization."""
    result = dict(config or {})
    if "merge_space" in result:
        result["merge_space"] = canonicalize_merge_space(result["merge_space"])

    if result.get("merge_method") is not None:
        method = str(result["merge_method"]).strip().lower()
        if method == "iso-c":
            method = "tv"
            result["isotropize"] = True
        if method not in MERGE_METHODS:
            raise ValueError(
                f"Unknown merge method {method!r}; choose from "
                f"{MERGE_METHODS} or 'iso-c'."
            )
        result["merge_method"] = method

    representation = result.get("representation", "matrix_per_layer")
    if representation != "matrix_per_layer":
        raise ValueError("Only representation='matrix_per_layer' is supported.")

    unsupported = sorted(
        key for key in result
        if key.startswith("pgsb_") and key not in PGSB_OPTIONS
    )
    if unsupported:
        raise ValueError(f"Unsupported PGSB configuration options: {unsupported}")

    rank_mode = str(result.get("pgsb_rank_mode", "exact")).lower()
    if rank_mode not in ("exact", "energy"):
        raise ValueError("pgsb_rank_mode must be 'exact' or 'energy'.")
    if "pgsb_rank_mode" in result:
        result["pgsb_rank_mode"] = rank_mode
    energy = float(result.get("pgsb_energy", 0.99))
    eps = float(result.get("pgsb_eps", 1e-8))
    if not math.isfinite(energy) or not 0 < energy <= 1:
        raise ValueError("pgsb_energy must be finite and in (0, 1].")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("pgsb_eps must be finite and positive.")
    return result


def get_pgsb_option(
    config: Mapping[str, Any] | None,
    suffix: str,
    default: Any = None,
) -> Any:
    """Read a validated PGSB option."""
    return canonicalize_merge_config(config).get(f"pgsb_{suffix}", default)
