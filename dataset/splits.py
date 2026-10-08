"""Validate existing evaluation partitions without generating replacements."""

from numbers import Integral
from pathlib import Path

import torch


def saved_split_indices(dataset, config):
    fraction = config.get("val_fraction", 0)
    if isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    path = Path(config["shuffled_idxs"])
    if not path.is_file():
        raise FileNotFoundError("Missing saved evaluation permutation: %s. Restore it; do not regenerate it randomly." % path)
    indices = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(indices, "ndim") and indices.ndim != 1:
        raise ValueError("Saved indices must be one-dimensional: %s" % path)
    indices = indices.tolist() if hasattr(indices, "tolist") else indices
    if not isinstance(indices, (list, tuple)) or any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
        raise ValueError("Saved indices must be integers: %s" % path)
    size = len(dataset)
    if len(indices) != size or set(indices) != set(range(size)):
        raise ValueError("Saved indices are not a complete, unique permutation of this dataset (%d samples): %s. Check the dataset version/order; no split was changed." % (size, path))
    n_valid = int(size * fraction)
    if not 0 < n_valid < size:
        raise ValueError("Validation fraction produces an empty partition")
    return indices[:n_valid], indices[n_valid:]


def validate_class_mapping(*datasets):
    mappings = [dataset.class_to_idx for dataset in datasets]
    if mappings and any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError("Dataset splits have different class-to-label mappings; check missing or renamed class folders")


def validate_disjoint_imagefolders(*datasets):
    """Check identical class IDs and duplicate image identities across splits."""
    validate_class_mapping(*datasets)
    seen = set()
    for dataset in datasets:
        current = set()
        filenames = set()
        for filename, label in dataset.samples:
            path = Path(filename)
            identity = (label, path.name)
            if identity in filenames:
                raise ValueError("Duplicate sample within a split: %s" % filename)
            filenames.add(identity)
            # Both copied filenames and shared symlink targets identify leakage.
            current.update((("name", identity), ("path", str(path.resolve()))))
        if seen.intersection(current):
            raise ValueError("Overlapping images in dataset partitions; existing files were not changed")
        seen.update(current)
