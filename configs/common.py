"""Main-experiment configuration builders; importing these never loads models."""

from pathlib import Path

from accuracies import get_vision_accuracies


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISION_TASKS = (
    "stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn",
)
NLI_TASKS = ("snli", "mnli", "sick", "qnli", "rte", "scitail")
SHUFFLED_INDICES = {
    "stanford_cars": "cars_shuffled_idxs.pt",
    "gtsrb": "gtsrb_shuffled_idxs.pt",
    "mnist": "mnist_shuffled_idxs.pt",
    "sun397": "sun397_shuffled_idxs.pt",
    "svhn": "svhn_shuffled_idxs.pt",
}


def merge_config(nli=False):
    """Keep the existing coordinate-search grids and their traversal order."""
    scaling = [0.1 + 0.1 * index for index in range(99)]
    return {
        "representation": "matrix_per_layer",
        "merge_space": "pgsb",
        "merge_method": "ties",
        "scaling_coeffs": 0.3 if nli else 0.1,
        "topK": 10,
        "isotropize": False,
        "linearsearch_initial_scaling_coeffs": 0.1,
        "linearsearch_initial_topK": 10,
        "linearsearch_initial_dare_pruning_coeffs": 1e-5,
        "dare_pruning_coeffs": 0.1,
        "cart_pruning_rank": 0.04,
        "cart_scaling_coeffs": 0.1,
        "search_scaling_coeffs": scaling,
        "search_topK": list(range(10, 101, 10)) if nli else list(range(100, 0, -10)),
        "search_dare_pruning_coeffs": [1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        "search_cart_pruning_rank": [0.04, 0.08, 0.16, 0.32],
        "search_cart_scaling_coeffs": scaling.copy(),
        "pgsb_rank_mode": "exact",
        "pgsb_eps": 1e-6,
        "model_parallel": False,
        "model_parallel_dtype": "float16",
        "model_parallel_num_gpus": 1,
    }


def vision_config(base_type, architecture, adapter_prefix=None):
    cache = REPOSITORY_ROOT / "data"
    siglip = adapter_prefix is None
    head_root = (
        cache / "siglip_heads" / "google--siglip-base-patch16-224"
        if siglip else cache / "heads" / architecture
    )
    datasets = []
    for task in VISION_TASKS:
        entry = {
            "name": task,
            "shuffle_train": True,
            "crop_ratio": 1.0,
            "clip_encodings": str(head_root / (task + "_head.pt")),
            "batch_size": 16 if architecture == "ViT-L-14-CLIP" else 32,
            "num_workers": 8 if task in ("mnist", "svhn") else 16,
        }
        if task in SHUFFLED_INDICES:
            entry.update({
                "val_fraction": 0.2,
                "shuffled_idxs": str(REPOSITORY_ROOT / "dataset" / "shuffled_idxs" / SHUFFLED_INDICES[task]),
            })
        datasets.append(entry)
    if siglip:
        checkpoint_root = REPOSITORY_ROOT / "checkpoints" / "siglip" / "checkpoints"
        bases = [str(checkpoint_root / task / "rank_16" / (task + "_siglip_lora_r16.pt")) for task in VISION_TASKS]
    else:
        bases = [adapter_prefix + task for task in VISION_TASKS]
    return {
        "seed": 420,
        "dataset": datasets,
        "model": {
            "name": "hf_siglip" if siglip else "hf_clip",
            "base_type": base_type,
            "cachedir": str(cache),
            "adapter_cachedir": str(cache),
            "bases": bases,
            "ft_config": {
                "type": "lora", "r": 16, "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "lora_dropout": 0.1, "bias": "none",
            },
        },
        "task_merge_config": merge_config(),
        # An explicit empty mapping prevents using CLIP baselines for SigLIP.
        "finetuned_accuracies": {} if siglip else get_vision_accuracies(base_type, 16, "lora"),
        "legacy_finetuned_accuracy_tasks": [],
        "eval_type": "clip",
    }
