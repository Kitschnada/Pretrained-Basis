"""LLaMA-3-8B with the six paper NLI adapters and original evaluation splits."""

from configs.common import NLI_TASKS, REPOSITORY_ROOT, merge_config

config = {
    "seed": 420,
    "dataset": [
        {"name": "snli", "mask_class": None},
        {"name": "mnli", "mask_class": None, "val_fraction": 0.2},
        {"name": "sick", "mask_class": None},
        {"name": "qnli", "mask_class": 1, "val_fraction": 0.2},
        {"name": "rte", "mask_class": 1, "val_fraction": 0.5},
        {"name": "scitail", "mask_class": 2},
    ],
    "model": {
        "name": "meta-llama/Meta-Llama-3-8B",
        "family": "llama",
        "ptm_path": "",
        "cachedir": str(REPOSITORY_ROOT / "data"),
        "adapter_cachedir": str(REPOSITORY_ROOT / "data"),
        "load_dtype": "float16",
        "bases": ["hoffman-lab/KnOTS-Llama3_8B_lora_R16_" + task for task in NLI_TASKS],
        "ft_config": {"type": "lora", "subtype": "peft"},
        "peft_config": {
            "task_type": "SEQ_CLS", "inference_mode": False,
            "r": 16, "lora_alpha": 16, "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    },
    "task_heads_path": str(REPOSITORY_ROOT / "heads.pt"),
    "finetuned_accuracies": {
        "snli": 92.49796416938111, "mnli": 90.30820173204279,
        "sick": 91.58173664900122, "qnli": 94.48512585812358,
        "rte": 89.85507246376812, "scitail": 96.51928504233303,
    },
    "task_merge_config": merge_config(nli=True),
    "eval_type": "logits",
}
