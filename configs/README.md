# Main experiment configurations

The four public configurations preserve the paper's task order, rank-16 LoRA
checkpoints, alpha=16, preprocessing, heads, and validation/test split definitions.

| Configuration | Model | Tasks |
| --- | --- | --- |
| `vit_b32.py` | CLIP ViT-B/32 | Eight vision tasks |
| `vit_l14.py` | CLIP ViT-L/14 | Eight vision tasks |
| `siglip.py` | SigLIP base patch16-224 | Eight vision tasks |
| `llama3_8b.py` | LLaMA-3-8B | Six NLI tasks |

`common.py` contains shared configuration builders. The supported merge spaces
are `full`, `knots`, `core`, and `pgsb`. Merge operators are `tv`, `ties`,
`dare-ties`, `tsv`, and `cart`; setting `isotropize=True` adds Iso-C. The runner's
`iso-c` method shorthand means `tv` with isotropization enabled.

Run from the repository root:

```bash
bash run_experiments.sh --model vit-b32 --spaces all --methods ties --run-id main_v1
bash run_experiments.sh --model vit-l14 --spaces pgsb --methods dare-ties --isotropize
bash run_experiments.sh --model llama3-8b --model-path /path/to/complete/model/snapshot
```

Linear search is the default. It retains the existing coordinate-search protocol,
including its early stopping, vision descending top-K grid, and NLI ascending
top-K grid. The seed defaults to 420. `--mode direct` evaluates fixed merge
parameters; it does not run validation search.

The runner writes a literal `config.py`, `resolved_config.json`, `command.json`,
`run.log`, and `result.json` for each case. Linear-search cases also write
`result.search.jsonl` containing every evaluated validation candidate.
Existing results are never overwritten; identical completed cases are skipped.
An interrupted/incomplete case must be rerun using a new `--run-id`.

Paths are repository-relative by default and can be overridden using
`--cache-dir`, `--adapter-cache-dir`, `--data-root`, `--nli-data-cache`,
`--head-root`, `--shuffled-index-root`, `--nli-head-path`, and `--model-path`.
`--head-root` is the exact model-specific directory containing `<task>_head.pt`.
Dataset ROOT defaults remain the original loader defaults unless explicitly
overridden. `--nli-data-cache` separately controls the NLI dataset cache.

CLIP and NLI use the original Hugging Face adapters by default. SigLIP uses
`checkpoints/siglip/checkpoints/<task>/rank_16/<task>_siglip_lora_r16.pt`.
For local checkpoints, set `--checkpoint-root` or supply an explicit
`--checkpoint-template` containing `{root}`, `{task}`, and/or `{rank}`.

Original Hub adapters use the recorded single-task baselines. Custom adapters
and SigLIP require matching adjacent `manifest.json` files with fractional
`test.accuracy`, or `--baseline-json` containing task-to-accuracy percentages.
For example, a baseline JSON may contain `{"stanford_cars": 74.0, ...}` with
one entry for every evaluated task. `--allow-missing-baselines` explicitly
allows raw-accuracy selection when complete baselines are unavailable.
SigLIP never inherits CLIP baselines.

Missing heads can optionally be downloaded from an explicitly supplied
`--heads-repo-id`. `--heads-file-template` selects the path inside that repository
(default `{task}_head.pt` for vision and `heads.pt` for NLI); `--heads-revision`
selects its revision. Existing heads are preserved. `--offline` disables Hub
network access. `--dry-run` prints the planned commands without downloads,
model loading, or result-file writes.
