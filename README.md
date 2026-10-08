# Pretrained-Basis

Pretrained-Basis provides the main experiments for **PGSB: Pretrained-Guided Shared Basis for LoRA Model Merging**, covering LoRA checkpoint loading, merging, validation-based linear search, and test evaluation on eight vision tasks and six NLI tasks.

| Model | Tasks | Configuration |
| --- | --- | --- |
| CLIP ViT-B/32 | Cars196, DTD, EuroSAT, GTSRB, MNIST, NWPU-RESISC45, SUN397, SVHN | `configs/vit_b32.py` |
| CLIP ViT-L/14 | Same eight vision tasks | `configs/vit_l14.py` |
| SigLIP base-patch16-224 | Same eight vision tasks | `configs/siglip.py` |
| LLaMA-3-8B | SNLI, MNLI, SICK, QNLI, RTE, SciTail | `configs/llama3_8b.py` |

Supported merge spaces: `full`, `knots`, `core`, and `pgsb`. Supported operators: `tv`, `ties`, `dare-ties`, `tsv`, and `cart`. Add `--isotropize` to combine any operator with Iso-C, or use `--methods iso-c` as shorthand for TV + Iso-C. TV corresponds to Task Arithmetic in the paper.

## Environment

```bash
conda env create -f environment.yml
conda activate pgsb
```

The environment uses Python 3.9, PyTorch 2.0.1, torchvision 0.15.2, Transformers 4.45.2, and PEFT 0.15.2. SentencePiece is included for SigLIP. An existing working merge environment can also be used without reinstalling.

The repository cleanup was checked with syntax, configuration, search-flow, and CPU numerical comparisons. Full GPU checkpoint experiments and a fresh Conda environment installation have not been rerun.

## Preparing datasets

Prepare data before launching GPU experiments. This command downloads missing files, reuses existing local data, and validates the evaluation partitions without loading any model:

```bash
python -m dataset.prepare --tasks vision --data-root ./data
python -m dataset.prepare --tasks nli --nli-data-cache ./data/datasets

# Prepare one task, or reuse local files/archives without network access
python -m dataset.prepare --tasks stanford_cars --data-root ./data
python -m dataset.prepare --tasks vision --data-root ./data --offline
```

Downloads use timeouts, retries, checksums where published, and atomic completion. Saved permutations are validated, not regenerated. Existing prepared folders and NLI caches take priority. Unique preparation reports are written under `data/.preparation/`; any failed task makes the command exit with an error. `--include-train` also downloads Cars, GTSRB, and SVHN training files, which are not needed for merging.

**EuroSAT requires special care:** keep your existing `EuroSAT_splits/` directory. The historical split membership is not stored in this repository, so fresh preparation requires your original filename manifests:

```bash
python -m dataset.prepare --tasks eurosat --data-root ./data \
  --eurosat-split-dir /path/to/original/manifests
```

The directory must contain `train.txt`, `val.txt`, and `test.txt`, with one `class/filename.jpg` per line. An explicitly selected `--eurosat-protocol torchgeo` is available for a new 60/20/20 split, **not** reproduction of the paper split; it requires a matching EuroSAT single-task baseline or raw-accuracy reporting. Existing folders are never resplit.

See [dataset/README.md](dataset/README.md) for sources, local raw-data reuse, NLI cache paths, and partition details. No multi-GB dataset downloads or full GPU experiments were run as part of these download repairs.

## Running experiments

Run commands from the repository root:

```bash
# ViT-B/32: TIES linear search across all four spaces
bash run_experiments.sh --model vit-b32 --spaces all --methods ties --run-id vitb_ties

# ViT-L/14: PGSB + TSV + Iso-C
bash run_experiments.sh --model vit-l14 --spaces pgsb --methods tsv --isotropize --run-id vitl_tsv_iso

# Run Core Space separately on another GPU
bash run_experiments.sh --model vit-b32 --spaces core --methods ties --gpu 1 --run-id vitb_core

# Preview commands for all spaces and five operators without downloading or evaluating
bash run_experiments.sh --model vit-b32 --spaces all --methods tv,ties,dare-ties,tsv,cart --dry-run
```

Linear search is the default. Experiments within each command run sequentially. To evaluate fixed parameters without validation search, use `--mode direct`:

```bash
bash run_experiments.sh --model vit-b32 --spaces pgsb --methods ties \
  --mode direct --scaling-coeffs 0.6 --top-k 10 --run-id vitb_direct
```

To run all four spaces and five operators, both with and without Iso-C:

```bash
bash run_experiments.sh --model vit-b32 --spaces all \
  --methods tv,ties,dare-ties,tsv,cart --run-id vitb_main

bash run_experiments.sh --model vit-b32 --spaces all \
  --methods tv,ties,dare-ties,tsv,cart --isotropize --run-id vitb_main
```

## Checkpoints, models, and paths

The main configurations use rank-16 LoRA. CLIP and NLI default to the original `hoffman-lab/KnOTS-*` Hugging Face adapters. Models and adapters are cached under the repository's `data/` directory by default. For a local base model, provide a complete model directory or snapshot:

```bash
bash run_experiments.sh --model llama3-8b --spaces pgsb --methods ties \
  --model-path /path/to/llama3-8b \
  --run-id llama_nli
```

NLI uses the included `heads.pt`; override it with `--nli-head-path`. FP16 is the default. For multi-GPU NLI inference, add `--gpu 0,1 --model-parallel-gpus 2`. Ensure sufficient GPU and host memory, particularly for LLaMA Full and KnOTS merging.

SigLIP defaults to the following local paths:

```text
checkpoints/siglip/checkpoints/<task>/rank_16/<task>_siglip_lora_r16.pt
data/siglip_heads/google--siglip-base-patch16-224/<task>_head.pt
```

```bash
bash run_experiments.sh --model siglip --spaces pgsb --methods ties \
  --checkpoint-root ./checkpoints/siglip/checkpoints \
  --head-root ./data/siglip_heads/google--siglip-base-patch16-224 \
  --run-id siglip_ties
```

`--head-root` must point directly to the directory containing the eight `<task>_head.pt` files. Default CLIP head directories are `data/heads/ViT-B-32-CLIP` and `data/heads/ViT-L-14-CLIP`. Reuse existing heads when available. To generate missing CLIP heads:

```bash
python dataset/parsing/generate_clip_heads.py --config vit_b32.py
python dataset/parsing/generate_clip_heads.py --config vit_l14.py
```

To download missing heads from Hugging Face, add `--heads-repo-id <your-repo-id>`. Vision head filenames default to `<task>_head.pt` within that repository. For a different layout, use an option such as `--heads-file-template 'siglip_heads/google--siglip-base-patch16-224/{task}_head.pt'`. Existing local heads are preserved. The default NLI head filename is `heads.pt`.

For custom adapters, use `--checkpoint-root` and `--checkpoint-template`. Templates support `{root}`, `{task}`, and `{rank}`:

```bash
bash run_experiments.sh --model vit-b32 --checkpoint-root /path/to/checkpoints \
  --checkpoint-template '{root}/vision/{task}/rank_{rank}/{task}_lora_r{rank}.pt' \
  --baseline-json /path/to/single_task_accuracy.json --run-id local_adapters
```

Model caches and dataset paths are configured separately. Use `--cache-dir` and `--adapter-cache-dir` for models and adapters, `--data-root` to override the vision data root, and `--nli-data-cache` to override the NLI dataset cache. Omitting the latter two preserves the original dataset-loader defaults. Saved shuffled indices are in `dataset/shuffled_idxs/`; override their location with `--shuffled-index-root`. See [dataset/README.md](dataset/README.md) for dataset layouts and splits.

`--offline` disables model and dataset network access and requires models, tokenizers, and dataset files to be available locally. Local dataset archives can still be extracted and prepared. Use `--python` to select the evaluator's interpreter; the Bash launcher's interpreter is controlled by the `PYTHON` environment variable.

## Metrics and linear search

Raw accuracy is reported as a percentage for each task. Normalized accuracy is:

```text
normalized_accuracy = merged_accuracy / single_task_accuracy × 100
```

Aggregate metrics are arithmetic means across tasks. Original Hub adapters use the recorded single-task baselines. SigLIP and custom adapters read `test.accuracy` (a fraction between 0 and 1) from matching adjacent `manifest.json` files, or accept `--baseline-json` with percentage values such as `{"stanford_cars": 74.0, ...}`. Missing baselines raise an error by default. Use `--allow-missing-baselines` explicitly to allow raw-accuracy reporting and raw validation accuracy for selection; missing baselines are not treated as normalized scores.

The evaluators retain the original parameter-by-parameter linear search, traversal order, and early stopping, rather than an exhaustive Cartesian-product grid search. Vision and NLI preserve their respective search protocols. Hyperparameters are selected on validation splits only; test evaluation follows selection. The original validation/test splits and saved indices are retained.

## Results

Each experiment writes to:

```text
results/main/<run-id>/<model>/<space>_<method>_<mode>/
    config.py
    resolved_config.json
    command.json
    run.log
    result.json
    result.search.jsonl       # Every evaluated validation candidate in linear-search mode
```

The final JSON contains test metrics and selected parameters; validation candidates are recorded separately in JSONL. Completed runs with matching configurations are skipped. Incomplete results or results with different configurations are never overwritten: use a new `--run-id` to rerun them. Ctrl+C stops the active subprocess group and preserves its logs.

## Repository layout

```text
run_experiments.{sh,py}   Unified experiment launcher
task_merger.py           Full, KnOTS, Core Space, and PGSB merging
merging_functions.py     Merge operators
configs/                Model configurations and shared settings
eval_scripts/           Vision/NLI evaluation and linear search
models/                 CLIP and SigLIP model interfaces
dataset/                Dataset loaders, split indices, and preparation tools
```

Local backups, datasets, checkpoints, generated results, and private tool configuration are excluded from version control. Do not include `backups/` or private configuration when publishing or packaging the repository.

## Acknowledgements and license

This repository builds on [Core Space Merging](https://github.com/apanariello4/core-space-merging) and [KnOTS](https://github.com/gstoica27/KnOTS), including their checkpoints and evaluation protocols. The upstream [Apache-2.0 license](LICENSE) is retained.

For the upstream Core Space implementation:

```bibtex
@inproceedings{panariello2025accurate,
  title = {Accurate and Efficient Low-Rank Model Merging in Core Space},
  author = {Panariello, Aniello and Marczak, Daniel and Magistri, Simone and Porrello, Angelo and Twardowski, Bart{\l}omiej and Bagdanov, Andrew D. and Calderara, Simone and van de Weijer, Joost},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025}
}
```
