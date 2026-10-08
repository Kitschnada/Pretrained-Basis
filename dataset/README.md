# Main-experiment datasets

The loaders retain the original task order, class mappings, tokenization, split fractions and saved vision indices. The unified launcher runs from the repository root so existing relative paths keep their meaning.

## Download and preparation

Run preparation before loading a large model:

```bash
python -m dataset.prepare --tasks vision --data-root ./data
python -m dataset.prepare --tasks nli --nli-data-cache ./data/datasets
python -m dataset.prepare --tasks dtd,resisc45 --data-root ./data
python -m dataset.prepare --tasks all --dry-run
```

Use `--offline` to forbid downloads while allowing preparation from local files or archives. Existing data is reused; invalid completed downloads are reported rather than overwritten. Transfers have bounded connection/read timeouts and retries. Checksum-verified partial transfers can resume. File locks prevent two preparations from racing, archives reject unsafe paths, and split generation uses symlinks rather than copying images. Keep raw image directories after preparation because those symlinks refer to them.

Reports are saved with unique names in `<data-root>/.preparation/`, including task status, source/protocol information, counts, and the evaluation permutation checksum. A companion JSONL records completed tasks incrementally. Failures return a nonzero exit status, but other selected tasks are still attempted. Use `--report /path/to/new.json` for a custom report location. No report is overwritten. Interrupted image-folder preparation leaves a marker and must finish before the tree can be treated as complete.

`--include-train` includes Cars, GTSRB, and SVHN training data; the default downloads only their evaluation data. The other loaders retain their required original layouts. NLI preparation downloads all original source splits without loading a tokenizer or model.

### Download sources

| Dataset | Preparation source and compatibility |
| --- | --- |
| Cars196 | [Original-file mirror](https://huggingface.co/datasets/iharabukhouski/stanford_cars), with its version pinned in the downloader; original JPEG filenames and MAT annotation order, not a reindexed HF image dataset. The old Stanford download URLs are unavailable. |
| DTD | [Oxford VGG archive](https://www.robots.ox.ac.uk/~vgg/data/dtd/); partition 1 train/validation/test lists remain separate. |
| EuroSAT | [Dataset author's Zenodo RGB archive](https://zenodo.org/records/7711810); original local split manifests are required for reproduction. |
| GTSRB | Original ERDA competition ZIPs and torchvision checksums; test labels stay in original CSV order. |
| MNIST | Torchvision's HTTPS OSSCI mirror; original IDX files and ordering. |
| RESISC45 | [TorchGeo's archive and lists](https://huggingface.co/datasets/isaaccorley/resisc45), with their version pinned in the downloader; mirrored split files match the original repository's checksums. |
| SUN397 | [Full-resolution archive and official partitions](https://huggingface.co/zhengli97/SUN397_Dataset), with their version pinned in the downloader; uses partition 01 and the saved test permutation. The full raw archive is approximately 39 GB. |
| SVHN | Original MAT files, with a Hugging Face mirror fallback; every source must match the original torchvision MD5. |

Hugging Face downloads honor `HF_ENDPOINT` if you explicitly configure a mirror. TLS certificate verification remains enabled. These sources were checked using upstream source/metadata; this repair did not download every complete archive.

### Reusing raw vision images

If raw data already exists elsewhere, supply its location for one task:

```bash
python -m dataset.prepare --tasks sun397 --data-root ./data \
  --source-dir /path/to/raw/SUN397
python -m dataset.prepare --tasks dtd --data-root ./data \
  --source-dir /path/to/dtd
```

For DTD, the raw directory contains `images/` and `labels/`. For Cars, it contains `cars_train/`, `cars_test/`, and the original annotations. For SUN397, it contains the original `a/`, `b/`, etc. hierarchy. For RESISC45 and EuroSAT, it contains the original class directories. The retained `parsing/` commands are also safe CLI wrappers; they no longer require editing empty paths in scripts.

### EuroSAT: do not silently change partitions

An existing `EuroSAT_splits/{train,val,test}` tree is reused unchanged after class/non-overlap checks. Its historical sample membership is not recorded in the repository; a fresh random split cannot reproduce it. Restore the original prepared tree, or provide original filename manifests:

```bash
python -m dataset.prepare --tasks eurosat --data-root ./data \
  --source-dir /path/to/EuroSAT_RGB \
  --eurosat-split-dir /path/to/original/splits
```

Manifests are named `train.txt`, `val.txt`, and `test.txt` (also accepts `eurosat-<split>.txt`), one `class/filename.jpg` per line. Duplicate/overlapping names and missing images fail validation.

For a deliberately different experiment, `--eurosat-protocol torchgeo` uses the published 16,200/5,400/5,400 split. It is **not the paper split**. The runner will not use the paper EuroSAT baseline for this recorded protocol: supply a matching value through `--baseline-json`, or explicitly select `--allow-missing-baselines`. New preparation records protocol and membership fingerprints in `.pgsb-preparation.json` inside the split directory.

## Eight vision tasks

| Task ID | Existing default data location |
| --- | --- |
| `stanford_cars` | `data/stanford_cars/` with the existing cars images and annotations |
| `dtd` | `data/dtd/{train,val,test}/` |
| `eurosat` | `data/EuroSAT_splits/{train,val,test}/` |
| `gtsrb` | `data/gtsrb/` |
| `mnist` | `data/MNIST/` (torchvision layout) |
| `resisc45` | `data/resisc45/` with existing image and split-list layout |
| `sun397` | `data/SUN397/{train,test}/` |
| `svhn` | `data/svhn/` |

The module-level loader `ROOT` values remain authoritative. Setting an entry's `dir` alone did not redirect these loaders in the original code. Use the launcher's explicit `--data-root` (environment variable `PGSB_VISION_DATA_ROOT`) to change their shared root. Cars and DTD append their task subdirectories; the other wrappers keep their existing subdirectory handling.

Cars, GTSRB, MNIST, SUN397 and SVHN use the saved permutations in `shuffled_idxs/` to reserve 20% of the original test partition for validation. The remaining 80% is the final test partition. DTD, EuroSAT and RESISC45 use their existing validation/test splits. No new random split is introduced during merging.

Keep your existing prepared data. The scripts in `parsing/` retain the DTD, SUN397 and RESISC45 preparation conventions and accept command-line paths. They are not a requirement to recreate already available datasets. Saved permutations must be complete, unique integer permutations of the original dataset size; missing or invalid indices cause an error rather than a new random split. Cars training now correctly reads training annotations; its legacy `cars_train` alias previously selected the test branch.

## Six NLI tasks

The task IDs and data remain SNLI (`stanfordnlp/snli`), MNLI (`SetFit/mnli`, matched validation), SICK (the original [author-published Zenodo archive](https://zenodo.org/records/2787612), also identified as `RobZamp/sick`), QNLI (`SetFit/qnli`), RTE (`SetFit/rte`) and SciTail (`allenai/scitail`, `tsv_format`). Fresh downloads use fixed source files/revisions rather than Hub recursive file discovery or a remote SICK loading script. SNLI/SciTail Parquet files and SICK's ZIP have published checksum verification.

SNLI, SICK and SciTail retain their dataset-provided validation/test partitions. MNLI and QNLI retain the deterministic 20% validation reservation from their original validation data; RTE retains 50%. Class masks and label mappings remain in the task-specific loaders and model configuration. The tokenizer and maximum lengths are unchanged.

The existing explicit NLI cache default is retained. To deliberately move or redownload the NLI dataset cache, use `--nli-data-cache /path/to/cache` (`PGSB_NLI_DATA_CACHE`). This is separate from the model/adapter cache. Point it at the cache containing the existing prepared datasets to reuse them.

Preparation uses `<repo>/data/datasets` by default; pass the same `--nli-data-cache` to preparation and merging when overriding it. Existing original Arrow builder caches are read without network access and without reading transformed `cache-*.arrow` files. Multiple complete revisions are reported as ambiguous rather than choosing one silently. Fresh prepared datasets are saved under `<cache>/pgsb_nli/<task>/`, with source metadata; later runs load them locally.

To choose a particular existing dataset/revision:

```bash
python -m dataset.prepare --tasks mnli --nli-data-cache ./data/datasets \
  --nli-local-path /path/to/original/mnli/cache/revision --offline

# Use the same explicit source during merging
export PGSB_NLI_MNLI_PATH=/path/to/original/mnli/cache/revision
```

The path may contain a saved DatasetDict, an original builder's Arrow files and `dataset_info.json`, or `train`, `validation`, and `test` JSONL/JSON/Parquet files. Equivalent per-task variables are available for the other five NLI tasks. Schemas, labels, counts and Arrow shard completeness are checked. Challenge test splits with hidden labels are not used for evaluation. QNLI/RTE retain the checkpoint label remapping from `1` to `2`; SciTail retains entailment=`0`, neutral=`1`.

## Classification heads

CLIP vision heads are class-vector matrices at `data/heads/ViT-B-32-CLIP/` or `data/heads/ViT-L-14-CLIP/`. Reuse existing heads to reproduce your experiments. The retained `parsing/generate_clip_heads.py` can create missing CLIP heads using the original prompt templates, prompt averaging, normalization and logit scale; it keeps existing files unless `--force` is supplied.

SigLIP uses its own head vectors at `data/siglip_heads/google--siglip-base-patch16-224/`; CLIP heads are not interchangeable with them. NLI uses the task-specific classification heads in the root `heads.pt`. All head paths can be overridden through the unified launcher.
