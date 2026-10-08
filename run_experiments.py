#!/usr/bin/env python3
"""Run the paper vision/NLI merging experiments with the existing evaluators."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pprint
import re
import shlex
import shutil
import signal
import subprocess
import sys

from configs.common import REPOSITORY_ROOT, SHUFFLED_INDICES


MODELS = {
    "vit-b32": "configs.vit_b32",
    "vit-l14": "configs.vit_l14",
    "siglip": "configs.siglip",
    "llama3-8b": "configs.llama3_8b",
}
SPACES = ("full", "knots", "core", "pgsb")
METHODS = ("tv", "ties", "dare-ties", "tsv", "cart", "iso-c")


def absolute_path(value):
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def model_source(value):
    """Retain Hub IDs, resolving explicit/existing local paths."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    if Path(expanded).is_absolute() or expanded.startswith(".") or Path(expanded).exists():
        return str(absolute_path(expanded))
    return expanded


def selected_values(value, allowed):
    values = list(allowed) if value.lower() == "all" else [part.strip().lower() for part in value.split(",")]
    if not values or any(part not in allowed for part in values):
        raise ValueError("Choose comma-separated values from " + ", ".join(allowed) + ", or all")
    if len(values) != len(set(values)):
        raise ValueError("Repeated entries are not allowed: " + value)
    return values


def read_baselines(path, tasks):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "finetuned_accuracies" in payload:
        payload = payload["finetuned_accuracies"]
    if not isinstance(payload, dict):
        raise ValueError("Baseline JSON must map task names to accuracy percentages")
    unknown = set(payload) - set(tasks)
    if unknown:
        raise ValueError("Unknown baseline tasks: " + ", ".join(sorted(unknown)))
    baselines = {}
    for task, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError("Baseline for %s must be a finite accuracy percentage in (0, 100]" % task)
        baselines[task] = float(value)
    return baselines


def manifest_baseline(checkpoint, task, model):
    """Read original training-export metadata without opening tensor weights."""
    checkpoint = Path(checkpoint)
    manifest = (checkpoint if checkpoint.is_dir() else checkpoint.parent) / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status", "complete") != "complete":
        raise ValueError("Checkpoint manifest is not complete: " + str(manifest))
    job = payload.get("job", {})
    expected = {"task": task, "rank": 16, "method": "lora"}
    if model == "siglip":
        expected["family"] = "siglip"
    for key, value in expected.items():
        if key in job and job[key] != value:
            raise ValueError("Manifest %s mismatch in %s: expected %r, got %r" % (key, manifest, value, job[key]))
    adapter = payload.get("adapter", {})
    for key, value in (("rank", 16), ("alpha", 16), ("method", "lora")):
        if key in adapter and adapter[key] != value:
            raise ValueError("Manifest adapter %s must be %r: %s" % (key, value, manifest))
    accuracy = payload.get("test", {}).get("accuracy")
    if accuracy is None:
        return None
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 < accuracy <= 1:
        raise ValueError("Manifest test.accuracy must be in (0, 1]: " + str(manifest))
    return 100.0 * accuracy


def prepare_base_config(args):
    config = deepcopy(importlib.import_module(MODELS[args.model]).config)
    model = config["model"]
    nli = args.model == "llama3-8b"
    tasks = [entry["name"] for entry in config["dataset"]]
    config["seed"] = args.seed
    model["cachedir"] = str(absolute_path(args.cache_dir))
    model["adapter_cachedir"] = str(absolute_path(args.adapter_cache_dir or args.cache_dir))
    model["local_files_only"] = args.offline
    if args.model_path:
        model["name" if nli else "base_type"] = model_source(args.model_path)
    if nli:
        config["task_heads_path"] = str(absolute_path(args.nli_head_path or config["task_heads_path"]))
        for entry in config["dataset"]:
            entry["model_name_or_path"] = model["name"]
        model["load_dtype"] = args.load_dtype
    for entry in config["dataset"]:
        if args.batch_size is not None:
            entry["batch_size"] = args.batch_size
        if args.num_workers is not None:
            entry["num_workers"] = args.num_workers
        if not nli and args.head_root:
            entry["clip_encodings"] = str(absolute_path(args.head_root) / (entry["name"] + "_head.pt"))
        if not nli and args.shuffled_index_root and entry["name"] in SHUFFLED_INDICES:
            entry["shuffled_idxs"] = str(absolute_path(args.shuffled_index_root) / SHUFFLED_INDICES[entry["name"]])

    custom_adapters = bool(args.checkpoint_root or args.checkpoint_template)
    if custom_adapters:
        root = absolute_path(args.checkpoint_root or (REPOSITORY_ROOT / "checkpoints"))
        if args.checkpoint_template:
            template = args.checkpoint_template
        elif args.model == "siglip":
            template = "{root}/{task}/rank_{rank}/{task}_siglip_lora_r{rank}.pt"
        elif nli:
            template = "{root}/{task}.pt"
        else:
            template = "{root}/vision/{task}/rank_{rank}/{task}_lora_r{rank}.pt"
        model["bases"] = [model_source(template.format(root=root, task=task, rank=16)) for task in tasks]
        # Main-experiment baseline scores describe the original Hub adapters.
        config["finetuned_accuracies"] = {}
    if custom_adapters or args.model == "siglip":
        for task, checkpoint in zip(tasks, model["bases"]):
            if Path(checkpoint).exists():
                accuracy = manifest_baseline(checkpoint, task, args.model)
                if accuracy is not None:
                    config["finetuned_accuracies"][task] = accuracy
    supplied_baselines = read_baselines(absolute_path(args.baseline_json), tasks) if args.baseline_json else {}
    config["finetuned_accuracies"].update(supplied_baselines)
    if not nli:
        provenance_path = absolute_path(args.data_root or (REPOSITORY_ROOT / "data")) / "EuroSAT_splits" / ".pgsb-preparation.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if "not paper split" in provenance.get("protocol", "") and "eurosat" not in supplied_baselines:
                config["finetuned_accuracies"].pop("eurosat", None)
                print("EuroSAT uses an explicitly selected non-paper split. Supply its matching single-task score via --baseline-json, or use --allow-missing-baselines for raw accuracy.", file=sys.stderr)
    missing = [task for task in tasks if task not in config["finetuned_accuracies"]]
    if missing and not args.allow_missing_baselines and not args.dry_run:
        raise ValueError(
            "Missing single-task baselines for %s. Supply --baseline-json (percentages), "
            "matching checkpoint manifest.json files, or --allow-missing-baselines for raw accuracy."
            % ", ".join(missing)
        )
    merge = config["task_merge_config"]
    merge["model_parallel"] = args.model_parallel_gpus > 1
    merge["model_parallel_num_gpus"] = args.model_parallel_gpus
    merge["model_parallel_dtype"] = args.load_dtype
    for arg, key in ((args.scaling_coeffs, "scaling_coeffs"), (args.top_k, "topK"),
                     (args.dare_pruning, "dare_pruning_coeffs"), (args.cart_pruning_rank, "cart_pruning_rank"),
                     (args.cart_scaling, "cart_scaling_coeffs")):
        if arg is not None:
            merge[key] = arg
    return config


def runtime_environment(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    cache = str(absolute_path(args.cache_dir))
    env["HF_HOME"] = cache
    env["HF_HUB_CACHE"] = cache
    env["HUGGINGFACE_HUB_CACHE"] = cache
    env["TRANSFORMERS_CACHE"] = cache
    if args.data_root:
        env["PGSB_VISION_DATA_ROOT"] = str(absolute_path(args.data_root))
    if args.nli_data_cache:
        env["PGSB_NLI_DATA_CACHE"] = str(absolute_path(args.nli_data_cache))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PGSB_DATA_OFFLINE"):
        env[key] = "1" if args.offline else "0"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    return env


def preflight(config):
    required = []
    if "task_heads_path" in config:
        required.append(config["task_heads_path"])
    for entry in config["dataset"]:
        for key in ("clip_encodings", "shuffled_idxs"):
            if key in entry:
                required.append(entry[key])
    for checkpoint in config["model"]["bases"]:
        if Path(checkpoint).is_absolute() or checkpoint.startswith("."):
            required.append(checkpoint)
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required heads, indices, or local checkpoints:\n" + "\n".join(missing))


def download_requested_heads(args, config):
    """Fetch explicitly requested heads; never replace an existing local head."""
    if not args.heads_repo_id:
        return
    from huggingface_hub import hf_hub_download

    if args.model == "llama3-8b":
        heads = [("nli", config["task_heads_path"])]
        template = args.heads_file_template or "heads.pt"
    else:
        heads = [(entry["name"], entry["clip_encodings"]) for entry in config["dataset"]]
        template = args.heads_file_template or "{task}_head.pt"
    for task, destination in heads:
        destination = Path(destination)
        if destination.exists():
            continue
        filename = template.format(task=task, model=args.model)
        cached = hf_hub_download(
            repo_id=args.heads_repo_id, filename=filename,
            revision=args.heads_revision, cache_dir=str(absolute_path(args.cache_dir)),
            local_files_only=args.offline,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            try:
                with Path(cached).open("rb") as source:
                    shutil.copyfileobj(source, target)
            except BaseException:
                destination.unlink()
                raise
        print("Downloaded head: " + str(destination), flush=True)


def write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def completed_result(path, tasks, require_normalized=False):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Result must be a JSON object: " + str(path))
    payload = payload.get("test_result", payload)
    if not isinstance(payload, dict):
        raise ValueError("test_result must be a JSON object: " + str(path))
    keys = list(tasks) + ["Average_acc"]
    if require_normalized:
        keys += [task + "_norm_acc" for task in tasks] + ["Average_norm_acc"]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Result is incomplete or invalid (%s): %s" % (key, path))


def run_and_tee(command, log_path, env):
    with log_path.open("x", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        child = subprocess.Popen(
            command, cwd=str(REPOSITORY_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        try:
            for line in child.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            returncode = child.wait()
        except BaseException:
            # A separate process group also reaches DataLoader worker children.
            for signum, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    break
                try:
                    child.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise
        finally:
            child.stdout.close()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--model", choices=tuple(MODELS), required=True)
    result.add_argument("--spaces", default="pgsb", help="Comma-separated full,knots,core,pgsb or all")
    result.add_argument("--methods", default="ties", help="Comma-separated tv,ties,dare-ties,tsv,cart,iso-c or all")
    result.add_argument("--isotropize", action="store_true", help="Add Iso-C isotropization to each selected merge operator")
    result.add_argument("--mode", choices=("linearsearch", "direct"), default="linearsearch")
    result.add_argument("--run-id", default=None, help="New result group; matching completed cases can be reused")
    result.add_argument("--output-root", default=str(REPOSITORY_ROOT / "results" / "main"))
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    result.add_argument("--model-parallel-gpus", type=int, default=1, help="NLI evaluation model sharding")
    result.add_argument("--model-path", default=os.getenv("PGSB_MODEL_PATH"), help="Hub base-model ID or complete local snapshot")
    result.add_argument("--cache-dir", default=os.getenv("PGSB_CACHE_DIR", str(REPOSITORY_ROOT / "data")))
    result.add_argument("--adapter-cache-dir", default=os.getenv("PGSB_ADAPTER_CACHE_DIR"))
    result.add_argument("--data-root", default=os.getenv("PGSB_VISION_DATA_ROOT"), help="Optional vision dataset ROOT override; existing loader defaults otherwise")
    result.add_argument("--nli-data-cache", default=os.getenv("PGSB_NLI_DATA_CACHE"), help="Optional NLI dataset cache override; existing cache otherwise")
    result.add_argument("--head-root", default=None, help="Exact vision head directory containing <task>_head.pt")
    result.add_argument("--shuffled-index-root", default=None)
    result.add_argument("--nli-head-path", default=os.getenv("PGSB_NLI_HEAD_PATH"))
    result.add_argument("--heads-repo-id", default=None, help="Optional Hugging Face repository from which to download missing heads")
    result.add_argument("--heads-file-template", default=None, help="Path inside the head repository; vision default {task}_head.pt, NLI default heads.pt")
    result.add_argument("--heads-revision", default="main")
    result.add_argument("--checkpoint-root", default=None, help="Local checkpoint root; see --checkpoint-template for layouts")
    result.add_argument("--checkpoint-template", default=None, help="Local path/Hub ID template accepting {root}, {task}, {rank}; rank is the paper's 16")
    result.add_argument("--baseline-json", default=None, help="Task -> single-task accuracy in percent, or {'finetuned_accuracies': {...}}")
    result.add_argument("--allow-missing-baselines", action="store_true", help="Allow raw-accuracy selection when matching single-task baselines are unavailable")
    result.add_argument("--merge-cache-dir", default=str(REPOSITORY_ROOT / "data" / "merge_cache"))
    result.add_argument("--no-merge-cache", action="store_true")
    result.add_argument("--offline", action="store_true", help="Disallow model and dataset network access; local archives may still be prepared")
    result.add_argument("--batch-size", type=int, default=None)
    result.add_argument("--num-workers", type=int, default=None)
    result.add_argument("--load-dtype", choices=("float16", "float32", "bfloat16"), default="float16", help="NLI model load/evaluation dtype")
    result.add_argument("--seed", type=int, default=420)
    result.add_argument("--scaling-coeffs", type=float, default=None, help="Fixed coefficient for direct mode; NLI search initial coefficient in linearsearch mode")
    result.add_argument("--top-k", type=float, default=None, help="TIES density percentage for direct mode; NLI search initial density in linearsearch mode")
    result.add_argument("--dare-pruning", type=float, default=None, help="DARE drop probability for direct mode; NLI search initial probability in linearsearch mode")
    result.add_argument("--cart-pruning-rank", type=float, default=None)
    result.add_argument("--cart-scaling", type=float, default=None)
    result.add_argument("--dry-run", action="store_true", help="Print commands/configuration summaries without writing files, loading models or downloading")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    spaces = selected_values(args.spaces, SPACES)
    methods = selected_values(args.methods, METHODS)
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.model_parallel_gpus < 1 or (args.model != "llama3-8b" and args.model_parallel_gpus != 1):
        raise ValueError("--model-parallel-gpus must be positive and is only supported for NLI")
    if args.seed < 0 or args.seed >= 2 ** 32:
        raise ValueError("--seed must be in [0, 2**32)")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("--run-id must contain only letters, numbers, underscores, dots and hyphens")
    base_config = prepare_base_config(args)
    environment = runtime_environment(args)
    if not args.dry_run:
        download_requested_heads(args, base_config)
        preflight(base_config)
    tasks = [entry["name"] for entry in base_config["dataset"]]
    require_normalized = all(task in base_config["finetuned_accuracies"] for task in tasks)
    output_root = absolute_path(args.output_root) / run_id / args.model
    evaluator = ("nli" if args.model == "llama3-8b" else "8vision") + "_pertask"
    if args.mode == "linearsearch":
        evaluator += "_linearsearch"
    evaluator_path = REPOSITORY_ROOT / "eval_scripts" / (evaluator + ".py")
    visited = set()
    for space in spaces:
        for requested_method in methods:
            method = "tv" if requested_method == "iso-c" else requested_method
            isotropize = args.isotropize or requested_method == "iso-c"
            if (space, method, isotropize) in visited:
                continue
            visited.add((space, method, isotropize))
            method_label = "iso-c" if method == "tv" and isotropize else method + ("-iso-c" if isotropize else "")
            case_dir = output_root / (space + "_" + method_label + "_" + args.mode)
            config_path = case_dir / "config.py"
            result_path = case_dir / "result.json"
            config = deepcopy(base_config)
            config["task_merge_config"].update({
                "merge_space": space, "merge_method": method, "isotropize": isotropize,
                "disk_cache": not args.no_merge_cache,
                "disk_cache_dir": str(absolute_path(args.merge_cache_dir)),
            })
            # Record environment inputs that affect data and evaluation; never tokens.
            recorded_env = {key: environment[key] for key in (
                "CUDA_VISIBLE_DEVICES", "PGSB_VISION_DATA_ROOT", "PGSB_NLI_DATA_CACHE",
                "NLI_BATCH_SIZE", "NLI_NUM_WORKERS", "HF_DATASETS_CACHE",
                "PGSB_NLI_SNLI_PATH", "PGSB_NLI_MNLI_PATH", "PGSB_NLI_SICK_PATH",
                "PGSB_NLI_QNLI_PATH", "PGSB_NLI_RTE_PATH", "PGSB_NLI_SCITAIL_PATH",
            ) if key in environment}
            config["experiment"] = {
                "model": args.model, "mode": args.mode, "seed": args.seed,
                "requested_method": requested_method, "tasks": tasks,
                "environment": recorded_env,
                "baseline_source": str(absolute_path(args.baseline_json)) if args.baseline_json else (
                    "checkpoint manifests" if args.model == "siglip" or args.checkpoint_root or args.checkpoint_template else "paper single-task baselines"
                ),
                "heads_repository": args.heads_repo_id,
                "heads_file_template": args.heads_file_template,
                "heads_revision": args.heads_revision if args.heads_repo_id else None,
            }
            fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            config["task_merge_config"]["cache_namespace"] = args.model + "-" + space + "-" + fingerprint[:20]
            serialized = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            config_source = '"""Resolved main-experiment configuration; generated by run_experiments.py."""\n\nconfig = ' + pprint.pformat(config, width=100, sort_dicts=False) + "\n"
            command = [args.python, str(evaluator_path), "--config", str(config_path), "--output_json", str(result_path)]
            print("\n[%s / %s / %s]" % (args.model, space, method_label), flush=True)
            print("$ " + shlex.join(command), flush=True)
            if args.dry_run:
                print("  base model: " + config["model"].get("base_type", config["model"]["name"]))
                print("  adapters: " + ", ".join(config["model"]["bases"]))
                print("  baseline tasks: " + ", ".join(config["finetuned_accuracies"]))
                print("  seed=%d; search grids and splits inherited from %s" % (args.seed, MODELS[args.model]))
                continue
            if case_dir.exists():
                saved = case_dir / "resolved_config.json"
                if saved.is_file() and saved.read_text(encoding="utf-8") == serialized and result_path.is_file():
                    completed_result(result_path, tasks, require_normalized)
                    print("Completed result exists; skipping " + str(result_path), flush=True)
                    continue
                raise FileExistsError("Existing case is incomplete or has different settings; choose a new --run-id: " + str(case_dir))
            case_dir.mkdir(parents=True)
            write_new(config_path, config_source)
            write_new(case_dir / "resolved_config.json", serialized)
            write_new(case_dir / "command.json", json.dumps({"argv": command, "cwd": str(REPOSITORY_ROOT), "fingerprint": fingerprint}, indent=2) + "\n")
            run_and_tee(command, case_dir / "run.log", environment)
            completed_result(result_path, tasks, require_normalized)
            print("Saved " + str(result_path), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; child processes stopped and partial logs retained. Use a new --run-id to rerun incomplete cases.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
