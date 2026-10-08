"""Prepare datasets before loading large models: python -m dataset.prepare --help."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VISION_TASKS = ("stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn")
NLI_TASKS = ("snli", "mnli", "sick", "qnli", "rte", "scitail")
INDEX_FILES = {"stanford_cars": "cars", "gtsrb": "gtsrb", "mnist": "mnist", "sun397": "sun397", "svhn": "svhn"}
EVAL_COUNTS = {"stanford_cars": 8041, "gtsrb": 12630, "mnist": 10000, "sun397": 19850, "svhn": 26032}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tasks", default="vision", help="vision, nli, all, or comma-separated task IDs")
    result.add_argument("--data-root", default=os.getenv("PGSB_VISION_DATA_ROOT", str(ROOT / "data")))
    result.add_argument("--nli-data-cache", default=os.getenv("PGSB_NLI_DATA_CACHE", os.getenv("HF_DATASETS_CACHE", str(ROOT / "data" / "datasets"))))
    result.add_argument("--source-dir", help="Existing raw vision images/annotations for one selected task")
    result.add_argument("--nli-local-path", help="Existing saved NLI dataset or raw Arrow directory for one selected task")
    result.add_argument("--eurosat-split-dir", help="Original train.txt, val.txt, test.txt manifests (class/filename.jpg)")
    result.add_argument("--eurosat-protocol", choices=("torchgeo",), help="Explicitly opt into a different 60/20/20 protocol; NOT the paper split")
    result.add_argument("--include-train", action="store_true", help="Also prepare Cars, GTSRB, and SVHN training files (not needed for merging)")
    result.add_argument("--shuffled-index-root", default=str(ROOT / "dataset" / "shuffled_idxs"))
    result.add_argument("--offline", action="store_true", help="Reuse local files/caches/archives only; never access the network")
    result.add_argument("--report", help="New report JSON path; existing reports are never overwritten")
    result.add_argument("--dry-run", action="store_true", help="Show selected tasks and paths without downloads or file writes")
    return result


def _tasks(value):
    groups = {"vision": VISION_TASKS, "nli": NLI_TASKS, "all": VISION_TASKS + NLI_TASKS}
    selected = list(groups[value]) if value in groups else [part.strip() for part in value.split(",")]
    if not selected or any(task not in VISION_TASKS + NLI_TASKS for task in selected):
        raise ValueError("Unknown task; choose from " + ", ".join(VISION_TASKS + NLI_TASKS))
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate task IDs are not allowed")
    return selected


def validate_saved_indices(task, index_root):
    from .download_utils import file_digest
    from .splits import saved_split_indices

    path = index_root / (INDEX_FILES[task] + "_shuffled_idxs.pt")
    validation, test = saved_split_indices(range(EVAL_COUNTS[task]), {"shuffled_idxs": str(path), "val_fraction": 0.2})
    return {"permutation": str(path), "sha256": file_digest(path), "val_fraction": 0.2,
            "validation_count": len(validation), "test_count": len(test), "regenerated": False}


def main(argv=None):
    args = parser().parse_args(argv)
    tasks = _tasks(args.tasks)
    data_root = Path(args.data_root).expanduser().resolve()
    nli_cache = Path(args.nli_data_cache).expanduser().resolve()
    if args.source_dir and (len(tasks) != 1 or tasks[0] not in VISION_TASKS):
        raise ValueError("--source-dir requires exactly one vision task")
    if args.nli_local_path and (len(tasks) != 1 or tasks[0] not in NLI_TASKS):
        raise ValueError("--nli-local-path requires exactly one NLI task")
    if args.eurosat_split_dir and args.eurosat_protocol:
        raise ValueError("Choose original EuroSAT manifests OR a new explicit protocol, not both")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = Path(args.report).expanduser().resolve() if args.report else data_root / ".preparation" / ("report-" + timestamp + ".json")
    settings = {"tasks": tasks, "data_root": str(data_root), "nli_data_cache": str(nli_cache),
                "offline": args.offline, "include_train": args.include_train,
                "eurosat_split_dir": args.eurosat_split_dir, "eurosat_protocol": args.eurosat_protocol,
                "source_dir": args.source_dir, "nli_local_path": args.nli_local_path,
                "shuffled_index_root": str(Path(args.shuffled_index_root).expanduser().resolve())}
    print(json.dumps(settings, indent=2))
    if args.dry_run:
        return 0
    # Reserve a unique output before any downloads. Incremental JSON lines also
    # retain completed task reports if preparation is interrupted.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists() or Path(str(report_path) + ".jsonl").exists():
        raise FileExistsError("Choose a new --report path: " + str(report_path))
    reports = []
    payload = {"settings": settings, "created_at": timestamp, "tasks": reports}
    with report_path.open("x", encoding="utf-8") as final, Path(str(report_path) + ".jsonl").open("x", encoding="utf-8") as progress:
        try:
            for task in tasks:
                print("\nPreparing " + task, flush=True)
                try:
                    index_report = validate_saved_indices(task, Path(settings["shuffled_index_root"])) if task in INDEX_FILES else None
                    if task in ("mnist", "gtsrb", "svhn"):
                        from .standard_downloads import prepare_standard_task
                        info = prepare_standard_task(task, data_root, offline=args.offline,
                                                     splits=("train", "test") if args.include_train else ("test",))
                    elif task in VISION_TASKS:
                        from .vision_downloads import prepare_vision_task
                        info = prepare_vision_task(task, data_root, offline=args.offline,
                                                   cars_splits=("train", "test") if args.include_train else ("test",),
                                                   source_dir=args.source_dir, eurosat_split_dir=args.eurosat_split_dir,
                                                   eurosat_protocol=args.eurosat_protocol)
                    else:
                        from .nli_downloads import prepare_nli_task
                        info = prepare_nli_task(task, cache_dir=str(nli_cache), offline=args.offline, location=args.nli_local_path)
                    result = {"task": task, "status": "ready", "details": info}
                    if index_report:
                        result["evaluation_split"] = index_report
                    print("Ready: " + task, flush=True)
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    result = {"task": task, "status": "error", "error": str(exc)}
                    print("ERROR [%s]: %s" % (task, exc), file=sys.stderr, flush=True)
                reports.append(result)
                progress.write(json.dumps(result, ensure_ascii=False) + "\n")
                progress.flush()
        finally:
            json.dump(payload, final, indent=2, ensure_ascii=False)
            final.write("\n")
    print("Preparation report: " + str(report_path))
    return 1 if any(result["status"] != "ready" for result in reports) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed files and task reports are preserved. Rerun to reuse them.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
