"""Local-first NLI data loading with fixed sources and unchanged row ordering."""

import json
import math
import os
import re
import tempfile
from pathlib import Path

from .download_utils import download_file, file_lock, offline_mode


TASK_IDS = {
    "snli": "stanfordnlp/snli", "mnli": "SetFit/mnli", "sick": "RobZamp/sick",
    "qnli": "SetFit/qnli", "rte": "SetFit/rte", "scitail": "allenai/scitail",
}
TASK_KEYS = {
    "snli": ("premise", "hypothesis"), "mnli": ("text1", "text2"),
    "sick": ("sentence_A", "sentence_B"), "qnli": ("text1", "text2"),
    "rte": ("text1", "text2"), "scitail": ("premise", "hypothesis"),
}
# Pin the same repositories used by the main experiments. Explicit filenames avoid
# remote globbing/recursive Hub listings; MNLI must remain the matched partition.
REVISIONS = {
    "snli": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
    "mnli": "1c01c732e1efb5d5090c5618276c010aecebefa3",
    "qnli": "e08ed82", "rte": "23f2a46",
    "scitail": "0cc4353235b289165dfde1c7c5d1be983f99ce44",
}
# Published Git-LFS object digests from the corresponding conversion commits.
PARQUET_SHA256 = {
    "snli": {
        "train": "ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620",
        "validation": "00f5ed8deaed007fef3022f0215b287efbab815b1bb31ac3f3ff4f4129d41ffe",
        "test": "4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c",
    },
    "scitail": {
        "train": "35ffcef823e42135a4fcee1b5ecb7c951e99f97b6f51c9363a23b537d41fb5d3",
        "validation": "7342d7d9c3f0c90b904b5fcfa37b909ed77fc3f9f0c4b87618d7718469f55b56",
        "test": "c2b4b8b5e258a30fe7d1f7861ad7154f1ebaf8f085f5e051db5e22352cf7ca96",
    },
}
SPLITS = ("train", "validation", "test")


def validate_nli_dataset(dataset, task):
    """Reject wrong schemas and unknown labels without sorting or resplitting."""
    missing = set(SPLITS) - set(dataset)
    if missing:
        raise ValueError("{}: missing NLI splits {}".format(task, sorted(missing)))
    required = set(TASK_KEYS[task]) | {"label"}
    allowed = {"entails", "neutral"} if task == "scitail" else {-1, 0, 1, 2}
    if task in ("qnli", "rte"):
        allowed = {-1, 0, 1}
    for split in SPLITS:
        current = dataset[split]
        missing = required - set(current.column_names)
        if missing:
            raise ValueError("{} {}: missing columns {}".format(task, split, sorted(missing)))
        if not len(current):
            raise ValueError("{} {}: empty source split".format(task, split))
        labels = set(current.unique("label"))
        if labels - allowed:
            raise ValueError("{} {}: unexpected labels {!r}".format(task, split, labels - allowed))
        # GLUE challenge test sets have no published labels and are never evaluated.
        unlabeled_challenge = task in ("mnli", "qnli", "rte") and split == "test"
        if labels <= {-1} and not unlabeled_challenge:
            raise ValueError("{} {}: no labeled examples".format(task, split))
        feature = getattr(current, "features", {}).get("label")
        names = getattr(feature, "names", None)
        expected = ["entailment", "not_entailment"] if task in ("qnli", "rte") else ["entailment", "neutral", "contradiction"]
        if names is not None and task != "scitail" and list(names) != expected:
            raise ValueError("{}: label names differ from the checkpoint convention: {!r}".format(task, names))
    return dataset


def _arrow_cache(directory, task, datasets):
    """Read original builder shards, never transformed cache-*.arrow files."""
    with (directory / "dataset_info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    if not set(SPLITS) <= set(info.get("splits", {})):
        return None
    if not (set(TASK_KEYS[task]) | {"label"}) <= set(info.get("features", {})):
        return None
    if task == "scitail" and info.get("config_name") != "tsv_format":
        return None
    result = {}
    for split in SPLITS:
        pattern = re.compile(r"^(?!cache-).+-" + split + r"(?:-(\d{5})-of-(\d{5}))?\.arrow$")
        files = [(path, pattern.match(path.name)) for path in sorted(directory.glob("*.arrow"))]
        files = [(path, match) for path, match in files if match]
        if not files:
            return None
        totals = {int(match.group(2)) for _, match in files if match.group(2)}
        if (len(files) > 1 and not totals) or (totals and (
                len(totals) != 1 or totals != {len(files)} or
                any(match.group(1) is None for _, match in files) or
                [int(match.group(1)) for _, match in files] != list(range(len(files))))):
            raise ValueError("Incomplete or mixed Arrow shards: {} ({})".format(directory, split))
        parts = [datasets.Dataset.from_file(str(path)) for path, _ in files]
        result[split] = parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)
        if len(result[split]) != info["splits"][split]["num_examples"]:
            raise ValueError("Arrow row count differs from dataset_info.json: {} ({})".format(directory, split))
    return validate_nli_dataset(datasets.DatasetDict(result), task)


def _local_dataset(location, task, datasets, cache_root):
    directory = Path(location).expanduser()
    if (directory / "dataset_dict.json").is_file():
        return validate_nli_dataset(datasets.load_from_disk(str(directory)), task)
    if (directory / "dataset_info.json").is_file() and not (directory / "state.json").exists():
        data = _arrow_cache(directory, task, datasets)
        if data is None:
            raise ValueError("Not a complete original {} Arrow cache: {}".format(task, directory))
        return data
    for extension, builder in (("jsonl", "json"), ("json", "json"), ("parquet", "parquet")):
        paths = {split: directory / (split + "." + extension) for split in SPLITS}
        if any(path.is_file() for path in paths.values()):
            if not all(path.is_file() for path in paths.values()):
                raise ValueError("{}: local NLI files must contain train, validation and test".format(directory))
            data = datasets.load_dataset(builder, data_files={split: str(path) for split, path in paths.items()}, cache_dir=str(cache_root))
            return validate_nli_dataset(data, task)
    return None


def _existing_cache(cache_root, task, datasets):
    aliases = {task, TASK_IDS[task].replace("/", "___").lower()}
    if task == "sick":
        aliases.add("rob_zamp___sick")
    matches = []
    if cache_root.is_dir():
        for directory in sorted(cache_root.iterdir()):
            if directory.is_dir() and directory.name.lower() in aliases:
                for metadata in sorted(directory.glob("*/*/*/dataset_info.json")):
                    if any(part.endswith(".incomplete") for part in metadata.parts):
                        continue
                    data = _arrow_cache(metadata.parent, task, datasets)
                    if data is not None:
                        matches.append((metadata.parent, data))
    if len(matches) > 1:
        raise ValueError("Multiple {} cache revisions found; choose the original via PGSB_NLI_{}_PATH or dataset config dir:\n{}".format(
            task, task.upper(), "\n".join(str(path) for path, _ in matches)))
    return matches[0] if matches else None


def _sick_dataset(source_root, datasets, offline):
    """Parse the author-published SICK archive with its original row order."""
    import zipfile

    archive = download_file([
        "https://zenodo.org/records/2787612/files/SICK.zip?download=1",
        "https://zenodo.org/record/2787612/files/SICK.zip?download=1",
    ], source_root / "SICK.zip", md5="b03210036cc64a4cf1bc7a0525357001", offline=offline)
    rows = {split: [] for split in SPLITS}
    split_names = {"TRAIN": "train", "TRIAL": "validation", "TEST": "test"}
    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    with zipfile.ZipFile(archive) as handle:
        names = [name for name in handle.namelist() if name.rstrip("/").split("/")[-1] == "SICK.txt"]
        if len(names) != 1:
            raise ValueError("SICK.zip must contain exactly one SICK.txt")
        with handle.open(names[0]) as contents:
            for line in contents:
                values = [value.strip() for value in line.decode("utf-8").split("\t")]
                if values[-1] in split_names:
                    if len(values) != 12 or values[3].lower() not in labels:
                        raise ValueError("Invalid SICK row: {}".format(values[0]))
                    rows[split_names[values[-1]]].append({
                        "id": values[0], "sentence_A": values[1], "sentence_B": values[2],
                        "label": labels[values[3].lower()],
                    })
    expected = {"train": 4439, "validation": 495, "test": 4906}
    for split in SPLITS:
        if len(rows[split]) != expected[split]:
            raise ValueError("SICK {}: expected {} rows, got {}".format(split, expected[split], len(rows[split])))
    return datasets.DatasetDict({split: datasets.Dataset.from_list(rows[split]) for split in SPLITS})


def _download_sources(task, cache_root, datasets, offline):
    source_root = cache_root / "pgsb_nli_sources" / task
    if task == "sick":
        return _sick_dataset(source_root, datasets, offline)
    files = {}
    builder = "parquet" if task in ("snli", "scitail") else "json"
    for split in SPLITS:
        if builder == "parquet":
            prefix = "plain_text" if task == "snli" else "tsv_format"
            filename = "{}/{}-00000-of-00001.parquet".format(prefix, split)
        else:
            filename = split + ("_matched" if task == "mnli" and split != "train" else "") + ".jsonl"
        endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = "{}/datasets/{}/resolve/{}/{}".format(endpoint, TASK_IDS[task], REVISIONS[task], filename)
        files[split] = str(download_file([url], source_root / filename,
                                         sha256=PARQUET_SHA256.get(task, {}).get(split), offline=offline))
    return datasets.load_dataset(builder, data_files=files, cache_dir=str(cache_root))


def load_nli_dataset(task, cache_dir=None, offline=False, location=None, return_report=False,
                     strict_location=False):
    """Load local data, an existing original Arrow cache, or pinned source files."""
    import datasets

    if task not in TASK_IDS:
        raise ValueError("Unknown NLI task: {}".format(task))
    cache_root = Path(cache_dir or os.getenv("PGSB_NLI_DATA_CACHE") or datasets.config.HF_DATASETS_CACHE).expanduser()
    is_offline = offline or offline_mode()
    explicit_location = os.getenv("PGSB_NLI_{}_PATH".format(task.upper()))
    location = location if strict_location else (explicit_location or location)
    data, source = None, None
    if location:
        data = _local_dataset(location, task, datasets, cache_root)
        if data is not None:
            source = str(Path(location).expanduser().resolve())
        elif explicit_location or strict_location:
            raise ValueError("Explicit {} local path does not contain a saved dataset, original Arrow cache, or split files: {}".format(task, location))
    elif strict_location:
        raise ValueError("Explicit {} local path is empty".format(task))
    prepared = cache_root / "pgsb_nli" / task
    if data is None and (prepared / "dataset_dict.json").is_file():
        data = validate_nli_dataset(datasets.load_from_disk(str(prepared)), task)
        source = str(prepared.resolve())
    if data is None:
        cached = _existing_cache(cache_root, task, datasets)
        if cached:
            source, data = cached
            source = str(source.resolve())
    if data is None:
        with file_lock(prepared.with_suffix(".lock")):
            if (prepared / "dataset_dict.json").is_file():
                data = validate_nli_dataset(datasets.load_from_disk(str(prepared)), task)
            else:
                try:
                    data = validate_nli_dataset(_download_sources(task, cache_root, datasets, is_offline), task)
                except (OSError, RuntimeError) as exc:
                    raise RuntimeError("Could not prepare {}. Cache: {}. Reuse local data via PGSB_NLI_{}_PATH; run dataset preparation online first if using --offline. {}".format(task, cache_root, task.upper(), exc)) from exc
                temporary = Path(tempfile.mkdtemp(prefix=task + "-", dir=str(prepared.parent)))
                data.save_to_disk(str(temporary))
                with (temporary / "source.json").open("w", encoding="utf-8") as handle:
                    json.dump({"task": task, "repository": TASK_IDS[task], "revision": REVISIONS.get(task, "zenodo:2787612"),
                               "splits": {split: len(data[split]) for split in SPLITS}}, handle, indent=2)
                os.replace(str(temporary), str(prepared))
                data = datasets.load_from_disk(str(prepared))
            source = str(prepared.resolve())
    report = {"task": task, "source": source, "cache_dir": str(cache_root.resolve()),
              "splits": {split: len(data[split]) for split in SPLITS},
              "labels": {split: sorted(data[split].unique("label")) for split in SPLITS}}
    print("NLI {}: {} (source splits: {})".format(task, source, report["splits"]))
    return (data, report) if return_report else data


def prepare_nli_task(task, cache_dir=None, offline=False, location=None):
    """Download/validate NLI data without importing a tokenizer or a model."""
    data, report = load_nli_dataset(task, cache_dir, offline, location, return_report=True,
                                    strict_location=(location is not None))
    valid_labels = {"entails", "neutral"} if task == "scitail" else {0, 1, 2}
    validation_count = sum(label in valid_labels for label in data["validation"]["label"])
    fraction = {"mnli": 0.2, "qnli": 0.2, "rte": 0.5}.get(task)
    if fraction is not None:
        reserved = math.ceil(validation_count * fraction)
        protocol = {"validation_source": "validation", "test_source": "validation",
                    "validation_fraction": fraction, "shuffle": task != "mnli", "seed": 42,
                    "validation_rows": reserved, "test_rows": validation_count - reserved,
                    "unlabeled_challenge_test_used": False}
    else:
        protocol = {"validation_source": "validation", "test_source": "test",
                    "validation_fraction": None, "shuffle": False, "seed": None,
                    "validation_rows": validation_count,
                    "test_rows": sum(label in valid_labels for label in data["test"]["label"])}
    protocol["configuration"] = "main-experiment defaults; an explicit task val_fraction overrides these counts"
    report["evaluation_protocol"] = protocol
    return report


def split_nli_validation(dataset, fraction, task):
    """Preserve the original HF split algorithm, including MNLI's no-shuffle rule."""
    if not 0 < fraction < 1:
        raise ValueError("{} val_fraction must lie strictly between 0 and 1".format(task))
    if len(dataset) < 2:
        raise ValueError("{}: not enough labeled validation rows to split".format(task))
    result = dataset.train_test_split(test_size=fraction, shuffle=(task != "mnli"), seed=42)
    if not len(result["train"]) or not len(result["test"]):
        raise ValueError("{}: empty validation/test partition".format(task))
    if len(result["train"]) + len(result["test"]) != len(dataset):
        raise ValueError("{}: split lost rows".format(task))
    print("NLI {}: validation={} test={} seed=42 shuffle={}".format(
        task, len(result["test"]), len(result["train"]), task != "mnli"))
    return result
