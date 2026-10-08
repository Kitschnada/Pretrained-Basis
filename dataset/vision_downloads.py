"""Prepare the original vision layouts without replacing existing data or splits.

This module deliberately does not depend on torchvision or Hugging Face Datasets:
raw filenames and annotation order are part of the evaluation protocol.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from .download_utils import download_file, extract_archive, file_lock, offline_mode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}
CARS_BASE = (
    "https://huggingface.co/datasets/iharabukhouski/stanford_cars/resolve/"
    "5e32750f7a3c226a5ab20653178f1278925b4ca9/"
)
CARS_METADATA = {
    "devkit/cars_meta.mat": "fe43dfda289448f97c25b09e09fafc0414e22fc4143831537780884a0a5f4a1a",
    "devkit/cars_train_annos.mat": "1c1973dd828a2964060b0bbc0e423466393fe3001a3f8e303f41fbf22067dfb1",
    "cars_test_annos_withlabels.mat": "790f75be8ea34eeded134cc559332baf23e30e91367e9ddca97d26ed9b895f05",
}
RESISC_BASE = (
    "https://huggingface.co/datasets/isaaccorley/resisc45/resolve/"
    "883edc0eee77b2c84225472f10f126e3ed83fa6e/"
)
RESISC_HASHES = {
    "train": "ecfa963be4d85eac83665f8be8634abcb4fe6f3546472cc0e87999e2cab4449b",
    "val": "08d81f642526bec240589000af7f49a47e8d071a6e7925b0f36246a78ab64342",
    "test": "e0927e80130b47317a2f18520d98382b6fc56f0d3edd3345140f7d02267c3805",
}
EUROSAT_BASE = (
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/"
    "1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/"
)
EUROSAT_HASHES = {
    "train": "1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695",
    "val": "b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330",
    "test": "cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592",
}
SUN_BASE = (
    "https://huggingface.co/zhengli97/SUN397_Dataset/resolve/"
    "99bbf6ad18ca4f01e47f8f96ca1a8b7e53cb235a/"
)


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _images(folder):
    folder = Path(folder)
    if folder.is_symlink() and not folder.exists():
        raise FileNotFoundError(f"Broken dataset symlink: {folder}")
    if not folder.is_dir():
        return []
    files = []

    def visit(directory, ancestors):
        info = directory.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in ancestors:
            raise ValueError(f"Directory symlink cycle in dataset: {directory}")
        ancestors = ancestors | {identity}
        with os.scandir(directory) as entries:
            entries = sorted(entries, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() and not path.exists():
                raise FileNotFoundError(f"Broken dataset symlink: {path}")
            # Match ImageFolder's followlinks=True traversal, but fail on cycles.
            if entry.is_dir(follow_symlinks=True):
                visit(path, ancestors)
            elif path.suffix.lower() in IMAGE_SUFFIXES:
                files.append(path)

    visit(folder, set())
    return sorted(files)


def _folder_report(base, splits, class_count, expected_counts=None):
    """Validate prepared ImageFolder trees; do not reorder or write them."""
    counts, classes, seen_names, seen_files = {}, None, set(), set()
    fingerprints = {}
    for split in splits:
        folder = base / split
        names = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        if len(names) != class_count:
            raise ValueError(f"{folder}: expected {class_count} class directories, found {len(names)}")
        if classes is not None and names != classes:
            raise ValueError(f"{folder}: class names do not match the other splits")
        classes = names
        files = _images(folder)
        for name in names:
            if not _images(folder / name):
                raise ValueError(f"{folder / name}: empty class directory")
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Missing image or broken symlink: {path}")
            identifier = path.relative_to(folder).as_posix()
            stat = path.stat()
            if not stat.st_size:
                raise ValueError(f"Empty image: {path}")
            physical = (stat.st_dev, stat.st_ino)
            if identifier in seen_names or physical in seen_files:
                raise ValueError(f"Duplicate image across splits: {path}")
            seen_names.add(identifier)
            seen_files.add(physical)
        counts[split] = len(files)
        fingerprints[split] = hashlib.sha256("\n".join(p.relative_to(folder).as_posix() for p in files).encode()).hexdigest()
        if expected_counts and len(files) != expected_counts[split]:
            raise ValueError(f"{folder}: expected {expected_counts[split]} images, found {len(files)}")
    return {"root": str(base), "counts": counts, "class_names": classes,
            "split_membership_sha256": fingerprints}


def _prepared(base, splits):
    return not (base / ".pgsb-preparing.json").exists() and all((base / split).is_dir() for split in splits)


def _manifest(path, leading_slash=False):
    rows = []
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        text = text.strip()
        if not text:
            continue
        if leading_slash:
            text = text.lstrip("/")
        parsed = PurePosixPath(text)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in text:
            raise ValueError(f"Unsafe image path in {path}: {text!r}")
        rows.append(parsed.as_posix())
    if not rows or len(set(rows)) != len(rows):
        raise ValueError(f"{path}: empty manifest or duplicate sample names")
    return rows


def _disjoint(manifests):
    seen = set()
    for split, rows in manifests.items():
        shared = seen.intersection(rows)
        if shared:
            raise ValueError(f"Overlapping split {split}: {next(iter(shared))}")
        seen.update(rows)


def _link(source, target):
    if not source.is_file():
        raise FileNotFoundError(f"Missing source image: {source}")
    if target.exists() or target.is_symlink():
        if not target.is_file():
            raise FileExistsError(f"Refusing to replace invalid existing path: {target}")
        if target.resolve() != source.resolve() and _digest(target) != _digest(source):
            raise FileExistsError(f"Existing file differs from source; not overwritten: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def _materialize(base, sources, provenance=None):
    """Preflight all files, then add missing symlinks without copying pixels."""
    for target, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing source image: {source}")
        if target.is_symlink() and not target.exists():
            raise FileExistsError(f"Broken existing symlink; repair explicitly: {target}")
    for split in {p.relative_to(base).parts[0] for p in sources}:
        extra = set(_images(base / split)).difference(sources)
        if extra:
            raise ValueError(f"Existing split has images absent from the selected manifest: {next(iter(extra))}")
    base.mkdir(parents=True, exist_ok=True)
    marker = base / ".pgsb-preparing.json"
    mapping = sorted((str(target), str(source.resolve())) for target, source in sources.items())
    plan = {"target_count": len(sources), "plan_sha256": hashlib.sha256(json.dumps(mapping).encode()).hexdigest(),
            "provenance": provenance or {}}
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError(f"Interrupted preparation has a different source/manifest plan: {marker}. "
                             "Repeat the original preparation command; existing files were not changed.")
    else:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    for target, source in sources.items():
        _link(source, target)


def _finished(base, report):
    metadata = base / ".pgsb-preparation.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        keys = ("counts", "class_names", "split_membership_sha256", "protocol", "manifest_sha256")
        if any(previous.get(key) != report.get(key) for key in keys):
            raise ValueError(f"Prepared split differs from recorded provenance; not overwritten: {metadata}")
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=".pgsb-preparation-", dir=str(base))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, metadata)
        finally:
            Path(temporary).unlink()
    # Only remove our in-progress marker after all membership/count checks pass.
    marker = base / ".pgsb-preparing.json"
    if marker.is_file():
        marker.unlink()
    return report


def _reuse(base, report, default_protocol):
    metadata = base / ".pgsb-preparation.json"
    if metadata.is_file():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        for key in ("counts", "class_names", "split_membership_sha256"):
            if previous.get(key) != report.get(key):
                raise ValueError(f"Dataset membership differs from recorded preparation: {metadata}")
        return dict(previous, **report, reused=True)
    return dict(report, reused=True, protocol=default_protocol)


def _find_raw(candidates, marker):
    for candidate in candidates:
        if candidate is not None:
            candidate = Path(candidate).expanduser().resolve()
            if marker(candidate):
                return candidate
    return None


def _cars(root, offline, cars_splits=("train", "test"), source_dir=None, cars_root=None, **unused):
    import numpy as np
    from scipy.io import loadmat

    base = Path(cars_root).expanduser().resolve() if cars_root else root / "stanford_cars"
    splits = tuple(cars_splits)
    if not splits or any(s not in ("train", "test") for s in splits):
        raise ValueError("cars_splits must contain train and/or test")
    if source_dir:
        source_dir = Path(source_dir).expanduser().resolve()
    annotations = {"train": "devkit/cars_train_annos.mat", "test": "cars_test_annos_withlabels.mat"}
    needed = ["devkit/cars_meta.mat"] + [annotations[s] for s in splits]
    reused = all((base / name).is_file() for name in needed)
    for name in needed:
        dest = base / name
        if not dest.is_file() and source_dir and (source_dir / name).is_file():
            _link(source_dir / name, dest)
        if not dest.is_file():
            download_file([CARS_BASE + name], dest, sha256=CARS_METADATA[name], offline=offline)
    class_names = np.atleast_1d(loadmat(base / "devkit/cars_meta.mat", squeeze_me=True)["class_names"]).tolist()
    if len(class_names) != 196 or len(set(class_names)) != 196:
        raise ValueError(f"{base}: expected 196 distinct Stanford Cars classes")
    counts, annotation_hashes, missing, physical_files = {}, {}, [], set()
    for split in splits:
        rows = np.atleast_1d(loadmat(base / annotations[split], squeeze_me=True)["annotations"])
        expected = 8144 if split == "train" else 8041
        if len(rows) != expected:
            raise ValueError(f"Cars {split}: expected {expected} annotations, got {len(rows)}")
        names = set()
        for row in rows:
            filename, label = str(row["fname"]), int(row["class"])
            if Path(filename).name != filename or filename in names or not 1 <= label <= 196:
                raise ValueError(f"Invalid/duplicate Cars annotation: {filename}, class={label}")
            names.add(filename)
            path = base / f"cars_{split}" / filename
            if not path.is_file() and source_dir and (source_dir / f"cars_{split}" / filename).is_file():
                _link(source_dir / f"cars_{split}" / filename, path)
            if not path.is_file():
                missing.append((f"cars_{split}/{filename}", path))
            else:
                stat = path.stat()
                if not stat.st_size:
                    raise ValueError(f"Empty Cars JPEG; move it aside before retrying: {path}")
                identity = (stat.st_dev, stat.st_ino)
                if identity in physical_files:
                    raise ValueError(f"Cars train/test share the same physical image: {path}")
                physical_files.add(identity)
        counts[split] = len(rows)
        annotation_hashes[split] = _digest(base / annotations[split])
    if missing:
        if offline:
            raise FileNotFoundError(f"Cars is missing {len(missing)} images; first: {missing[0][1]}. Run preparation without --offline.")
        print(f"Stanford Cars: downloading {len(missing)} missing original JPEGs from the pinned mirror.", flush=True)
        def fetch(item):
            from PIL import Image

            name, destination = item
            download_file([CARS_BASE + name], destination, offline=offline)
            try:
                with Image.open(destination) as image:
                    if image.format != "JPEG":
                        raise ValueError("Not a JPEG image")
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Invalid downloaded Cars JPEG; move it aside before retrying: {destination}") from exc
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(fetch, missing):
                pass
    return {"root": str(base), "counts": counts, "reused": reused and not missing,
            "protocol": "original MAT annotation order; official train/test", "sources": [CARS_BASE],
            "annotation_sha256": annotation_hashes}


def _dtd(root, offline, source_dir=None, **unused):
    base, splits = root / "dtd", ("train", "val", "test")
    if _prepared(base, splits):
        return _reuse(base, _folder_report(base, splits, 47), "existing prepared splits (unchanged)")
    raw = _find_raw([source_dir, base, base / "dtd"], lambda p: (p / "labels/train1.txt").is_file() and (p / "images").is_dir())
    source = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    if raw is None:
        archive = root / ".downloads" / "dtd-r1.0.1.tar.gz"
        download_file([source], archive, md5="fff73e5086ae6bdbea199a49dfb8a4c1", offline=offline)
        extract_archive(archive, root)
        raw = base
    manifests = {s: _manifest(raw / "labels" / f"{s}1.txt") for s in splits}
    _disjoint(manifests)
    _materialize(base, {base / s / row: raw / "images" / row for s, rows in manifests.items() for row in rows},
                 {s: str(raw / "labels" / f"{s}1.txt") for s in splits})
    return _finished(base, dict(_folder_report(base, splits, 47, {s: 1880 for s in splits}), reused=False,
                protocol="DTD partition 1; train1/val1/test1 kept separate", sources=[source],
                manifest_sha256={s: _digest(raw / "labels" / f"{s}1.txt") for s in splits}))


def _eurosat(root, offline, source_dir=None, eurosat_split_dir=None, eurosat_protocol=None, **unused):
    base, splits = root / "EuroSAT_splits", ("train", "val", "test")
    if _prepared(base, splits):
        return _reuse(base, _folder_report(base, splits, 10), "existing prepared splits (unchanged)")
    if not eurosat_split_dir and eurosat_protocol != "torchgeo":
        marker = base / ".pgsb-preparing.json"
        pending = (f" Interrupted preparation details: {marker.read_text(encoding='utf-8')}" if marker.is_file() else "")
        raise ValueError(
            "EuroSAT_splits is not prepared. The historical paper split was random and its membership "
            "is not stored in this repository. Supply --eurosat-split-dir containing train.txt, val.txt, "
            "test.txt (class/filename.jpg), or explicitly choose --eurosat-protocol torchgeo. "
            "TorchGeo is a DIFFERENT 60/20/20 protocol, not a recreation of the paper split. "
            "Existing EuroSAT_splits directories are always reused unchanged." + pending
        )
    if eurosat_protocol not in (None, "torchgeo"):
        raise ValueError(f"Unsupported EuroSAT protocol: {eurosat_protocol}")
    if not eurosat_split_dir:
        print("WARNING: explicitly selected EuroSAT TorchGeo 60/20/20; this is not the historical paper split. Re-evaluate baselines on this protocol.", flush=True)
    manifest_dir = Path(eurosat_split_dir).expanduser().resolve() if eurosat_split_dir else root / "eurosat" / "splits"
    paths = {}
    for split in splits:
        if eurosat_split_dir:
            candidates = [manifest_dir / f"{split}.txt", manifest_dir / f"eurosat-{split}.txt"]
            paths[split] = next((p for p in candidates if p.is_file()), candidates[0])
        else:
            paths[split] = manifest_dir / f"eurosat-{split}.txt"
            download_file([EUROSAT_BASE + paths[split].name], paths[split], sha256=EUROSAT_HASHES[split], offline=offline)
    manifests = {split: _manifest(path) for split, path in paths.items()}
    # TorchGeo lists bare names; names are unambiguous in the RGB archive.
    manifests = {s: [f"{row.rsplit('_', 1)[0]}/{row}" if "/" not in row else row for row in rows]
                 for s, rows in manifests.items()}
    _disjoint(manifests)
    raw = _find_raw([source_dir, root / "EuroSAT_RGB", root / "2750", root / "eurosat/2750", root / "eurosat/EuroSAT_RGB"],
                    lambda p: (p / "AnnualCrop").is_dir())
    url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
    if raw is None:
        archive = root / ".downloads" / "EuroSAT_RGB.zip"
        download_file([url], archive, md5="f46e308c4d50d4bf32fedad2d3d62f3b", offline=offline)
        extract_archive(archive, root / "eurosat")
        raw = _find_raw([root / "eurosat/EuroSAT_RGB", root / "eurosat/2750"], lambda p: (p / "AnnualCrop").is_dir())
        if raw is None:
            raise FileNotFoundError("EuroSAT archive did not contain the expected RGB class folders")
    _materialize(base, {base / s / row: raw / row for s, rows in manifests.items() for row in rows},
                 {s: str(p) for s, p in paths.items()})
    return _finished(base, dict(_folder_report(base, splits, 10, {s: len(rows) for s, rows in manifests.items()}), reused=False,
                protocol="user-supplied manifests" if eurosat_split_dir else "torchgeo-60-20-20 (not paper split)",
                sources=[url] + ([] if eurosat_split_dir else [EUROSAT_BASE]),
                manifest_sha256={s: _digest(p) for s, p in paths.items()}))


def _sun397(root, offline, source_dir=None, **unused):
    base, splits = root / "SUN397", ("train", "test")
    if _prepared(base, splits):
        return _reuse(base, _folder_report(base, splits, 397, {s: 19850 for s in splits}),
                      "existing prepared splits (unchanged); saved test permutation applied by loader")
    partition_dir = _find_raw([base, base / "Partitions", root / "Partitions", Path(source_dir) if source_dir else None],
                              lambda p: (p / "Training_01.txt").is_file() and (p / "Testing_01.txt").is_file())
    if partition_dir is None:
        archive = root / ".downloads" / "SUN397-Partitions.zip"
        download_file([SUN_BASE + "Partitions.zip"], archive,
                      sha256="e3311ea5f812d4cc6faef194688343d1111f6cd6f70f2c62f393c2b8d6ba1ec9", offline=offline)
        extract_archive(archive, base)
        partition_dir = _find_raw([base / "Partitions", base], lambda p: (p / "Training_01.txt").is_file())
        if partition_dir is None:
            raise FileNotFoundError("SUN397 partition archive lacks Training_01.txt")
    paths = {"train": partition_dir / "Training_01.txt", "test": partition_dir / "Testing_01.txt"}
    manifests = {s: _manifest(p, leading_slash=True) for s, p in paths.items()}
    _disjoint(manifests)
    first = manifests["train"][0]
    raw = _find_raw([source_dir, base, base / "SUN397", root / "sun397/SUN397"], lambda p: (p / first).is_file())
    if raw is None:
        archive = root / ".downloads" / "SUN397.tar.gz"
        print("SUN397 requires the original full-resolution archive (about 39 GB); existing data is reused when available.", flush=True)
        download_file([SUN_BASE + "SUN397.tar.gz"], archive,
                      sha256="f404130965a7ad77bed5ececc71e720ab50f3cc1e1bb257257610d38f3b928ec", offline=offline)
        extract_archive(archive, root)
        raw = base
    def flattened(row):
        parts = PurePosixPath(row).parts
        return Path("_".join(parts[:-1])) / parts[-1]
    sources = {base / s / flattened(row): raw / row for s, rows in manifests.items() for row in rows}
    if len(sources) != sum(map(len, manifests.values())):
        raise ValueError("SUN397 flattened class names collide")
    _materialize(base, sources, {s: str(p) for s, p in paths.items()})
    return _finished(base, dict(_folder_report(base, splits, 397, {s: 19850 for s in splits}), reused=False,
                protocol="SUN397 official partition 01; saved test permutation applied by loader", sources=[SUN_BASE],
                manifest_sha256={s: _digest(p) for s, p in paths.items()}))


def _resisc45(root, offline, source_dir=None, **unused):
    base = root / "resisc45"
    manifests, paths = {}, {}
    for split in ("train", "val", "test"):
        path = base / f"resisc45-{split}.txt"
        if not path.is_file():
            download_file([RESISC_BASE + path.name], path, sha256=RESISC_HASHES[split], offline=offline)
        paths[split], manifests[split] = path, _manifest(path)
        if any("/" in row for row in manifests[split]):
            raise ValueError(f"{path}: RESISC45 manifests must contain bare image filenames, matching the existing loader")
    _disjoint(manifests)
    def relative(row):
        return Path(row.rsplit("_", 1)[0]) / row if "/" not in row else Path(row)
    needed = [relative(row) for rows in manifests.values() for row in rows]
    raw = _find_raw([base / "NWPU-RESISC45", source_dir, root / "NWPU-RESISC45"],
                    lambda p: (p / needed[0]).is_file() or (p / needed[0].name).is_file())
    reused = raw == base / "NWPU-RESISC45"
    if raw is None:
        archive = root / ".downloads" / "NWPU-RESISC45.zip"
        download_file([RESISC_BASE + archive.name], archive,
                      sha256="beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53", offline=offline)
        extract_archive(archive, base)
        raw = base / "NWPU-RESISC45"
    image_root = base / "NWPU-RESISC45"
    for rel in needed:
        source = raw / rel if (raw / rel).is_file() else raw / rel.name
        _link(source, image_root / rel)
    classes = sorted({rel.parts[0] for rel in needed})
    if len(classes) != 45:
        raise ValueError(f"RESISC45: expected 45 classes, got {len(classes)}")
    for split, rows in manifests.items():
        present = {relative(row).parts[0] for row in rows}
        if present != set(classes):
            raise ValueError(f"RESISC45 {split}: missing classes")
    actual_classes = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    if actual_classes != classes:
        raise ValueError("RESISC45 image class folders differ from the split manifests")
    physical = set()
    for rel in needed:
        info = (image_root / rel).stat()
        identity = (info.st_dev, info.st_ino)
        if not info.st_size or identity in physical:
            raise ValueError(f"Empty or shared RESISC45 image: {image_root / rel}")
        physical.add(identity)
    # Loader filters by basename; duplicate names would silently join two samples.
    if len({rel.name for rel in needed}) != len(needed):
        raise ValueError("RESISC45: duplicate basenames in split manifests")
    hashes = {s: _digest(p) for s, p in paths.items()}
    original = hashes == RESISC_HASHES
    return {"root": str(base), "counts": {s: len(r) for s, r in manifests.items()}, "reused": reused,
            "protocol": "original remote-sensing train/val/test manifests" if original else "existing custom split manifests (unchanged)",
            "sources": [RESISC_BASE] if original or not reused else [], "manifest_sha256": hashes}


def prepare_vision_task(task, root, offline=False, **options):
    """Download missing files and prepare the repository's legacy layout.

    Options: source_dir, cars_splits, eurosat_split_dir, eurosat_protocol.
    Existing prepared trees take precedence over alternate source/protocol options.
    """
    handlers = {"cars": _cars, "stanford_cars": _cars, "dtd": _dtd, "eurosat": _eurosat,
                "sun397": _sun397, "resisc45": _resisc45}
    if task not in handlers:
        raise ValueError(f"Unsupported preparation task: {task}")
    root = Path(root).expanduser().resolve()
    if task == "cars":
        task = "stanford_cars"
    with file_lock(root / ".preparation" / f"{task}.lock"):
        result = handlers[task](root, offline=offline or offline_mode(), **options)
    return dict(result, task=task)
