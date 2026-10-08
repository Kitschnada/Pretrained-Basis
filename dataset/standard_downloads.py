"""Prepare byte-compatible MNIST, GTSRB, and SVHN files in their old layouts."""

import csv
import gzip
from pathlib import Path
import struct

from .download_utils import download_file, extract_archive, file_lock, _publish_member


MNIST_RESOURCES = (
    ("train-images-idx3-ubyte", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 60000, 2051),
    ("train-labels-idx1-ubyte", "d53e105ee54ea40749a09fcbcd1e9432", 60000, 2049),
    ("t10k-images-idx3-ubyte", "9fb629c4189551a2d022fa330f9573f3", 10000, 2051),
    ("t10k-labels-idx1-ubyte", "ec29112dd5afa0611ce80d1b7f02629c", 10000, 2049),
)
GTSRB_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"


def _check_mnist(path, count, magic):
    with path.open("rb") as source:
        header = source.read(8)
        if len(header) != 8 or struct.unpack(">II", header) != (magic, count):
            raise ValueError("Invalid MNIST IDX header: %s" % path)
        if magic == 2051 and source.read(8) != struct.pack(">II", 28, 28):
            raise ValueError("Invalid MNIST image dimensions: %s" % path)
    expected_size = 16 + count * 28 * 28 if magic == 2051 else 8 + count
    if path.stat().st_size != expected_size:
        raise ValueError("Truncated MNIST IDX file (not overwritten): %s" % path)


def prepare_mnist(root, offline=False):
    root = Path(root)
    processed = root / "MNIST" / "processed"
    if all((processed / name).is_file() for name in ("training.pt", "test.pt")):
        return {"task": "mnist", "source": "existing torchvision processed files", "root": str(root)}
    raw = root / "MNIST" / "raw"
    with file_lock(raw / ".pgsb-prepare.lock"):
        for name, md5, count, magic in MNIST_RESOURCES:
            path = raw / name
            if not path.exists():
                archive = download_file("https://ossci-datasets.s3.amazonaws.com/mnist/" + name + ".gz", raw / (name + ".gz"), md5=md5, offline=offline)
                with gzip.open(archive, "rb") as source:
                    _publish_member(source, path)
            _check_mnist(path, count, magic)
    return {"task": "mnist", "root": str(root), "source": "torchvision OSSCI mirror; original IDX order", "counts": {"train": 60000, "test_before_validation": 10000}}


def prepare_svhn(root, splits=("train", "test"), offline=False):
    root = Path(root) / "svhn"
    counts = {}
    for split in splits:
        if split not in ("train", "test"):
            raise ValueError("SVHN split must be train or test")
        filename = split + "_32x32.mat"
        checksum = "e26dedcc434d2e4c54c9b2d4a06d8373" if split == "train" else "eb5a983be6a315427106f1b164d9cef3"
        # The checksum is the original torchvision MAT checksum on every mirror.
        download_file([
            "https://ufldl.stanford.edu/housenumbers/" + filename,
            "https://huggingface.co/datasets/darcook/DCPV-Artifacts/resolve/main/data/svhn/" + filename,
            "http://ufldl.stanford.edu/housenumbers/" + filename,
        ], root / filename, md5=checksum, offline=offline)
        counts[split] = 73257 if split == "train" else 26032
    return {"task": "svhn", "root": str(root), "source": "original checksum-verified MAT files", "counts_before_validation": counts}


def _gtsrb_valid(root, split):
    if split == "train":
        directory = root / "GTSRB" / "Training"
        for label in range(43):
            class_dir = directory / ("%05d" % label)
            if not class_dir.is_dir() or not any(class_dir.glob("*.ppm")):
                return False
            annotation = class_dir / ("GT-%05d.csv" % label)
            if annotation.is_file():
                with annotation.open(newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter=";", skipinitialspace=True))
                if any(Path(row["Filename"]).name != row["Filename"] for row in rows):
                    raise ValueError("Invalid GTSRB training filename")
                if len({row["Filename"] for row in rows}) != len(rows) or any(int(row["ClassId"]) != label for row in rows):
                    raise ValueError("Duplicate names or incorrect GTSRB training labels")
                if not rows or any(not (class_dir / row["Filename"]).is_file() for row in rows):
                    return False
        # The competition archive has 26,640 images; preserve an existing
        # 39,209-image full-training layout too, without replacing either one.
        return len(list(directory.glob("*/*.ppm"))) in (26640, 39209)
    directory = root / "GTSRB" / "Final_Test" / "Images"
    annotations = root / "GT-final_test.csv"
    if not annotations.is_file():
        return False
    with annotations.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";", skipinitialspace=True))
    names = [row["Filename"] for row in rows]
    if len(rows) != 12630 or len(set(names)) != len(rows):
        raise ValueError("GTSRB test annotations are incomplete or contain duplicates: %s" % annotations)
    if any(Path(name).name != name for name in names) or any(not 0 <= int(row["ClassId"]) < 43 for row in rows):
        raise ValueError("Invalid GTSRB filenames/labels")
    return all((directory / name).is_file() for name in names)


def prepare_gtsrb(root, splits=("train", "test"), offline=False):
    root = Path(root) / "gtsrb"
    resources = {
        "train": [("GTSRB-Training_fixed.zip", "513f3c79a4c5141765e10e952eaa2478")],
        "test": [("GTSRB_Final_Test_Images.zip", "c7e4e6327067d32654124b0fe9e82185"),
                 ("GTSRB_Final_Test_GT.zip", "fe31e9c9270bbcd7b84b7f21a9d9d9e5")],
    }
    with file_lock(root / ".pgsb-prepare.lock"):
        for split in splits:
            if split not in resources:
                raise ValueError("GTSRB split must be train or test")
            if _gtsrb_valid(root, split):
                continue
            for filename, md5 in resources[split]:
                archive = download_file(GTSRB_URL + filename, root / filename, md5=md5, offline=offline)
                extract_archive(archive, root)
            if not _gtsrb_valid(root, split):
                raise ValueError("Incomplete GTSRB %s preparation at %s" % (split, root))
    return {"task": "gtsrb", "root": str(root), "source": GTSRB_URL, "splits": list(splits), "order": "original ImageFolder/GT CSV order"}


def prepare_standard_task(task, root, offline=False, splits=("train", "test")):
    if task == "mnist":
        return prepare_mnist(root, offline=offline)
    if task == "svhn":
        return prepare_svhn(root, splits=splits, offline=offline)
    if task == "gtsrb":
        return prepare_gtsrb(root, splits=splits, offline=offline)
    raise ValueError("Unknown standard vision dataset: %s" % task)
