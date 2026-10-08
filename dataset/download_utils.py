"""Safe, cache-first dataset transfers. No model or CUDA dependencies."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
import time
import zipfile
from urllib.parse import urlsplit

import requests


def offline_mode():
    return any(os.getenv(key, "").lower() in ("1", "true", "yes") for key in (
        "PGSB_DATA_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE",
    ))


@contextmanager
def file_lock(path):
    """Serialize processes preparing the same files (Linux experiment servers)."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path, md5=None, sha256=None):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Empty or missing download: %s" % path)
    for algorithm, expected in (("md5", md5), ("sha256", sha256)):
        if expected and file_digest(path, algorithm) != expected.lower():
            raise ValueError("%s mismatch: %s" % (algorithm, path))


def download_file(urls, destination, md5=None, sha256=None, offline=False):
    """Download from verified mirrors with timeouts, retries, and atomic publish.

    Existing valid files win without a request. Existing invalid completed files
    are never overwritten. Partial transfers can resume when a digest is known;
    HTTP range replies are checked before appending. TLS verification stays on.
    """
    destination = Path(destination).expanduser()
    urls = [urls] if isinstance(urls, str) else list(urls)
    if not urls:
        raise ValueError("No download sources supplied for %s" % destination)
    with file_lock(str(destination) + ".lock"):
        if destination.exists():
            try:
                _verify(destination, md5, sha256)
            except ValueError as exc:
                raise ValueError("%s; move this file aside before retrying (it was not overwritten)." % exc) from exc
            return destination
        if offline or offline_mode():
            raise FileNotFoundError("Offline: missing %s. Prepare it online first." % destination)
        partial = destination.with_name(destination.name + ".part")
        errors = []
        for url in urls:
            endpoint = os.getenv("HF_ENDPOINT")
            parsed = urlsplit(url)
            if endpoint and parsed.hostname in ("huggingface.co", "hf.co"):
                url = endpoint.rstrip("/") + parsed.path + (("?" + parsed.query) if parsed.query else "")
            if not url.startswith("https://") and not url.startswith("http://"):
                raise ValueError("Expected an HTTP(S) dataset URL: %s" % url)
            for attempt in range(2):
                try:
                    # A previous interrupted transfer may already be complete.
                    if partial.exists() and (md5 or sha256):
                        try:
                            _verify(partial, md5, sha256)
                        except ValueError:
                            pass
                        else:
                            os.replace(str(partial), str(destination))
                            return destination
                    offset = partial.stat().st_size if partial.exists() and (md5 or sha256) else 0
                    headers = {"Accept-Encoding": "identity", "User-Agent": "PGSB-dataset-prepare/1.0"}
                    if offset:
                        headers["Range"] = "bytes=%d-" % offset
                    print("Downloading %s -> %s" % (url, destination), flush=True)
                    with requests.get(url, stream=True, timeout=(15, 60), headers=headers) as response:
                        if response.status_code == 416 and offset:
                            # An interrupted/corrupt partial can exceed the remote
                            # object. Restart the owned partial, never a completed file.
                            partial.unlink()
                            raise ValueError("Range rejected; retrying without the invalid partial")
                        response.raise_for_status()
                        if "text/html" in response.headers.get("Content-Type", "").lower():
                            raise ValueError("Received an HTML page instead of dataset bytes")
                        append = response.status_code == 206
                        if append:
                            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if not content_range or int(content_range[1]) != offset or int(content_range[2]) + 1 != int(content_range[3]):
                                raise ValueError("Invalid or incomplete HTTP Content-Range")
                        written = 0
                        with partial.open("ab" if append else "wb") as output:
                            for block in response.iter_content(chunk_size=1024 * 1024):
                                if block:
                                    output.write(block)
                                    written += len(block)
                            output.flush()
                            os.fsync(output.fileno())
                        length = response.headers.get("Content-Length")
                        if length is not None and written != int(length):
                            raise ValueError("Incomplete HTTP response: %d / %s bytes" % (written, length))
                    _verify(partial, md5, sha256)
                    os.replace(str(partial), str(destination))
                    return destination
                except (requests.RequestException, ValueError, OSError) as exc:
                    errors.append("%s: %s" % (url, exc))
                    # A completed corrupt response must restart, not be appended.
                    if isinstance(exc, ValueError) and partial.exists():
                        partial.unlink()
                    if attempt == 0:
                        time.sleep(1)
        raise RuntimeError("Dataset download failed; existing data is unchanged.\n" + "\n".join(errors))


def _safe_target(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe archive path: %s" % name)
    target = root / relative
    if root.resolve() not in (target.resolve(), *target.resolve().parents):
        raise ValueError("Archive path escapes destination: %s" % name)
    if target.is_symlink():
        raise ValueError("Refusing archive extraction over a symlink: %s" % target)
    return target


def _publish_member(stream, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite existing images, annotations, or user-prepared split files.
    if target.exists():
        if not target.is_file():
            raise ValueError("Archive file conflicts with existing directory: %s" % target)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != file_digest(target):
            raise ValueError("Archive differs from existing file (not overwritten): %s" % target)
        return
    descriptor, name = tempfile.mkstemp(prefix=".pgsb-extract-", dir=str(target.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
        os.link(str(temporary), str(target))  # Fails rather than replacing a concurrent file.
    finally:
        temporary.unlink()


def extract_archive(archive, destination):
    """Extract ZIP/TAR archives, rejecting traversal, links, and file conflicts."""
    archive, destination = Path(archive), Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    with file_lock(destination / ".pgsb-extract.lock"):
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for item in source.infolist():
                    target = _safe_target(destination, item.filename)
                    if stat.S_ISLNK(item.external_attr >> 16):
                        raise ValueError("Archive symlinks are not accepted: %s" % item.filename)
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        with source.open(item) as stream:
                            _publish_member(stream, target)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as source:
                for item in source:
                    target = _safe_target(destination, item.name)
                    if item.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif item.isfile():
                        with source.extractfile(item) as stream:
                            _publish_member(stream, target)
                    else:
                        raise ValueError("Only ordinary files/directories are accepted: %s" % item.name)
        else:
            raise ValueError("Not a supported ZIP/TAR archive: %s" % archive)
    return destination
