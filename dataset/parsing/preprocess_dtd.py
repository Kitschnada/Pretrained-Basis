"""Prepare DTD train1/val1/test1 without manual path edits."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.vision_downloads import prepare_vision_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-dir", help="Existing raw DTD directory containing images/ and labels/")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_vision_task("dtd", args.data_root, offline=args.offline,
                                       source_dir=args.source_dir), indent=2))


if __name__ == "__main__":
    main()
