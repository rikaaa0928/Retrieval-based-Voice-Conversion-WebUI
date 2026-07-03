from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip a generated dataset for Google Drive upload.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"目录不存在: {dataset_dir}")

    out = args.out or dataset_dir.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in dataset_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_dir.parent))
    print(f"Created: {out}")


if __name__ == "__main__":
    main()
