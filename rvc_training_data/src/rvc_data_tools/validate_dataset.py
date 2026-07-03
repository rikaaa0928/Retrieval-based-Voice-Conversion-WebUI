from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .audio_utils import audio_duration_seconds, is_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate clip durations in a generated dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_dir = args.dataset_dir / "audio"
    if not audio_dir.exists():
        raise SystemExit(f"缺少 audio 目录: {audio_dir}")

    rows = []
    bad = []
    total = 0.0
    for path in sorted(audio_dir.iterdir()):
        if not is_audio_file(path):
            continue
        duration = audio_duration_seconds(path)
        total += duration
        item = {
            "file": path.name,
            "duration_seconds": f"{duration:.3f}",
            "ok": str(args.min_seconds <= duration <= args.max_seconds),
        }
        rows.append(item)
        if item["ok"] != "True":
            bad.append(item)

    report_path = args.dataset_dir / "validation_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "duration_seconds", "ok"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "clips": len(rows),
        "bad_clips": len(bad),
        "duration_seconds": round(total, 3),
        "duration_minutes": round(total / 60, 3),
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
