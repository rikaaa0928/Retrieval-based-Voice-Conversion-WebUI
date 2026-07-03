from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from .audio_utils import audio_duration_seconds
from .text_sources import LANGUAGES, iter_text_chunks, load_language_sentences
from .tts_bridge import generate_speech


FIELDNAMES = [
    "file",
    "duration_seconds",
    "language",
    "voice",
    "source_ids",
    "utf8_bytes",
    "estimated_seconds",
    "text",
]

FATAL_ERROR_MARKERS = (
    "缺少 TTS_API_KEY",
    "Incorrect API key",
    "invalid_api_key",
    "authentication",
    "Unauthorized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 5-15s TTS clips for RVC training.")
    parser.add_argument("--language", choices=sorted(LANGUAGES), required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--minutes", type=float, default=30.0, help="Target accepted audio duration.")
    parser.add_argument("--voice", default=None, help="TTS voice id. Defaults to TTS_VOICE or the bundled client default.")
    parser.add_argument("--model", default=None, help="TTS model id. Defaults to TTS_MODEL or the bundled client default.")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "flac", "opus", "aac"])
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-utf8-bytes", type=int, default=900)
    parser.add_argument("--refresh-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only write planned text chunks.")
    parser.add_argument("--max-clips", type=int, default=0, help="Optional cap for testing.")
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--tts-retries", type=int, default=8)
    return parser.parse_args()


def read_existing_metadata(path: Path) -> tuple[float, int]:
    if not path.exists():
        return 0.0, 0

    total = 0.0
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            total += float(row["duration_seconds"])
    return total, rows


def append_metadata(path: Path, row: dict[str, str | float]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def is_fatal_generation_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker.lower() in message.lower() for marker in FATAL_ERROR_MARKERS)


def main() -> None:
    args = parse_args()
    if args.minutes < 30 or args.minutes > 60:
        print("提示: RVC 推荐本工具按 30-60 分钟生成；当前 --minutes 不在该范围内，仍继续执行。")

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    rejected_dir = args.out / "rejected"
    text_dir = args.out / "texts"
    cache_dir = args.out / "source_cache"
    for directory in (audio_dir, rejected_dir, text_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sentences, specs = load_language_sentences(
        args.language,
        cache_dir,
        refresh=args.refresh_sources,
    )
    source_ids = ",".join(spec.id for spec in specs)
    metadata_path = args.out / "metadata.csv"
    total_seconds, existing_rows = read_existing_metadata(metadata_path)
    target_seconds = args.minutes * 60
    accepted = existing_rows
    rejected = 0

    chunk_iter = iter_text_chunks(
        sentences,
        args.language,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        seed=args.seed + existing_rows,
        max_utf8_bytes=args.max_utf8_bytes,
    )

    planned_text_path = text_dir / "planned_chunks.txt"
    progress = tqdm(total=target_seconds, initial=min(total_seconds, target_seconds), unit="s")
    try:
        for attempt in range(1, args.max_attempts + 1):
            if total_seconds >= target_seconds:
                break
            if args.max_clips and accepted >= args.max_clips:
                break

            text, estimated_seconds = next(chunk_iter)
            clip_index = accepted + rejected + 1
            stem = f"{args.language}_{clip_index:05d}"
            output_path = audio_dir / f"{stem}.{args.format}"
            text_path = text_dir / f"{stem}.txt"
            text_path.write_text(text, encoding="utf-8")

            with planned_text_path.open("a", encoding="utf-8") as handle:
                handle.write(text.replace("\n", " ") + "\n")

            if args.dry_run:
                accepted += 1
                continue

            try:
                generate_speech(
                    text,
                    output_path,
                    voice=args.voice,
                    model=args.model,
                    response_format=args.format,
                    max_retries=args.tts_retries,
                )
                duration = audio_duration_seconds(output_path)
            except Exception as exc:  # noqa: BLE001
                if is_fatal_generation_error(exc):
                    raise RuntimeError(f"TTS 配置错误，已停止生成: {exc}") from exc
                rejected += 1
                print(f"生成失败，跳过 {stem}: {exc}")
                continue

            if not (args.min_seconds <= duration <= args.max_seconds):
                rejected += 1
                rejected_path = rejected_dir / output_path.name
                shutil.move(str(output_path), rejected_path)
                print(f"时长 {duration:.2f}s 不在范围内，已移到 rejected/: {rejected_path.name}")
                continue

            accepted += 1
            total_seconds += duration
            append_metadata(
                metadata_path,
                {
                    "file": str(output_path.relative_to(args.out)),
                    "duration_seconds": f"{duration:.3f}",
                    "language": args.language,
                    "voice": args.voice or "",
                    "source_ids": source_ids,
                    "utf8_bytes": len(text.encode("utf-8")),
                    "estimated_seconds": f"{estimated_seconds:.3f}",
                    "text": text,
                },
            )
            progress.update(duration)
        else:
            raise RuntimeError(f"达到 max attempts={args.max_attempts}，仍未生成到目标时长")
    finally:
        progress.close()

    summary = {
        "language": args.language,
        "voice": args.voice,
        "target_minutes": args.minutes,
        "accepted_clips": accepted,
        "rejected_clips": rejected,
        "duration_seconds": round(total_seconds, 3),
        "duration_minutes": round(total_seconds / 60, 3),
        "sources": [spec.__dict__ for spec in specs],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
