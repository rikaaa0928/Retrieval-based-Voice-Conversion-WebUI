from __future__ import annotations

import argparse
from pathlib import Path

from .text_sources import LANGUAGES, load_language_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize public-domain text sources.")
    parser.add_argument("--language", choices=sorted(LANGUAGES), required=True)
    parser.add_argument("--out", type=Path, default=Path("data/source_cache"))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences, specs = load_language_sentences(args.language, args.out, refresh=args.refresh)
    normalized = args.out / f"{args.language}_sentences.txt"
    normalized.write_text("\n".join(sentences), encoding="utf-8")
    print(f"Downloaded {len(specs)} source(s), parsed {len(sentences)} sentence(s).")
    print(f"Normalized sentences: {normalized}")


if __name__ == "__main__":
    main()
