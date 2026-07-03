from __future__ import annotations

import json
import random
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests


CATALOG_PATH = Path(__file__).resolve().parents[2] / "sources" / "catalog.json"
LANGUAGES = {"zh", "en", "ja"}
LANGUAGE_PROFILES = {
    "zh": {"units_per_second": 4.2, "min_units": 18},
    "en": {"units_per_second": 2.35, "min_units": 12},
    "ja": {"units_per_second": 4.8, "min_units": 20},
}


@dataclass(frozen=True)
class SourceSpec:
    id: str
    title: str
    author: str
    format: str
    url: str = ""
    path: str = ""
    encoding: str = "utf-8"
    license_note: str = ""


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, list[SourceSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        lang: [SourceSpec(**item) for item in items]
        for lang, items in raw.items()
    }


def source_cache_path(cache_dir: Path, spec: SourceSpec) -> Path:
    suffix = ".zip" if spec.format.endswith("zip") else ".txt"
    return cache_dir / f"{spec.id}{suffix}"


def download_source(spec: SourceSpec, cache_dir: Path, refresh: bool = False) -> Path:
    if spec.path:
        local_path = Path(spec.path)
        if not local_path.is_absolute():
            local_path = CATALOG_PATH.parent / local_path
        if not local_path.exists():
            raise FileNotFoundError(f"本地文本源不存在: {local_path}")
        return local_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = source_cache_path(cache_dir, spec)
    if target.exists() and not refresh:
        return target

    if not spec.url:
        raise ValueError(f"{spec.id} 缺少 url 或 path")

    response = requests.get(spec.url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def read_source_text(path: Path, spec: SourceSpec) -> str:
    if spec.format == "aozora_zip":
        with zipfile.ZipFile(path) as archive:
            txt_names = [name for name in archive.namelist() if name.endswith(".txt")]
            if not txt_names:
                raise ValueError(f"{path} 中没有 txt 文件")
            data = archive.read(txt_names[0])
        return data.decode(spec.encoding, errors="replace")

    return path.read_text(encoding=spec.encoding, errors="replace")


def strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*", text, re.I | re.S)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text


def strip_aozora_markup(text: str) -> str:
    text = re.sub(r"-{20,}.*?-{20,}", "", text, flags=re.S)
    text = re.sub(r"［＃.*?］", "", text)
    text = re.sub(r"｜", "", text)
    text = re.sub(r"《.*?》", "", text)
    text = re.sub(r"※［.*?］", "", text)
    return text


def normalize_text(text: str, language: str) -> str:
    text = strip_gutenberg_boilerplate(text)
    if language == "ja":
        text = strip_aozora_markup(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def split_sentences(text: str, language: str) -> list[str]:
    if language == "en":
        pieces = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    else:
        pieces = re.split(r"(?<=[。！？!?])\s*|\n+", text)

    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        sentences.extend(split_long_sentence(piece, language))
    return [s for s in sentences if unit_count(s, language) >= 3]


def split_long_sentence(sentence: str, language: str, max_units: int = 80) -> list[str]:
    if unit_count(sentence, language) <= max_units:
        return [sentence]

    separators = r"[,;:]" if language == "en" else r"[，、；：,;:]"
    parts = [p.strip() for p in re.split(separators, sentence) if p.strip()]
    if len(parts) <= 1:
        return [sentence]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if language == "en" else f"{current}{part}"
        if current and unit_count(candidate, language) > max_units:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def unit_count(text: str, language: str) -> int:
    if language == "en":
        return len(re.findall(r"[A-Za-z0-9']+", text))
    return len(re.sub(r"\s+", "", text))


def estimate_seconds(text: str, language: str) -> float:
    profile = LANGUAGE_PROFILES[language]
    return unit_count(text, language) / profile["units_per_second"]


def load_language_sentences(
    language: str,
    cache_dir: Path,
    *,
    refresh: bool = False,
    catalog_path: Path = CATALOG_PATH,
) -> tuple[list[str], list[SourceSpec]]:
    if language not in LANGUAGES:
        raise ValueError(f"language 必须是 {sorted(LANGUAGES)} 之一")

    catalog = load_catalog(catalog_path)
    specs = catalog.get(language, [])
    if not specs:
        raise ValueError(f"catalog 中没有 {language} 数据源")

    all_sentences: list[str] = []
    for spec in specs:
        raw_path = download_source(spec, cache_dir, refresh=refresh)
        raw_text = read_source_text(raw_path, spec)
        text = normalize_text(raw_text, language)
        all_sentences.extend(split_sentences(text, language))

    if not all_sentences:
        raise RuntimeError(f"没有从 {language} 数据源中解析出句子")
    return all_sentences, specs


def iter_text_chunks(
    sentences: Iterable[str],
    language: str,
    *,
    min_seconds: float = 5.0,
    max_seconds: float = 15.0,
    seed: int = 1234,
    max_utf8_bytes: int = 900,
):
    rng = random.Random(seed)
    pool = list(sentences)
    rng.shuffle(pool)
    index = 0
    multiplier = 1.0

    while True:
        if index >= len(pool):
            rng.shuffle(pool)
            index = 0

        target_seconds = rng.uniform(min_seconds, max_seconds)
        target_units = max(
            LANGUAGE_PROFILES[language]["min_units"],
            int(target_seconds * LANGUAGE_PROFILES[language]["units_per_second"] * multiplier),
        )

        chunk_parts: list[str] = []
        while index < len(pool):
            candidate_part = pool[index]
            index += 1
            candidate = join_sentences(chunk_parts + [candidate_part], language)
            if len(candidate.encode("utf-8")) > max_utf8_bytes and chunk_parts:
                break
            chunk_parts.append(candidate_part)
            if unit_count(candidate, language) >= target_units:
                break

        if not chunk_parts:
            continue

        text = join_sentences(chunk_parts, language)
        estimated = estimate_seconds(text, language)
        yield text, estimated


def join_sentences(parts: list[str], language: str) -> str:
    if language == "en":
        return " ".join(parts)
    return "".join(parts)
