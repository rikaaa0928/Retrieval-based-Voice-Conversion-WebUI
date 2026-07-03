from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency for tiny demo use
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()
    tool_env = Path(__file__).resolve().parents[2] / ".env"
    repo_env = Path(__file__).resolve().parents[3] / ".env"
    if tool_env.exists():
        load_dotenv(tool_env)
    if repo_env.exists():
        load_dotenv(repo_env)


DEFAULT_BASE_URL = "https://tts.api.c.yiling.top/v1"
DEFAULT_MODEL = "utf-8-tts"
DEFAULT_VOICE = "leijun"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def create_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    resolved_api_key = api_key or os.getenv("TTS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "缺少 TTS_API_KEY。请在环境变量或 rvc_training_data/.env 中设置 TTS_API_KEY。"
        )

    return OpenAI(
        api_key=resolved_api_key,
        base_url=base_url or os.getenv("TTS_BASE_URL") or DEFAULT_BASE_URL,
        timeout=float(os.getenv("TTS_TIMEOUT_SECONDS", "120")),
    )


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers: Any = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def _is_retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status in RETRYABLE_STATUS_CODES:
        return True

    name = exc.__class__.__name__.lower()
    return "timeout" in name or "connection" in name or "ratelimit" in name


def generate_speech(
    text: str,
    output_file: str | Path,
    *,
    voice: str | None = None,
    model: str | None = None,
    response_format: str = "mp3",
    max_retries: int = 8,
    initial_delay: float = 2.0,
    max_delay: float = 90.0,
    client: OpenAI | None = None,
) -> Path:
    """Generate one TTS audio file with 429-aware retries."""

    text = text.strip()
    if not text:
        raise ValueError("text 不能为空")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_client = client or create_client()
    resolved_model = model or os.getenv("TTS_MODEL") or DEFAULT_MODEL
    resolved_voice = voice or os.getenv("TTS_VOICE") or DEFAULT_VOICE

    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            response = resolved_client.audio.speech.create(
                model=resolved_model,
                voice=resolved_voice,
                input=text,
                response_format=response_format,
            )
            response.stream_to_file(output_path)
            return output_path
        except Exception as exc:  # noqa: BLE001 - OpenAI-compatible servers vary
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise

            retry_after = _retry_after_seconds(exc)
            if retry_after is None:
                retry_after = min(max_delay, initial_delay * (2**attempt))
                retry_after += random.uniform(0.0, min(1.0, retry_after * 0.1))

            status = _status_code(exc)
            status_text = f"HTTP {status}" if status is not None else exc.__class__.__name__
            print(
                f"TTS 临时失败({status_text})，{retry_after:.1f}s 后重试 "
                f"({attempt + 1}/{max_retries})"
            )
            time.sleep(retry_after)

    assert last_exc is not None
    raise last_exc


def text_to_speech_demo() -> None:
    print("正在调用 TTS 服务...")
    input_text = "你好，这是一条测试音频"
    print(f"Input UTF-8 bytes length: {len(input_text.encode('utf-8'))}")
    output_file = generate_speech(input_text, "openai_output.mp3")
    print(f"成功！音频已保存到: {output_file}")
