"""Nemotron 3.5 Transcribe backend (Linux + NVIDIA via local vLLM).

Talks to an OpenAI-compatible server started with vLLM + NeMo Speech
(``nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B``). scribe-md does not
manage that server process — see docs/LINUX.md.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from ..transcriber import TranscriptionError
from .segments import segments_from_text, segments_from_timed

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
DEFAULT_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: omit fillers and "
    "accidental repetitions; keep the remaining spoken wording unchanged. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)
DEFAULT_TIMEOUT = 600.0
HEALTH_TIMEOUT = 1.0


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def health_ok(base_url: str = DEFAULT_BASE_URL, timeout: float = HEALTH_TIMEOUT) -> bool:
    """True when the local OpenAI-compatible server answers ``/models``."""
    url = f"{_normalize_base_url(base_url)}/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise TranscriptionError(
            f"Nemotron server HTTP {e.code}: {detail[:500]}"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TranscriptionError(
            f"Nemotron server unreachable at {url}: {e}. "
            "Start vLLM + NeMo Speech (see docs/LINUX.md), or use "
            "--engine whispercpp."
        ) from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise TranscriptionError(
            f"Nemotron server returned non-JSON: {body[:200]!r}"
        ) from e


def _extract_content(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def _extract_timed_segments(response: dict) -> list[dict] | None:
    """Best-effort pull of word/segment timestamps from non-standard fields."""
    for key in ("segments", "words"):
        raw = response.get(key)
        if isinstance(raw, list) and raw:
            return raw
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    for key in ("segments", "words"):
        raw = message.get(key)
        if isinstance(raw, list) and raw:
            return raw
    return None


def _wav_duration_seconds(path: Path) -> float:
    """Read duration from a PCM WAV header; 0.0 if unavailable."""
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (OSError, wave.Error):
        return 0.0


class NemotronBackend:
    """Transcription via a local Nemotron 3.5 Transcribe OpenAI server."""

    name = "nemotron"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        timeout_seconds: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.served_model = model
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds

    def describe(self) -> str:
        return f"Nemotron 3.5 Transcribe ({self.base_url})"

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None,
        device: str | None = None,
    ) -> dict:
        # model (Whisper preset) is ignored; the served checkpoint is configured
        # via [nemotron].model. device is ignored — vLLM owns GPU placement.
        _ = (model, device)
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        user_text = self.prompt
        if language:
            user_text = f"{self.prompt}\nLanguage code hint: {language}."

        payload = {
            "model": self.served_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_b64}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,
        }
        response = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            timeout=self.timeout_seconds,
        )
        timed = _extract_timed_segments(response)
        if timed:
            normalized = []
            for item in timed:
                if not isinstance(item, dict):
                    continue
                if "from" in item or "to" in item:
                    # whisper.cpp-style millisecond offsets
                    start = float(item.get("from", 0)) / 1000.0
                    end = float(item.get("to", item.get("from", 0))) / 1000.0
                else:
                    start = float(item.get("start", item.get("begin", 0.0)) or 0.0)
                    end = float(item.get("end", start) or start)
                normalized.append(
                    {
                        "start": start,
                        "end": end,
                        "text": item.get("text", item.get("word", "")),
                    }
                )
            result = segments_from_timed(normalized)
            if result["segments"]:
                return result

        text = _extract_content(response)
        duration = _wav_duration_seconds(audio_path)
        return segments_from_text(text, duration=duration)
