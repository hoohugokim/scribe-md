"""Post-processing for transcription output (Phase 4.4).

Provides two levels of post-processing:

- **Rule-based cleaning** (``clean_transcription``): removes repeated phrases
  and common Whisper hallucination artifacts without any external dependencies.
- **LLM summarization** (``summarize_with_llm``): generates a concise summary
  using a local LLM via ``mlx-lm``.  This is entirely optional — the core tool
  works without it.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Known Whisper hallucination phrases
# ---------------------------------------------------------------------------

# These are common artifacts that Whisper produces when there is silence,
# music, or non-speech audio.  They appear across many languages.
_HALLUCINATION_PHRASES: list[str] = [
    "Thank you for watching",
    "Thank you for watching.",
    "Thank you for watching!",
    "Thanks for watching",
    "Thanks for watching.",
    "Thanks for watching!",
    "Please subscribe",
    "Please subscribe.",
    "Subscribe",
    "Subscribe.",
    "Like and subscribe",
    "Like and subscribe.",
    "Please like and subscribe",
    "자막 제공자",
    "자막 제작",
    "시청해 주셔서 감사합니다",
    "시청해주셔서 감사합니다",
    "구독과 좋아요",
    "구독과 좋아요 부탁드립니다",
    "MBC 뉴스 이덕영입니다",
    "Sous-titres réalisés par",
    "Sous-titres par",
    "Subtítulos realizados por",
    "Untertitel von",
    "ご視聴ありがとうございました",
    "字幕提供",
]

_LINE_ONLY_HALLUCINATION_PHRASES: list[str] = [
    "you",
]

# Compile a set for fast O(1) lookups (case-insensitive by lowering)
_HALLUCINATION_SET: set[str] = {
    p.lower().strip()
    for p in [*_HALLUCINATION_PHRASES, *_LINE_ONLY_HALLUCINATION_PHRASES]
}

_SENTENCE_TOKEN_RE = re.compile(r'(.+?(?:[.!?。！？]+|$))(\s*)', re.DOTALL)
_TIMESTAMP_PREFIX_RE = re.compile(r'^\s*\[\d{2}:\d{2}:\d{2}\]\s*')
_CJK_RE = re.compile(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]')
_SENTENCE_END_CHARS = ".!?。！？"


# ---------------------------------------------------------------------------
# Rule-based cleaning
# ---------------------------------------------------------------------------


def _remove_consecutive_duplicates(text: str) -> str:
    """Remove consecutive duplicate sentences or phrases.

    Splits the text into sentences (on ``.``, ``!``, ``?`` followed by
    whitespace or end-of-string) and removes any sentence that is identical
    to the one immediately before it.
    """
    tokens = _SENTENCE_TOKEN_RE.findall(text)
    if not tokens:
        return text

    deduped: list[str] = []
    previous_key = ""
    for sentence, separator in tokens:
        sentence = _remove_repeated_cjk_phrases(sentence)
        key = _dedupe_key(sentence)
        if key and key == previous_key:
            continue
        deduped.append(sentence)
        deduped.append(separator)
        if key:
            previous_key = key

    return "".join(deduped)


def _dedupe_key(sentence: str) -> str:
    """Normalize a sentence for adjacent-duplicate comparison."""
    key = _TIMESTAMP_PREFIX_RE.sub("", sentence.strip().lower(), count=1)
    return re.sub(r'\s+', ' ', key)


def _remove_repeated_cjk_phrases(sentence: str) -> str:
    """Collapse exact repeated CJK token phrases in unpunctuated fragments."""
    if not _CJK_RE.search(sentence) or any(ch in sentence for ch in _SENTENCE_END_CHARS):
        return sentence

    match = _TIMESTAMP_PREFIX_RE.match(sentence)
    prefix = match.group(0) if match else ""
    body = sentence[len(prefix):]
    tokens = body.split()
    if len(tokens) < 2:
        return sentence

    deduped: list[str] = []
    i = 0
    while i < len(tokens):
        phrase_size = 0
        max_size = min(8, (len(tokens) - i) // 2)
        for size in range(max_size, 0, -1):
            if tokens[i:i + size] == tokens[i + size:i + 2 * size]:
                phrase_size = size
                break

        if phrase_size == 0:
            deduped.append(tokens[i])
            i += 1
            continue

        phrase = tokens[i:i + phrase_size]
        deduped.extend(phrase)
        i += phrase_size
        while tokens[i:i + phrase_size] == phrase:
            i += phrase_size

    return prefix + " ".join(deduped)


def _remove_hallucination_lines(text: str) -> str:
    """Remove lines that consist entirely of a known hallucination phrase."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Check if the entire line (ignoring leading/trailing whitespace and
        # punctuation) matches a known hallucination phrase
        normalized = stripped.lower().rstrip(".!?,;:")
        if normalized in _HALLUCINATION_SET or stripped.lower() in _HALLUCINATION_SET:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _remove_hallucination_phrases_inline(text: str) -> str:
    """Remove hallucination phrases that appear inline within text.

    This handles cases where the hallucination appears as a sentence within
    a larger block of text rather than on its own line.
    """
    for phrase in _HALLUCINATION_PHRASES:
        # Remove the phrase when it appears as a standalone sentence
        # (preceded by sentence boundary or start, followed by punctuation)
        pattern = re.compile(
            r'(?:^|\.\s+)' + re.escape(phrase) + r'[.!?]*(?:\s+|$)',
            re.IGNORECASE,
        )
        text = pattern.sub(lambda m: ". " if m.group().startswith(".") else "", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Strip excessive whitespace and blank lines."""
    # Collapse runs of 3+ newlines into exactly 2 (one blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse runs of spaces/tabs (but not newlines) into a single space
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Remove trailing whitespace on each line
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


def clean_transcription(text: str) -> str:
    """Apply rule-based cleaning to transcription text.

    This does NOT require an LLM.  It performs:
    1. Removal of known Whisper hallucination phrases
    2. Removal of consecutive duplicate sentences
    3. Normalization of excessive whitespace

    Parameters
    ----------
    text : str
        Raw transcription text (Markdown).

    Returns
    -------
    str
        Cleaned transcription text.
    """
    if not text:
        return text

    text = _remove_hallucination_lines(text)
    text = _remove_hallucination_phrases_inline(text)
    text = _remove_consecutive_duplicates(text)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# LLM summarization (optional — requires mlx-lm)
# ---------------------------------------------------------------------------

_DEFAULT_SUMMARY_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def summarize_with_llm(text: str, model: str | None = None) -> str:
    """Generate a concise summary of the transcription using a local LLM.

    Requires ``mlx-lm`` to be installed.  If it is not available, a helpful
    error message is raised.

    Parameters
    ----------
    text : str
        The transcription text to summarize.
    model : str or None
        Model identifier (HuggingFace repo path).  If *None*, the default
        model ``mlx-community/Qwen2.5-1.5B-Instruct-4bit`` is used.

    Returns
    -------
    str
        The summary text.

    Raises
    ------
    ImportError
        If ``mlx-lm`` is not installed.
    """
    try:
        from mlx_lm import load, generate  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "mlx-lm is required for summarization but is not installed.\n"
            "Install it with: pip install mlx-lm"
        )

    model_name = model or _DEFAULT_SUMMARY_MODEL

    mlx_model, tokenizer = load(model_name)

    # Build a chat-style prompt if the tokenizer supports it, otherwise
    # fall back to a plain prompt.
    prompt_text = (
        "Summarize the following transcription concisely:\n\n" + text
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt = prompt_text

    summary = generate(
        mlx_model,
        tokenizer,
        prompt=prompt,
        max_tokens=512,
    )

    return summary.strip()
