#!/usr/bin/env python3
"""Transcribe WAV files to Markdown using mlx-whisper, with chunk merge support."""

import argparse
import json
import sys
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def transcribe(args):
    """Transcribe a single WAV file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    default_ext = ".json" if args.format == "json" else ".md"
    output_path = Path(args.output) if args.output else input_path.with_suffix(default_ext)

    import mlx_whisper

    kwargs = {"path_or_hf_repo": args.model}
    if args.language:
        kwargs["language"] = args.language

    print(f"Transcribing {input_path}...", file=sys.stderr)
    result = mlx_whisper.transcribe(str(input_path), **kwargs)

    if args.format == "json":
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        out = {"chunk_index": args.chunk_index, "segments": segments}
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        lines = []
        if args.timestamps and "segments" in result:
            for seg in result["segments"]:
                lines.append(f"{format_timestamp(seg['start'])} {seg['text'].strip()}")
        else:
            lines.append(result.get("text", "").strip())
        output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}", file=sys.stderr)


def merge(args):
    """Merge chunk JSON files into a single markdown file."""
    chunk_files = sorted(args.merge)
    if not chunk_files:
        print("Error: no chunk files to merge", file=sys.stderr)
        sys.exit(1)

    chunk_dur = args.chunk_duration
    overlap = args.overlap
    all_segments = []

    for cf in chunk_files:
        path = Path(cf)
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        idx = data.get("chunk_index", 0)

        # Chunk 0: absolute = whisper_ts, keep all
        # Chunk N>0: audio starts at N*chunk_dur - overlap
        #   skip segments where whisper_ts < overlap (already covered by previous chunk)
        #   absolute = whisper_ts + (N*chunk_dur - overlap)
        offset = 0.0 if idx == 0 else idx * chunk_dur - overlap

        for seg in data.get("segments", []):
            if idx > 0 and seg["start"] < overlap:
                continue
            all_segments.append({
                "start": offset + seg["start"],
                "end": offset + seg["end"],
                "text": seg["text"],
            })

    output_path = Path(args.output)
    lines = []
    if args.timestamps:
        for seg in all_segments:
            lines.append(f"{format_timestamp(seg['start'])} {seg['text']}")
    else:
        lines = [seg["text"] for seg in all_segments]

    output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged {len(chunk_files)} chunks -> {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV to Markdown")

    # Merge mode
    parser.add_argument("--merge", nargs="+", metavar="JSON", help="Merge chunk JSON files")
    parser.add_argument("--chunk-duration", type=float, default=60, help="Chunk duration (for merge)")
    parser.add_argument("--overlap", type=float, default=5, help="Overlap seconds (for merge)")

    # Transcription mode
    parser.add_argument("input", nargs="?", help="Input WAV file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--language", "-l", default=None, help="Language code (en, ko, etc.)")
    parser.add_argument("--timestamps", "-t", action="store_true", help="Include timestamps")
    parser.add_argument(
        "--model", "-m",
        default="mlx-community/whisper-large-v3-mlx",
        help="Model path or HF repo",
    )
    parser.add_argument("--format", "-f", choices=["md", "json"], default="md", help="Output format")
    parser.add_argument("--chunk-index", type=int, default=0, help="Chunk index (for JSON output)")

    args = parser.parse_args()

    if args.merge:
        merge(args)
    elif args.input:
        transcribe(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
