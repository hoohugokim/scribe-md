#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAPTURE_BIN="$SCRIPT_DIR/capture/.build/release/appaudio-capture"

# Defaults
DURATION=""
OUTPUT="transcription.md"
LANGUAGE=""
TIMESTAMPS=""
MODEL=""
KEEP_AUDIO=false
CHUNK_SECONDS=""
OVERLAP_SECONDS="5"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --duration, -d SEC        Recording duration in seconds (omit for manual stop)
  --output, -o PATH         Output markdown file (default: transcription.md)
  --language, -l LANG       Language code (en, ko, etc.)
  --timestamps, -t          Include timestamps in output
  --model, -m MODEL         Whisper model to use
  --keep-audio              Keep the intermediate WAV file(s)
  --chunk-seconds SEC       Enable chunked pipeline (transcribe every N seconds)
  --overlap-seconds SEC     Overlap between chunks (default: 5)
  -h, --help                Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration|-d) DURATION="$2"; shift 2 ;;
        --output|-o) OUTPUT="$2"; shift 2 ;;
        --language|-l) LANGUAGE="$2"; shift 2 ;;
        --timestamps|-t) TIMESTAMPS="--timestamps"; shift ;;
        --model|-m) MODEL="$2"; shift 2 ;;
        --keep-audio) KEEP_AUDIO=true; shift ;;
        --chunk-seconds) CHUNK_SECONDS="$2"; shift 2 ;;
        --overlap-seconds) OVERLAP_SECONDS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Build capture binary if needed
if [[ ! -x "$CAPTURE_BIN" ]]; then
    echo "Building capture tool..." >&2
    (cd "$SCRIPT_DIR/capture" && swift build -c release)
fi

# Common transcribe args
COMMON_ARGS=()
[[ -n "$LANGUAGE" ]] && COMMON_ARGS+=(--language "$LANGUAGE")
[[ -n "$MODEL" ]] && COMMON_ARGS+=(--model "$MODEL")

if [[ -n "$CHUNK_SECONDS" ]]; then
    # ── Chunked pipeline ──────────────────────────────────────────────
    CHUNK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/appaudio-chunks-XXXXXX")
    FIFO="$CHUNK_DIR/fifo"
    mkfifo "$FIFO"

    chunk_cleanup() {
        rm -f "$FIFO"
        if [[ "$KEEP_AUDIO" == true ]]; then
            echo "Audio chunks saved: $CHUNK_DIR" >&2
        else
            rm -rf "$CHUNK_DIR"
        fi
    }
    trap chunk_cleanup EXIT

    # Use trap-with-handler (not trap '') so children get default SIGINT
    trap : INT

    CAPTURE_ARGS=(--output "$CHUNK_DIR/chunk.wav" --chunk-seconds "$CHUNK_SECONDS" --overlap-seconds "$OVERLAP_SECONDS")
    [[ -n "$DURATION" ]] && CAPTURE_ARGS+=(--duration "$DURATION")

    "$CAPTURE_BIN" "${CAPTURE_ARGS[@]}" > "$FIFO" &
    CAPTURE_PID=$!

    CHUNK_IDX=0
    while IFS= read -r CHUNK_RAW; do
        IDX=$(printf '%03d' "$CHUNK_IDX")
        CHUNK_16K="$CHUNK_DIR/chunk_${IDX}_16k.wav"
        CHUNK_JSON="$CHUNK_DIR/chunk_${IDX}.json"

        echo "Chunk $CHUNK_IDX: processing..." >&2
        ffmpeg -y -i "$CHUNK_RAW" -ar 16000 -ac 1 -sample_fmt s16 "$CHUNK_16K" -loglevel error || { CHUNK_IDX=$((CHUNK_IDX + 1)); continue; }
        pixi run python "$SCRIPT_DIR/transcribe.py" "$CHUNK_16K" -o "$CHUNK_JSON" -f json --chunk-index "$CHUNK_IDX" "${COMMON_ARGS[@]}" || true

        CHUNK_IDX=$((CHUNK_IDX + 1))
    done < "$FIFO"

    CAPTURE_EXIT=0
    wait "$CAPTURE_PID" || CAPTURE_EXIT=$?
    trap - INT

    if [[ "$CAPTURE_EXIT" -ne 0 ]]; then
        echo "Recording cancelled." >&2
    fi

    # Merge whatever chunks were transcribed
    shopt -s nullglob
    CHUNK_JSONS=("$CHUNK_DIR"/chunk_*.json)
    shopt -u nullglob

    if [[ ${#CHUNK_JSONS[@]} -eq 0 ]]; then
        echo "No chunks were transcribed." >&2
        exit 1
    fi

    MERGE_ARGS=(--merge "${CHUNK_JSONS[@]}" --chunk-duration "$CHUNK_SECONDS" --overlap "$OVERLAP_SECONDS" -o "$OUTPUT")
    [[ -n "$TIMESTAMPS" ]] && MERGE_ARGS+=("$TIMESTAMPS")
    pixi run python "$SCRIPT_DIR/transcribe.py" "${MERGE_ARGS[@]}"

    echo "Done: $OUTPUT (${#CHUNK_JSONS[@]} chunks)" >&2

else
    # ── Single-file pipeline ──────────────────────────────────────────
    TMPWAV_RAW=$(mktemp "${TMPDIR:-/tmp}/appaudio-raw-XXXXXX.wav")
    TMPWAV=$(mktemp "${TMPDIR:-/tmp}/appaudio-XXXXXX.wav")

    cleanup() {
        rm -f "$TMPWAV_RAW"
        if [[ "$KEEP_AUDIO" == false && -f "$TMPWAV" ]]; then
            rm -f "$TMPWAV"
        elif [[ "$KEEP_AUDIO" == true ]]; then
            echo "Audio saved: $TMPWAV" >&2
        fi
    }
    trap cleanup EXIT

    trap : INT

    CAPTURE_ARGS=(--output "$TMPWAV_RAW")
    [[ -n "$DURATION" ]] && CAPTURE_ARGS+=(--duration "$DURATION")

    CAPTURE_EXIT=0
    "$CAPTURE_BIN" "${CAPTURE_ARGS[@]}" || CAPTURE_EXIT=$?

    trap - INT

    if [[ "$CAPTURE_EXIT" -ne 0 ]]; then
        echo "Recording cancelled." >&2
        exit 1
    fi

    echo "Converting to 16kHz mono..." >&2
    ffmpeg -y -i "$TMPWAV_RAW" -ar 16000 -ac 1 -sample_fmt s16 "$TMPWAV" -loglevel error

    TRANSCRIBE_ARGS=("$TMPWAV" --output "$OUTPUT" "${COMMON_ARGS[@]}")
    [[ -n "$TIMESTAMPS" ]] && TRANSCRIBE_ARGS+=("$TIMESTAMPS")
    pixi run python "$SCRIPT_DIR/transcribe.py" "${TRANSCRIBE_ARGS[@]}"

    echo "Done: $OUTPUT" >&2
fi
