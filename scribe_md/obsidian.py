"""Obsidian vault integration for scribe-md.

Provides helpers for:
- Writing Markdown files with YAML frontmatter
- Appending transcriptions to daily notes
- Resolving output paths within an Obsidian vault
"""

from datetime import datetime
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as 'H:MM:SS' or 'M:SS'.

    Examples:
        332.0 -> '5:32'
        3661.0 -> '1:01:01'
        45.0 -> '0:45'
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def build_frontmatter(metadata: dict) -> str:
    """Build a YAML frontmatter string from a metadata dict.

    Expected keys (all optional):
        date: str (ISO date, e.g. '2026-03-13')
        source: str (e.g. 'YouTube: Video Title', 'file: recording.wav')
        duration: str (e.g. '5:32')
        language: str (e.g. 'ko')
        model: str (e.g. 'large-v3')
        tags: list[str]

    Returns the frontmatter block including the ``---`` delimiters and a
    trailing newline.
    """
    lines = ["---"]
    for key in ("date", "source", "duration", "language", "model"):
        if key in metadata and metadata[key]:
            value = metadata[key]
            # Quote strings that contain special YAML characters
            if isinstance(value, str) and (":" in value or '"' in value):
                value = f'"{value}"'
            lines.append(f"{key}: {value}")
    if "tags" in metadata and metadata["tags"]:
        tag_list = ", ".join(metadata["tags"])
        lines.append(f"tags: [{tag_list}]")
    lines.append("---")
    return "\n".join(lines) + "\n"


def write_with_frontmatter(
    output_path: Path,
    text: str,
    metadata: dict,
) -> None:
    """Write a Markdown file with YAML frontmatter prepended.

    Args:
        output_path: Path to the output ``.md`` file.
        text: The Markdown body (transcription text).
        metadata: Dict of frontmatter fields (see ``build_frontmatter``).
    """
    frontmatter = build_frontmatter(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(frontmatter + "\n" + text, encoding="utf-8")


def append_to_daily_note(
    vault_path: Path,
    daily_folder: str,
    text: str,
    metadata: dict,
) -> Path:
    """Append a transcription section to today's daily note.

    Creates the daily note file if it does not exist yet. Appends a section
    with a ``## Transcription (HH:MM)`` header followed by the transcription
    text.

    Args:
        vault_path: Root path of the Obsidian vault.
        daily_folder: Subfolder name for daily notes (e.g. 'Daily Notes').
        text: The Markdown body (transcription text).
        metadata: Dict of frontmatter fields — used for the source label.

    Returns:
        Path to the daily note file.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    daily_dir = vault_path / daily_folder
    daily_dir.mkdir(parents=True, exist_ok=True)
    note_path = daily_dir / f"{date_str}.md"

    # Build the section to append
    source = metadata.get("source", "")
    header = f"## Transcription ({time_str})"
    if source:
        header += f" - {source}"

    section = f"\n{header}\n\n{text}\n"

    if note_path.exists():
        # Append to existing daily note
        with open(note_path, "a", encoding="utf-8") as f:
            f.write(section)
    else:
        # Create new daily note with the section
        note_path.write_text(section.lstrip("\n"), encoding="utf-8")

    return note_path


def resolve_vault_output(vault_path: Path, filename: str) -> Path:
    """Resolve an output file path within the Obsidian vault.

    Args:
        vault_path: Root path of the Obsidian vault.
        filename: The desired filename (e.g. 'recording.md').

    Returns:
        Absolute path within the vault.
    """
    vault_path = Path(vault_path).expanduser().resolve()
    return vault_path / filename
