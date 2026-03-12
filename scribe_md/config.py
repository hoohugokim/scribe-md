"""Configuration file loading and merging for scribe-md.

Config search order (highest priority first):
  1. CLI flags
  2. Project-local `.scribe-md.toml`
  3. User config `~/.config/scribe-md/config.toml`
  4. Built-in defaults
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

USER_CONFIG_DIR = Path.home() / ".config" / "scribe-md"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.toml"
PROJECT_CONFIG_NAME = ".scribe-md.toml"

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_TOML = """\
[defaults]
model = "large-v3"        # or full HF path
language = ""             # auto-detect when empty
timestamps = true
timestamp_mode = "segment" # segment, paragraph, minute, or none
paragraph_gap = 2.0       # seconds of silence to trigger a paragraph break
chunk_seconds = 1800
overlap_seconds = 5
incremental = false       # append chunks to output file as they complete
parallel = true           # parallelize chunk transcription (file/url only)
workers = 2               # max parallel transcription workers (1-4)
clean = false             # apply rule-based artifact cleaning to transcription
summary_model = ""        # LLM model for --summarize (empty = default model)

[output]
directory = "."           # default output directory

[obsidian]
vault = ""                # path to Obsidian vault (empty = disabled)
daily_note_folder = "Daily Notes"  # subfolder for daily notes within vault

[live]
keep_audio = false
incremental = true        # live mode defaults to incremental output
"""


@dataclass
class ScribeMdConfig:
    """Resolved configuration for scribe-md."""

    # [defaults]
    model: str = "large-v3"
    language: str = ""
    timestamps: bool = True
    timestamp_mode: str = "segment"
    paragraph_gap: float = 2.0
    chunk_seconds: float = 1800
    overlap_seconds: float = 5
    incremental: bool = False
    parallel: bool = True
    workers: int = 2
    clean: bool = False
    summary_model: str = ""

    # [output]
    output_directory: str = "."

    # [obsidian]
    vault: str = ""
    daily_note_folder: str = "Daily Notes"

    # [live]
    keep_audio: bool = False
    live_incremental: bool = True

    # Metadata — which files contributed to this config
    _sources: list[str] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _find_project_config() -> Path | None:
    """Walk up from cwd looking for `.scribe-md.toml`."""
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        candidate = parent / PROJECT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def _apply_toml(cfg: ScribeMdConfig, data: dict, source: str) -> None:
    """Merge a parsed TOML dict into an existing config, mutating *cfg*."""
    defaults = data.get("defaults", {})
    output = data.get("output", {})
    obsidian = data.get("obsidian", {})
    live = data.get("live", {})

    if "model" in defaults:
        cfg.model = str(defaults["model"])
    if "language" in defaults:
        cfg.language = str(defaults["language"])
    if "timestamps" in defaults:
        cfg.timestamps = bool(defaults["timestamps"])
    if "chunk_seconds" in defaults:
        cfg.chunk_seconds = float(defaults["chunk_seconds"])
    if "overlap_seconds" in defaults:
        cfg.overlap_seconds = float(defaults["overlap_seconds"])
    if "timestamp_mode" in defaults:
        cfg.timestamp_mode = str(defaults["timestamp_mode"])
    if "paragraph_gap" in defaults:
        cfg.paragraph_gap = float(defaults["paragraph_gap"])
    if "incremental" in defaults:
        cfg.incremental = bool(defaults["incremental"])
    if "parallel" in defaults:
        cfg.parallel = bool(defaults["parallel"])
    if "workers" in defaults:
        cfg.workers = max(1, min(4, int(defaults["workers"])))
    if "clean" in defaults:
        cfg.clean = bool(defaults["clean"])
    if "summary_model" in defaults:
        cfg.summary_model = str(defaults["summary_model"])

    if "directory" in output:
        cfg.output_directory = str(output["directory"])

    if "vault" in obsidian:
        cfg.vault = str(obsidian["vault"])
    if "daily_note_folder" in obsidian:
        cfg.daily_note_folder = str(obsidian["daily_note_folder"])

    if "keep_audio" in live:
        cfg.keep_audio = bool(live["keep_audio"])
    if "incremental" in live:
        cfg.live_incremental = bool(live["incremental"])

    cfg._sources.append(source)


def _load_toml_file(path: Path) -> dict | None:
    """Read and parse a TOML file, returning None on any error."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config() -> ScribeMdConfig:
    """Load configuration by merging built-in defaults, user config, and
    project-local config.  CLI flags are NOT applied here — the caller should
    override individual fields after calling this function.

    Returns a ``ScribeMdConfig`` instance.
    """
    cfg = ScribeMdConfig()
    cfg._sources.append("built-in defaults")

    # User-level config
    if USER_CONFIG_PATH.is_file():
        data = _load_toml_file(USER_CONFIG_PATH)
        if data is not None:
            _apply_toml(cfg, data, str(USER_CONFIG_PATH))

    # Project-local config
    project_cfg = _find_project_config()
    if project_cfg is not None:
        data = _load_toml_file(project_cfg)
        if data is not None:
            _apply_toml(cfg, data, str(project_cfg))

    return cfg


def config_as_toml(cfg: ScribeMdConfig) -> str:
    """Render a ScribeMdConfig as a TOML string (for display purposes)."""
    lines = [
        "[defaults]",
        f'model = "{cfg.model}"',
        f'language = "{cfg.language}"',
        f"timestamps = {'true' if cfg.timestamps else 'false'}",
        f'timestamp_mode = "{cfg.timestamp_mode}"',
        f"paragraph_gap = {cfg.paragraph_gap}",
        f"chunk_seconds = {cfg.chunk_seconds}",
        f"overlap_seconds = {cfg.overlap_seconds}",
        f"incremental = {'true' if cfg.incremental else 'false'}",
        f"parallel = {'true' if cfg.parallel else 'false'}",
        f"workers = {cfg.workers}",
        f"clean = {'true' if cfg.clean else 'false'}",
        f'summary_model = "{cfg.summary_model}"',
        "",
        "[output]",
        f'directory = "{cfg.output_directory}"',
        "",
        "[obsidian]",
        f'vault = "{cfg.vault}"',
        f'daily_note_folder = "{cfg.daily_note_folder}"',
        "",
        "[live]",
        f"keep_audio = {'true' if cfg.keep_audio else 'false'}",
        f"incremental = {'true' if cfg.live_incremental else 'false'}",
    ]
    return "\n".join(lines) + "\n"


def init_user_config() -> Path:
    """Create the default user config file at ~/.config/scribe-md/config.toml.

    Returns the path to the created file.
    Raises FileExistsError if the file already exists.
    """
    if USER_CONFIG_PATH.exists():
        raise FileExistsError(f"Config file already exists: {USER_CONFIG_PATH}")
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_PATH.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return USER_CONFIG_PATH
