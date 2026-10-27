# AGENT — scribe-md output organization

When you receive a raw scribe-md transcript (timestamped segment format), reorganize it into a human-readable Obsidian note before presenting it or saving it.

## What "organize the last output file" means

1. **Locate the most recent file** in `output/` by modification time. That is the "last output file."
2. **Read the full raw transcript** — all lines, no truncation.
3. **Produce a cleaned Obsidian Markdown file** overwriting the same path, with:
   - YAML frontmatter (`date`, `source`, `duration`, `language`, `model`, `tags`)
   - A top-level `# Title` heading derived from the filename
   - Section `##` headings that reflect the content structure (not timestamps)
   - Timestamps stripped from prose; inline timestamps only preserved when the user explicitly wants them
   - Repetitive Whisper artifacts and stutter repeats collapsed (mirror the spirit of `postprocess.clean_transcription` but applied by hand where needed)
   - Paragraphs broken for readability; list items where the source is enumerating

4. **Do not change the substance.** Reorganize — do not rewrite, summarize, or interpret unless asked.

## When to ask before organizing

- If the file is already clean (no timestamps, `---` frontmatter present, section headings present) — leave it.
- If the output file is a PDF or non-text format — the task is out of scope; tell the user.
- If there are multiple output files modified in the same batch and the user says "the last output file," pick the one with the newest mtime.

## Persisting the setting

This AGENT.md is the persistence mechanism. As long as it exists at the repo root, the behavior above is the default whenever the user says "organize the last output file," "clean up the transcript," or "make this readable."

If the user wants frontmatter fields tailored (e.g., extra tags, a different source label), ask them; do not invent tags beyond what the content supports.
