"""Shared filesystem layout helpers for SERVIS run outputs."""

import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path


_MAX_SLUG_LEN = 80


def slugify(value, default="run"):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    if not text:
        text = str(default)
    return text[:_MAX_SLUG_LEN].strip("_") or str(default)


def timestamp_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_id(parts, timestamp=None):
    timestamp = timestamp or timestamp_now()
    slugs = [slugify(part) for part in parts if part is not None and str(part).strip()]
    return "_".join([timestamp] + slugs)


def unique_run_root(runs_root, task, parts, timestamp=None):
    base_dir = Path(runs_root) / str(task)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(parts, timestamp=timestamp)
    candidate = base_dir / run_id
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    return candidate


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_command(path, argv=None):
    argv = list(sys.argv if argv is None else argv)
    text = shlex.join(argv) if argv else ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def relative_artifact(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_run_readme(path, title, fields=None, artifacts=None, notes=None):
    fields = fields or []
    artifacts = artifacts or []
    notes = notes or []
    lines = [f"# {title}", ""]
    if fields:
        lines.append("## Details")
        lines.append("")
        for key, value in fields:
            lines.append(f"- {key}: {value}")
        lines.append("")
    if artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for label, value in artifacts:
            lines.append(f"- {label}: `{value}`")
        lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
