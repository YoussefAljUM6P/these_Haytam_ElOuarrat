"""Shared filesystem layout helpers for SERVIS run outputs."""

import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path


_MAX_SLUG_LEN = 80


_ANSI_COLORS = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "grey": "90",
    "gray": "90",
    "bold": "1",
}


def color(text, *names):
    """Wrap ``text`` in ANSI color/style codes for terminal output.

    Accepts zero or more style names, e.g. ``color(x)``, ``color(x, "cyan")``,
    or ``color(x, "red", "bold")``. Falls back to plain text when stdout is not
    a TTY or no known styles are given, so piped/captured output (e.g. the
    compare runner reading console logs) stays clean.
    """
    codes = [c for c in (_ANSI_COLORS.get(str(n).lower()) for n in names) if c]
    if not codes or not sys.stdout.isatty():
        return str(text)
    return f"\033[{';'.join(codes)}m{text}\033[0m"


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


def run_folder_name(scene, controller, depth, feature_matcher=None, renderer=None):
    """Flat run-folder name: SCENE[-RENDERER]-CONTROLLER-DEPTH[-FEATUREMATCHER]."""
    parts = [scene]
    if renderer:
        parts.append(renderer)
    parts.extend([controller, depth])
    if feature_matcher:
        parts.append(feature_matcher)
    return "-".join(str(part).upper() for part in parts)


def run_folder(runs_root, scene, controller, depth, feature_matcher=None, renderer=None):
    """Return a fresh RUNS/<NAME> path, using -NN to avoid collisions.

    The first run for a config gets the bare name; subsequent runs get
    <NAME>-01, <NAME>-02, ... so prior runs are never overwritten.
    Mirrors the matching done by find_latest_resumable_run.
    """
    runs_root = Path(runs_root)
    name = run_folder_name(
        scene, controller, depth, feature_matcher, renderer=renderer
    )
    candidate = runs_root / name
    suffix = 1
    while candidate.exists():
        candidate = runs_root / f"{name}-{suffix:02d}"
        suffix += 1
    return candidate

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
