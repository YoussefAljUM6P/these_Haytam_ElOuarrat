"""Detect and persist capture "takes" for a scene.

Some datasets are one flat list of frames, but the capture is often
several passes ("takes"): the camera moves smoothly, then teleports to start a
new pass. There is no metadata for this — the only signal is a large jump in
camera position between consecutive frames.

This module detects those jumps from the COLMAP poses (the same source the
trajectory runner uses, so frame ordering and centers match exactly) and writes
the segmentation to ``DATA/<scene>/takes.json``. The trajectory runner calls
``ensure_takes`` before a run: it reuses a valid file and regenerates a stale or
missing one, so the file is always present and consistent before a simulation.

Use via the CLI:
    python cli.py takes <scene> [--jump-factor 5.0] [--force] [--show]
    python cli.py takes --all
"""

import json
import re
from pathlib import Path

from run_layout import color
from runners.servo_frames import frame_id_from_path, frame_number

# numpy / pycolmap are heavy; import them inside the functions that need them so
# `cli.py` can register this subcommand without pulling them in.


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"

# A step between consecutive camera centers is a take boundary when it exceeds
# this many times the median step over the whole sequence. 5x is loose enough to
# never cut a smooth orbit (speed varies maybe 2-3x) yet catches real teleports.
DEFAULT_JUMP_FACTOR = 5.0

TAKES_FILENAME = "takes.json"
TAKES_VERSION = 1

# Some datasets (Stanford/SUN3D "gates*" StructureSensor captures) ship the
# authoritative take boundaries in a split.txt alongside the frames, one line
# per pass, e.g. "sequence0 [frames=1053]  [start=0 ; end=1052]". When present
# this is preferred over jump detection: it needs no threshold and stays correct
# even when the COLMAP model only registered some of the sequences.
SPLIT_FILENAME = "split.txt"
_SPLIT_RANGE_RE = re.compile(r"start\s*=\s*(\d+)\s*;\s*end\s*=\s*(\d+)")


def parse_split(scene_dir):
    """Parse a dataset ``split.txt`` into inclusive ``(start, end)`` frame ranges.

    Returns the ranges in file order, or ``None`` when the file is absent or has
    no parseable ``start=.. ; end=..`` lines. Ranges are by frame *number* (the
    integer in ``frame-NNNNNN``), matching how the captures are numbered on disk.
    """
    path = Path(scene_dir) / SPLIT_FILENAME
    if not path.exists():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    ranges = [(int(m.group(1)), int(m.group(2)))
              for m in _SPLIT_RANGE_RE.finditer(text)]
    return ranges or None


def takes_path(scene_dir):
    return Path(scene_dir) / TAKES_FILENAME


def _camera_centers(frame_ids, camera_of):
    """(frame_ids already sorted) -> (N, 3) world-space camera centers."""
    import numpy as np

    return np.array(
        [np.asarray(camera_of(fid).T_world_cam, dtype=np.float64)[:3, 3]
         for fid in frame_ids],
        dtype=np.float64,
    )


def detect_takes(frame_ids, centers, jump_factor=DEFAULT_JUMP_FACTOR, scene=None):
    """Split an ordered frame list into takes at large camera-center jumps.

    ``frame_ids`` must already be in capture order and ``centers[i]`` is the
    world-space camera center of ``frame_ids[i]``. Returns the JSON-serializable
    takes document.
    """
    import numpy as np

    centers = np.asarray(centers, dtype=np.float64)
    n = len(frame_ids)
    if n == 0:
        raise ValueError("detect_takes: no frames")

    if n > 1:
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        median_step = float(np.median(steps))
    else:
        steps = np.zeros(0, dtype=np.float64)
        median_step = 0.0

    threshold = jump_factor * median_step if median_step > 0.0 else float("inf")

    # Cut *after* frame i whenever the step from i to i+1 exceeds the threshold.
    cut_after = [i for i, s in enumerate(steps) if s > threshold]
    bounds = [0] + [i + 1 for i in cut_after] + [n]

    takes = []
    for k in range(len(bounds) - 1):
        start, end = bounds[k], bounds[k + 1]
        seg = list(frame_ids[start:end])
        internal = steps[start:end - 1] if end - 1 > start else np.zeros(0)
        takes.append({
            "index": k,
            "start_frame": seg[0],
            "end_frame": seg[-1],
            "count": len(seg),
            "max_internal_step": float(internal.max()) if internal.size else 0.0,
            # step that separates this take from the next one (None for the last)
            "boundary_step": float(steps[end - 1]) if end < n else None,
            "frames": seg,
        })

    return {
        "version": TAKES_VERSION,
        "scene": scene,
        "source": "colmap",
        "num_frames": n,
        "jump_factor": float(jump_factor),
        "median_step": median_step,
        "threshold": (None if threshold == float("inf") else threshold),
        "num_takes": len(takes),
        "takes": takes,
    }


def detect_takes_from_split(frame_ids, centers, ranges, scene=None):
    """Segment frames using dataset-provided ``split.txt`` ranges.

    Each frame is assigned to the range whose inclusive ``[start, end]`` (by
    frame number) contains it. Only frames present in ``frame_ids`` are emitted,
    so a scene whose COLMAP model registered just some sequences yields exactly
    the takes it has frames for (the rest simply have no frames yet). Ranges with
    no present frame are skipped and takes are re-indexed contiguously.

    ``frame_ids`` must be in capture order with ``centers[i]`` the world-space
    center of ``frame_ids[i]``. Returns the same JSON document shape as
    :func:`detect_takes`.
    """
    import numpy as np

    centers = np.asarray(centers, dtype=np.float64)
    n = len(frame_ids)
    if n == 0:
        raise ValueError("detect_takes_from_split: no frames")

    def range_of(num):
        for r, (start, end) in enumerate(ranges):
            if start <= num <= end:
                return r
        return None

    assigned = [range_of(frame_number(f)) for f in frame_ids]
    dropped = sum(1 for a in assigned if a is None)

    steps = (np.linalg.norm(np.diff(centers, axis=0), axis=1)
             if n > 1 else np.zeros(0, dtype=np.float64))
    median_step = float(np.median(steps)) if steps.size else 0.0

    # Collect the frame indices belonging to each range, in order.
    grouped = []  # (range_idx, [frame index into frame_ids, ...])
    for r in range(len(ranges)):
        idxs = [i for i, a in enumerate(assigned) if a == r]
        if idxs:
            grouped.append((r, idxs))

    takes = []
    for k, (r, idxs) in enumerate(grouped):
        seg = [frame_ids[i] for i in idxs]
        internal = np.array(
            [float(np.linalg.norm(centers[idxs[j + 1]] - centers[idxs[j]]))
             for j in range(len(idxs) - 1)],
            dtype=np.float64,
        ) if len(idxs) > 1 else np.zeros(0)
        # Distance from this take's last present frame to the next take's first.
        boundary_step = None
        if k + 1 < len(grouped):
            next_idxs = grouped[k + 1][1]
            boundary_step = float(
                np.linalg.norm(centers[next_idxs[0]] - centers[idxs[-1]])
            )
        takes.append({
            "index": k,
            "start_frame": seg[0],
            "end_frame": seg[-1],
            "count": len(seg),
            "max_internal_step": float(internal.max()) if internal.size else 0.0,
            "boundary_step": boundary_step,
            "split_range": list(ranges[r]),
            "frames": seg,
        })

    return {
        "version": TAKES_VERSION,
        "scene": scene,
        "source": "split",
        "num_frames": n,
        "num_ranges": len(ranges),
        "dropped_frames": dropped,
        "jump_factor": None,
        "median_step": median_step,
        "threshold": None,
        "num_takes": len(takes),
        "takes": takes,
    }


def build_takes_document(scene_dir, frame_ids, centers,
                         jump_factor=DEFAULT_JUMP_FACTOR):
    """Detect takes for ``scene_dir``, preferring ``split.txt`` over jumps.

    Uses the dataset's authoritative ``split.txt`` boundaries when present,
    otherwise falls back to camera-center jump detection.
    """
    ranges = parse_split(scene_dir)
    if ranges:
        return detect_takes_from_split(
            frame_ids, centers, ranges, scene=Path(scene_dir).name
        )
    return detect_takes(
        frame_ids, centers, jump_factor=jump_factor, scene=Path(scene_dir).name
    )


def validate_takes(data, frame_ids):
    """Check a loaded takes document against the scene's current frame list.

    Returns ``(ok, reason)``. Validity means the takes partition exactly the
    frames present now, in order — so a dataset that changed on disk is caught
    and the file is regenerated rather than silently mis-segmenting a run.
    """
    if not isinstance(data, dict):
        return False, "not a JSON object"
    takes = data.get("takes")
    if not isinstance(takes, list) or not takes:
        return False, "missing or empty 'takes'"

    concat = []
    for take in takes:
        if not isinstance(take, dict):
            return False, "a take entry is not an object"
        frames = take.get("frames")
        if not isinstance(frames, list) or not frames:
            return False, f"take {take.get('index', '?')} has no 'frames'"
        concat.extend(frames)

    expected = list(frame_ids)
    if concat != expected:
        return False, (
            f"frame mismatch: takes cover {len(concat)} frame(s), "
            f"scene has {len(expected)} (order/content differs)"
        )
    return True, "ok"


def load_takes(scene_dir):
    path = takes_path(scene_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_takes(scene_dir, data):
    path = takes_path(scene_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def ensure_takes(scene_dir, frame_index, jump_factor=DEFAULT_JUMP_FACTOR,
                 regenerate=False):
    """Return the list of takes (each a list of frame ids) for ``scene_dir``.

    Reuses ``takes.json`` when it is present and still valid for the loaded
    frames; otherwise detects takes from ``frame_index`` camera centers, writes
    the file, and returns the result. ``frame_index`` maps frame id ->
    ``{"camera": Camera, ...}`` (the trajectory runner's index), so detection
    uses the exact poses the run will use.
    """
    scene_dir = Path(scene_dir)
    frame_ids = sorted(frame_index, key=frame_number)
    path = takes_path(scene_dir)

    if not regenerate:
        existing = load_takes(scene_dir)
        if existing is not None:
            ok, reason = validate_takes(existing, frame_ids)
            if ok:
                takes = [t["frames"] for t in existing["takes"]]
                print(color(
                    f"[takes] {path.name}: {len(takes)} take(s), "
                    f"{len(frame_ids)} frames (reused)",
                    "cyan",
                ))
                return takes
            print(color(
                f"[takes] {path.name} invalid ({reason}); regenerating",
                "yellow",
            ))

    centers = _camera_centers(
        frame_ids, lambda fid: frame_index[fid]["camera"]
    )
    data = build_takes_document(
        scene_dir, frame_ids, centers, jump_factor=jump_factor
    )
    save_takes(scene_dir, data)
    thr = data["threshold"]
    print(color(
        f"[takes] wrote {path.name}: {data['num_takes']} take(s) from "
        f"{data['num_frames']} frames "
        f"(median step={data['median_step']:.4f}, "
        f"threshold={'inf' if thr is None else f'{thr:.4f}'})",
        "cyan",
    ))
    return [t["frames"] for t in data["takes"]]


# ---- standalone CLI --------------------------------------------------------


def compute_takes_for_scene(scene_dir, jump_factor=DEFAULT_JUMP_FACTOR):
    """Load COLMAP poses for a scene and detect takes (no file written)."""
    from dataset import load_colmap

    scene_dir = Path(scene_dir)
    records = load_colmap(scene_dir)
    if not records:
        raise RuntimeError(f"No COLMAP frames loaded from {scene_dir}")

    camera_by_id = {}
    for record in records:
        camera, rgb_path = record[0], record[1]
        camera_by_id[frame_id_from_path(rgb_path)] = camera

    frame_ids = sorted(camera_by_id, key=frame_number)
    centers = _camera_centers(frame_ids, lambda fid: camera_by_id[fid])
    return build_takes_document(
        scene_dir, frame_ids, centers, jump_factor=jump_factor
    )


def print_takes_summary(data):
    thr = data["threshold"]
    source = data.get("source", "colmap")
    if source == "split":
        dropped = data.get("dropped_frames", 0)
        detail = f"source=split.txt ({data.get('num_ranges', '?')} ranges"
        detail += f", {dropped} frames outside any range)" if dropped else ")"
    else:
        detail = (
            f"median step={data['median_step']:.4f}, "
            f"threshold={'inf' if thr is None else f'{thr:.4f}'}"
        )
    print(
        f"  {data.get('scene', '?')}: {data['num_takes']} take(s), "
        f"{data['num_frames']} frames ({detail})"
    )
    for take in data["takes"]:
        boundary = take["boundary_step"]
        boundary_str = "-" if boundary is None else f"{boundary:.3f}"
        print(
            f"    take {take['index']:2d}: {take['count']:4d} frames  "
            f"{take['start_frame']} .. {take['end_frame']}  "
            f"(max internal step={take['max_internal_step']:.3f}, "
            f"jump to next={boundary_str})"
        )


def _resolve_scene_dir(scene):
    path = Path(scene).expanduser()
    if path.is_dir():
        return path
    candidate = DATA_ROOT / scene
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Scene {scene!r} not found (looked at {path} and {candidate})"
    )


def _scene_has_colmap(scene_dir):
    return (Path(scene_dir) / "sparse" / "0").exists()


def add_arguments(parser):
    parser.add_argument(
        "scene",
        nargs="?",
        help="Scene name under DATA/ (or a path). Omit with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every scene under DATA/ that has a COLMAP sparse model.",
    )
    parser.add_argument(
        "--jump-factor",
        type=float,
        default=DEFAULT_JUMP_FACTOR,
        help=(
            "A step is a take boundary when it exceeds this multiple of the "
            f"median step (default {DEFAULT_JUMP_FACTOR})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if a valid takes.json already exists.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the detected takes without writing takes.json.",
    )


def run(args):
    if args.all:
        scene_dirs = sorted(
            entry for entry in DATA_ROOT.iterdir()
            if entry.is_dir() and _scene_has_colmap(entry)
        )
        if not scene_dirs:
            print(f"No scenes with a COLMAP sparse model under {DATA_ROOT}")
            return
    else:
        if not args.scene:
            raise SystemExit("takes: provide a scene name or use --all")
        scene_dirs = [_resolve_scene_dir(args.scene)]

    for scene_dir in scene_dirs:
        if not args.force and not args.show:
            existing = load_takes(scene_dir)
            if existing is not None:
                # Validate against the scene's actual frames before trusting it.
                from dataset import load_colmap
                frame_ids = sorted(
                    (frame_id_from_path(r[1]) for r in load_colmap(scene_dir)),
                    key=frame_number,
                )
                ok, reason = validate_takes(existing, frame_ids)
                if ok:
                    print(color(f"[takes] {scene_dir.name}: up to date", "cyan"))
                    print_takes_summary(existing)
                    continue
                print(color(
                    f"[takes] {scene_dir.name}: stale ({reason}); regenerating",
                    "yellow",
                ))

        data = compute_takes_for_scene(scene_dir, jump_factor=args.jump_factor)
        if args.show:
            print(color(f"[takes] {scene_dir.name} (not written):", "cyan"))
        else:
            path = save_takes(scene_dir, data)
            print(color(f"[takes] wrote {path}", "green"))
        print_takes_summary(data)
