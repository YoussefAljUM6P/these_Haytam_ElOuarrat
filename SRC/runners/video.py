"""Compile a run's saved visualizations into a single MP4 at a chosen FPS.

This never re-servoes. It gathers the per-task visualization PNGs a trajectory
run already wrote -- preferring the ``per_task_errors.csv`` manifest, falling
back to globbing ``visualizations/*.png`` -- and encodes them into one video.

FPS is free to choose: a low FPS makes a slow, readable convergence reel; a
high FPS makes a quick flythrough. The same frames can be re-encoded at any
FPS without re-running anything.

    python cli.py video --run RUNS/<dir> [--fps 8] [--scene kitchen] \
        [--output out.mp4] [--render-missing]
    python cli.py video --latest --fps 12

The heavy lifting (frame discovery, optional reconstruction, encoding) lives in
``trajectory_visuals_to_video``; this runner only adapts it to the unified CLI.
"""

from pathlib import Path

from runners.servo_frames import PROJECT_ROOT, RUNS_ROOT
from trajectory_visuals_to_video import (
    collect_frames,
    latest_trajectory_run,
    render_missing_visuals,
    resolve_existing_run_dir,
    write_video,
)


def add_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--run",
        help=(
            "Run directory containing saved visualizations. Resolved relative "
            "to RUNS/ if not found as given."
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest run under RUNS/trajectory/.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional scene subdirectory to compile (multi-scene runs).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Output frames per second (default 6).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default <run>/trajectory_visualizations.mp4).",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc codec. mp4v works for .mp4 on most installs.",
    )
    parser.add_argument(
        "--pattern",
        default="visualizations/*.png",
        help="Fallback glob under each scene dir when per_task_errors.csv is absent.",
    )
    parser.add_argument(
        "--render-missing",
        action="store_true",
        help=(
            "If no saved visualizations exist, rebuild final-vs-target frames "
            "from sim_traj.tum + per_task_errors.csv before encoding."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames to encode or reconstruct.",
    )


def resolve_run_dir(run_arg):
    """Accept a run dir or a RUNS-relative name."""
    candidate = Path(run_arg)
    if not candidate.exists():
        candidate = RUNS_ROOT / run_arg
    return resolve_existing_run_dir(candidate)


def run(args):
    if getattr(args, "latest", False) or getattr(args, "run", None) is None:
        run_dir = resolve_existing_run_dir(latest_trajectory_run())
    else:
        run_dir = resolve_run_dir(args.run)

    if float(args.fps) <= 0:
        raise ValueError("--fps must be > 0")

    max_frames = getattr(args, "max_frames", None)
    try:
        frame_paths = collect_frames(run_dir, scene=args.scene, pattern=args.pattern)
    except FileNotFoundError:
        if not getattr(args, "render_missing", False):
            raise
        frame_paths = render_missing_visuals(
            run_dir, scene=args.scene, max_frames=max_frames
        )
        if not frame_paths:
            raise FileNotFoundError(
                f"No saved or reconstructable visualization frames found in {run_dir}"
            )

    if max_frames is not None:
        frame_paths = frame_paths[: int(max_frames)]

    output_path = (
        Path(args.output) if args.output else run_dir / "trajectory_visualizations.mp4"
    )
    width, height = write_video(
        frame_paths,
        output_path=output_path,
        fps=args.fps,
        codec=args.codec,
    )

    try:
        rel = Path(output_path).relative_to(PROJECT_ROOT)
    except ValueError:
        rel = output_path
    print(
        f"Wrote {rel} from {len(frame_paths)} frames "
        f"at {args.fps:g} fps ({width}x{height})"
    )
