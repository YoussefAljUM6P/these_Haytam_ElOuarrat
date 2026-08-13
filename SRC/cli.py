"""SERVIS unified CLI.

Single entry point for every experiment. Subcommands dispatch into
`runners/`; the bare invocation (no args) launches the interactive
questionary wizard, which builds a config and runs the chosen runner
in-process.

Examples:
    python cli.py                                  # interactive wizard
    python cli.py wizard                           # same, explicit
    python cli.py smoke [--scene kitchen]
    python cli.py servo-frames --config CONFIGS/x.json
    python cli.py trajectory --config CONFIGS/x.json [--resume]
    python cli.py matrix --dataset kitchen --iterations 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import config_schema
from wizard import build_config as wizard_build_config
from runners import compare as runner_compare
from runners import inspect as runner_inspect
from runners import matrix as runner_matrix
from runners import mesh_check as runner_mesh_check
from runners import plot as runner_plot
from runners import servo_frames as runner_servo_frames
from runners import smoke as runner_smoke
from runners import trajectory as runner_trajectory
from runners import video as runner_video
from scene_assets import has_nerfstudio_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"
RUNS_ROOT = PROJECT_ROOT / "RUNS"
CONFIG_ROOT = PROJECT_ROOT / "CONFIGS"


SUBCOMMANDS = {
    "smoke": {
        "runner": runner_smoke,
        "help": "Mesh + GS smoke render/servo test.",
    },
    "servo-frames": {
        "runner": runner_servo_frames,
        "help": "Single frame-to-frame servo experiment.",
    },
    "trajectory": {
        "runner": runner_trajectory,
        "help": "Chained mini servos along a GT trajectory + evo eval.",
    },
    "matrix": {
        "runner": runner_matrix,
        "help": "Servo matrix sweep (scene x depth x matcher).",
    },
    "compare": {
        "runner": runner_compare,
        "help": "Variant x dataset benchmark sweep (renderer/depth/feature axes).",
    },
    "inspect": {
        "runner": runner_inspect,
        "help": "Render pose at frame N and compare to real image (optional features).",
    },
    "plot": {
        "runner": runner_plot,
        "help": "Regenerate detailed evo APE/RPE plots from a previous run's trajectories.",
    },
    "video": {
        "runner": runner_video,
        "help": "Compile a run's saved visualizations into one MP4 at a chosen FPS.",
    },
    "mesh-check": {
        "runner": runner_mesh_check,
        "help": "Check mesh.ply visibility in COLMAP camera frames.",
    },
}


# ---- subcommand wiring -----------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    for name, info in SUBCOMMANDS.items():
        sp = sub.add_parser(name, help=info["help"])
        info["runner"].add_arguments(sp)

    sub.add_parser("wizard", help="Interactive questionary wizard (default).")
    sub.add_parser("bot", help="Telegram phone-control bot (remote launcher).")

    return parser


def dispatch(args):
    info = SUBCOMMANDS[args.command]
    info["runner"].run(args)


# ---- interactive wizard ----------------------------------------------------


TASKS = {
    "resume_trajectory": {
        "kind": "trajectory",
        "command": "trajectory",
        "label": "Resume     — continue a trajectory run",
    },
    "compare_table": {
        "kind": "trajectory",
        "command": "trajectory",
        "label": "Compare    — trajectories on all datasets + Markdown table",
    },
    "trajectory": {
        "kind": "trajectory",
        "command": "trajectory",
        "label": "Trajectory  — chained mini servos along a GT trajectory",
    },
    "servo_frames": {
        "kind": "servo_frames",
        "command": "servo-frames",
        "label": "Servo frames — single frame-to-frame servo",
    },
    "inspect": {
        "kind": "inspect",
        "command": "inspect",
        "label": "Inspect     — render pose at frame N vs real image",
    },
    "plot_run": {
        "kind": "plot",
        "command": "plot",
        "label": "Plots      — detailed evo plots of a previous run",
    },
    "video_run": {
        "kind": "video",
        "command": "video",
        "label": "Video      — compile a run's visualizations into one MP4 (variable FPS)",
    },
    "filter_runs": {
        "kind": "filter",
        "command": "filter",
        "label": "Filter     — browse runs by recon type / controller / scene",
    },
}

CONTROLLERS = {
    "ibvs": "FBVS (feature-based / IBVS)",
    "photometric": "PVS  (photometric)",
}



def detect_renderers(scene_dir):
    available = []
    if (scene_dir / "mesh.ply").exists():
        available.append("mesh")
    if (scene_dir / "gs.ply").exists():
        available.append("gs")
    if has_nerfstudio_checkpoint(scene_dir):
        available.append("nerf")
    return available


def scene_pose_count(scene_dir):
    """Number of registered camera poses in the COLMAP sparse model.

    This is the source of trajectory tasks (consecutive pose pairs). Prefers the
    pose count cached in takes.json (num_frames); falls back to reading sparse/0
    directly. Returns None when the scene has no sparse model.
    """
    takes = scene_dir / "takes.json"
    if takes.is_file():
        try:
            n = json.loads(takes.read_text()).get("num_frames")
            if isinstance(n, int) and n > 0:
                return n
        except (OSError, ValueError):
            pass
    if (scene_dir / "sparse" / "0").is_dir():
        try:
            import pycolmap

            return len(pycolmap.Reconstruction(str(scene_dir / "sparse" / "0")).images)
        except Exception:
            return None
    return None


def scene_task_count(scene_dir):
    """Trajectory tasks for a scene = consecutive pose pairs = poses - 1."""
    poses = scene_pose_count(scene_dir)
    if poses is None or poses < 2:
        return None
    return poses - 1


def scene_takes(scene_dir):
    """Return the scene's takes as ``[{index, count, start_frame, end_frame}]``.

    Prefers a cached ``takes.json``; if absent, detects takes from the COLMAP
    poses and writes it (so the trajectory runner reuses the same segmentation).
    ``index`` is 1-based to match how takes are shown and selected. Returns an
    empty list when the scene has no COLMAP model or detection fails.
    """
    scene_dir = Path(scene_dir)
    from takes import load_takes, compute_takes_for_scene, save_takes

    data = load_takes(scene_dir)
    if data is None:
        if not (scene_dir / "sparse" / "0").is_dir():
            return []
        try:
            data = compute_takes_for_scene(scene_dir)
            save_takes(scene_dir, data)
        except Exception:
            return []

    out = []
    for k, take in enumerate(data.get("takes", []), start=1):
        count = int(take.get("count", len(take.get("frames", []))))
        out.append({
            "index": k,
            "count": count,
            "start_frame": take.get("start_frame"),
            "end_frame": take.get("end_frame"),
        })
    return out


def list_scenes():
    if not DATA_ROOT.is_dir():
        return []
    scenes = []
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        scenes.append((entry.name, detect_renderers(entry)))
    return scenes


def renderable_scenes(scenes):
    return [(name, renderers) for name, renderers in scenes if renderers]


def ordered_renderers(scene_items, common=False):
    if not scene_items:
        return []
    if common:
        available = set(scene_items[0][1])
        for _, renderers in scene_items[1:]:
            available &= set(renderers)
    else:
        available = set()
        for _, renderers in scene_items:
            available.update(renderers)
    return [r for r in ("mesh", "gs", "nerf") if r in available]


def write_config(cfg, task_key, scene_name):
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    controller = cfg.get("controller", "ibvs")
    name = f"cli_{task_key}_{scene_name}_{controller}_{stamp}.json"
    path = CONFIG_ROOT / name
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def wizard():
    """Lazy-import questionary/rich; only the wizard pulls them in."""
    import questionary
    from questionary import Choice, Style
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    console = Console()
    copied_cfg = {}

    QSTYLE = Style(
        [
            ("qmark", "fg:#00d7ff bold"),
            ("question", "bold"),
            ("answer", "fg:#5fd75f bold"),
            ("pointer", "fg:#00d7ff bold"),
            ("highlighted", "fg:#00d7ff bold"),
            ("selected", "fg:#5fd75f"),
            ("instruction", "fg:#808080 italic"),
        ]
    )

    class GoBack(Exception):
        pass

    def ask_select(message, choices, default=None):
        result = questionary.select(
            message, choices=choices, style=QSTYLE, default=default, qmark="›"
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    def ask_select_with_back(message, choices, default=None):
        choices_with_back = list(choices) + [Choice("« Back", value="__back__")]
        res = ask_select(message, choices_with_back, default)
        if res == "__back__":
            raise GoBack()
        return res

    def ask_confirm(message, default=True):
        result = questionary.confirm(
            message, default=default, style=QSTYLE, qmark="›"
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    def _validate_factory(caster, optional=False):
        def _v(text):
            text = text.strip()
            if not text:
                return True
            if optional and text.lower() in {"none", "null"}:
                return True
            try:
                caster(text)
                return True
            except Exception as exc:
                return str(exc)
        return _v

    def ask_text(message, default, caster=str, optional=False):
        default_str = "" if default is None else str(default)
        result = questionary.text(
            message,
            default=default_str,
            style=QSTYLE,
            qmark="›",
            validate=_validate_factory(caster, optional=optional),
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        result = result.strip()
        if optional and (not result or result.lower() in {"none", "null"}):
            return None
        if not result:
            return default
        return caster(result)

    def banner():
        text = Text()
        text.append("  SERVIS  ", style="bold cyan on grey15")
        text.append("  Interactive Experiment Launcher", style="bold white")
        console.print()
        console.print(Panel(text, border_style="cyan", padding=(0, 2)))

    def scene_table(scenes):
        psnr_data = {}
        # Load custom PSNRs if any
        try:
            with open(DATA_ROOT / "psnr.json") as f:
                psnr_data = json.load(f)
        except Exception:
            pass

        def _format_cell(scene_name, renderer, rs):
            if renderer not in rs:
                return "[grey50]·[/]"
            val = psnr_data.get(scene_name, {}).get(renderer)
            if val is not None:
                return f"[green]{val}[/]"
            return "[green]✓[/]"

        table = Table(
            title="Detected scenes in DATA/",
            title_style="bold cyan",
            border_style="grey50",
            header_style="bold",
        )
        table.add_column("scene", style="green")
        table.add_column("tasks", justify="right", style="cyan")
        table.add_column("mesh", justify="center")
        table.add_column("gs", justify="center")
        table.add_column("nerf", justify="center")
        for name, rs in scenes:
            n_tasks = scene_task_count(DATA_ROOT / name)
            table.add_row(
                name,
                str(n_tasks) if n_tasks is not None else "[grey50]·[/]",
                _format_cell(name, "mesh", rs),
                _format_cell(name, "gs", rs),
                _format_cell(name, "nerf", rs),
            )
        console.print(table)

    def read_json_file(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def read_resume_scene(run_dir, resolved_config, summary):
        if isinstance(summary, dict) and len(summary) == 1:
            scene_key, scene_summary = next(iter(summary.items()))
            if isinstance(scene_summary, dict) and scene_summary.get("scene"):
                return str(scene_summary["scene"])
            return str(scene_key)

        datasets = resolved_config.get("datasets")
        if isinstance(datasets, list) and len(datasets) == 1:
            return str(datasets[0])
        if isinstance(datasets, str) and datasets.strip():
            return datasets.split(",")[0].strip()

        return run_dir.name.split("-")[0].lower()

    def read_resume_progress(run_dir):
        csv_path = run_dir / "per_task_errors.csv"
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error):
            rows = []

        if not rows:
            return {
                "tasks_done": 0,
                "last_task": None,
                "last_target": None,
                "last_failure": None,
            }

        last = rows[-1]
        failure = last.get("failure_type") or None
        return {
            "tasks_done": len(rows),
            "last_task": last.get("task_index"),
            "last_target": last.get("target_frame"),
            "last_failure": failure,
        }

    def build_resume_info(run_dir):
        resolved = read_json_file(run_dir / "config.resolved.json") or {}
        summary = read_json_file(run_dir / "summary.json") or {}
        scene = read_resume_scene(run_dir, resolved, summary)
        progress = read_resume_progress(run_dir)

        csv_path = run_dir / "per_task_errors.csv"
        mtime = max(
            p.stat().st_mtime
            for p in (run_dir, csv_path, run_dir / "summary.json")
            if p.exists()
        )

        return {
            "path": run_dir,
            "scene": scene,
            "resolved": resolved,
            "summary": summary,
            "progress": progress,
            "mtime": mtime,
        }

    def has_resume_config(info):
        cfg = info["resolved"]
        required = (
            "renderer",
            "controller",
            "depth_mode",
            "stride",
            "mini_iterations",
            "start_index",
        )
        if not all(key in cfg for key in required):
            return False
        if cfg.get("controller") == "ibvs" and "feature_method" not in cfg:
            return False
        return True

    _MANUAL_RUN = "__manual_run_path__"
    # The picker discovers a bounded pool of the most-recent runs and shows up to
    # select_run's display_limit of them; anything older is reached by typing a
    # path via the manual-path entry.
    _RUN_POOL_LIMIT = 200

    def run_choice_label(info):
        when = datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M")
        return f"{when}  {info['path'].relative_to(PROJECT_ROOT)}  [{info['scene']}]"

    def _resolve_run_path(raw):
        """Resolve a user-typed run path to an existing directory.

        Accepts an absolute path, a ``~`` path, or a path relative to either
        the project root or ``RUNS/`` (so both ``RUNS/FOO`` and ``FOO`` work).
        Returns the resolved directory, or None if nothing matches.
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        candidates = [p] if p.is_absolute() else [p, PROJECT_ROOT / p, RUNS_ROOT / p]
        for cand in candidates:
            if cand.is_dir():
                return cand.resolve()
        return None

    def select_run(message, runs, choice_label, requires=(), validate=None,
                   display_limit=15):
        """Run picker with a manual-path escape hatch.

        Shows up to ``display_limit`` runs (most recent first), a hint when more
        exist, and a "📁 Enter a run path manually…" entry. Picking that prompts
        for a path (absolute / ~ / relative to RUNS/), checks it contains every
        file in ``requires`` and passes the optional
        ``validate(info) -> error_or_None`` gate, then returns a fresh
        ``build_resume_info`` dict — same shape the discover_* functions yield.
        A blank path goes back. Listed runs are returned as-is. Raises GoBack
        on « Back.
        """
        choices = [
            Choice(choice_label(info), value=info)
            for info in runs[:display_limit]
        ]
        hidden = len(runs) - display_limit
        if hidden > 0:
            choices.append(
                Choice(
                    f"   … {hidden} older run(s) not shown — use the manual path",
                    value=_MANUAL_RUN,
                )
            )
        choices.append(Choice("📁 Enter a run path manually…", value=_MANUAL_RUN))

        selected = ask_select_with_back(
            message, choices, default=(runs[0] if runs else None)
        )

        if selected is not _MANUAL_RUN:
            return selected

        while True:
            raw = ask_text(
                "Run path (absolute, ~, or relative to RUNS/; blank = back):",
                None,
                str,
                optional=True,
            )
            if raw is None:
                raise GoBack()
            run_dir = _resolve_run_path(raw)
            if run_dir is None:
                console.print(f"[red]No such directory:[/] {raw}")
                continue
            missing = [name for name in requires if not (run_dir / name).exists()]
            if missing:
                console.print(
                    f"[red]{run_dir} is missing required file(s):[/] "
                    f"{', '.join(missing)}"
                )
                continue
            info = build_resume_info(run_dir)
            if validate is not None:
                err = validate(info)
                if err:
                    console.print(f"[red]{err}[/]")
                    continue
            return info

    def discover_resume_runs(limit=15):
        runs_root = PROJECT_ROOT / "RUNS"
        if not runs_root.is_dir():
            return []

        runs = []
        for entry in runs_root.iterdir():
            if not entry.is_dir():
                continue
            required = (
                entry / "config.resolved.json",
                entry / "per_task_errors.csv",
                entry / "sim_traj.tum",
                entry / "gt_traj.tum",
            )
            if not all(path.exists() for path in required):
                continue
            info = build_resume_info(entry)
            if not has_resume_config(info):
                continue
            runs.append(info)

        runs.sort(key=lambda item: item["mtime"], reverse=True)
        return runs[:limit]

    def resume_label(info):
        cfg = info["resolved"]
        progress = info["progress"]
        modified = datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M")
        controller = cfg.get("controller", "?")
        renderer = cfg.get("renderer", "?")
        depth = cfg.get("depth_mode", "?")
        feature = cfg.get("feature_method")
        method = f"{renderer}/{controller}/{depth}"
        if controller == "ibvs" and feature:
            method += f"/{feature}"

        status = f"{progress['tasks_done']} tasks"
        if progress["last_task"] is not None:
            status += f", next {int(progress['last_task']) + 1}"
        if progress["last_failure"]:
            status += f", last failed: {progress['last_failure']}"

        rel = info["path"].relative_to(PROJECT_ROOT)
        return f"{modified}  {rel}  [{info['scene']} {method}]  {status}"

    def resume_config_for_runner(info):
        from experiment_config import TRAJECTORY_CONFIG_KEYS

        cfg = {
            key: value
            for key, value in info["resolved"].items()
            if key in TRAJECTORY_CONFIG_KEYS
        }
        cfg["datasets"] = [info["scene"]]
        cfg.setdefault("controller", "ibvs")
        cfg.setdefault("renderer", "gs")
        cfg.setdefault("depth_mode", "intrinsic")
        return cfg

    def config_to_set_args(cfg):
        return [f"{key}={json.dumps(value)}" for key, value in sorted(cfg.items())]

    def run_resume_wizard():
        runs = discover_resume_runs(limit=_RUN_POOL_LIMIT)
        if not runs:
            console.print(
                f"[yellow]No resumable trajectory runs auto-detected under "
                f"{PROJECT_ROOT / 'RUNS'}; you can still enter a run path manually.[/]"
            )

        if runs:
            table = Table(
                title="Last 15 Resumable Trajectory Runs",
                title_style="bold cyan",
                border_style="grey50",
                header_style="bold",
            )
            table.add_column("#", justify="right", style="cyan")
            table.add_column("modified", style="green")
            table.add_column("run")
            table.add_column("scene")
            table.add_column("method")
            table.add_column("progress", justify="right")
            for idx, info in enumerate(runs, start=1):
                cfg = info["resolved"]
                progress = info["progress"]
                method = "/".join(
                    str(part)
                    for part in (
                        cfg.get("renderer", "?"),
                        cfg.get("controller", "?"),
                        cfg.get("depth_mode", "?"),
                        cfg.get("feature_method") if cfg.get("controller") == "ibvs" else None,
                    )
                    if part is not None
                )
                progress_text = str(progress["tasks_done"])
                if progress["last_failure"]:
                    progress_text += f" ({progress['last_failure']})"
                table.add_row(
                    str(idx),
                    datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M"),
                    str(info["path"].relative_to(PROJECT_ROOT)),
                    info["scene"],
                    method,
                    progress_text,
                )
            console.print(table)

        try:
            selected = select_run(
                "Resume run:",
                runs,
                resume_label,
                requires=(
                    "config.resolved.json",
                    "per_task_errors.csv",
                    "sim_traj.tum",
                    "gt_traj.tum",
                ),
                validate=lambda info: (
                    None
                    if has_resume_config(info)
                    else "That run has no usable trajectory config (missing keys)."
                ),
            )
        except GoBack:
            return "__back__"

        cfg = resume_config_for_runner(selected)

        console.print()
        console.print(
            Panel(
                Syntax(json.dumps(cfg, indent=2), "json", theme="ansi_dark"),
                title="[bold]Resume config[/]",
                border_style="cyan",
            )
        )
        if not ask_confirm("Resume this run now?", default=True):
            console.print("[yellow]resume cancelled[/]")
            return 0

        console.rule("[bold green]Resuming trajectory[/]")
        runner_args = SimpleNamespace(
            command="trajectory",
            config=None,
            set=config_to_set_args(cfg),
            resume=str(selected["path"]),
            diverge_mse_per_px=None,
            diverge_residual_px=None,
            continue_on_task_failure=None,
        )
        runner_trajectory.run(runner_args)
        return 0

    # --- declarative question helpers (questionary.prompt with when=) ------

    def _text_validator(caster, optional):
        def _v(text):
            t = text.strip()
            if not t:
                return True
            if optional and t.lower() in {"none", "null"}:
                return True
            try:
                caster(t)
                return True
            except Exception as exc:
                return str(exc)
        return _v

    def _text_filter(default, caster, optional):
        def _f(text):
            t = text.strip() if isinstance(text, str) else text
            if optional and (not t or (isinstance(t, str) and t.lower() in {"none", "null"})):
                return None
            if not t:
                return default
            return caster(t)
        return _f

    def q_select(name, message, choices, default=None, when=None):
        actual_val = copied_cfg.get(name, default)
        matched_choice = None
        for c in choices:
            if getattr(c, "value", c) == actual_val:
                matched_choice = c
                break
        q = {
            "type": "select",
            "name": name,
            "message": message,
            "choices": choices,
            "qmark": "›",
            "style": QSTYLE,
        }
        if matched_choice is not None:
            q["default"] = matched_choice
        elif default is not None:
            q["default"] = default
        if when is not None:
            q["when"] = when
        return q

    def q_text(name, message, default, caster=str, optional=False, when=None):
        actual_val = copied_cfg.get(name, default)
        q = {
            "type": "text",
            "name": name,
            "message": message,
            "default": "" if actual_val is None else str(actual_val),
            "validate": _text_validator(caster, optional),
            "filter": _text_filter(actual_val, caster, optional),
            "qmark": "›",
            "style": QSTYLE,
        }
        if when is not None:
            q["when"] = when
        return q

    def q_confirm(name, message, default=True, when=None):
        actual_val = copied_cfg.get(name, default)
        q = {
            "type": "confirm",
            "name": name,
            "message": message,
            "default": bool(actual_val) if actual_val is not None else default,
            "qmark": "›",
            "style": QSTYLE,
        }
        if when is not None:
            q["when"] = when
        return q

    def run_prompt(questions):
        if not questions:
            return {}

        answers = {}
        ALWAYS_ASK = {"renderer"}

        for q in questions:
            if "when" in q and callable(q["when"]):
                if not q["when"](answers):
                    continue

            name = q["name"]

            if fast_forward and copied_cfg and name not in ALWAYS_ASK and name in copied_cfg:
                answers[name] = copied_cfg[name]
                continue

            q_copy = dict(q)
            q_copy.pop("when", None)

            res = questionary.prompt([q_copy])
            if not res:
                raise KeyboardInterrupt
            answers[name] = res[name]

        return answers

    # --- per-controller capability ------------------------------------------

    # Experiment-config prompts are generated from config_schema (single source
    # of truth) by wizard.build_config, which reuses these prompt helpers so the
    # copy-settings / fast-forward behaviour is preserved. See SRC/wizard.py.
    wizard_ctx = SimpleNamespace(
        q_text=q_text,
        q_select=q_select,
        q_confirm=q_confirm,
        run_prompt=run_prompt,
        console=console,
        Choice=Choice,
    )

    def run_inspect_wizard(scene_name, scene_renderers):
        try:
            if len(scene_renderers) == 1:
                renderer = scene_renderers[0]
                console.print(f"[grey50]renderer auto-picked:[/] [bold]{renderer}[/]")
            else:
                renderer = ask_select_with_back(
                    "Renderer:",
                    [Choice(r, value=r) for r in scene_renderers],
                    default=scene_renderers[0],
                )

            index = ask_text("Frame index (1-based):", 1, int)

            features = ask_select(
                "Feature overlay:",
                [
                    Choice("none   — just side-by-side render vs real", value="none"),
                    Choice("sift   — overlay SIFT matches (RANSAC filtered)", value="sift"),
                    Choice("xfeat  — overlay XFeat matches (RANSAC filtered)", value="xfeat"),
                ],
                default="none",
            )

            output = ask_text(
                "Output PNG path (blank = auto under RUNS/inspect/):",
                None,
                str,
                optional=True,
            )

            console.rule("[bold green]Launching inspect[/]")
            runner_args = SimpleNamespace(
                command="inspect",
                scene=scene_name,
                index=int(index),
                renderer=renderer,
                features=features,
                output=output,
            )
            runner_inspect.run(runner_args)
            return 0

        except GoBack:
            return "__back__"

    def discover_plot_runs(limit=15):
        """Any run dir that has both TUM trajectory files can be re-plotted."""
        runs_root = PROJECT_ROOT / "RUNS"
        if not runs_root.is_dir():
            return []
        runs = []
        for entry in runs_root.iterdir():
            if not entry.is_dir():
                continue
            if not (entry / "sim_traj.tum").exists() or not (entry / "gt_traj.tum").exists():
                continue
            runs.append(build_resume_info(entry))
        runs.sort(key=lambda item: item["mtime"], reverse=True)
        return runs[:limit]

    def run_plot_wizard():
        runs = discover_plot_runs(limit=_RUN_POOL_LIMIT)
        if not runs:
            console.print(
                f"[yellow]No runs with sim_traj.tum + gt_traj.tum auto-detected under "
                f"{PROJECT_ROOT / 'RUNS'}; you can still enter a run path manually.[/]"
            )

        if runs:
            table = Table(
                title="Last 15 Runs Available for Plotting",
                title_style="bold cyan",
                border_style="grey50",
                header_style="bold",
            )
            table.add_column("#", justify="right", style="cyan")
            table.add_column("modified", style="green")
            table.add_column("run")
            table.add_column("scene")
            for idx, info in enumerate(runs, start=1):
                table.add_row(
                    str(idx),
                    datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M"),
                    str(info["path"].relative_to(PROJECT_ROOT)),
                    info["scene"],
                )
            console.print(table)

        try:
            selected = select_run(
                "Plot run:",
                runs,
                run_choice_label,
                requires=("sim_traj.tum", "gt_traj.tum"),
            )

            # Multi-take runs are always saved in full: one combined trajectory
            # plus one independent plot set per take (evo_plots/take_<k>/). No
            # take selection — the runner emits them all.
            windows = runner_plot.read_take_windows(selected["path"])
            if len(windows) > 1:
                console.print(
                    f"[grey50]This run has {len(windows)} takes — saving the "
                    f"combined trajectory plus all {len(windows)} takes "
                    f"independently.[/]"
                )

            rpe_delta = ask_text("RPE delta (frames):", 1, int)
            out = ask_text(
                "Output dir (blank = <run>/evo_plots):", None, str, optional=True
            )
        except GoBack:
            return "__back__"

        console.rule("[bold green]Generating detailed evo plots[/]")
        runner_args = SimpleNamespace(
            command="plot",
            run=str(selected["path"]),
            rpe_delta=int(rpe_delta),
            out=out,
            list_takes=False,
        )
        runner_plot.run(runner_args)
        return 0

    def run_has_visuals(run_dir):
        """A run is encodable if it has saved viz PNGs or rebuildable frames."""
        if list(run_dir.glob("visualizations/*.png")):
            return True
        if list(run_dir.glob("*/visualizations/*.png")) or list(
            run_dir.glob("scenes/*/visualizations/*.png")
        ):
            return True
        return (
            (run_dir / "summary.json").exists()
            and (run_dir / "sim_traj.tum").exists()
            and (run_dir / "per_task_errors.csv").exists()
        )

    def discover_video_runs(limit=15):
        runs_root = PROJECT_ROOT / "RUNS"
        if not runs_root.is_dir():
            return []
        runs = []
        for entry in runs_root.iterdir():
            if not entry.is_dir() or not run_has_visuals(entry):
                continue
            runs.append(build_resume_info(entry))
        runs.sort(key=lambda item: item["mtime"], reverse=True)
        return runs[:limit]

    def run_video_wizard():
        runs = discover_video_runs(limit=_RUN_POOL_LIMIT)
        if not runs:
            console.print(
                f"[yellow]No runs with saved visualizations auto-detected under "
                f"{PROJECT_ROOT / 'RUNS'}; you can still enter a run path manually.[/]"
            )

        if runs:
            table = Table(
                title="Last 15 Runs Available for Video",
                title_style="bold cyan",
                border_style="grey50",
                header_style="bold",
            )
            table.add_column("#", justify="right", style="cyan")
            table.add_column("modified", style="green")
            table.add_column("run")
            table.add_column("scene")
            for idx, info in enumerate(runs, start=1):
                table.add_row(
                    str(idx),
                    datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M"),
                    str(info["path"].relative_to(PROJECT_ROOT)),
                    info["scene"],
                )
            console.print(table)

        try:
            selected = select_run(
                "Video run:",
                runs,
                run_choice_label,
                validate=lambda info: (
                    None
                    if run_has_visuals(info["path"])
                    else "That run has no saved or rebuildable visualizations."
                ),
            )
            fps = ask_text("FPS (frames per second):", 6.0, float)
            scene = ask_text(
                "Scene subdirectory (blank = all):", None, str, optional=True
            )
            output = ask_text(
                "Output path (blank = <run>/trajectory_visualizations.mp4):",
                None,
                str,
                optional=True,
            )
            render_missing = ask_confirm(
                "Rebuild frames from trajectories if none are saved?", default=False
            )
        except GoBack:
            return "__back__"

        if float(fps) <= 0:
            console.print("[red]FPS must be > 0[/]")
            return 1

        console.rule("[bold green]Compiling visualizations into MP4[/]")
        runner_args = SimpleNamespace(
            command="video",
            run=str(selected["path"]),
            latest=False,
            scene=scene,
            fps=float(fps),
            output=output,
            codec="mp4v",
            pattern="visualizations/*.png",
            render_missing=render_missing,
            max_frames=None,
        )
        runner_video.run(runner_args)
        return 0

    def discover_all_runs(limit=500):
        """Every run dir under RUNS/ carrying a resolved config, newest first."""
        runs_root = PROJECT_ROOT / "RUNS"
        if not runs_root.is_dir():
            return []
        runs = []
        for entry in runs_root.iterdir():
            if not entry.is_dir():
                continue
            if not (entry / "config.resolved.json").exists():
                continue
            runs.append(build_resume_info(entry))
        runs.sort(key=lambda item: item["mtime"], reverse=True)
        return runs[:limit]

    def run_filter_wizard():
        """Browse runs filtered by recon type (renderer), controller and scene.

        Every filter is a menu choice (never free text); "— any —" is a
        wildcard that ANDs with the others. Matching runs are printed as a
        table. This is a read-only browser, so it always returns to the main
        menu (via "__back__").
        """
        runs = discover_all_runs()
        if not runs:
            console.print(
                f"[yellow]No runs with a resolved config found under "
                f"{PROJECT_ROOT / 'RUNS'}.[/]"
            )
            return "__back__"

        scenes = sorted({r["scene"] for r in runs if r.get("scene")})
        ANY = "__any__"

        def _choices(pairs):
            return [Choice("— any —", value=ANY)] + [
                Choice(label, value=value) for label, value in pairs
            ]

        while True:
            try:
                recon = ask_select_with_back(
                    "Reconstruction type (renderer):",
                    _choices([("mesh", "mesh"), ("gs", "gs"), ("nerf", "nerf")]),
                    default=ANY,
                )
                controller = ask_select_with_back(
                    "Controller:",
                    _choices([
                        ("ibvs  (FBVS / feature-based)", "ibvs"),
                        ("photometric  (PVS)", "photometric"),
                    ]),
                    default=ANY,
                )
                scene = ask_select_with_back(
                    "Scene:",
                    _choices([(s, s) for s in scenes]),
                    default=ANY,
                )
            except GoBack:
                return "__back__"

            def _keep(info, _recon=recon, _controller=controller, _scene=scene):
                cfg = info["resolved"]
                if _recon is not ANY and cfg.get("renderer") != _recon:
                    return False
                if _controller is not ANY and cfg.get("controller") != _controller:
                    return False
                if _scene is not ANY and info.get("scene") != _scene:
                    return False
                return True

            matches = [r for r in runs if _keep(r)]
            active = " ".join(
                f"{name}={val}"
                for name, val in (("recon", recon), ("controller", controller),
                                  ("scene", scene))
                if val is not ANY
            ) or "none"

            if not matches:
                console.print(
                    f"[yellow]No runs match filter[/] ([grey50]{active}[/]); "
                    f"{len(runs)} run(s) total."
                )
            else:
                table = Table(
                    title=f"Runs matching: {active}  ({len(matches)} of {len(runs)})",
                    title_style="bold cyan",
                    border_style="grey50",
                    header_style="bold",
                )
                table.add_column("#", justify="right", style="cyan")
                table.add_column("modified", style="green")
                table.add_column("run")
                table.add_column("scene")
                table.add_column("recon")
                table.add_column("controller")
                table.add_column("depth")
                table.add_column("tasks", justify="right")
                for idx, info in enumerate(matches, start=1):
                    cfg = info["resolved"]
                    table.add_row(
                        str(idx),
                        datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M"),
                        str(info["path"].relative_to(PROJECT_ROOT)),
                        info["scene"],
                        str(cfg.get("renderer", "?")),
                        str(cfg.get("controller", "?")),
                        str(cfg.get("depth_mode", "?")),
                        str(info["progress"]["tasks_done"]),
                    )
                console.print(table)

            nxt = ask_select(
                "Next:",
                [
                    Choice("Filter again", value="again"),
                    Choice("Back to main menu", value="menu"),
                ],
                default="menu",
            )
            if nxt == "menu":
                return "__back__"

    def discover_copy_runs(controller_type, limit=15):
        runs = discover_resume_runs(limit=100)
        filtered = [r for r in runs if r["resolved"].get("controller") == controller_type]
        return filtered[:limit]

    def ask_checkbox(message, choices):
        result = questionary.checkbox(
            message, choices=choices, style=QSTYLE, qmark="›"
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

    def run_takes_wizard(scene_name, default_indices=None):
        """Choose which capture takes to run for a trajectory.

        Returns ``(selection, interactive)`` where ``selection`` is None to run
        the whole trajectory (all takes, back-to-back) or a 1-based list of take
        indices, and ``interactive`` says whether the user was actually prompted.
        Raises GoBack for the « Back choice. When the scene has 0/1 takes there
        is nothing to choose, so returns ``(None, False)``.
        """
        takes = scene_takes(DATA_ROOT / scene_name)
        if len(takes) <= 1:
            if len(takes) == 1:
                console.print(
                    "[grey50]Scene has a single take — running the whole "
                    "trajectory.[/]"
                )
            return None, False

        total_tasks = sum(max(0, t["count"] - 1) for t in takes)
        console.print(
            f"[grey50]{len(takes)} takes detected "
            f"(~{total_tasks} tasks total @ stride 1).[/]"
        )
        mode = ask_select_with_back(
            "Which takes?",
            [
                Choice(
                    f"Whole trajectory — all {len(takes)} takes "
                    f"(~{total_tasks} tasks)",
                    value="whole",
                ),
                Choice("Pick specific takes…", value="pick"),
            ],
            default="whole",
        )
        if mode == "whole":
            return None, True

        default_set = set(default_indices or [])
        take_choices = [
            Choice(
                f"take {t['index']}  ·  ~{max(0, t['count'] - 1)} tasks  "
                f"({t['count']} frames)  "
                f"[{t['start_frame']} → {t['end_frame']}]",
                value=t["index"],
                checked=(t["index"] in default_set),
            )
            for t in takes
        ]
        while True:
            picked = ask_checkbox(
                "Select takes to run (space to toggle, enter to confirm):",
                take_choices,
            )
            if picked:
                return sorted(picked), True
            console.print("[yellow]Select at least one take.[/]")

    banner()
    state = "TASK"
    task_key = "trajectory"
    scene_name = None
    controller = "ibvs"
    cfg = {}
    fast_forward = False
    take_indices_sel = None
    takes_selectable = False

    while True:
        try:
            if state == "TASK":
                task_key = ask_select(
                    "Task type:",
                    [Choice(v["label"], value=k) for k, v in TASKS.items()] + [Choice("Exit", value="__exit__")],
                    default=task_key,
                )
                if task_key == "__exit__":
                    return 0
                if task_key == "resume_trajectory":
                    state = "RESUME"
                elif task_key == "inspect":
                    state = "INSPECT_SCENE"
                elif task_key == "plot_run":
                    state = "PLOT_RUN"
                elif task_key == "video_run":
                    state = "VIDEO_RUN"
                elif task_key == "filter_runs":
                    state = "FILTER_RUNS"
                else:
                    state = "CONTROLLER"

            elif state == "RESUME":
                res = run_resume_wizard()
                if res == "__back__":
                    state = "TASK"
                else:
                    return res

            elif state == "PLOT_RUN":
                res = run_plot_wizard()
                if res == "__back__":
                    state = "TASK"
                else:
                    return res

            elif state == "VIDEO_RUN":
                res = run_video_wizard()
                if res == "__back__":
                    state = "TASK"
                else:
                    return res

            elif state == "FILTER_RUNS":
                res = run_filter_wizard()
                if res == "__back__":
                    state = "TASK"
                else:
                    return res

            elif state == "INSPECT_SCENE":
                scenes = list_scenes()
                if not scenes:
                    console.print(f"[red]No scene directories found under {DATA_ROOT}[/]")
                    return 1
                scene_table(scenes)
                console.print()
                scene_name = ask_select_with_back(
                    "Scene:",
                    [Choice(f"{name}   [{', '.join(rs) if rs else 'no renderable assets'}]", value=name) for name, rs in scenes],
                    default=scene_name or scenes[0][0],
                )
                scene_renderers = dict(scenes)[scene_name]
                if not scene_renderers:
                    console.print(f"[red]Scene {scene_name!r} has no renderable assets.[/]")
                    continue
                state = "INSPECT_OPTIONS"

            elif state == "INSPECT_OPTIONS":
                res = run_inspect_wizard(scene_name, scene_renderers)
                if res == "__back__":
                    state = "INSPECT_SCENE"
                else:
                    return res

            elif state == "CONTROLLER":
                copied_cfg.clear()
                controller = ask_select_with_back(
                    "Controller:",
                    [Choice(v, value=k) for k, v in CONTROLLERS.items()],
                    default=controller,
                )
                state = "COPY_SETTINGS"

            elif state == "COPY_SETTINGS":
                runs = discover_copy_runs(controller, limit=15)
                choices = [Choice("None - start fresh", value={})]
                for r in runs:
                    choices.append(Choice(resume_label(r), value=r["resolved"]))
                choices.append(
                    Choice("📁 Enter a run path manually…", value=_MANUAL_RUN)
                )

                selected_copy = ask_select_with_back(
                    "Copy settings from previous run?",
                    choices,
                    default=choices[0]
                )
                if selected_copy is _MANUAL_RUN:
                    selected_copy = {}
                    while True:
                        raw = ask_text(
                            "Run path to copy settings from "
                            "(absolute, ~, or relative to RUNS/; blank = start fresh):",
                            None,
                            str,
                            optional=True,
                        )
                        if raw is None:
                            break
                        run_dir = _resolve_run_path(raw)
                        if run_dir is None:
                            console.print(f"[red]No such directory:[/] {raw}")
                            continue
                        resolved = build_resume_info(run_dir)["resolved"]
                        if not resolved:
                            console.print(
                                f"[red]{run_dir} has no config.resolved.json to copy.[/]"
                            )
                            continue
                        other = resolved.get("controller")
                        if other and other != controller:
                            console.print(
                                f"[yellow]Note: that run's controller is {other!r}, "
                                f"not {controller!r}; copying its settings anyway.[/]"
                            )
                        selected_copy = resolved
                        break
                copied_cfg.update(selected_copy)

                if copied_cfg:
                    action = ask_select_with_back(
                        "How to apply copied settings?",
                        [
                            Choice("Fast-forward (skip prompts for copied values)", value="fast"),
                            Choice("Review and edit all settings manually", value="review")
                        ],
                        default="fast"
                    )
                    fast_forward = (action == "fast")
                else:
                    fast_forward = False

                if task_key == "compare_table":
                    state = "QUESTIONS"
                else:
                    state = "SCENE"

            elif state == "SCENE":
                scenes = list_scenes()
                if not scenes:
                    console.print(f"[red]No scene directories found under {DATA_ROOT}[/]")
                    return 1
                scene_table(scenes)
                console.print()
                scene_name = ask_select_with_back(
                    "Scene:",
                    [Choice(f"{name}   [{', '.join(rs) if rs else 'no renderable assets'}]", value=name) for name, rs in scenes],
                    default=scene_name or scenes[0][0],
                )
                scene_renderers = dict(scenes)[scene_name]
                if not scene_renderers:
                    console.print(f"[red]Scene {scene_name!r} has no renderable assets.[/]")
                    continue
                # Only trajectory runs are take-partitioned; servo_frames isn't.
                state = "TAKES" if task_key == "trajectory" else "QUESTIONS"

            elif state == "TAKES":
                take_indices_sel, takes_selectable = run_takes_wizard(
                    scene_name, default_indices=take_indices_sel
                )
                state = "QUESTIONS"

            elif state == "QUESTIONS":
                if task_key == "compare_table":
                    comparison_scenes = renderable_scenes(list_scenes())
                    if not comparison_scenes:
                        console.print("[red]No renderable datasets found under DATA/[/]")
                        return 1
                    comparison_datasets = [name for name, _ in comparison_scenes]
                    scene_renderers = ordered_renderers(comparison_scenes, common=True)
                    if not scene_renderers:
                        scene_renderers = ordered_renderers(comparison_scenes, common=False)
                        console.print("[yellow]No renderer is available in every dataset.[/]")
                    console.print("[grey50]datasets:[/] " + ", ".join(comparison_datasets))
                    scene_name = "all_datasets"
                    cfg = wizard_build_config(
                        config_schema.TRAJECTORY,
                        controller,
                        {"datasets": comparison_datasets},
                        scene_renderers,
                        wizard_ctx,
                        run_tag_default="compare_all",
                    )
                else:
                    if task_key == "servo_frames":
                        cfg = wizard_build_config(
                            config_schema.SERVO_FRAMES,
                            controller,
                            {"scene_dir": scene_name},
                            scene_renderers,
                            wizard_ctx,
                        )
                    else:
                        cfg = wizard_build_config(
                            config_schema.TRAJECTORY,
                            controller,
                            {"datasets": [scene_name]},
                            scene_renderers,
                            wizard_ctx,
                        )
                        # A specific take selection implies take-partitioning;
                        # "whole trajectory" leaves the schema default (all takes).
                        if take_indices_sel:
                            cfg["use_takes"] = True
                            cfg["take_indices"] = list(take_indices_sel)

                console.print()
                console.print(
                    Panel(
                        Syntax(json.dumps(cfg, indent=2), "json", theme="ansi_dark"),
                        title="[bold]Generated config[/]",
                        border_style="cyan",
                    )
                )
                state = "ACTION"

            elif state == "ACTION":
                action = ask_select_with_back(
                    "Next:",
                    [
                        Choice("Write config + run now", value="run"),
                        Choice("Write config only (no run)", value="save"),
                        Choice("Abort (discard)", value="abort"),
                    ],
                    default="run",
                )
                if action == "abort":
                    console.print("[yellow]aborted[/]")
                    return 0

                config_path = write_config(cfg, task_key, scene_name)
                console.print(f"[green]✓[/] wrote [bold]{config_path.relative_to(PROJECT_ROOT)}[/]")

                if action == "save":
                    console.print("[grey50]save-only mode; not running[/]")
                    return 0

                command_name = TASKS[task_key]["command"]
                console.rule(f"[bold green]Launching {command_name}[/]")

                runner_args = SimpleNamespace(
                    command=command_name,
                    config=str(config_path),
                    set=[],
                    resume=None,
                )
                SUBCOMMANDS[command_name]["runner"].run(runner_args)
                return 0

        except GoBack:
            back_map = {
                "RESUME": "TASK",
                "PLOT_RUN": "TASK",
                "VIDEO_RUN": "TASK",
                "FILTER_RUNS": "TASK",
                "INSPECT_SCENE": "TASK",
                "INSPECT_OPTIONS": "INSPECT_SCENE",
                "CONTROLLER": "TASK",
                "COPY_SETTINGS": "CONTROLLER",
                "SCENE": "COPY_SETTINGS",
                "TAKES": "SCENE",
                "ACTION": "QUESTIONS",
            }
            if state == "QUESTIONS":
                if task_key == "compare_table":
                    back = "COPY_SETTINGS"
                elif task_key == "trajectory" and takes_selectable:
                    back = "TAKES"
                else:
                    back = "SCENE"
                state = back
            else:
                state = back_map.get(state, "TASK")


# ---- entry point -----------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "wizard"):
        try:
            return wizard() or 0
        except KeyboardInterrupt:
            print("\nabort (Ctrl-C)")
            return 130

    if args.command == "bot":
        # Lazy import: only the bot needs python-telegram-bot.
        import phone_bot
        phone_bot.run(args)
        return 0

    dispatch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
