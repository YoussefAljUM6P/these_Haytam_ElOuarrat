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
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from runners import inspect as runner_inspect
from runners import matrix as runner_matrix
from runners import servo_frames as runner_servo_frames
from runners import smoke as runner_smoke
from runners import trajectory as runner_trajectory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"
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
    "inspect": {
        "runner": runner_inspect,
        "help": "Render pose at frame N and compare to real image (optional features).",
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

    return parser


def dispatch(args):
    info = SUBCOMMANDS[args.command]
    info["runner"].run(args)


# ---- interactive wizard ----------------------------------------------------


TASKS = {
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
}

CONTROLLERS = {
    "ibvs": "FBVS (feature-based / IBVS)",
    "photometric": "PVS  (photometric, NumPy)",
    "photometric_torch": "PVS  (photometric, PyTorch / ViSP port)",
}

DEPTH_MODES = ["intrinsic", "learned"]
FEATURE_METHODS = ["sift", "xfeat"]


def detect_renderers(scene_dir):
    available = []
    if (scene_dir / "mesh.ply").exists():
        available.append("mesh")
    if (scene_dir / "gs.ply").exists():
        available.append("gs")
    if (
        (scene_dir / "nerf").is_dir()
        or list(scene_dir.glob("*-instant-ngp-tcnn"))
        or list(scene_dir.glob("step-*.ckpt"))
    ):
        available.append("nerf")
    return available


def list_scenes():
    if not DATA_ROOT.is_dir():
        return []
    scenes = []
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        scenes.append((entry.name, detect_renderers(entry)))
    return scenes


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

    def ask_select(message, choices, default=None):
        result = questionary.select(
            message, choices=choices, style=QSTYLE, default=default, qmark="›"
        ).ask()
        if result is None:
            raise KeyboardInterrupt
        return result

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
        table = Table(
            title="Detected scenes in DATA/",
            title_style="bold cyan",
            border_style="grey50",
            header_style="bold",
        )
        table.add_column("scene", style="green")
        table.add_column("mesh", justify="center")
        table.add_column("gs", justify="center")
        table.add_column("nerf", justify="center")
        for name, rs in scenes:
            table.add_row(
                name,
                "[green]✓[/]" if "mesh" in rs else "[grey50]·[/]",
                "[green]✓[/]" if "gs" in rs else "[grey50]·[/]",
                "[green]✓[/]" if "nerf" in rs else "[grey50]·[/]",
            )
        console.print(table)

    def build_common(controller, renderers):
        renderer = ask_select(
            "Renderer:",
            [Choice(r, value=r) for r in renderers],
            default=renderers[0],
        )

        cfg = {"renderer": renderer, "controller": controller}

        cfg["depth_mode"] = ask_select(
            "Depth mode:",
            [
                Choice("intrinsic  — scene.render_depth()", value="intrinsic"),
                Choice("learned    — MoGe2", value="learned"),
            ],
            default="intrinsic",
        )

        cfg["feature_method"] = ask_select(
            "Feature method:",
            [Choice(m, value=m) for m in FEATURE_METHODS],
            default="sift",
        )

        cfg["gain_ibvs"] = ask_text("Gain IBVS:", 0.75, float)

        if controller in ("photometric", "photometric_torch"):
            console.rule("[bold magenta]Photometric controller knobs[/]")
            cfg["gain_photo"] = ask_text("Gain photometric:", 0.005, float)
            cfg["sigma_blur"] = ask_text("Sigma blur:", 1.0, float)
            cfg["use_gzn"] = ask_confirm("Use GZN?", default=True)
            cfg["grad_percentile"] = ask_text("Grad percentile:", 50.0, float)
            cfg["photometric_max_pixels"] = ask_text(
                "Photometric max pixels:", 50000, int
            )
            cfg["use_huber"] = ask_confirm("Use Huber loss?", default=True)
            cfg["huber_k"] = ask_text(
                "Huber k (blank = auto):", None, float, optional=True
            )
        return cfg

    def build_servo_frames_config(controller, scene_name, renderers):
        cfg = {"kind": "servo_frames", "scene_dir": scene_name}
        cfg.update(build_common(controller, renderers))

        console.rule("[bold magenta]Frame selection[/]")
        cfg["start_index"] = ask_text("Start index:", 1, int)
        target = ask_text(
            "Target index (blank → use index_away):", None, int, optional=True
        )
        if target is None:
            cfg["index_away"] = ask_text("Index away:", 1, int)
            cfg["target_index"] = None
        else:
            cfg["target_index"] = target
            cfg["index_away"] = 1

        console.rule("[bold magenta]Servo loop[/]")
        cfg["iterations"] = ask_text("Iterations:", 100, int)
        cfg["dt"] = ask_text("dt:", 1.0, float)
        cfg["min_features"] = ask_text("Min features:", 3, int)
        cfg["ratio"] = ask_text("Match ratio (0 = match once):", 1, int)
        cfg["viz_iter"] = ask_text("Viz every N iters (0 disables):", 1, int)
        cfg["stop_residual_px"] = ask_text(
            "IBVS stop: RMS reprojection error (px):", 0.5, float
        )
        cfg["stop_mse_per_px"] = ask_text(
            "Photometric stop: mean(e^2) per pixel on [0,1]:", 2.0e-6, float
        )
        cfg["run_name"] = ask_text(
            "Run name (blank = auto):", None, str, optional=True
        )
        return cfg

    def build_trajectory_config(controller, scene_name, renderers):
        cfg = {"kind": "trajectory", "datasets": [scene_name]}
        cfg.update(build_common(controller, renderers))

        if cfg["renderer"] == "nerf":
            cfg["nerf_render_scale"] = ask_text("NeRF render scale:", 0.25, float)

        console.rule("[bold magenta]Trajectory pacing[/]")
        cfg["stride"] = ask_text("Stride between frames:", 1, int)
        cfg["mini_iterations"] = ask_text("Iterations per mini task:", 30, int)
        cfg["dt"] = ask_text("dt:", 1.0, float)
        cfg["min_features"] = ask_text("Min features:", 3, int)
        cfg["ratio"] = ask_text("Match ratio:", 1, int)
        cfg["start_index"] = ask_text("Start index:", 1, int)
        cfg["max_pairs"] = ask_text(
            "Max pairs (blank = all):", None, int, optional=True
        )
        cfg["rpe_delta"] = ask_text("RPE delta:", 1, int)

        console.rule("[bold magenta]Stopping + viz[/]")
        cfg["stop_residual_px"] = ask_text(
            "IBVS stop: RMS reprojection error (px):", 0.5, float
        )
        cfg["stop_mse_per_px"] = ask_text(
            "Photometric stop: mean(e^2) per pixel on [0,1]:", 2.0e-6, float
        )
        cfg["save_task_viz"] = ask_confirm("Save per-task viz?", default=True)
        cfg["task_viz_every"] = ask_text("Save viz every N tasks:", 1, int)
        cfg["run_tag"] = ask_text(
            "Run tag (blank = auto):", None, str, optional=True
        )
        return cfg

    def run_inspect_wizard(scene_name, scene_renderers):
        if len(scene_renderers) == 1:
            renderer = scene_renderers[0]
            console.print(f"[grey50]renderer auto-picked:[/] [bold]{renderer}[/]")
        else:
            renderer = ask_select(
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

    banner()
    scenes = list_scenes()
    if not scenes:
        console.print(f"[red]No scene directories found under {DATA_ROOT}[/]")
        return 1

    scene_table(scenes)
    console.print()

    task_key = ask_select(
        "Task type:",
        [Choice(v["label"], value=k) for k, v in TASKS.items()],
        default="trajectory",
    )

    if task_key == "inspect":
        scene_name = ask_select(
            "Scene:",
            [
                Choice(
                    f"{name}   [{', '.join(rs) if rs else 'no renderable assets'}]",
                    value=name,
                )
                for name, rs in scenes
            ],
            default=scenes[0][0],
        )
        scene_renderers = dict(scenes)[scene_name]
        if not scene_renderers:
            console.print(
                f"[red]Scene {scene_name!r} has no renderable assets "
                f"(expected mesh.ply, gs.ply, or nerf/).[/]"
            )
            return 1
        return run_inspect_wizard(scene_name, scene_renderers)

    controller = ask_select(
        "Controller:",
        [Choice(v, value=k) for k, v in CONTROLLERS.items()],
        default="ibvs",
    )

    scene_name = ask_select(
        "Scene:",
        [
            Choice(
                f"{name}   [{', '.join(rs) if rs else 'no renderable assets'}]",
                value=name,
            )
            for name, rs in scenes
        ],
        default=scenes[0][0],
    )
    scene_renderers = dict(scenes)[scene_name]
    if not scene_renderers:
        console.print(
            f"[red]Scene {scene_name!r} has no renderable assets "
            f"(expected mesh.ply, gs.ply, or nerf/).[/]"
        )
        return 1

    if task_key == "servo_frames":
        cfg = build_servo_frames_config(controller, scene_name, scene_renderers)
    else:
        cfg = build_trajectory_config(controller, scene_name, scene_renderers)

    console.print()
    console.print(
        Panel(
            Syntax(json.dumps(cfg, indent=2), "json", theme="ansi_dark"),
            title="[bold]Generated config[/]",
            border_style="cyan",
        )
    )

    action = ask_select(
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
    console.print(
        f"[green]✓[/] wrote [bold]{config_path.relative_to(PROJECT_ROOT)}[/]"
    )

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

    dispatch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
