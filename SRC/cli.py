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
from runners import mesh_check as runner_mesh_check
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

    return parser


def dispatch(args):
    info = SUBCOMMANDS[args.command]
    info["runner"].run(args)


# ---- interactive wizard ----------------------------------------------------


TASKS = {
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
}

CONTROLLERS = {
    "ibvs": "FBVS (feature-based / IBVS)",
    "photometric": "PVS  (photometric, NumPy)",
    "photometric_torch": "PVS  (photometric, PyTorch / ViSP port)",
}

DEPTH_MODES = ["intrinsic"]
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
        q = {
            "type": "select",
            "name": name,
            "message": message,
            "choices": choices,
            "qmark": "›",
            "style": QSTYLE,
        }
        if default is not None:
            q["default"] = default
        if when is not None:
            q["when"] = when
        return q

    def q_text(name, message, default, caster=str, optional=False, when=None):
        q = {
            "type": "text",
            "name": name,
            "message": message,
            "default": "" if default is None else str(default),
            "validate": _text_validator(caster, optional),
            "filter": _text_filter(default, caster, optional),
            "qmark": "›",
            "style": QSTYLE,
        }
        if when is not None:
            q["when"] = when
        return q

    def q_confirm(name, message, default=True, when=None):
        q = {
            "type": "confirm",
            "name": name,
            "message": message,
            "default": default,
            "qmark": "›",
            "style": QSTYLE,
        }
        if when is not None:
            q["when"] = when
        return q

    def run_prompt(questions):
        if not questions:
            return {}
        ans = questionary.prompt(questions)
        if not ans and "when" not in questions[0]:
            raise KeyboardInterrupt
        return ans

    # --- per-controller capability ------------------------------------------

    def common_questions(controller, renderers):
        is_ibvs = controller == "ibvs"
        is_photo = controller in ("photometric", "photometric_torch")

        qs = [
            q_select(
                "renderer",
                "Renderer:",
                [Choice(r, value=r) for r in renderers],
                default=renderers[0],
            ),
            q_select(
                "depth_mode",
                "Depth mode:",
                [
                    Choice("intrinsic  — scene.render_depth()", value="intrinsic"),
                ],
                default="intrinsic",
            ),
        ]
        if is_ibvs:
            qs += [
                q_select(
                    "feature_method",
                    "Feature method:",
                    [Choice(m, value=m) for m in FEATURE_METHODS],
                    default="sift",
                ),
                q_text("gain_ibvs", "Gain IBVS:", 0.75, float),
            ]
        if is_photo:
            qs += [
                q_text("gain_photo", "Gain photometric:", 0.005, float),
                q_text("sigma_blur", "Sigma blur:", 1.0, float),
                q_confirm("use_gzn", "Use GZN?", default=True),
                q_text("grad_percentile", "Grad percentile:", 50.0, float),
                q_text("photometric_max_pixels", "Photometric max pixels:", 50000, int),
                q_confirm("use_huber", "Use Huber loss?", default=True),
                q_text("huber_k", "Huber k (blank = auto):", None, float, optional=True),
            ]
        return qs

    def build_servo_frames_config(controller, scene_name, renderers):
        is_ibvs = controller == "ibvs"
        is_photo = controller in ("photometric", "photometric_torch")

        cfg = {"kind": "servo_frames", "scene_dir": scene_name, "controller": controller}

        if is_photo:
            console.rule("[bold magenta]Photometric controller knobs[/]")
        cfg.update(run_prompt(common_questions(controller, renderers)))

        console.rule("[bold magenta]Frame selection[/]")
        frame_qs = [
            q_text("start_index", "Start index:", 1, int),
            q_text(
                "target_index",
                "Target index (blank → use index_away):",
                None,
                int,
                optional=True,
            ),
            q_text(
                "index_away",
                "Index away:",
                1,
                int,
                when=lambda a: a.get("target_index") is None,
            ),
        ]
        frame_ans = run_prompt(frame_qs)
        cfg["start_index"] = frame_ans["start_index"]
        cfg["target_index"] = frame_ans.get("target_index")
        cfg["index_away"] = frame_ans.get("index_away", 1)

        console.rule("[bold magenta]Servo loop[/]")
        loop_qs = [
            q_text("iterations", "Iterations:", 100, int),
            q_text("dt", "dt:", 1.0, float),
        ]
        if is_ibvs:
            loop_qs += [
                q_text("min_features", "Min features:", 3, int),
                q_text("ratio", "Match ratio (0 = match once):", 1, int),
            ]
        loop_qs += [q_text("viz_iter", "Viz every N iters (0 disables):", 1, int)]
        if is_ibvs:
            loop_qs += [
                q_text(
                    "stop_residual_px",
                    "IBVS stop: RMS reprojection error (px):",
                    0.5,
                    float,
                )
            ]
        if is_photo:
            loop_qs += [
                q_text(
                    "stop_mse_per_px",
                    "Photometric stop: mean(e^2) per pixel on [0,1]:",
                    2.0e-6,
                    float,
                )
            ]
        loop_qs += [
            q_text("run_name", "Run name (blank = auto):", None, str, optional=True)
        ]
        cfg.update(run_prompt(loop_qs))
        return cfg

    def build_trajectory_config(controller, datasets, renderers, run_tag_default=None):
        is_ibvs = controller == "ibvs"
        is_photo = controller in ("photometric", "photometric_torch")

        cfg = {"kind": "trajectory", "datasets": list(datasets), "controller": controller}

        if is_photo:
            console.rule("[bold magenta]Photometric controller knobs[/]")
        cfg.update(run_prompt(common_questions(controller, renderers)))

        if cfg["renderer"] == "nerf":
            cfg.update(
                run_prompt([q_text("nerf_render_scale", "NeRF render scale:", 0.25, float)])
            )

        console.rule("[bold magenta]Trajectory pacing[/]")
        pacing_qs = [
            q_text("stride", "Stride between frames:", 1, int),
            q_text("mini_iterations", "Iterations per mini task:", 30, int),
            q_text("dt", "dt:", 1.0, float),
        ]
        if is_ibvs:
            pacing_qs += [
                q_text("min_features", "Min features:", 3, int),
                q_text("ratio", "Match ratio:", 1, int),
            ]
        pacing_qs += [
            q_text("start_index", "Start index:", 1, int),
            q_text("max_pairs", "Max pairs (blank = all):", None, int, optional=True),
            q_text("rpe_delta", "RPE delta:", 1, int),
        ]
        cfg.update(run_prompt(pacing_qs))

        console.rule("[bold magenta]Stopping + viz[/]")
        stop_qs = []
        if is_ibvs:
            stop_qs += [
                q_text(
                    "stop_residual_px",
                    "IBVS stop: RMS reprojection error (px):",
                    0.5,
                    float,
                )
            ]
        if is_photo:
            stop_qs += [
                q_text(
                    "stop_mse_per_px",
                    "Photometric stop: mean(e^2) per pixel on [0,1]:",
                    2.0e-6,
                    float,
                )
            ]
        stop_qs += [
            q_confirm("save_task_viz", "Save per-task viz?", default=True),
            q_text("task_viz_every", "Save viz every N tasks:", 1, int),
            q_text(
                "run_tag",
                "Run tag (blank = auto):",
                run_tag_default,
                str,
                optional=True,
            ),
        ]
        cfg.update(run_prompt(stop_qs))
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

    if task_key == "compare_table":
        comparison_scenes = renderable_scenes(scenes)
        if not comparison_scenes:
            console.print(
                "[red]No renderable datasets found under DATA/ "
                "(expected mesh.ply, gs.ply, or nerf/).[/]"
            )
            return 1
        comparison_datasets = [name for name, _ in comparison_scenes]
        scene_renderers = ordered_renderers(comparison_scenes, common=True)
        if not scene_renderers:
            scene_renderers = ordered_renderers(comparison_scenes, common=False)
            console.print(
                "[yellow]No renderer is available in every dataset; datasets "
                "missing the selected renderer will be marked failed in the table.[/]"
            )
        console.print(
            "[grey50]datasets:[/] " + ", ".join(comparison_datasets)
        )
        scene_name = "all_datasets"
        cfg = build_trajectory_config(
            controller,
            comparison_datasets,
            scene_renderers,
            run_tag_default="compare_all",
        )
    else:
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
            cfg = build_trajectory_config(controller, [scene_name], scene_renderers)

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
