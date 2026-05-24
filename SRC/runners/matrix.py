"""Run the IBVS servo matrix and collect evo pose metrics.

Launches `python cli.py trajectory ...` once per condition:

    scene/render source in {GS, Poisson Mesh}
    depth in {Intrinsic, MoGe2}
    matcher in {SIFT, XFeat}

Each condition gets a directory under ``RUNS/servo_matrix/<batch_id>/`` with:

    command.json       exact command and condition metadata
    console.txt        full stdout/stderr from trajectory runner
    evo_output.txt     extracted evo metrics block from console.txt
    evo_metrics.txt    concise APE translation/rotation stats
    run_root.txt       trajectory run directory written by trajectory runner

The batch root also gets combined long-form CSV and matrix CSV/Markdown tables.

Use via the CLI:
    python cli.py matrix --dataset kitchen [--iterations 30 ...]
"""

import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ROOT = PROJECT_ROOT / "RUNS"
CLI_SCRIPT = PROJECT_ROOT / "SRC" / "cli.py"
DEFAULT_CONFIG = PROJECT_ROOT / "CONFIGS" / "trajectory_kitchen_mesh.json"
METRIC_KEYS = ("rmse", "mean", "median", "std", "max")
METRIC_LABELS = {
    "rmse": "RMSE",
    "mean": "Mean",
    "median": "Median",
    "std": "Std",
    "max": "Max",
}


@dataclass(frozen=True)
class SceneSpec:
    label: str
    renderer: str


@dataclass(frozen=True)
class DepthSpec:
    label: str
    value: str


@dataclass(frozen=True)
class MatcherSpec:
    label: str
    value: str


@dataclass(frozen=True)
class Condition:
    scene: SceneSpec
    depth: DepthSpec
    matcher: MatcherSpec

    @property
    def slug(self):
        return slugify(f"{self.scene.label}_{self.depth.label}_{self.matcher.label}")


SCENES = (
    SceneSpec("GS", "gs"),
    SceneSpec("Poisson Mesh", "mesh"),
)

DEPTHS = (
    DepthSpec("Intrinsic", "intrinsic"),
    DepthSpec("MoGe2", "learned"),
)

MATCHERS = (
    MatcherSpec("SIFT", "sift"),
    MatcherSpec("XFeat", "xfeat"),
)


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def add_arguments(parser):
    parser.add_argument(
        "--dataset",
        default="kitchen",
        help="Dataset folder under DATA/ to run. Default: kitchen.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base trajectory JSON config.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch directory name under RUNS/servo_matrix/.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the trajectory runner.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override mini_iterations for every condition.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit mini-servo tasks for every condition.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override frame stride for every condition.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Override 1-based start index for every condition.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Override IBVS gain for every condition.",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=None,
        help="Override feature refresh ratio for every condition.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=None,
        help="Override minimum IBVS feature count for every condition.",
    )
    parser.add_argument(
        "--save-task-viz",
        action="store_true",
        help="Keep per-task final-vs-target images. Disabled by default.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume conditions that already have run_root.txt in the batch dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands and tables without running experiments.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch after the first failed condition.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra --set override applied to every condition.",
    )


def all_conditions():
    return [
        Condition(scene, depth, matcher)
        for scene in SCENES
        for depth in DEPTHS
        for matcher in MATCHERS
    ]


def condition_overrides(condition, args, batch_id):
    tag = f"matrix_{batch_id}_{condition.slug}"
    overrides = {
        "datasets": args.dataset,
        "controller": "ibvs",
        "renderer": condition.scene.renderer,
        "depth_mode": condition.depth.value,
        "feature_method": condition.matcher.value,
        "run_tag": tag,
        "save_task_viz": str(bool(args.save_task_viz)).lower(),
    }

    optional = {
        "mini_iterations": args.iterations,
        "max_pairs": args.max_pairs,
        "stride": args.stride,
        "start_index": args.start_index,
        "gain": args.gain,
        "ratio": args.ratio,
        "min_features": args.min_features,
    }
    for key, value in optional.items():
        if value is not None:
            overrides[key] = value

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Expected --set KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value

    return overrides


def build_command(condition, args, batch_id):
    command = [
        args.python,
        str(CLI_SCRIPT),
        "trajectory",
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    for key, value in condition_overrides(condition, args, batch_id).items():
        command.extend(["--set", f"{key}={value}"])
    return command


def scene_asset_error(condition, dataset):
    scene_dir = PROJECT_ROOT / "DATA" / dataset
    if not scene_dir.exists():
        return f"Missing dataset directory: {scene_dir}"
    if not has_colmap_reconstruction(dataset):
        return f"Missing COLMAP reconstruction: {scene_dir / 'sparse' / '0'}"
    if condition.scene.renderer == "gs":
        path = scene_dir / "gs.ply"
    else:
        path = scene_dir / "mesh.ply"
    if not path.exists():
        return f"Missing scene asset: {path}"
    return None


def has_colmap_reconstruction(dataset):
    return (PROJECT_ROOT / "DATA" / dataset / "sparse" / "0").exists()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_condition(condition, args, batch_id, batch_dir):
    condition_dir = batch_dir / condition.slug
    condition_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(condition, args, batch_id)

    existing_run_root = condition_dir / "run_root.txt"
    if args.resume_existing and existing_run_root.exists():
        run_root = Path(existing_run_root.read_text().strip())
        command.extend(["--resume", str(run_root)])

    metadata = {
        "condition": {
            "scene": condition.scene.label,
            "depth": condition.depth.label,
            "matcher": condition.matcher.label,
            "slug": condition.slug,
        },
        "command": command,
        "cwd": str(PROJECT_ROOT),
    }
    write_json(condition_dir / "command.json", metadata)

    asset_error = scene_asset_error(condition, args.dataset)
    if asset_error is not None:
        text = f"SKIPPED: {asset_error}\n"
        (condition_dir / "console.txt").write_text(text)
        (condition_dir / "evo_output.txt").write_text(text)
        write_condition_metrics_txt(
            condition_dir / "evo_metrics.txt",
            condition,
            {},
            asset_error,
        )
        return {
            "condition": condition,
            "condition_dir": condition_dir,
            "returncode": None,
            "run_root": None,
            "error": asset_error,
            "metrics": {},
            "skipped": True,
        }

    if args.dry_run:
        text = "DRY RUN\n" + " ".join(command) + "\n"
        (condition_dir / "console.txt").write_text(text)
        (condition_dir / "evo_output.txt").write_text(text)
        write_condition_metrics_txt(
            condition_dir / "evo_metrics.txt",
            condition,
            {},
            "dry_run",
        )
        return {
            "condition": condition,
            "condition_dir": condition_dir,
            "returncode": 0,
            "run_root": None,
            "error": "dry_run",
            "metrics": {},
            "skipped": True,
        }

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout
    (condition_dir / "console.txt").write_text(output)
    (condition_dir / "evo_output.txt").write_text(extract_evo_output(output))

    run_root = parse_run_root(output)
    if run_root is not None:
        (condition_dir / "run_root.txt").write_text(str(run_root) + "\n")
        summary_path = run_root / "trajectory_summary.json"
        if summary_path.exists():
            shutil.copy2(summary_path, condition_dir / "trajectory_summary.json")

    metrics, error = load_condition_metrics(condition_dir, run_root, args.dataset)
    write_condition_metrics_txt(condition_dir / "evo_metrics.txt", condition, metrics, error)

    return {
        "condition": condition,
        "condition_dir": condition_dir,
        "returncode": process.returncode,
        "run_root": run_root,
        "error": error,
        "metrics": metrics,
        "skipped": False,
    }


def parse_run_root(output):
    matches = re.findall(r"^Wrote (.+)$", output, flags=re.MULTILINE)
    if not matches:
        return None
    return Path(matches[-1]).expanduser().resolve()


def extract_evo_output(output):
    lines = output.splitlines()
    selected = []
    capture = False
    for line in lines:
        if "evo metrics" in line:
            capture = True
        if capture:
            if line.startswith("Wrote ") and selected:
                break
            selected.append(line)
    if selected:
        return "\n".join(selected).rstrip() + "\n"
    return "No evo metrics block found in console output.\n"


def load_condition_metrics(condition_dir, run_root, dataset):
    summary_path = None
    if run_root is not None and (run_root / "trajectory_summary.json").exists():
        summary_path = run_root / "trajectory_summary.json"
    elif (condition_dir / "trajectory_summary.json").exists():
        summary_path = condition_dir / "trajectory_summary.json"

    if summary_path is None:
        return {}, "trajectory_summary.json not found"

    with open(summary_path) as f:
        summary = json.load(f)

    scene_summary = summary.get(dataset)
    if scene_summary is None:
        if len(summary) == 1:
            scene_summary = next(iter(summary.values()))
        else:
            return {}, f"dataset {dataset!r} not found in summary"

    if "error" in scene_summary and "metrics" not in scene_summary:
        return {}, str(scene_summary["error"])

    metrics = scene_summary.get("metrics", {})
    ape_translation = metrics.get("ape_translation")
    ape_rotation = metrics.get("ape_rotation_deg")
    if not isinstance(ape_translation, dict):
        return {}, "ape_translation metrics not found"

    return {
        "translation": ape_translation,
        "rotation": ape_rotation if isinstance(ape_rotation, dict) else {},
    }, None


def metric_value(metrics, family, key):
    values = metrics.get(family, {})
    value = values.get(key)
    if value is None:
        return None
    value = float(value)
    return value


def metric_unit(family):
    if family == "translation":
        return "scene scale"
    if family == "rotation":
        return "deg"
    raise ValueError(f"Unknown metric family {family!r}")


def metric_title(family):
    if family == "translation":
        return "APE Translation"
    if family == "rotation":
        return "APE Rotation"
    raise ValueError(f"Unknown metric family {family!r}")


def format_metric(value):
    if value is None:
        return "X"
    value = float(value)
    if value != value:
        return "-"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0.0 else "-inf"
    mag = abs(value)
    if mag == 0.0:
        return "0"
    if mag < 1.0e-3:
        return f"{value:.2e}"
    if mag < 1.0:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if mag < 10.0:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def write_condition_metrics_txt(path, condition, metrics, error):
    lines = [
        f"Scene: {condition.scene.label}",
        f"Depth: {condition.depth.label}",
        f"Matcher: {condition.matcher.label}",
        "",
    ]
    if error:
        lines.append(f"ERROR: {error}")
    else:
        lines.append("APE translation")
        for key in METRIC_KEYS:
            lines.append(
                f"{METRIC_LABELS[key]}: "
                f"{format_metric(metric_value(metrics, 'translation', key))}"
            )
        lines.append("")
        lines.append("APE rotation (deg)")
        for key in METRIC_KEYS:
            lines.append(
                f"{METRIC_LABELS[key]}: "
                f"{format_metric(metric_value(metrics, 'rotation', key))}"
            )
    Path(path).write_text("\n".join(lines).rstrip() + "\n")


def write_batch_outputs(batch_dir, results):
    write_json(
        batch_dir / "manifest.json",
        [
            {
                "scene": r["condition"].scene.label,
                "depth": r["condition"].depth.label,
                "matcher": r["condition"].matcher.label,
                "slug": r["condition"].slug,
                "condition_dir": str(r["condition_dir"]),
                "run_root": None if r["run_root"] is None else str(r["run_root"]),
                "returncode": r["returncode"],
                "error": r["error"],
                "skipped": r["skipped"],
            }
            for r in results
        ],
    )
    write_long_csv(batch_dir / "conditions_ape_metrics.csv", results)
    write_matrix_csv(batch_dir / "matrix_ape_translation_gap.csv", results, "translation")
    write_matrix_markdown(
        batch_dir / "matrix_ape_translation_gap.md",
        results,
        "translation",
    )
    write_matrix_csv(batch_dir / "matrix_ape_rotation_deg.csv", results, "rotation")
    write_matrix_markdown(
        batch_dir / "matrix_ape_rotation_deg.md",
        results,
        "rotation",
    )


def result_by_slug(results):
    return {r["condition"].slug: r for r in results}


def ordered_conditions():
    return all_conditions()


def write_long_csv(path, results):
    fields = [
        "scene",
        "depth",
        "matcher",
        "status",
        "ape_translation_rmse",
        "ape_translation_mean",
        "ape_translation_median",
        "ape_translation_std",
        "ape_translation_max",
        "ape_rotation_rmse_deg",
        "ape_rotation_mean_deg",
        "ape_rotation_median_deg",
        "ape_rotation_std_deg",
        "ape_rotation_max_deg",
        "run_root",
        "condition_dir",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            error = result["error"]
            row = {
                "scene": result["condition"].scene.label,
                "depth": result["condition"].depth.label,
                "matcher": result["condition"].matcher.label,
                "status": "ok" if error is None else "skipped" if result["skipped"] else "failed",
                "run_root": "" if result["run_root"] is None else str(result["run_root"]),
                "condition_dir": str(result["condition_dir"]),
                "error": "" if error is None else error,
            }
            for key in METRIC_KEYS:
                t_value = None if error else metric_value(metrics, "translation", key)
                r_value = None if error else metric_value(metrics, "rotation", key)
                row[f"ape_translation_{key}"] = (
                    "" if t_value is None else f"{t_value:.6f}"
                )
                row[f"ape_rotation_{key}_deg"] = (
                    "" if r_value is None else f"{r_value:.6f}"
                )
            writer.writerow(row)


def write_matrix_csv(path, results, family):
    by_slug = result_by_slug(results)
    conditions = ordered_conditions()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scene"] + [c.scene.label for c in conditions])
        writer.writerow(["Depth"] + [c.depth.label for c in conditions])
        writer.writerow(["Matcher"] + [c.matcher.label for c in conditions])
        for key in METRIC_KEYS:
            row = [METRIC_LABELS[key]]
            for condition in conditions:
                result = by_slug.get(condition.slug)
                value = None
                if result is not None and result["error"] is None:
                    value = metric_value(result["metrics"], family, key)
                row.append(format_metric(value))
            writer.writerow(row)


def write_matrix_markdown(path, results, family):
    by_slug = result_by_slug(results)
    conditions = ordered_conditions()
    rows = [
        ["Scene"] + [c.scene.label for c in conditions],
        ["Depth"] + [c.depth.label for c in conditions],
        ["Matcher"] + [c.matcher.label for c in conditions],
    ]
    for key in METRIC_KEYS:
        row = [METRIC_LABELS[key]]
        for condition in conditions:
            result = by_slug.get(condition.slug)
            value = None
            if result is not None and result["error"] is None:
                value = metric_value(result["metrics"], family, key)
            row.append(format_metric(value))
        rows.append(row)

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    def fmt(row):
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    lines = [
        f"# Servo Matrix: {metric_title(family)} ({metric_unit(family)})",
        "",
        fmt(rows[0]),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows[1:])
    lines.append("")
    Path(path).write_text("\n".join(lines))


def run(args):
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = RUNS_ROOT / "servo_matrix" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, condition in enumerate(all_conditions(), start=1):
        print(
            f"[{index:02d}/12] {condition.scene.label} | "
            f"{condition.depth.label} | {condition.matcher.label}"
        )
        result = run_condition(condition, args, batch_id, batch_dir)
        results.append(result)
        if result["error"] is None:
            metrics = result["metrics"]
            print(
                "  ok: "
                f"trans_rmse={format_metric(metric_value(metrics, 'translation', 'rmse'))} "
                f"rot_rmse={format_metric(metric_value(metrics, 'rotation', 'rmse'))}deg"
            )
        else:
            status = "skipped" if result["skipped"] else "failed"
            print(f"  {status}: {result['error']}")
            if args.fail_fast:
                break
        write_batch_outputs(batch_dir, results)

    write_batch_outputs(batch_dir, results)
    print(f"\nWrote batch outputs to {batch_dir}")
    print(f"Translation matrix: {batch_dir / 'matrix_ape_translation_gap.md'}")
    print(f"Rotation matrix: {batch_dir / 'matrix_ape_rotation_deg.md'}")
