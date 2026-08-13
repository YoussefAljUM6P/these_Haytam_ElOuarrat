"""Generic variant x dataset benchmarking sweep.

Runs the `trajectory` runner once per (variant x dataset) cell, holding a shared
base config fixed and varying a single user-defined axis (the *variant*). A
variant is a named bundle of config overrides, e.g. ``renderer=gs`` vs
``renderer=mesh``, or ``depth_mode=intrinsic`` vs ``depth_mode=learned``, or
``feature_method=sift`` vs ``feature_method=xfeat``.

Variants come from preset axes and/or explicit ``--variant`` flags:

    # compare renderers on three datasets
    python cli.py compare --datasets bonsai,counter,kitchen --axis renderer

    # compare depth estimators (renderer fixed by --config)
    python cli.py compare --datasets bonsai,counter --axis depth --config CONFIGS/base.json

    # custom axis
    python cli.py compare --datasets bonsai \
        --variant sift:feature_method=sift \
        --variant xfeat:feature_method=xfeat

Each cell gets ``RUNS/compare/<batch_id>/<variant>/<dataset>/`` with the exact
command, console log, and the resolved trajectory run root. The batch root gets:

    compare_<variant>.csv   rows = metric, cols = dataset
    compare_all.csv         rows = metric x variant, cols = dataset
    manifest.json           per-cell status / run_root / error

Failed, unfinished, or skipped (missing-asset) cells are recorded as ``N/A``.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scene_assets import nerfstudio_checkpoint_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ROOT = PROJECT_ROOT / "RUNS"
DATA_ROOT = PROJECT_ROOT / "DATA"
CLI_SCRIPT = PROJECT_ROOT / "SRC" / "cli.py"

# CSV row order: (row label, summary metric family, stat key).
METRIC_ROWS = (
    ("ape_trans_rmse", "ape_translation", "rmse"),
    ("ape_rot_rmse", "ape_rotation_deg", "rmse"),
    ("rpe_trans_rmse", "rpe_translation", "rmse"),
    ("rpe_rot_rmse", "rpe_rotation_deg", "rmse"),
    ("avg_fps", "timing", "fps"),
    ("avg_render_ms", "timing", "render_ms_mean"),
)

# Preset axes -> ordered list of (variant name, override dict).
PRESET_AXES = {
    "renderer": [
        ("gs", {"renderer": "gs"}),
        ("mesh", {"renderer": "mesh"}),
        ("nerf", {"renderer": "nerf"}),
    ],
    "depth": [
        ("intrinsic", {"depth_mode": "intrinsic"}),
        ("learned", {"depth_mode": "learned"}),
    ],
    "feature": [
        ("sift", {"feature_method": "sift"}),
        ("xfeat", {"feature_method": "xfeat"}),
    ],
}

NA = "N/A"


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: Dict[str, str]


# ---- argument parsing ------------------------------------------------------


def add_arguments(parser):
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset folders under DATA/, or 'all'.",
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        choices=sorted(PRESET_AXES),
        help="Preset comparison axis (repeatable): renderer | depth | feature.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME:key=val[,key=val]",
        help="Custom variant (repeatable): a name and its config overrides.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base trajectory JSON config shared by every cell.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional batch directory name under RUNS/compare/.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the trajectory runner.",
    )
    # Pass-through overrides applied to every cell.
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override mini_iterations for every cell.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit mini-servo tasks for every cell.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override frame stride for every cell.")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Override 1-based start index for every cell.")
    parser.add_argument("--gain", type=float, default=None,
                        help="Override IBVS gain for every cell.")
    parser.add_argument("--ratio", type=int, default=None,
                        help="Override feature refresh ratio for every cell.")
    parser.add_argument("--min-features", type=int, default=None,
                        help="Override minimum IBVS feature count for every cell.")
    parser.add_argument("--diverge-translation-error", type=float, default=None,
                        help="Override final translation-error divergence threshold for every cell.")
    parser.add_argument("--diverge-rotation-error-deg", type=float, default=None,
                        help="Override final rotation-error divergence threshold for every cell.")
    parser.add_argument("--save-task-viz", action="store_true",
                        help="Keep per-task final-vs-target images.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra override applied to every cell.",
    )
    parser.add_argument("--resume-existing", action="store_true",
                        help="Resume cells that already have run_root.txt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write commands and CSVs without running.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop the batch after the first failed cell.")


def parse_variant_flag(spec: str) -> Variant:
    """Parse ``NAME:key=val[,key=val ...]`` into a Variant."""
    if ":" not in spec:
        raise ValueError(
            f"Expected --variant NAME:key=val[,key=val], got {spec!r}"
        )
    name, body = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty variant name in {spec!r}")
    overrides: Dict[str, str] = {}
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Expected key=value in variant {name!r}, got {item!r}"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty override key in variant {name!r}: {item!r}")
        overrides[key] = value.strip()
    if not overrides:
        raise ValueError(f"Variant {name!r} has no overrides")
    return Variant(name, overrides)


def build_variants(args) -> List[Variant]:
    variants: List[Variant] = []
    for axis in args.axis:
        for name, overrides in PRESET_AXES[axis]:
            variants.append(Variant(name, dict(overrides)))
    for spec in args.variant:
        variants.append(parse_variant_flag(spec))
    if not variants:
        raise ValueError("No variants: pass at least one --axis or --variant.")
    seen = set()
    for v in variants:
        if v.name in seen:
            raise ValueError(f"Duplicate variant name {v.name!r}.")
        seen.add(v.name)
    return variants


# ---- asset checks ----------------------------------------------------------


def has_colmap_reconstruction(dataset: str) -> bool:
    return (DATA_ROOT / dataset / "sparse" / "0").exists()


def base_renderer(config_path: Optional[Path]) -> str:
    if config_path is None or not Path(config_path).exists():
        return "gs"
    try:
        cfg = json.loads(Path(config_path).read_text())
    except (json.JSONDecodeError, OSError):
        return "gs"
    return str(cfg.get("renderer", "gs"))


def cell_renderer(variant: Variant, config_path: Optional[Path]) -> str:
    return variant.overrides.get("renderer", base_renderer(config_path))


def asset_error(renderer: str, dataset: str) -> Optional[str]:
    scene_dir = DATA_ROOT / dataset
    if not scene_dir.exists():
        return f"Missing dataset directory: {scene_dir}"
    if not has_colmap_reconstruction(dataset):
        return f"Missing COLMAP reconstruction: {scene_dir / 'sparse' / '0'}"
    if renderer == "gs":
        asset = scene_dir / "gs.ply"
    elif renderer == "mesh":
        asset = scene_dir / "mesh.ply"
    elif renderer == "nerf":
        return nerfstudio_checkpoint_error(scene_dir)
    else:
        return f"Unknown renderer {renderer!r}"
    if not asset.exists():
        return f"Missing scene asset for {renderer}: {asset}"
    return None


# ---- command construction & execution --------------------------------------


def optional_overrides(args) -> Dict[str, object]:
    optional = {
        "mini_iterations": args.iterations,
        "max_pairs": args.max_pairs,
        "stride": args.stride,
        "start_index": args.start_index,
        "gain": args.gain,
        "ratio": args.ratio,
        "min_features": args.min_features,
        "diverge_translation_error": args.diverge_translation_error,
        "diverge_rotation_error_deg": args.diverge_rotation_error_deg,
    }
    return {k: v for k, v in optional.items() if v is not None}


def cell_overrides(variant, dataset, renderer, args, batch_id) -> Dict[str, str]:
    overrides: Dict[str, object] = {
        "datasets": dataset,
        "renderer": renderer,
        "run_tag": f"compare_{batch_id}_{variant.name}_{dataset}",
        "save_task_viz": str(bool(args.save_task_viz)).lower(),
    }
    overrides.update(variant.overrides)
    overrides.update(optional_overrides(args))
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Expected --set KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value
    # renderer must reflect the resolved cell renderer even if --set tried to change it
    overrides["renderer"] = renderer
    overrides["datasets"] = dataset
    return {k: str(v) for k, v in overrides.items()}


def build_command(variant, dataset, renderer, args, batch_id) -> List[str]:
    command = [args.python, str(CLI_SCRIPT), "trajectory"]
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    for key, value in cell_overrides(variant, dataset, renderer, args, batch_id).items():
        command.extend(["--set", f"{key}={value}"])
    return command


def parse_run_root(output: str) -> Optional[Path]:
    matches = re.findall(r"^Wrote (.+)$", output, flags=re.MULTILINE)
    if not matches:
        return None
    return Path(matches[-1].strip()).expanduser().resolve()


def empty_metrics() -> Dict[str, Optional[float]]:
    return {label: None for label, _, _ in METRIC_ROWS}


def metrics_from_summary(run_root: Optional[Path], dataset: str):
    """Return (metrics dict, error str|None) from a run's trajectory_summary.json."""
    if run_root is None:
        return empty_metrics(), "no run_root parsed from console"
    summary_path = run_root / "trajectory_summary.json"
    if not summary_path.exists():
        return empty_metrics(), "trajectory_summary.json not found"
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return empty_metrics(), f"summary unreadable: {e}"

    scene = summary.get(dataset)
    if scene is None and len(summary) == 1:
        scene = next(iter(summary.values()))
    if scene is None:
        return empty_metrics(), f"dataset {dataset!r} not in summary"
    if "error" in scene and "metrics" not in scene:
        return empty_metrics(), str(scene["error"])

    metrics = scene.get("metrics", {})
    if "error" in metrics:
        return empty_metrics(), str(metrics["error"])

    out = empty_metrics()
    timing = scene.get("timing", {})
    for label, family, stat in METRIC_ROWS:
        fam = timing if family == "timing" else metrics.get(family)
        if isinstance(fam, dict) and fam.get(stat) is not None:
            out[label] = float(fam[stat])
    return out, None


def run_cell(variant, dataset, args, batch_id, batch_dir):
    cell_dir = batch_dir / variant.name / dataset
    cell_dir.mkdir(parents=True, exist_ok=True)
    renderer = cell_renderer(variant, args.config)

    result = {
        "variant": variant.name,
        "dataset": dataset,
        "renderer": renderer,
        "status": "ok",
        "run_root": None,
        "error": None,
        "metrics": empty_metrics(),
    }

    err = asset_error(renderer, dataset)
    command = build_command(variant, dataset, renderer, args, batch_id)

    run_root_file = cell_dir / "run_root.txt"
    if args.resume_existing and run_root_file.exists():
        command.extend(["--resume", run_root_file.read_text().strip()])

    (cell_dir / "command.json").write_text(json.dumps({
        "variant": variant.name,
        "overrides": variant.overrides,
        "dataset": dataset,
        "renderer": renderer,
        "command": command,
        "cwd": str(PROJECT_ROOT),
    }, indent=2))

    if err is not None:
        result["status"] = "skipped"
        result["error"] = err
        (cell_dir / "console.txt").write_text(f"SKIPPED: {err}\n")
        return result

    if args.dry_run:
        result["status"] = "dry_run"
        (cell_dir / "console.txt").write_text("DRY RUN\n" + " ".join(command) + "\n")
        return result

    proc = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (cell_dir / "console.txt").write_text(proc.stdout)
    run_root = parse_run_root(proc.stdout)
    if run_root is not None:
        run_root_file.write_text(str(run_root) + "\n")
    result["run_root"] = None if run_root is None else str(run_root)

    if proc.returncode != 0:
        result["status"] = "failed"
        result["error"] = f"exit code {proc.returncode}"

    metrics, metrics_err = metrics_from_summary(run_root, dataset)
    result["metrics"] = metrics
    if metrics_err and result["error"] is None:
        result["status"] = "failed"
        result["error"] = metrics_err
    return result


# ---- output ----------------------------------------------------------------


def fmt(value: Optional[float]) -> str:
    if value is None:
        return NA
    value = float(value)
    if value != value:  # NaN
        return NA
    return f"{value:.6g}"


def _cell_metric(results, variant_name, dataset, label):
    """Metric value for a cell, or None if the cell has not run yet."""
    cell = results.get((variant_name, dataset))
    if cell is None:
        return None
    return cell["metrics"].get(label)


def write_variant_csv(path, variant, datasets, results):
    lines = ["metric," + ",".join(datasets)]
    for label, _, _ in METRIC_ROWS:
        row = [label]
        for ds in datasets:
            row.append(fmt(_cell_metric(results, variant.name, ds, label)))
        lines.append(",".join(row))
    Path(path).write_text("\n".join(lines) + "\n")


def write_combined_csv(path, variants, datasets, results):
    lines = ["metric,variant," + ",".join(datasets)]
    for label, _, _ in METRIC_ROWS:
        for variant in variants:
            row = [label, variant.name]
            for ds in datasets:
                row.append(fmt(_cell_metric(results, variant.name, ds, label)))
            lines.append(",".join(row))
    Path(path).write_text("\n".join(lines) + "\n")


def write_outputs(batch_dir, variants, datasets, results):
    for variant in variants:
        write_variant_csv(
            batch_dir / f"compare_{variant.name}.csv", variant, datasets, results
        )
    write_combined_csv(batch_dir / "compare_all.csv", variants, datasets, results)
    manifest = [
        {k: cell[k] for k in ("variant", "dataset", "renderer", "status", "run_root", "error")}
        | {"metrics": cell["metrics"]}
        for cell in results.values()
    ]
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def resolve_datasets(spec: str) -> List[str]:
    if spec == "all":
        datasets = [
            d.name for d in sorted(DATA_ROOT.iterdir())
            if d.is_dir() and has_colmap_reconstruction(d.name)
        ]
        if not datasets:
            raise ValueError("No datasets under DATA/ with a COLMAP reconstruction.")
        return datasets
    datasets = [d.strip() for d in spec.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets specified.")
    return datasets


def run(args):
    variants = build_variants(args)
    datasets = resolve_datasets(args.datasets)
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = RUNS_ROOT / "compare" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Compare batch {batch_id}")
    print(f"  variants: {', '.join(v.name for v in variants)}")
    print(f"  datasets: {', '.join(datasets)}")

    results: Dict[Tuple[str, str], dict] = {}
    cells = [(v, ds) for v in variants for ds in datasets]
    stop = False
    for index, (variant, dataset) in enumerate(cells, start=1):
        print(f"[{index:02d}/{len(cells)}] {variant.name} | {dataset}")
        cell = run_cell(variant, dataset, args, batch_id, batch_dir)
        results[(variant.name, dataset)] = cell
        status = cell["status"]
        if status in ("ok",):
            print(f"  ok: ape_trans_rmse={fmt(cell['metrics']['ape_trans_rmse'])} "
                  f"ape_rot_rmse={fmt(cell['metrics']['ape_rot_rmse'])} "
                  f"avg_fps={fmt(cell['metrics']['avg_fps'])} "
                  f"avg_render_ms={fmt(cell['metrics']['avg_render_ms'])}")
        else:
            print(f"  {status}: {cell['error']}")
            if status == "failed" and args.fail_fast:
                stop = True
        write_outputs(batch_dir, variants, datasets, results)
        if stop:
            break

    # Fill any cells skipped by fail-fast so the CSVs stay rectangular.
    for variant, dataset in cells:
        results.setdefault((variant.name, dataset), {
            "variant": variant.name, "dataset": dataset,
            "renderer": cell_renderer(variant, args.config),
            "status": "not_run", "run_root": None,
            "error": "fail-fast stopped batch", "metrics": empty_metrics(),
        })
    write_outputs(batch_dir, variants, datasets, results)
    print(f"\nWrote batch outputs to {batch_dir}")
