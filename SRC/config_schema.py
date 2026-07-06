"""Single source of truth for SERVIS experiment parameters.

Every knob that the ``trajectory`` and ``servo_frames`` runners accept is
declared exactly once, here, as a :class:`Field`.  From this one registry the
rest of the CLI is *generated* rather than restated:

* ``experiment_config`` derives the snake->UPPER key maps, the type coercion /
  validation, the aliases, and the ``--set``/config loading from it;
* the runners receive their default module-globals via ``apply_defaults`` and
  their per-flag argparse surface via ``add_schema_arguments``;
* the interactive wizard walks the same registry to build its prompts.

Because there is one declaration per parameter, the runner defaults, the wizard
defaults, the CLI flag help, and the validation rules can no longer drift apart
(historically the wizard carried its own copy of e.g. ``mini_iterations`` /
``gain_photo`` and disagreed with the runner).

Adding a parameter is a one-line entry in :data:`FIELDS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Runner "kinds" that share this schema.
TRAJECTORY = "trajectory"
SERVO_FRAMES = "servo_frames"

# Controller applicability for a field. ``None`` => used by both controllers.
IBVS = "ibvs"
PHOTOMETRIC = "photometric"

RENDERERS = ("mesh", "gs", "nerf")
GS_MODELS = ("standard", "moge")
DEPTH_MODES = ("intrinsic", "learned")
CONTROLLERS = ("ibvs", "photometric")
FEATURE_METHODS = ("sift", "xfeat")  # suggested; runner accepts any lowercased str


# --- value types ------------------------------------------------------------
#
# Each type tag names how a raw config/CLI value is coerced and (optionally)
# validated. The behaviour mirrors the historical ``coerce_config_value`` ladder
# exactly, so migrating configs stay byte-for-byte compatible.

INT = "int"                # int, bounds via vmin/vmax
FLOAT = "float"            # float, bounds via vmin/vmax
BOOL = "bool"              # truthy/falsey words -> bool
STR = "str"                # free string
OPT_INT = "opt_int"        # null | int (>= 1 unless vmin overrides)
OPT_FLOAT = "opt_float"    # null | float, bounds via vmin/vmax
OPT_STR = "opt_str"        # null | string
CHOICE = "choice"          # lowercased, must be in `choices`
FEATURE = "feature"        # lowercased string (open set)
STR_LIST = "str_list"      # comma string or list -> non-empty list[str]
SCENE_DIR = "scene_dir"    # -> resolved Path under DATA/


@dataclass(frozen=True)
class Field:
    """One experiment parameter, declared once.

    name     snake_case config key (also the ``--name`` flag, dashed)
    const    UPPER_CASE module-global the runners read
    type     one of the value-type tags above
    kinds    which runners accept it (TRAJECTORY / SERVO_FRAMES)
    default  authoritative default; a dict keyed by kind when they differ
    applies  controller gate (IBVS / PHOTOMETRIC / None=both) — drives wizard
             ``when`` and flag help, never validation
    section  wizard grouping key; ``None`` => never prompted by the wizard
             (still reachable via flag / --set / config file)
    ask      whether the wizard prompts for it (scene/controller are chosen in
             dedicated wizard states, so their fields are ask=False)
    aliases  extra accepted key spellings for config / --set
    """

    name: str
    const: str
    type: str
    kinds: tuple[str, ...]
    default: Any = None
    help: str = ""
    applies: Optional[str] = None
    section: Optional[str] = None
    ask: bool = True
    optional: bool = False
    choices: Optional[tuple[str, ...]] = None
    vmin: Optional[float] = None
    vmin_inclusive: bool = True
    vmax: Optional[float] = None
    vmax_inclusive: bool = True
    aliases: tuple[str, ...] = ()

    # -- helpers ----------------------------------------------------------
    def applies_to(self, kind: str) -> bool:
        return kind in self.kinds

    def default_for(self, kind: str) -> Any:
        if isinstance(self.default, dict):
            return self.default.get(kind)
        return self.default

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")


# Ordered wizard sections. The wizard renders prompts in this order and prints a
# rule between groups.
SECTION_ORDER = (
    "common",
    "render",
    "controller_ibvs",
    "controller_photo",
    "frame",
    "loop",
    "loop_shared",
    "pacing",
    "stopping",
    "advanced",
)


# ============================================================================
# THE REGISTRY
# ============================================================================
#
# Defaults are the authoritative runner values (source of truth). Where the two
# runners historically used different defaults, `default` is a per-kind dict.

FIELDS: tuple[Field, ...] = (
    # --- scene selection (chosen in dedicated wizard states; ask=False) -----
    Field("datasets", "DATASETS", STR_LIST, (TRAJECTORY,),
          default=["living"], ask=False,
          help="Datasets/scenes to run (comma-separated or repeated).",
          aliases=("dataset", "scene", "scenes")),
    Field("scene_dir", "SCENE_DIR", SCENE_DIR, (SERVO_FRAMES,),
          default=PROJECT_ROOT / "DATA" / "kitchen", ask=False,
          help="Scene directory (bare name resolves under DATA/).",
          aliases=("dataset", "scene")),

    # --- common (both controllers) ------------------------------------------
    Field("renderer", "RENDERER", CHOICE, (TRAJECTORY, SERVO_FRAMES),
          default={TRAJECTORY: "gs", SERVO_FRAMES: "mesh"},
          choices=RENDERERS, section="common", ask=False,
          help="Rendering backend."),
    Field("controller", "CONTROLLER", CHOICE, (TRAJECTORY, SERVO_FRAMES),
          default="ibvs", choices=CONTROLLERS, section="common", ask=False,
          help="Servo controller (ibvs=feature-based, photometric=PVS)."),
    Field("depth_mode", "DEPTH_MODE", CHOICE, (TRAJECTORY, SERVO_FRAMES),
          default="intrinsic", choices=DEPTH_MODES, section="common",
          help="Depth source: intrinsic=scene.render_depth(), learned=MoGe2.",
          aliases=("depth",)),
    Field("gs_model", "GS_MODEL", CHOICE, (TRAJECTORY, SERVO_FRAMES),
          default="standard", choices=GS_MODELS, section="advanced",
          help="GS variant: standard=gs.ply, moge=gs_moge.ply."),

    # --- render scales (renderer-conditional in the wizard) -----------------
    Field("nerf_render_scale", "NERF_RENDER_SCALE", FLOAT, (TRAJECTORY,),
          default=0.25, vmin=0, vmin_inclusive=False, section="render",
          help="NeRF render scale (>0).", aliases=("nerf_scale", "render_scale")),
    Field("gs_render_scale", "GS_RENDER_SCALE", FLOAT, (TRAJECTORY,),
          default=1.0, vmin=0, vmin_inclusive=False, section="render",
          help="GS render scale (>0)."),
    Field("mesh_render_scale", "MESH_RENDER_SCALE", FLOAT, (TRAJECTORY,),
          default=1.0, vmin=0, vmin_inclusive=False, section="render",
          help="Mesh render scale (>0)."),

    # --- IBVS controller knobs ----------------------------------------------
    Field("feature_method", "FEATURE_METHOD", FEATURE, (TRAJECTORY, SERVO_FRAMES),
          default="sift", applies=IBVS, section="controller_ibvs",
          help="Feature matcher (sift, xfeat, ...).",
          aliases=("feature", "matcher")),
    Field("gain_ibvs", "GAIN_IBVS", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=0.75, applies=IBVS, section="controller_ibvs",
          help="IBVS control gain.", aliases=("gain",)),

    # --- photometric controller knobs ---------------------------------------
    Field("gain_photo", "GAIN_PHOTO", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=0.75, applies=PHOTOMETRIC, section="controller_photo",
          help="Photometric control gain."),
    Field("sigma_blur", "SIGMA_BLUR", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=1.0, vmin=0, applies=PHOTOMETRIC, section="controller_photo",
          help="Gaussian blur sigma before photometric error (>=0)."),
    Field("use_gzn", "USE_GZN", BOOL, (TRAJECTORY, SERVO_FRAMES),
          default=True, applies=PHOTOMETRIC, section="controller_photo",
          help="Use gradient-based Z normalization."),
    Field("grad_percentile", "GRAD_PERCENTILE", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=50.0, vmin=0, vmax=100, vmax_inclusive=False,
          applies=PHOTOMETRIC, section="controller_photo",
          help="Gradient percentile pixel selection, [0, 100)."),
    Field("photometric_max_pixels", "PHOTOMETRIC_MAX_PIXELS", INT,
          (TRAJECTORY, SERVO_FRAMES), default=50000, vmin=0,
          applies=PHOTOMETRIC, section="controller_photo",
          help="Max pixels used by the photometric controller."),
    Field("use_huber", "USE_HUBER", BOOL, (TRAJECTORY, SERVO_FRAMES),
          default=True, applies=PHOTOMETRIC, section="controller_photo",
          help="Use Huber robust loss."),
    Field("huber_k", "HUBER_K", OPT_FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=None, optional=True, applies=PHOTOMETRIC,
          section="controller_photo",
          help="Huber threshold k (blank=auto)."),
    Field("stop_ssd", "STOP_SSD", OPT_FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=None, optional=True, applies=PHOTOMETRIC, section="advanced",
          help="ViSP-style SSD stop threshold (blank=use stop_mse_per_px)."),

    # --- servo_frames: frame selection --------------------------------------
    Field("start_index", "START_INDEX", INT, (TRAJECTORY, SERVO_FRAMES),
          default=1, vmin=1, section="frame",
          help="1-based start frame index.", aliases=("start",)),
    Field("target_index", "TARGET_INDEX", OPT_INT, (SERVO_FRAMES,),
          default=None, optional=True, vmin=1, section="frame",
          help="Target frame index (blank => use index_away).",
          aliases=("target",)),
    Field("index_away", "INDEX_AWAY", INT, (SERVO_FRAMES,),
          default=1, section="frame",
          help="Offset from start to target when target_index is blank.",
          aliases=("away",)),

    # --- servo_frames: loop --------------------------------------------------
    Field("iterations", "ITERATIONS", INT, (SERVO_FRAMES,),
          default=100, vmin=0, section="loop",
          help="Servo iterations.", aliases=("iters",)),
    Field("viz_iter", "VIZ_ITER", INT, (SERVO_FRAMES,),
          default=1, section="loop",
          help="Save a viz frame every N iterations (0 disables).",
          aliases=("viz_every", "visualize_every")),
    Field("run_name", "RUN_NAME", OPT_STR, (SERVO_FRAMES,),
          default=None, optional=True, section="loop",
          help="Run name (blank=auto).", aliases=("tag",)),

    # --- trajectory: pacing --------------------------------------------------
    Field("stride", "STRIDE", INT, (TRAJECTORY,),
          default=1, vmin=1, section="pacing",
          help="Frames between consecutive servo targets."),
    Field("mini_iterations", "MINI_ITERATIONS", INT, (TRAJECTORY,),
          default=500, vmin=0, section="pacing",
          help="Iterations per mini servo task.",
          aliases=("iters", "iterations")),
    Field("max_pairs", "MAX_PAIRS", OPT_INT, (TRAJECTORY,),
          default=None, optional=True, vmin=1, section="pacing",
          help="Max consecutive pose pairs to run (blank=all).",
          aliases=("max_tasks", "pairs")),
    Field("rpe_delta", "RPE_DELTA", INT, (TRAJECTORY,),
          default=1, vmin=1, section="pacing",
          help="evo RPE delta in frames."),
    Field("use_takes", "USE_TAKES", BOOL, (TRAJECTORY,),
          default=True, section="advanced",
          help="Split capture into takes and reset at take boundaries."),
    Field("takes_jump_factor", "TAKES_JUMP_FACTOR", FLOAT, (TRAJECTORY,),
          default=5.0, section="advanced",
          help="Motion factor that marks a take boundary."),
    Field("dynamic_ibvs_iters", "DYNAMIC_IBVS_ITERS", BOOL, (TRAJECTORY,),
          default=True, section="advanced",
          help="Use the paper eq.7 dynamic IBVS iteration schedule."),

    # --- shared loop knobs (both kinds) -------------------------------------
    Field("dt", "DT", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=1.0, vmin=0, vmin_inclusive=False, section="loop_shared",
          help="Integration timestep (>0)."),
    Field("min_features", "MIN_FEATURES", INT, (TRAJECTORY, SERVO_FRAMES),
          default=3, vmin=1, applies=IBVS, section="loop_shared",
          help="Minimum matched features to keep servoing.",
          aliases=("min_matches",)),
    Field("ratio", "RATIO", INT, (TRAJECTORY, SERVO_FRAMES),
          default=1, vmin=0, applies=IBVS, section="loop_shared",
          help="Re-match ratio (0 = match once)."),

    # --- stopping / divergence (both kinds) ---------------------------------
    Field("stop_residual_px", "STOP_RESIDUAL_PX", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=0.5, vmin=0, vmin_inclusive=False, applies=IBVS,
          section="stopping",
          help="IBVS stop: RMS reprojection error (px, >0)."),
    Field("stop_mse_per_px", "STOP_MSE_PER_PX", FLOAT, (TRAJECTORY, SERVO_FRAMES),
          default=2.0e-6, vmin=0, vmin_inclusive=False, applies=PHOTOMETRIC,
          section="stopping",
          help="Photometric stop: mean(e^2)/px on [0,1] (>0)."),
    Field("stop_plateau_iters", "STOP_PLATEAU_ITERS", OPT_INT,
          (TRAJECTORY, SERVO_FRAMES), default=None, optional=True, vmin=1,
          section="stopping",
          help="Stop after N plateauing iterations (blank=disabled)."),
    Field("stop_plateau_ratio", "STOP_PLATEAU_RATIO", FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=None, section="advanced",
          help="Relative-improvement threshold for the plateau detector."),
    Field("diverge_residual_px", "DIVERGE_RESIDUAL_PX", FLOAT, (TRAJECTORY,),
          default=1000.0, vmin=0, vmin_inclusive=False, applies=IBVS,
          section="stopping",
          help="IBVS abort if final residual exceeds px (>0)."),
    Field("diverge_mse_per_px", "DIVERGE_MSE_PER_PX", FLOAT, (TRAJECTORY,),
          default=0.01, vmin=0, vmin_inclusive=False, applies=PHOTOMETRIC,
          section="stopping",
          help="Photometric abort if final MSE/px exceeds (>0)."),
    Field("diverge_translation_error", "DIVERGE_TRANSLATION_ERROR", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=0.5, optional=True,
          vmin=0, vmin_inclusive=False, section="stopping",
          help="Abort if final translation error exceeds (blank=disabled).",
          aliases=("diverge_tr_error",)),
    Field("diverge_rotation_error_deg", "DIVERGE_ROTATION_ERROR_DEG", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=30.0, optional=True,
          vmin=0, vmin_inclusive=False, section="stopping",
          help="Abort if final rotation error exceeds deg (blank=disabled).",
          aliases=("diverge_rot_error", "diverge_rot_error_deg")),

    # --- trajectory: run bookkeeping / viz ----------------------------------
    Field("continue_on_task_failure", "CONTINUE_ON_TASK_FAILURE", BOOL,
          (TRAJECTORY,), default=False, section="stopping",
          help="Cut to target pose and continue after a failed task.",
          aliases=("continue_after_failure", "cut_on_failure")),
    Field("save_task_viz", "SAVE_TASK_VIZ", BOOL, (TRAJECTORY,),
          default=True, section="stopping",
          help="Save per-task visualizations.",
          aliases=("save_viz", "task_viz")),
    Field("task_viz_every", "TASK_VIZ_EVERY", INT, (TRAJECTORY,),
          default=1, vmin=0, section="stopping",
          help="Save task viz every N tasks.",
          aliases=("task_viz_stride",)),
    Field("run_tag", "RUN_TAG", OPT_STR, (TRAJECTORY,),
          default="GS_SIFT_INTRINSIC", optional=True, section="stopping",
          help="Run tag (blank=auto).", aliases=("tag",)),

    # --- interaction-matrix guards (advanced; both kinds) -------------------
    Field("min_interaction_rank", "MIN_INTERACTION_RANK", INT,
          (TRAJECTORY, SERVO_FRAMES), default=6, vmin=1, vmax=6,
          section="advanced", help="Minimum interaction-matrix rank, [1,6]."),
    Field("max_interaction_condition", "MAX_INTERACTION_CONDITION", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=1.0e8, optional=True,
          vmin=0, vmin_inclusive=False, section="advanced",
          help="Max interaction-matrix condition number (>0 or blank)."),
    Field("max_translation_step", "MAX_TRANSLATION_STEP", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=0.5, optional=True,
          vmin=0, vmin_inclusive=False, section="advanced",
          help="Soft per-step translation clamp (>0 or blank)."),
    Field("max_rotation_step_deg", "MAX_ROTATION_STEP_DEG", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=30.0, optional=True,
          vmin=0, vmin_inclusive=False, section="advanced",
          help="Soft per-step rotation clamp in degrees (>0 or blank)."),
    Field("hard_translation_step", "HARD_TRANSLATION_STEP", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=5.0, optional=True,
          vmin=0, vmin_inclusive=False, section="advanced",
          help="Hard per-step translation clamp (>0 or blank)."),
    Field("hard_rotation_step_deg", "HARD_ROTATION_STEP_DEG", OPT_FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=180.0, optional=True,
          vmin=0, vmin_inclusive=False, section="advanced",
          help="Hard per-step rotation clamp in degrees (>0 or blank)."),
    Field("adaptive_step_fraction", "ADAPTIVE_STEP_FRACTION", FLOAT,
          (TRAJECTORY, SERVO_FRAMES), default=0.25, vmin=0, section="advanced",
          help="Adaptive step fraction (>=0)."),
)


# --- derived lookups --------------------------------------------------------

def fields_for(kind: str) -> list[Field]:
    return [f for f in FIELDS if f.applies_to(kind)]


def field_by_name(name: str) -> Optional[Field]:
    return _BY_NAME.get(name)


def key_map(kind: str) -> dict[str, str]:
    """snake_case config key -> UPPER module-global, for one kind."""
    return {f.name: f.const for f in FIELDS if f.applies_to(kind)}


def alias_map(kind: str) -> dict[str, str]:
    """alias -> canonical name, for one kind (kind aliases win over common)."""
    aliases: dict[str, str] = {}
    for f in FIELDS:
        if not f.applies_to(kind):
            continue
        for a in f.aliases:
            aliases[a] = f.name
    return aliases


def defaults_for(kind: str) -> dict[str, str]:
    """UPPER const -> default value, for one kind."""
    return {f.const: f.default_for(kind) for f in FIELDS if f.applies_to(kind)}


_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}


def _sanity_check() -> None:
    """Guard against duplicate names/consts within a kind at import time."""
    for kind in (TRAJECTORY, SERVO_FRAMES):
        names: dict[str, str] = {}
        consts: dict[str, str] = {}
        for f in fields_for(kind):
            if f.name in names:
                raise ValueError(f"duplicate field name {f.name!r} in {kind}")
            if f.const in consts:
                raise ValueError(f"duplicate const {f.const!r} in {kind}")
            names[f.name] = f.const
            consts[f.const] = f.name


_sanity_check()
