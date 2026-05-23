"""JSON config helpers for SERVIS experiment entrypoints."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "CONFIGS"


_PHOTOMETRIC_KEYS = {
    "controller": "CONTROLLER",
    "sigma_blur": "SIGMA_BLUR",
    "use_gzn": "USE_GZN",
    "grad_percentile": "GRAD_PERCENTILE",
    "photometric_max_pixels": "PHOTOMETRIC_MAX_PIXELS",
    "use_huber": "USE_HUBER",
    "huber_k": "HUBER_K",
}

TRAJECTORY_CONFIG_KEYS = {
    "datasets": "DATASETS",
    "renderer": "RENDERER",
    "nerf_render_scale": "NERF_RENDER_SCALE",
    "stride": "STRIDE",
    "mini_iterations": "MINI_ITERATIONS",
    "dt": "DT",
    "depth_mode": "DEPTH_MODE",
    "feature_method": "FEATURE_METHOD",
    "gain_ibvs": "GAIN_IBVS",
    "gain_photo": "GAIN_PHOTO",
    "min_features": "MIN_FEATURES",
    "ratio": "RATIO",
    "start_index": "START_INDEX",
    "max_pairs": "MAX_PAIRS",
    "stop_residual_px": "STOP_RESIDUAL_PX",
    "stop_mse_per_px": "STOP_MSE_PER_PX",
    "dynamic_ibvs_iters": "DYNAMIC_IBVS_ITERS",
    "rpe_delta": "RPE_DELTA",
    "run_tag": "RUN_TAG",
    "save_task_viz": "SAVE_TASK_VIZ",
    "task_viz_every": "TASK_VIZ_EVERY",
    **_PHOTOMETRIC_KEYS,
}

SERVO_FRAMES_CONFIG_KEYS = {
    "scene_dir": "SCENE_DIR",
    "renderer": "RENDERER",
    "start_index": "START_INDEX",
    "index_away": "INDEX_AWAY",
    "target_index": "TARGET_INDEX",
    "iterations": "ITERATIONS",
    "dt": "DT",
    "depth_mode": "DEPTH_MODE",
    "feature_method": "FEATURE_METHOD",
    "viz_iter": "VIZ_ITER",
    "gain_ibvs": "GAIN_IBVS",
    "gain_photo": "GAIN_PHOTO",
    "min_features": "MIN_FEATURES",
    "ratio": "RATIO",
    "run_name": "RUN_NAME",
    "stop_residual_px": "STOP_RESIDUAL_PX",
    "stop_mse_per_px": "STOP_MSE_PER_PX",
    **_PHOTOMETRIC_KEYS,
}

COMMON_ALIASES = {
    "feature": "feature_method",
    "matcher": "feature_method",
    "depth": "depth_mode",
    "viz_every": "viz_iter",
    "visualize_every": "viz_iter",
    "min_matches": "min_features",
    "gain": "gain_ibvs",
}

KIND_ALIASES = {
    "trajectory": {
        "dataset": "datasets",
        "scene": "datasets",
        "scenes": "datasets",
        "iters": "mini_iterations",
        "iterations": "mini_iterations",
        "max_tasks": "max_pairs",
        "pairs": "max_pairs",
        "save_viz": "save_task_viz",
        "task_viz": "save_task_viz",
        "task_viz_stride": "task_viz_every",
        "tag": "run_tag",
        "nerf_scale": "nerf_render_scale",
        "render_scale": "nerf_render_scale",
    },
    "servo_frames": {
        "dataset": "scene_dir",
        "scene": "scene_dir",
        "target": "target_index",
        "start": "start_index",
        "away": "index_away",
        "iters": "iterations",
        "tag": "run_name",
    },
}

RENDERERS = {"mesh", "gs", "nerf"}
DEPTH_MODES = {"learned", "intrinsic"}
CONTROLLERS = {"ibvs", "photometric", "photometric_torch"}

INT_KEYS = {
    "stride",
    "mini_iterations",
    "iterations",
    "min_features",
    "ratio",
    "start_index",
    "index_away",
    "viz_iter",
    "rpe_delta",
    "task_viz_every",
    "photometric_max_pixels",
}
OPTIONAL_INT_KEYS = {"max_pairs", "target_index"}
FLOAT_KEYS = {
    "dt",
    "gain_ibvs",
    "gain_photo",
    "nerf_render_scale",
    "stop_residual_px",
    "stop_mse_per_px",
    "sigma_blur",
    "grad_percentile",
}
OPTIONAL_FLOAT_KEYS = {"huber_k"}
BOOL_KEYS = {"save_task_viz", "use_gzn", "use_huber", "dynamic_ibvs_iters"}
OPTIONAL_STR_KEYS = {"run_tag", "run_name"}


def resolve_config_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    return CONFIG_ROOT / path


def load_config_file(path, expected_kind=None):
    config_path = resolve_config_path(path)
    with open(config_path) as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    kind = config.get("kind")
    if expected_kind is not None and kind not in (None, expected_kind):
        raise ValueError(
            f"Config kind must be {expected_kind!r}, got {kind!r} in {config_path}"
        )

    return config


def parse_literal(text):
    text = str(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def canonical_config_key(raw_key, kind):
    key = str(raw_key).strip().lower().replace("-", "_")
    key = COMMON_ALIASES.get(key, key)
    key = KIND_ALIASES.get(kind, {}).get(key, key)
    return key


def parse_cli_overrides(overrides, key_map, kind):
    config = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Expected --set KEY=VALUE, got {override!r}"
            )
        key, value = override.split("=", 1)
        config[key] = parse_literal(value)
    return normalize_config(config, key_map, kind)


def load_cli_config(config_path, overrides, key_map, kind):
    config = {}
    if config_path is not None:
        config.update(normalize_config(
            load_config_file(config_path, expected_kind=kind),
            key_map,
            kind,
        ))
    config.update(parse_cli_overrides(overrides, key_map, kind))
    return config


def apply_config(config, module_globals, key_map):
    for key, value in config.items():
        module_globals[key_map[key]] = value


def normalize_config(config, key_map, kind):
    normalized = {}
    for raw_key, raw_value in config.items():
        key = canonical_config_key(raw_key, kind)
        if key == "kind":
            continue
        if key not in key_map:
            allowed = ", ".join(sorted(key_map))
            raise KeyError(
                f"Unknown {kind} config key {raw_key!r}. Allowed keys: {allowed}"
            )
        normalized[key] = coerce_config_value(key, raw_value)
    return normalized


def coerce_config_value(key, value):
    if key == "datasets":
        return coerce_str_list(value)
    if key == "scene_dir":
        return coerce_scene_dir(value)
    if key in OPTIONAL_INT_KEYS:
        if is_null_value(value):
            return None
        value = int(value)
        if value < 1:
            raise ValueError(f"{key} must be >= 1 or null")
        return value
    if key in INT_KEYS:
        value = int(value)
        validate_int_value(key, value)
        return value
    if key in FLOAT_KEYS:
        value = float(value)
        if key == "dt" and value <= 0.0:
            raise ValueError("dt must be > 0")
        if key == "nerf_render_scale" and value <= 0.0:
            raise ValueError("nerf_render_scale must be > 0")
        if key == "sigma_blur" and value < 0.0:
            raise ValueError("sigma_blur must be >= 0")
        if key == "grad_percentile" and not (0.0 <= value < 100.0):
            raise ValueError("grad_percentile must be in [0, 100)")
        return value
    if key in OPTIONAL_FLOAT_KEYS:
        if is_null_value(value):
            return None
        return float(value)
    if key in BOOL_KEYS:
        return coerce_bool(value)
    if key == "controller":
        value = str(value).lower()
        if value not in CONTROLLERS:
            raise ValueError(f"controller must be one of {sorted(CONTROLLERS)}")
        return value
    if key in OPTIONAL_STR_KEYS:
        if is_null_value(value):
            return None
        return str(value)
    if key == "renderer":
        value = str(value).lower()
        if value not in RENDERERS:
            raise ValueError(f"renderer must be one of {sorted(RENDERERS)}")
        return value
    if key == "depth_mode":
        value = str(value).lower()
        if value not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {sorted(DEPTH_MODES)}")
        return value
    if key == "feature_method":
        return str(value).lower()
    return value


def coerce_str_list(value):
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError("datasets must be a string or list of strings")

    items = [item for item in items if item]
    if not items:
        raise ValueError("datasets must not be empty")
    return items


def coerce_scene_dir(value):
    if value is None:
        raise ValueError("scene_dir must not be null")

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return PROJECT_ROOT / "DATA" / path
    return PROJECT_ROOT / path


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def is_null_value(value):
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def validate_int_value(key, value):
    if key in {"start_index", "min_features", "stride", "rpe_delta"} and value < 1:
        raise ValueError(f"{key} must be >= 1")
    if key in {
        "mini_iterations",
        "iterations",
        "ratio",
        "viz_iter",
        "task_viz_every",
        "photometric_max_pixels",
    } and value < 0:
        raise ValueError(f"{key} must be >= 0")


def format_applied_config(config):
    if not config:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
