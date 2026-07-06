"""JSON config helpers for SERVIS experiment entrypoints.

All parameter knowledge (names, UPPER module-globals, types, bounds, defaults,
aliases) now lives in :mod:`config_schema`. This module derives the key maps,
coercion/validation, and CLI plumbing from that single registry, and keeps the
same public API the runners and ``cli.py`` already import:

    TRAJECTORY_CONFIG_KEYS / SERVO_FRAMES_CONFIG_KEYS  snake -> UPPER maps
    load_cli_config(config_path, overrides, key_map, kind) -> dict
    apply_config(config, module_globals, key_map)
    format_applied_config(config)
    normalize_config(config, key_map, kind)

plus the schema-driven additions used by the refactor:

    apply_defaults(module_globals, kind)        seed a runner's module-globals
    add_schema_arguments(parser, kind)          generate per-flag argparse
    overrides_from_args(args, kind)             fold named flags + --set together
"""

import argparse
import json
from pathlib import Path

import config_schema as schema
from config_schema import (
    BOOL,
    CHOICE,
    FEATURE,
    FLOAT,
    INT,
    OPT_FLOAT,
    OPT_INT,
    OPT_STR,
    SCENE_DIR,
    STR,
    STR_LIST,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "CONFIGS"

# --- derived from the schema (single source of truth) -----------------------

TRAJECTORY_CONFIG_KEYS = schema.key_map(schema.TRAJECTORY)
SERVO_FRAMES_CONFIG_KEYS = schema.key_map(schema.SERVO_FRAMES)

RENDERERS = set(schema.RENDERERS)
GS_MODELS = set(schema.GS_MODELS)
DEPTH_MODES = set(schema.DEPTH_MODES)
CONTROLLERS = set(schema.CONTROLLERS)

# Map a key_map object back to its kind so the coercion path can find fields.
_KIND_BY_KEYMAP_ID = {
    id(TRAJECTORY_CONFIG_KEYS): schema.TRAJECTORY,
    id(SERVO_FRAMES_CONFIG_KEYS): schema.SERVO_FRAMES,
}


# --- config file loading ----------------------------------------------------


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


# --- key / alias resolution -------------------------------------------------


def canonical_config_key(raw_key, kind):
    key = str(raw_key).strip().lower().replace("-", "_")
    return schema.alias_map(kind).get(key, key)


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


# --- coercion / validation (derived from Field type + bounds) ---------------


def coerce_config_value(key, value):
    field = schema.field_by_name(key)
    if field is None:
        return value
    return coerce_field_value(field, value)


def coerce_field_value(field, value):
    t = field.type
    if t == STR_LIST:
        return coerce_str_list(value)
    if t == SCENE_DIR:
        return coerce_scene_dir(value)
    if t == BOOL:
        return coerce_bool(value)
    if t == CHOICE:
        text = str(value).lower()
        if text not in field.choices:
            raise ValueError(f"{field.name} must be one of {sorted(field.choices)}")
        return text
    if t == FEATURE:
        return str(value).lower()
    if t == STR:
        return str(value)
    if t == OPT_STR:
        return None if is_null_value(value) else str(value)
    if t == OPT_INT:
        if is_null_value(value):
            return None
        return _bounded(field, int(value))
    if t == INT:
        return _bounded(field, int(value))
    if t == OPT_FLOAT:
        if is_null_value(value):
            return None
        return _bounded(field, float(value))
    if t == FLOAT:
        return _bounded(field, float(value))
    return value


def _bounded(field, value):
    if field.vmin is not None:
        if field.vmin_inclusive:
            if value < field.vmin:
                raise ValueError(f"{field.name} must be >= {_num(field.vmin)}")
        elif value <= field.vmin:
            raise ValueError(f"{field.name} must be > {_num(field.vmin)}")
    if field.vmax is not None:
        if field.vmax_inclusive:
            if value > field.vmax:
                raise ValueError(f"{field.name} must be <= {_num(field.vmax)}")
        elif value >= field.vmax:
            raise ValueError(f"{field.name} must be < {_num(field.vmax)}")
    return value


def _num(x):
    return int(x) if float(x).is_integer() else x


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


# --- overrides + loading ----------------------------------------------------


def parse_cli_overrides(overrides, key_map, kind):
    config = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Expected --set KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        config[key] = parse_literal(value)
    return normalize_config(config, key_map, kind)


def load_cli_config(config_path, overrides, key_map, kind):
    config = {}
    if config_path is not None:
        config.update(
            normalize_config(
                load_config_file(config_path, expected_kind=kind), key_map, kind
            )
        )
    config.update(parse_cli_overrides(overrides, key_map, kind))
    return config


def apply_config(config, module_globals, key_map):
    for key, value in config.items():
        module_globals[key_map[key]] = value


def format_applied_config(config):
    if not config:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))


# --- schema-driven CLI plumbing (new) ---------------------------------------


def apply_defaults(module_globals, kind):
    """Seed a runner's UPPER module-globals with the schema defaults.

    Called at runner import so the runtime defaults and the wizard/flag defaults
    come from one place and cannot drift.
    """
    for const, value in schema.defaults_for(kind).items():
        module_globals[const] = value


def _flag_help(field, kind):
    default = field.default_for(kind)
    hint = "blank/none" if default is None else repr(default)
    parts = [field.help or field.name, f"(default: {hint})"]
    if field.applies:
        parts.append(f"[{field.applies} only]")
    return " ".join(parts)


def add_schema_arguments(parser, kind):
    """Generate one --flag per schema field for a runner's subparser.

    Flags default to argparse.SUPPRESS so an unset flag stays absent from the
    namespace; ``overrides_from_args`` then only forwards the ones the user
    actually passed. ``--set`` remains the escape hatch and wins over flags.
    """
    for field in schema.fields_for(kind):
        kwargs = {
            "dest": field.name,
            "default": argparse.SUPPRESS,
            "help": _flag_help(field, kind),
        }
        if field.type == BOOL:
            parser.add_argument(
                field.flag, action=argparse.BooleanOptionalAction, **kwargs
            )
        else:
            metavar = field.name.upper()
            parser.add_argument(field.flag, metavar=metavar, type=str, **kwargs)


def overrides_from_args(args, kind):
    """Fold explicitly-passed named flags + repeated --set into one list.

    Returns a list of ``KEY=VALUE`` strings suitable for ``load_cli_config``.
    Named flags come first, then --set entries, so --set overrides a flag.
    """
    overrides = []
    for field in schema.fields_for(kind):
        if not hasattr(args, field.name):
            continue
        value = getattr(args, field.name)
        if isinstance(value, bool):
            value = "true" if value else "false"
        overrides.append(f"{field.name}={value}")
    overrides.extend(getattr(args, "set", []) or [])
    return overrides
