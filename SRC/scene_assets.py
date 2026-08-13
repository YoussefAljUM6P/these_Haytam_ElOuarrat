"""Lightweight renderer asset probes."""

from pathlib import Path


def nerfstudio_checkpoint_configs(scene_dir):
    """Return config.yml files with adjacent nerfstudio checkpoints."""
    scene_dir = Path(scene_dir)
    configs = [
        config_path
        for config_path in scene_dir.rglob("config.yml")
        if list((config_path.parent / "nerfstudio_models").glob("step-*.ckpt"))
    ]
    configs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return configs


def has_nerfstudio_checkpoint(scene_dir):
    return bool(nerfstudio_checkpoint_configs(scene_dir))


def nerfstudio_checkpoint_error(scene_dir):
    """Return a user-facing missing-asset error, or None when runnable."""
    scene_dir = Path(scene_dir)
    if has_nerfstudio_checkpoint(scene_dir):
        return None

    configs = list(scene_dir.rglob("config.yml"))
    if not configs:
        return f"Missing nerfstudio config.yml under {scene_dir}"

    expected = ", ".join(
        str(config_path.parent / "nerfstudio_models" / "step-*.ckpt")
        for config_path in configs
    )
    return f"Missing nerfstudio checkpoint; expected {expected}"
