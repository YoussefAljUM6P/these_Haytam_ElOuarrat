import numpy as np


def get_depth(image, scene=None, use_intrinsic=False):
    if not use_intrinsic:
        raise NotImplementedError(
            "Learned (MoGe) depth has been removed; use intrinsic depth via "
            "scene.render_depth() (use_intrinsic=True)."
        )

    render_depth = None if scene is None else getattr(scene, "render_depth", None)
    if not callable(render_depth):
        raise NotImplementedError("Intrinsic depth requires a scene with render_depth()")

    return np.asarray(render_depth(), dtype=np.float32)
