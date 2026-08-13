import sys
from pathlib import Path

import numpy as np
import torch

# Keep MoGe on CUDA, but avoid CUDNN_STATUS_NOT_INITIALIZED on this setup.
torch.backends.cudnn.enabled = False


def estimate_depth_moge(image):
    image = np.asarray(image, dtype=np.float32)

    moge_root = Path(__file__).resolve().parent / "third_party" / "moge"
    if not moge_root.exists():
        moge_root = Path(__file__).resolve().parent / "third_party" / "MoGe"
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))

    if not hasattr(estimate_depth_moge, "_model"):
        from moge.model.v2 import MoGeModel

        estimate_depth_moge._model = (
            MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
            .to("cuda")
            .eval()
        )

    image_t = torch.as_tensor(image, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    with torch.inference_mode():
        output = estimate_depth_moge._model.infer(image_t)

    return output["depth"].detach().cpu().numpy().astype(np.float32, copy=False)


def get_depth(image, scene=None, use_intrinsic=False):
    if not use_intrinsic:
        return estimate_depth_moge(image)

    render_depth = None if scene is None else getattr(scene, "render_depth", None)
    if not callable(render_depth):
        raise NotImplementedError("Intrinsic depth requires a scene with render_depth()")

    return np.asarray(render_depth(), dtype=np.float32)
