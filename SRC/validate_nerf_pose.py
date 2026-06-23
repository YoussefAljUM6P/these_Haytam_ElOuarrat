"""Regression check for NeRFScene's OpenCV -> nerfstudio pose mapping.

Verifies that NeRFScene._opencv_to_nerfstudio_c2w reproduces the camera poses
that nerfstudio's own ColmapDataParser produces for the same images. If this
passes, the in-framework NeRF renderer is using the exact same camera convention
as `ns-render`, so any remaining servoing discrepancy is not a pose-mapping bug.

Run with the env that has nerfstudio installed, e.g.:
    /home/haytam-elourrat/miniconda3/envs/viservo/bin/python \
        SRC/validate_nerf_pose.py DATA/bonsai
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_colmap  # noqa: E402
from scenes.nerf import NeRFScene  # noqa: E402

TOL = 1e-3


def validate(scene_dir):
    scene_dir = Path(scene_dir)
    scene = NeRFScene(scene_dir)

    # Ground truth: nerfstudio's own dataparser cameras (normalized space).
    dpo = scene.pipeline.datamanager.train_dataparser_outputs
    gt_c2w = dpo.cameras.camera_to_worlds.cpu().numpy()  # (N, 3, 4)
    gt = {Path(f).name: gt_c2w[i] for i, f in enumerate(dpo.image_filenames)}

    errs = []
    for cam, rgb_path in load_colmap(scene_dir):
        if rgb_path.name not in gt:
            continue
        mine = scene._opencv_to_nerfstudio_c2w(cam.T_world_cam)  # (3, 4)
        errs.append(np.abs(mine - gt[rgb_path.name]).max())

    if not errs:
        raise SystemExit("No overlapping frames between COLMAP and dataparser.")

    errs = np.array(errs)
    max_err = errs.max()
    print(
        f"{scene_dir}: matched {len(errs)} frames | "
        f"max err {max_err:.3e} | mean {errs.mean():.3e}"
    )
    ok = max_err < TOL
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "DATA/bonsai"
    sys.exit(0 if validate(scene) else 1)
