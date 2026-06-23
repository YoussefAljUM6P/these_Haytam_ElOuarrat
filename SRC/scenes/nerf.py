"""Nerfstudio-backed NeRF scene wrapper.

Mirrors the MeshScene / GSScene interface so the same servo loop can drive
a NeRF-rendered target.

Finds the most recent nerfstudio output by searching for:
    <scene_dir>/**/config.yml
where a nerfstudio_models directory with step-*.ckpt files is adjacent.

Requires `dataparser_transforms.json` next to config.yml for coordinate mapping.
"""

import json
from pathlib import Path

import numpy as np
import torch

from scene_assets import nerfstudio_checkpoint_configs


class NeRFScene:
    def __init__(self, scene_dir):
        self.scene_dir = Path(scene_dir)
        config_path = self._find_config(self.scene_dir)
        
        dp_path = config_path.parent / "dataparser_transforms.json"
        if not dp_path.exists():
            raise FileNotFoundError(
                f"Missing {dp_path}. Required for OpenCV pose -> nerfstudio mapping."
            )
            
        with open(dp_path) as f:
            dp = json.load(f)
            
        self.scale = float(dp.get("scale", 1.0))
        self.transform = np.array(dp["transform"], dtype=np.float32)  # 3x4
        
        # Lazy-load nerfstudio.
        from nerfstudio.utils.eval_utils import eval_setup

        self._config_path = config_path
        _, self.pipeline, _, _ = eval_setup(
            config_path,
            test_mode="inference",
            update_config_callback=self._patch_config_paths,
        )
        self.pipeline.eval()
        self._last_camera = None
        self._last_render_key = None
        self._last_depth = None

    def _patch_config_paths(self, config):
        """Rewrite saved relative paths so inference works regardless of CWD.

        The saved config stores relative ``data`` and ``output_dir`` paths
        based on the directory nerfstudio was launched from. At inference time
        we anchor ``data`` at ``self.scene_dir``, anchor ``output_dir`` so the
        checkpoint dir resolves to the actual ``nerfstudio_models`` folder
        beside ``config.yml``, and remap ``images_path``/``colmap_path`` if the
        recorded subdirectories have moved on disk.
        """
        # nerfstudio rebuilds the checkpoint dir as
        #   output_dir / experiment_name / method_name / timestamp / relative_model_dir
        # from strings baked in at train time, which don't survive the model being
        # moved or symlinked under DATA/ (the timestamp may not even match the real
        # dir name). Collapse the experiment/method/timestamp segments to "." and
        # anchor output_dir at this config's own directory, so the checkpoint dir
        # always resolves to the nerfstudio_models folder beside config.yml.
        config.output_dir = self._config_path.parent.resolve()
        config.experiment_name = "."
        config.method_name = "."
        config.timestamp = "."
        config.relative_model_dir = Path("nerfstudio_models")

        # The datamanager carries its own ``data`` field which, during setup,
        # overrides ``dataparser.data``. Both are baked as absolute training-time
        # paths, so anchor both at the local scene dir or the stale path wins.
        if getattr(config.pipeline.datamanager, "data", None) is not None:
            config.pipeline.datamanager.data = self.scene_dir

        dp = getattr(config.pipeline.datamanager, "dataparser", None)
        if dp is None:
            return config

        dp.data = self.scene_dir

        images_path = getattr(dp, "images_path", None)
        if images_path is not None:
            candidate = self.scene_dir / images_path
            if not candidate.exists():
                fallback = self._find_images_dir()
                if fallback is not None:
                    dp.images_path = fallback.relative_to(self.scene_dir)

        colmap_path = getattr(dp, "colmap_path", None)
        if colmap_path is not None and not (self.scene_dir / colmap_path).exists():
            fallback = self.scene_dir / "sparse" / "0"
            if fallback.exists():
                dp.colmap_path = Path("sparse") / "0"

        return config

    def _find_images_dir(self):
        """Locate a usable images directory under the scene root."""
        preferred = [
            self.scene_dir / "images",
            self.scene_dir / "mesh" / "dense" / "images",
            self.scene_dir / "mesh_v2" / "dense" / "images",
        ]
        for cand in preferred:
            if cand.is_dir():
                return cand
        for cand in self.scene_dir.rglob("images"):
            if cand.is_dir():
                return cand
        return None

    def _find_config(self, scene_dir):
        valid = nerfstudio_checkpoint_configs(scene_dir)
        if not valid:
            candidates = list(scene_dir.rglob("config.yml"))
            if not candidates:
                raise FileNotFoundError(
                    f"No nerfstudio config.yml found under {scene_dir}."
                )
            expected = ", ".join(
                str(config.parent / "nerfstudio_models" / "step-*.ckpt")
                for config in candidates
            )
            raise FileNotFoundError(
                f"No nerfstudio checkpoints found under {scene_dir}. "
                f"Expected {expected}"
            )
        return valid[0]

    def _opencv_to_nerfstudio_c2w(self, T_world_cam_opencv):
        """OpenCV cam2world -> nerfstudio normalized-space cam2world (3x4).

        Replicates nerfstudio's ColmapDataParser pose pipeline exactly:
          1. Flip the camera Y/Z axes (OpenCV -> OpenGL convention).
          2. Left-multiply by the dataparser transform. The saved transform
             already folds in the colmap world-coordinate swap
             (transform = auto_orient @ applied_transform), so it is the full
             original-colmap-world -> nerfstudio-world map.
          3. Scale the translation *after* the transform.
        """
        c2w = T_world_cam_opencv.copy().astype(np.float32)

        # 1. OpenCV camera convention -> OpenGL (nerfstudio): negate y,z axes.
        c2w[0:3, 1:3] *= -1.0

        # 2. Apply the nerfstudio dataparser world transform.
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = self.transform
        c2w_ns = T @ c2w

        # 3. Scale only the translation, matching the dataparser ordering.
        c2w_ns[0:3, 3] *= self.scale

        return c2w_ns[:3, :]

    def _camera_key(self, camera):
        return (
            camera.W,
            camera.H,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.T_world_cam.tobytes(),
        )

    def _make_ns_camera(self, camera):
        from nerfstudio.cameras.cameras import Cameras, CameraType

        c2w = self._opencv_to_nerfstudio_c2w(camera.T_world_cam)

        return Cameras(
            camera_to_worlds=torch.from_numpy(c2w).unsqueeze(0),
            fx=torch.tensor([camera.fx]),
            fy=torch.tensor([camera.fy]),
            cx=torch.tensor([camera.cx]),
            cy=torch.tensor([camera.cy]),
            height=torch.tensor([camera.H]),
            width=torch.tensor([camera.W]),
            camera_type=CameraType.PERSPECTIVE,
        ).to(self.pipeline.device)

    def render_tensor(self, camera):
        """Render an HWC float32 image on the pipeline device."""
        self._last_camera = camera
        key = self._camera_key(camera)

        ns_cam = self._make_ns_camera(camera)
        
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(ns_cam)
            
        if "rgb" not in outputs:
            raise RuntimeError("nerfstudio model outputs have no 'rgb' key")

        self._last_render_key = key
        self._last_depth = None
        if "depth" in outputs:
            self._last_depth = outputs["depth"].squeeze(0)

        return outputs["rgb"].squeeze(0).to(torch.float32)

    def render(self, camera):
        return self.render_tensor(camera).cpu().numpy().astype(
            np.float32, copy=False
        )

    @staticmethod
    def _ray_depth_to_z(depth, camera):
        """Convert nerfstudio ray distance to OpenCV optical-axis Z.

        Nerfstudio depth is an expected distance along each camera ray. The
        SERVIS interaction matrices use OpenCV Zc, so off-axis pixels must be
        scaled by the ray's z-component. MeshScene and GSScene already return
        Zc directly; this conversion is NeRF-specific.
        """
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        x = (xx - float(camera.cx)) / float(camera.fx)
        y = (yy - float(camera.cy)) / float(camera.fy)
        ray_norm = np.sqrt(1.0 + x * x + y * y, dtype=np.float32)
        return (depth / ray_norm).astype(np.float32, copy=False)

    def render_depth(self, camera=None):
        if camera is None:
            camera = self._last_camera
        if camera is None:
            raise NotImplementedError("NeRFScene.render_depth requires a camera")

        key = self._camera_key(camera)
        if self._last_render_key == key and self._last_depth is not None:
            depth = self._last_depth.cpu().numpy().astype(np.float32, copy=False)
        else:
            ns_cam = self._make_ns_camera(camera)

            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(ns_cam)

            if "depth" not in outputs:
                raise RuntimeError("nerfstudio model outputs have no 'depth' key")

            depth = outputs["depth"].squeeze(0).cpu().numpy().astype(
                np.float32, copy=False
            )

        depth = depth.squeeze(-1)  # H, W

        # Nerfstudio depth is in normalized scene units and measured along each
        # ray. Undo scene scale, then convert ray distance to camera-frame Zc.
        if self.scale > 0.0:
            depth = depth / self.scale
        return self._ray_depth_to_z(depth, camera)
