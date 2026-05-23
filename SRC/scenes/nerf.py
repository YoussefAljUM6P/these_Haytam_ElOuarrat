"""Nerfstudio-backed NeRF scene wrapper.

Mirrors the MeshScene / GSScene interface so the same servo loop can drive
a NeRF-rendered target.

Finds the most recent nerfstudio output by searching for:
    <scene_dir>/**/config.yml  (where a nerfstudio_models dir is adjacent)

Requires `dataparser_transforms.json` next to config.yml for coordinate mapping.
"""

import json
from pathlib import Path

import numpy as np
import torch


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
        # output_dir / experiment_name / method_name / timestamp / relative_model_dir
        # must equal <config_path.parent>/nerfstudio_models. Anchor output_dir
        # three parents above config_path so the structure resolves correctly.
        config.output_dir = self._config_path.parents[3].resolve()

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
        candidates = list(scene_dir.rglob("config.yml"))
        valid = [c for c in candidates if (c.parent / "nerfstudio_models").exists()]
        if not valid:
            raise FileNotFoundError(f"No nerfstudio config.yml found under {scene_dir}.")
        valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return valid[0]

    def _opencv_to_nerfstudio_c2w(self, T_world_cam_opencv):
        """OpenCV cam2world -> nerfstudio normalized-space cam2world (3x4)."""
        c2w = T_world_cam_opencv.copy()
        
        # Scale only the translation (nerfstudio's dataparser does this on poses).
        c2w[:3, 3] *= self.scale
        
        # Apply the nerfstudio dataparser world remap.
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = self.transform
        
        c2w_ns = T @ c2w
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

    def render(self, camera):
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
            self._last_depth = outputs["depth"].squeeze(0).cpu().numpy().astype(np.float32)

        rgb = outputs["rgb"].squeeze(0).cpu().numpy().astype(np.float32)
        return rgb

    def render_depth(self, camera=None):
        if camera is None:
            camera = self._last_camera
        if camera is None:
            raise NotImplementedError("NeRFScene.render_depth requires a camera")

        key = self._camera_key(camera)
        if self._last_render_key == key and self._last_depth is not None:
            depth = self._last_depth.copy()
        else:
            ns_cam = self._make_ns_camera(camera)

            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(ns_cam)

            if "depth" not in outputs:
                raise RuntimeError("nerfstudio model outputs have no 'depth' key")

            depth = outputs["depth"].squeeze(0).cpu().numpy().astype(np.float32)

        depth = depth.squeeze(-1)  # H, W
        
        # nerfstudio depth is in normalized space; undo dataparser scale
        if self.scale > 0.0:
            depth = depth / self.scale
            
        return depth
