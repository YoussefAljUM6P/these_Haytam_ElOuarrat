import numpy as np
import torch

from camera import Camera
from scenes.gs import GSScene


def make_camera(T_world_cam=None):
    if T_world_cam is None:
        T_world_cam = np.eye(4, dtype=np.float32)
    return Camera(T_world_cam, fx=80.0, fy=80.0, cx=32.0, cy=32.0, H=64, W=64)


def make_synthetic_gs_scene(xyz, opacities, scales=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if scales is None:
        scales = np.full_like(xyz, 0.18, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1, 1)

    scene = GSScene.__new__(GSScene)
    scene.xyz = torch.tensor(xyz).float().cuda()
    scene.opacities = torch.tensor(opacities).float().cuda()
    scene.scales = torch.tensor(scales).float().cuda()
    scene.rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(xyz)).float().cuda()
    scene.sh_degree = 0
    scene._shs = torch.zeros((len(xyz), 1, 3), device="cuda")
    scene._means2D = torch.zeros_like(scene.xyz)
    scene._depth_colors = torch.empty_like(scene.xyz)
    scene._depth_colors[:, 2] = 1.0
    scene._bg = torch.zeros(3, device="cuda")
    scene._proj_cache = {}
    scene._last_camera = None
    scene._last_depth_key = None
    scene._last_depth = None
    return scene


def covered(depth):
    mask = np.isfinite(depth) & (depth > 0.0)
    if not mask.any():
        raise AssertionError("Synthetic Gaussian did not cover any pixels")
    return mask


def assert_single_gaussian_depth_is_z(z, opacity, camera=None, xyz=None):
    if camera is None:
        camera = make_camera()
    if xyz is None:
        xyz = [[0.0, 0.0, z]]

    scene = make_synthetic_gs_scene(xyz, [opacity])
    depth = scene.render_depth(camera)
    mask = covered(depth)
    max_error = np.max(np.abs(depth[mask] - z))
    if max_error > 1e-4:
        raise AssertionError(
            f"Single Gaussian depth should be {z}, got max error {max_error}"
        )
    return depth, mask


def test_single_gaussian_opacity_invariance():
    depth_low, mask_low = assert_single_gaussian_depth_is_z(2.0, 0.25)
    depth_high, mask_high = assert_single_gaussian_depth_is_z(2.0, 0.90)

    mask = mask_low & mask_high
    max_delta = np.max(np.abs(depth_low[mask] - depth_high[mask]))
    if max_delta > 1e-4:
        raise AssertionError(
            "Changing opacity changed single-Gaussian depth by "
            f"{max_delta}; depth is not normalized by alpha"
        )


def test_camera_space_z_transform():
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[2, 3] = 1.0
    camera = make_camera(T_world_cam)
    assert_single_gaussian_depth_is_z(
        z=2.0,
        opacity=0.75,
        camera=camera,
        xyz=[[0.0, 0.0, 3.0]],
    )


def test_two_layer_depth_bounds():
    scene = make_synthetic_gs_scene(
        xyz=[
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
        ],
        opacities=[0.55, 0.55],
    )
    depth = scene.render_depth(make_camera())
    mask = covered(depth)
    min_depth = float(depth[mask].min())
    max_depth = float(depth[mask].max())
    if min_depth < 2.0 - 1e-4 or max_depth > 4.0 + 1e-4:
        raise AssertionError(
            "Two-layer alpha-weighted mean depth must stay between visible "
            f"layer depths [2, 4], got [{min_depth}, {max_depth}]"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GS depth validation requires CUDA")

    test_single_gaussian_opacity_invariance()
    test_camera_space_z_transform()
    test_two_layer_depth_bounds()
    print("GS render_depth validation passed")


if __name__ == "__main__":
    main()
