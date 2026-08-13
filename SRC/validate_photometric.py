import tempfile
from pathlib import Path

import numpy as np
import torch

from camera import Camera
from photometric import PhotometricController, PhotometricControllerTorch
from servo import run_servo_loop


class ConstantDepthScene:
    def __init__(self, depth=2.0):
        self.depth = float(depth)
        self.calls = 0

    def render_depth(self, camera=None):
        if camera is None:
            raise AssertionError("photometric controller must pass the active camera")
        self.calls += 1
        return np.full((camera.H, camera.W), self.depth, dtype=np.float32)


class ZeroDepthScene:
    def render_depth(self, camera=None):
        return np.zeros((camera.H, camera.W), dtype=np.float32)


class TensorScene(ConstantDepthScene):
    def __init__(self, image, depth=2.0):
        super().__init__(depth=depth)
        self.image = np.asarray(image, dtype=np.float32)

    def render_tensor(self, camera):
        return torch.as_tensor(self.image, dtype=torch.float32)

    def render(self, camera):
        return self.image.copy()


class DummyMatcher:
    def match(self, img1, img2):
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty.copy()


def make_camera(height=48, width=64):
    return Camera(
        np.eye(4, dtype=np.float32),
        fx=60.0,
        fy=58.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        H=height,
        W=width,
    )


def make_texture(height=48, width=64):
    rng = np.random.default_rng(7)
    base = rng.random((height, width), dtype=np.float32)
    image = np.stack([
        0.10 + 0.80 * base,
        0.10 + 0.80 * np.roll(base, 1, axis=0),
        0.10 + 0.80 * np.roll(base, 1, axis=1),
    ], axis=-1)
    return image.astype(np.float32, copy=False)


def make_controller(scene, **kwargs):
    defaults = dict(
        scene=scene,
        gain=0.01,
        sigma_blur=0.0,
        use_gzn=False,
        grad_percentile=0.0,
        max_pixels=0,
        use_huber=False,
        use_intrinsic_depth=True,
        method="lm",
        stop_mse_per_px=1.0e-12,
        min_interaction_rank=6,
        max_interaction_condition=None,
        device="cpu",
    )
    defaults.update(kwargs)
    return PhotometricControllerTorch(**defaults)


def assert_valid_velocity(controller, velocity):
    if velocity.shape != (6,) or not np.isfinite(velocity).all():
        raise AssertionError(f"invalid velocity: {velocity}")
    info = controller.last_info
    if not info.get("measurement_valid", False):
        raise AssertionError(f"measurement should be valid, got {info}")
    if info.get("num_pixels_used", 0) < 6:
        raise AssertionError(f"expected dense pixels, got {info}")


def test_aliases_are_implemented():
    if PhotometricController is not PhotometricControllerTorch:
        raise AssertionError("PhotometricController alias should resolve to Torch backend")


def test_zero_residual_returns_zero_velocity_and_stops():
    camera = make_camera()
    image = make_texture(camera.H, camera.W)
    scene = ConstantDepthScene()
    controller = make_controller(scene, stop_ssd=1.0e-9)

    velocity = controller(image, image.copy(), camera, iteration=0)
    assert_valid_velocity(controller, velocity)
    if not np.allclose(velocity, 0.0, atol=1.0e-7):
        raise AssertionError(f"zero residual should return zero velocity, got {velocity}")
    if controller.should_stop() != "photometric_ssd_below_threshold":
        raise AssertionError(f"zero residual should stop, got {controller.last_info}")
    if scene.calls != 1:
        raise AssertionError("controller did not request current-camera intrinsic depth")


def test_shifted_image_returns_finite_nonzero_velocity():
    camera = make_camera()
    rendered = make_texture(camera.H, camera.W)
    target = np.roll(rendered, 1, axis=1).copy()
    controller = make_controller(ConstantDepthScene())

    velocity = controller(rendered, target, camera, iteration=0)
    assert_valid_velocity(controller, velocity)
    if np.linalg.norm(velocity) <= 0.0:
        raise AssertionError("nonzero photometric error should produce nonzero velocity")
    if controller.last_info.get("interaction_rank", 0) < 6:
        raise AssertionError(f"interaction matrix should be full rank: {controller.last_info}")


def test_invalid_depth_faults_without_fallback():
    camera = make_camera()
    image = make_texture(camera.H, camera.W)
    controller = make_controller(ZeroDepthScene())

    velocity = controller(image, np.roll(image, 1, axis=0), camera, iteration=0)
    if not np.allclose(velocity, 0.0):
        raise AssertionError(f"invalid depth should zero velocity, got {velocity}")
    if controller.last_info.get("fault_reason") != "measurement_invalid_not_enough_pixels":
        raise AssertionError(f"expected depth fault, got {controller.last_info}")
    if controller.should_stop() != "measurement_invalid_not_enough_pixels":
        raise AssertionError("fault reason should propagate through should_stop()")


def test_huber_weights_reduce_outlier_cost():
    camera = make_camera()
    rendered = make_texture(camera.H, camera.W)
    target = rendered.copy()
    target[8:18, 8:18, :] = 1.0 - target[8:18, 8:18, :]
    controller = make_controller(
        ConstantDepthScene(),
        use_huber=True,
        huber_k=0.01,
        stop_mse_per_px=0.0,
    )

    velocity = controller(rendered, target, camera, iteration=0)
    assert_valid_velocity(controller, velocity)
    info = controller.last_info
    if info.get("huber_k_active") != 0.01:
        raise AssertionError(f"expected fixed Huber k, got {info}")
    if info["weighted_residual_ssd"] > info["residual_ssd"] + 1.0e-9:
        raise AssertionError(f"Huber weighting should not increase SSD: {info}")


def test_servo_loop_uses_tensor_render_and_photometric_viz():
    camera = make_camera()
    image = make_texture(camera.H, camera.W)
    scene = TensorScene(image)
    controller = make_controller(scene, stop_ssd=1.0e-9)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_servo_loop(
            scene,
            camera,
            image.copy(),
            controller,
            iterations=2,
            visualization_dir=tmpdir,
            matcher=DummyMatcher(),
            viz_iter=1,
        )
        if not result.get("timing", {}).get("tensor_render", False):
            raise AssertionError("servo loop did not use render_tensor fast path")
        if result["stop_reason"] != "photometric_ssd_below_threshold":
            raise AssertionError(f"unexpected stop reason: {result['stop_reason']}")
        expected = Path(tmpdir) / "iter_0000_photometric.png"
        if not expected.exists():
            raise AssertionError(f"missing photometric visualization {expected}")


def main():
    test_aliases_are_implemented()
    test_zero_residual_returns_zero_velocity_and_stops()
    test_shifted_image_returns_finite_nonzero_velocity()
    test_invalid_depth_faults_without_fallback()
    test_huber_weights_reduce_outlier_cost()
    test_servo_loop_uses_tensor_render_and_photometric_viz()
    print("Photometric controller validation passed")


if __name__ == "__main__":
    main()
