"""Smoke render + servo test (mesh + GS).

Use via the CLI:
    python cli.py smoke
"""

import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEED = 42
FEATURE_METHOD = "xfeat"
SERVO_ITERATIONS = 100
# Velocity built inside run() once numpy is imported there.


def add_arguments(parser):
    parser.add_argument(
        "--scene",
        default="kitchen",
        help="Scene folder under DATA/. Default: kitchen.",
    )


def run(args):
    import numpy as np
    from PIL import Image
    from dataset import load_colmap
    from features import FeatureMatcher, filter_matches
    from scenes.gs import GSScene
    from scenes.mesh import MeshScene
    from servo import FixedVelocityController, run_servo_loop
    from viz import save_match_visualization

    SERVO_VELOCITY = np.array([0.0, 0.0, 0.005, 0.0, 0.0, 0.0], dtype=np.float32)

    def load_real_rgb(rgb_path, camera):
        real_rgb = Image.open(rgb_path).convert("RGB")
        real_rgb = real_rgb.resize((camera.W, camera.H))
        return np.array(real_rgb, dtype=np.float32) / 255.0

    def save_render_matches(rendered, real, camera, matcher, output_path):
        kpts1, kpts2 = matcher.match(rendered, real)
        kpts1_f, kpts2_f, H_matrix, _, _, inliers = filter_matches(kpts1, kpts2, camera)
        kpts1_removed = kpts1[~inliers]
        kpts2_removed = kpts2[~inliers]

        matches_removed = [(i, i) for i in range(len(kpts1_removed))]
        matches_kept = [(i, i) for i in range(len(kpts1_f))]

        save_match_visualization(
            rendered, real,
            kpts1_removed, kpts2_removed, matches_removed,
            kpts1_f, kpts2_f, matches_kept,
            output_path,
        )
        return len(kpts1), len(kpts1_f), H_matrix

    def run_servo_smoke(scene, camera, target, matcher, output_dir):
        controller = FixedVelocityController(SERVO_VELOCITY)
        result = run_servo_loop(
            scene, camera, target, controller,
            iterations=SERVO_ITERATIONS,
            visualization_dir=output_dir,
            matcher=matcher,
        )
        last = result["history"][-1]
        return len(result["history"]), last["num_matches"], last["num_inliers"]

    random.seed(SEED)
    np.random.seed(SEED)

    scene_name = getattr(args, "scene", "kitchen")
    scene_dir = PROJECT_ROOT / "DATA" / scene_name
    matcher = FeatureMatcher(method=FEATURE_METHOD)
    print(f"Using {FEATURE_METHOD} feature matcher")

    colmap_data = load_colmap(scene_dir)
    if not colmap_data:
        raise RuntimeError(f"No COLMAP images in {scene_dir}")

    print("Running Mesh test...")
    camera, rgb_path = random.choice(colmap_data)

    mesh_scene = MeshScene(scene_dir / "mesh.ply")
    rendered_mesh = mesh_scene.render(camera)
    real_mesh = load_real_rgb(rgb_path, camera)

    n_mesh, n_mesh_f, H_mesh = save_render_matches(
        rendered_mesh, real_mesh, camera, matcher, "output_mesh.png"
    )
    print(
        f"Saved output_mesh.png ({n_mesh} matches, {n_mesh_f} RANSAC inliers, "
        f"H={'ok' if H_mesh is not None else 'failed'})"
    )
    n_steps, n_matches, n_inliers = run_servo_smoke(
        mesh_scene, camera, real_mesh, matcher,
        PROJECT_ROOT / "servo_mesh",
    )
    print(
        f"Saved servo_mesh/ ({n_steps} iterations, last frame "
        f"{n_matches} matches, {n_inliers} RANSAC inliers)"
    )

    print("Running GS test...")
    camera_gs, rgb_path_gs = random.choice(colmap_data)

    gs_scene = GSScene(scene_dir / "gs.ply")
    rendered_gs = gs_scene.render(camera_gs)
    real_gs = load_real_rgb(rgb_path_gs, camera_gs)

    n_gs, n_gs_f, H_gs = save_render_matches(
        rendered_gs, real_gs, camera_gs, matcher, "output_gs.png"
    )
    print(
        f"Saved output_gs.png ({n_gs} matches, {n_gs_f} RANSAC inliers, "
        f"H={'ok' if H_gs is not None else 'failed'})"
    )
    n_steps_gs, n_matches_gs, n_inliers_gs = run_servo_smoke(
        gs_scene, camera_gs, real_gs, matcher,
        PROJECT_ROOT / "servo_gs",
    )
    print(
        f"Saved servo_gs/ ({n_steps_gs} iterations, last frame "
        f"{n_matches_gs} matches, {n_inliers_gs} RANSAC inliers)"
    )
