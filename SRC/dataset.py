import numpy as np
from pathlib import Path
from camera import Camera

IMAGE_DIR_NAMES = ("images", "data")


def _resolve_image_path(scene_dir, image_name):
    for image_dir_name in IMAGE_DIR_NAMES:
        rgb_path = scene_dir / image_dir_name / image_name
        if rgb_path.is_file():
            return rgb_path
    return None


def load_colmap(scene_dir):
    import pycolmap
    import sys
    scene_dir = Path(scene_dir)
    reconstruction = pycolmap.Reconstruction(scene_dir / "sparse" / "0")

    data = []
    skipped = []
    for _, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]

        model_name = camera.model.name
        if model_name == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif model_name == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        elif model_name in ("SIMPLE_RADIAL", "RADIAL"):
            # SIMPLE_RADIAL: [f, cx, cy, k]
            # RADIAL:        [f, cx, cy, k1, k2]
            f = camera.params[0]
            fx = fy = f
            cx = camera.params[1]
            cy = camera.params[2]
        elif model_name in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            # First four params are fx, fy, cx, cy for these models.
            fx, fy, cx, cy = camera.params[:4]
        else:
            raise NotImplementedError(
                f"COLMAP camera model {model_name!r} not supported"
            )

        W = camera.width
        H = camera.height

        T_cam_world = np.eye(4, dtype=np.float32)
        T_cam_world[:3, :4] = image.cam_from_world().matrix()
        T_world_cam = np.linalg.inv(T_cam_world)

        # COLMAP image names are usually relative to images/, but Replica-style
        # scenes in this repo store RGB/depth/pose triples under data/.
        rgb_path = _resolve_image_path(scene_dir, image.name)
        if rgb_path is None:
            skipped.append(image.name)
            continue

        cam = Camera(T_world_cam, fx, fy, cx, cy, H, W)
        data.append((cam, rgb_path))

    if skipped:
        preview = ", ".join(skipped[:5])
        more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
        searched = ", ".join(str(scene_dir / name) for name in IMAGE_DIR_NAMES)
        print(
            f"[load_colmap] skipped {len(skipped)} registered image(s) with "
            f"missing files under {searched}: {preview}{more}",
            file=sys.stderr,
        )
    return data
