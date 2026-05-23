import numpy as np
from pathlib import Path
from camera import Camera

def load_scannet(scene_dir):
    scene_dir = Path(scene_dir)
    info_path = scene_dir / "info.txt"

    fx = fy = cx = cy = 0.0
    W = H = 0

    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("m_colorWidth"):
                W = int(line.split("=")[1].strip())
            elif line.startswith("m_colorHeight"):
                H = int(line.split("=")[1].strip())
            elif line.startswith("m_calibrationColorIntrinsic"):
                vals = [float(v) for v in line.split("=")[1].strip().split()]
                intrinsics = np.array(vals).reshape(4, 4)
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]

    pose_files = sorted(list((scene_dir / "data").glob("frame-*.pose.txt")))

    data = []
    for pose_file in pose_files:
        T_world_cam = np.loadtxt(pose_file, dtype=np.float32)
        # ScanNet emits non-finite poses for tracking failures; skip them.
        if not np.isfinite(T_world_cam).all():
            continue
        stem = pose_file.name.split(".pose.txt")[0]
        rgb_path = scene_dir / "data" / f"{stem}.color.jpg"
        depth_path = scene_dir / "data" / f"{stem}.depth.png"

        cam = Camera(T_world_cam, fx, fy, cx, cy, H, W)
        data.append((cam, rgb_path, depth_path))

    return data

def load_colmap(scene_dir):
    import pycolmap
    scene_dir = Path(scene_dir)
    reconstruction = pycolmap.Reconstruction(scene_dir / "sparse" / "0")

    data = []
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

        # COLMAP image names are relative to the reconstruction's images/ dir.
        rgb_path = scene_dir / "images" / image.name

        cam = Camera(T_world_cam, fx, fy, cx, cy, H, W)
        data.append((cam, rgb_path))

    return data
