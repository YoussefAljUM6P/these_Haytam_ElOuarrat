"""Check that mesh.ply is expressed in the COLMAP reconstruction frame.

Use via the CLI:
    python cli.py mesh-check --scene living
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"


def add_arguments(parser):
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene folder under DATA/ (e.g. living).",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=10,
        help="Number of sorted COLMAP frames to project into. Use <=0 for all.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=200000,
        help="Deterministic point sample per asset before projection.",
    )
    parser.add_argument(
        "--min-mesh-visible",
        type=int,
        default=1,
        help="Minimum mesh vertices that must project inside sampled cameras.",
    )
    parser.add_argument(
        "--skip-gs",
        action="store_true",
        help="Do not compare against gs.ply even when it exists.",
    )


def _sample_points(points, limit):
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if limit is None or limit <= 0 or len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, int(limit)).astype(np.int64)
    return points[indices]


def _robust_bounds(points):
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return None
    q01, q50, q99 = np.percentile(points, [1, 50, 99], axis=0)
    extent = q99 - q01
    return {
        "p01": q01,
        "median": q50,
        "p99": q99,
        "extent": extent,
        "diag": float(np.linalg.norm(extent)),
    }


def _format_vec(vec):
    return "[" + ", ".join(f"{float(v):.5g}" for v in vec) + "]"


def _print_bounds(label, points):
    stats = _robust_bounds(points)
    if stats is None:
        print(f"  {label}: empty")
        return
    print(
        f"  {label}: n={len(points)} "
        f"p01={_format_vec(stats['p01'])} "
        f"p50={_format_vec(stats['median'])} "
        f"p99={_format_vec(stats['p99'])} "
        f"diag={stats['diag']:.5g}"
    )


def _project_points(points, records):
    import numpy as np

    points_h = np.c_[points, np.ones(len(points), dtype=np.float64)]
    rows = []
    for camera, rgb_path in records:
        cam_points = points_h @ camera.T_cam_world.astype(np.float64).T
        z = cam_points[:, 2]
        front = z > 0.01
        inside = np.zeros(len(points), dtype=bool)
        if np.any(front):
            u = camera.fx * (cam_points[:, 0] / z) + camera.cx
            v = camera.fy * (cam_points[:, 1] / z) + camera.cy
            inside = front & (u >= 0) & (u < camera.W) & (v >= 0) & (v < camera.H)
        rows.append(
            {
                "frame": Path(rgb_path).name,
                "front": int(front.sum()),
                "inside": int(inside.sum()),
            }
        )
    return rows


def _fail(message):
    print(f"ERROR: {message}")
    raise SystemExit(1)


def _print_visibility(label, rows):
    total_inside = sum(row["inside"] for row in rows)
    total_front = sum(row["front"] for row in rows)
    best = max(rows, key=lambda row: row["inside"])
    print(
        f"  {label}: front_total={total_front} inside_total={total_inside} "
        f"best={best['frame']}:{best['inside']}"
    )
    for row in rows[:5]:
        print(
            f"    {row['frame']}: front={row['front']} inside={row['inside']}"
        )
    return total_inside


def _read_sparse_points(scene_dir):
    import numpy as np
    import pycolmap

    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.exists():
        raise RuntimeError(f"COLMAP sparse directory not found: {sparse_dir}")
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    points = [point.xyz for point in reconstruction.points3D.values()]
    if not points:
        raise RuntimeError(f"COLMAP sparse reconstruction has no points: {sparse_dir}")
    return np.stack(points).astype(np.float64)


def _read_mesh_vertices(scene_dir):
    import numpy as np
    import trimesh

    mesh_path = scene_dir / "mesh.ply"
    if not mesh_path.exists():
        raise RuntimeError(f"mesh.ply not found: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise RuntimeError(f"mesh.ply has no vertices: {mesh_path}")
    return vertices


def _read_gs_points(scene_dir):
    import numpy as np
    from plyfile import PlyData

    gs_path = scene_dir / "gs.ply"
    if not gs_path.exists():
        return None
    ply = PlyData.read(str(gs_path))
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float64
    )


def run(args):
    from dataset import load_colmap

    scene_dir = DATA_ROOT / args.scene
    if not scene_dir.exists():
        raise RuntimeError(f"Scene directory not found: {scene_dir}")

    records = sorted(load_colmap(scene_dir), key=lambda item: str(item[1]))
    if not records:
        raise RuntimeError(f"No COLMAP image records with RGB files in {scene_dir}")
    if args.max_cameras > 0:
        records = records[: args.max_cameras]

    sparse_points = _sample_points(_read_sparse_points(scene_dir), args.sample_points)
    mesh_vertices = _sample_points(_read_mesh_vertices(scene_dir), args.sample_points)
    gs_points = None if args.skip_gs else _read_gs_points(scene_dir)
    if gs_points is not None:
        gs_points = _sample_points(gs_points, args.sample_points)

    print(f"Scene: {scene_dir}")
    print(f"COLMAP cameras sampled: {len(records)}")
    print("Robust COLMAP-frame bounds (1/50/99 percentiles):")
    _print_bounds("sparse", sparse_points)
    _print_bounds("mesh", mesh_vertices)
    if gs_points is not None:
        _print_bounds("gs", gs_points)

    print("Projected visibility in sampled COLMAP cameras:")
    sparse_inside = _print_visibility(
        "sparse", _project_points(sparse_points, records)
    )
    mesh_inside = _print_visibility("mesh", _project_points(mesh_vertices, records))
    gs_inside = None
    if gs_points is not None:
        gs_inside = _print_visibility("gs", _project_points(gs_points, records))

    if mesh_inside < args.min_mesh_visible:
        reference = "COLMAP sparse points"
        if gs_inside is not None and gs_inside > 0:
            reference += " and gs.ply"
        if sparse_inside == 0 and (gs_inside is None or gs_inside == 0):
            _fail(
                "No reference points project into the sampled COLMAP cameras; "
                "check sparse/0, image records, and camera intrinsics first."
            )
        _fail(
            f"mesh.ply projects {mesh_inside} vertices inside sampled COLMAP "
            f"cameras, below required {args.min_mesh_visible}, while {reference} "
            "are visible. Use a mesh exported in the same COLMAP frame as "
            "sparse/0/cameras; do not hide this with a runtime transform."
        )

    print("mesh.ply projects into the sampled COLMAP cameras")
