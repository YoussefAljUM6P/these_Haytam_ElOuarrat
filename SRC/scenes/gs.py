import sys
import os
import math
import numpy as np
import torch

# Ensure diff-gaussian-rasterization is importable
third_party_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party'))
sys.path.append(third_party_dir)

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


# PLY scalar type -> numpy type code (binary formats only).
_PLY_TYPE_MAP = {
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
}


def _memmap_ply_vertices(ply_path):
    """Parse a binary PLY header and memory-map its vertex block.

    Returns a read-only structured ``np.memmap`` over the vertex element. The
    file is never copied into RAM wholesale: individual property columns page
    in lazily from disk (reclaimable page cache) as they are read, so loading a
    mult-GB splat does not balloon resident memory.
    """
    with open(ply_path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f"{ply_path}: not a PLY file")
        fmt = f.readline().split()
        if len(fmt) < 2 or fmt[0] != b'format':
            raise ValueError(f"{ply_path}: malformed PLY format line")
        byte_order = {b'binary_little_endian': '<', b'binary_big_endian': '>'}.get(fmt[1])
        if byte_order is None:
            raise ValueError(
                f"{ply_path}: only binary PLY is supported, got {fmt[1].decode()!r}"
            )

        count = None
        props = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{ply_path}: unexpected EOF in PLY header")
            parts = line.split()
            if not parts:
                continue
            kw = parts[0]
            if kw == b'comment' or kw == b'obj_info':
                continue
            if kw == b'element':
                in_vertex = parts[1] == b'vertex'
                if in_vertex:
                    count = int(parts[2])
            elif kw == b'property' and in_vertex:
                if parts[1] == b'list':
                    raise ValueError(f"{ply_path}: list properties unsupported in vertex")
                ply_type = parts[1].decode()
                if ply_type not in _PLY_TYPE_MAP:
                    raise ValueError(f"{ply_path}: unsupported property type {ply_type!r}")
                props.append((parts[2].decode(), byte_order + _PLY_TYPE_MAP[ply_type]))
            elif kw == b'end_header':
                break
        if count is None:
            raise ValueError(f"{ply_path}: no 'vertex' element in header")
        data_offset = f.tell()

    dtype = np.dtype(props)
    return np.memmap(ply_path, dtype=dtype, mode='r', offset=data_offset, shape=(count,))


def _stack_columns(vertices, names):
    """Copy the named memmap columns into one contiguous (N, len(names)) f32 array."""
    out = np.empty((vertices.shape[0], len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        out[:, i] = vertices[name]
    return out


def getProjectionMatrix(znear, zfar, fx, fy, cx, cy, W, H):
    """OpenCV-convention perspective projection with asymmetric frustum.

    Matches the INRIA gaussian-splatting CUDA rasterizer (z_sign = +1, depth in
    [0, 1] over [znear, zfar]) but allows an off-center principal point.
    Pixel mapping in CUDA is (NDC + 1) * size / 2, so:
        u = 0  -> NDC.x = -1, x/z = -cx / fx
        u = W  -> NDC.x = +1, x/z = (W - cx) / fx
        v = 0  -> NDC.y = -1, y/z = -cy / fy   (OpenCV y-down)
        v = H  -> NDC.y = +1, y/z = (H - cy) / fy
    """
    left = -cx / fx * znear
    right = (W - cx) / fx * znear
    top = -cy / fy * znear           # OpenCV y-down: top of image -> negative y_cam
    bottom = (H - cy) / fy * znear

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (bottom - top)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = -(bottom + top) / (bottom - top)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class GSScene:
    def __init__(self, ply_path):
        # Memory-mapped vertex block: columns stream from disk on demand so a
        # multi-GB splat never lands in RAM all at once. Each attribute is then
        # copied once into a contiguous host buffer, moved to the GPU, and freed
        # before the next, keeping the host peak to roughly a single column block.
        v = _memmap_ply_vertices(ply_path)

        prop_names = {name for name in v.dtype.names}
        f_rest_count = sum(1 for n in prop_names if n.startswith('f_rest_'))
        if f_rest_count % 3 != 0:
            raise ValueError(
                f"Expected f_rest_* count divisible by 3, got {f_rest_count}"
            )
        n_per_channel = f_rest_count // 3
        # SH coefs per channel = (deg+1)^2; minus 1 DC term => n_per_channel.
        sh_degree = int(round(math.sqrt(n_per_channel + 1) - 1))
        if (sh_degree + 1) ** 2 - 1 != n_per_channel:
            raise ValueError(
                f"f_rest count {f_rest_count} does not match any SH degree"
            )
        self.sh_degree = sh_degree
        N = v.shape[0]

        xyz_np = _stack_columns(v, ['x', 'y', 'z'])
        self.xyz = torch.from_numpy(xyz_np).cuda()
        del xyz_np

        opacity_np = np.ascontiguousarray(v['opacity'], dtype=np.float32)
        self.opacities = torch.sigmoid(torch.from_numpy(opacity_np).unsqueeze(-1)).cuda()
        del opacity_np

        scales_np = _stack_columns(v, ['scale_0', 'scale_1', 'scale_2'])
        self.scales = torch.exp(torch.from_numpy(scales_np)).cuda()
        del scales_np

        rot_np = _stack_columns(v, ['rot_0', 'rot_1', 'rot_2', 'rot_3'])
        rot = torch.from_numpy(rot_np).cuda()
        self.rotations = rot / rot.norm(dim=-1, keepdim=True)
        del rot_np

        # Assemble the SH buffer directly on the GPU as (N, 1 + n_per_channel, 3):
        # slot 0 is the DC term, the rest follow. Building it in place avoids
        # keeping a second full copy of the SH coefficients around (a host cat or
        # a separate sh_rest tensor), which on a degree-3 splat is ~1.3 GB.
        self._shs = torch.empty((N, 1 + n_per_channel, 3), device='cuda')
        dc_np = _stack_columns(v, ['f_dc_0', 'f_dc_1', 'f_dc_2'])
        self._shs[:, 0, :] = torch.from_numpy(dc_np).cuda()
        del dc_np
        if n_per_channel > 0:
            f_rest_names = [f'f_rest_{i}' for i in range(f_rest_count)]
            rest_np = _stack_columns(v, f_rest_names)
            # INRIA PLY storage: (N, 3, n_per_channel) flattened row-major to
            # (N, 3*n_per_channel). Reverse: reshape -> (N, 3, n_per_channel),
            # transpose -> (N, n_per_channel, 3).
            rest = torch.from_numpy(rest_np).cuda().reshape(N, 3, n_per_channel).transpose(1, 2)
            self._shs[:, 1:, :] = rest
            del rest_np, rest

        del v  # release the memmap

        # Static per-render buffer that never changes across calls.
        self._means2D = torch.zeros_like(self.xyz)
        self._bg = torch.zeros(3, device='cuda')
        self._depth_colors = torch.empty_like(self.xyz)
        self._depth_colors[:, 2] = 1.0
        self._proj_cache = {}
        self._last_camera = None
        self._last_depth_key = None
        self._last_depth = None

    @staticmethod
    def _camera_key(camera):
        return (
            camera.W,
            camera.H,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.T_world_cam.tobytes(),
        )

    def _get_proj(self, fx, fy, cx, cy, W, H):
        key = (fx, fy, cx, cy, W, H)
        proj = self._proj_cache.get(key)
        if proj is None:
            proj = (
                getProjectionMatrix(0.01, 100.0, fx, fy, cx, cy, W, H)
                .float()
                .transpose(0, 1)
                .cuda()
            )
            self._proj_cache[key] = proj
        return proj

    def render_tensor(self, camera):
        """Render an HWC float32 image on CUDA without a host readback."""
        self._last_camera = camera

        # tan(fov/2) values are used by the rasterizer to recover focal lengths
        # for the screen-space covariance Jacobian: focal = size / (2 * tanfov).
        # Off-center cx, cy is handled via the asymmetric projection matrix below.
        tanfovx = camera.W / (2.0 * camera.fx)
        tanfovy = camera.H / (2.0 * camera.fy)

        viewmatrix = (
            torch.from_numpy(np.ascontiguousarray(camera.T_cam_world))
            .float()
            .transpose(0, 1)
            .cuda(non_blocking=True)
        )
        projmatrix = self._get_proj(
            camera.fx, camera.fy, camera.cx, camera.cy, camera.W, camera.H
        )

        # Pytorch's bmm expects (B, N, M), we unsqueeze and squeeze
        full_projmatrix = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)

        campos = (
            torch.from_numpy(np.ascontiguousarray(camera.T_world_cam[:3, 3]))
            .float()
            .cuda(non_blocking=True)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.H,
            image_width=camera.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self._bg,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_projmatrix,
            sh_degree=self.sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(raster_settings)

        with torch.inference_mode():
            outputs = rasterizer(
                means3D=self.xyz,
                means2D=self._means2D,
                shs=self._shs,
                opacities=self.opacities,
                scales=self.scales,
                rotations=self.rotations,
                colors_precomp=None
            )
        rendered_image = outputs[0]

        return rendered_image.permute(1, 2, 0).clamp(0, 1)

    def render(self, camera):
        return self.render_tensor(camera).cpu().numpy()

    def render_depth(self, camera=None):
        if camera is None:
            camera = self._last_camera
        if camera is None:
            raise NotImplementedError("GSScene.render_depth requires a camera")

        depth_key = self._camera_key(camera)
        if depth_key == self._last_depth_key:
            return self._last_depth.copy()

        tanfovx = camera.W / (2.0 * camera.fx)
        tanfovy = camera.H / (2.0 * camera.fy)

        T_cam_world = (
            torch.from_numpy(np.ascontiguousarray(camera.T_cam_world))
            .float()
            .cuda(non_blocking=True)
        )
        viewmatrix = T_cam_world.transpose(0, 1)
        projmatrix = self._get_proj(
            camera.fx, camera.fy, camera.cx, camera.cy, camera.W, camera.H
        )

        full_projmatrix = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)

        campos = (
            torch.from_numpy(np.ascontiguousarray(camera.T_world_cam[:3, 3]))
            .float()
            .cuda(non_blocking=True)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.H,
            image_width=camera.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self._bg,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_projmatrix,
            sh_degree=self.sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(raster_settings)

        z_cam = self.xyz @ T_cam_world[2, :3] + T_cam_world[2, 3]
        self._depth_colors[:, 0].copy_(z_cam)
        self._depth_colors[:, 1].copy_(z_cam)
        depth_colors = self._depth_colors

        with torch.inference_mode():
            outputs = rasterizer(
                means3D=self.xyz,
                means2D=self._means2D,
                shs=None,
                opacities=self.opacities,
                scales=self.scales,
                rotations=self.rotations,
                colors_precomp=depth_colors
            )
        depth_numerator = outputs[0][0]
        accum_alpha = outputs[0][2]
        rendered_depth = torch.zeros_like(depth_numerator)
        valid = accum_alpha > 1e-6
        rendered_depth[valid] = depth_numerator[valid] / accum_alpha[valid]

        depth = rendered_depth.detach().cpu().numpy().astype(np.float32, copy=False)
        self._last_depth_key = depth_key
        self._last_depth = depth
        return depth.copy()
