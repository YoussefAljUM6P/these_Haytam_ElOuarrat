import numpy as np

class Camera:
    def __init__(self, T_world_cam, fx, fy, cx, cy, H, W):
        self.T_world_cam = np.array(T_world_cam, dtype=np.float32)  # (4,4)
        assert self.T_world_cam.shape == (4, 4), \
            f"T_world_cam must be 4x4, got {self.T_world_cam.shape}"
        assert np.isfinite(self.T_world_cam).all(), \
            "T_world_cam contains non-finite values"
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.H = int(H)
        self.W = int(W)
        self._T_cam_world = None

    @property
    def K(self):
        return np.array([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1]
        ], dtype=np.float32)

    @property
    def T_cam_world(self):
        if self._T_cam_world is None:
            self._T_cam_world = np.linalg.inv(self.T_world_cam).astype(np.float32)
        return self._T_cam_world
