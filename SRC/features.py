import sys
from pathlib import Path
import numpy as np
import cv2
import torch  # noqa: F401  (XFeat requires torch on the env; keep explicit import)

# XFeat lives in third_party/accelerated_features.
_XFEAT_DIR = Path(__file__).resolve().parent / "third_party" / "accelerated_features"
if str(_XFEAT_DIR) not in sys.path:
    sys.path.append(str(_XFEAT_DIR))


class FeatureMatcher:
    def __init__(self, method='xfeat'):
        if method == 'xfeat':
            from modules.xfeat import XFeat
            self.extractor = XFeat()
        elif method == 'sift':
            self.extractor = cv2.SIFT_create()
        else:
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self._target_image = None
        self._target_features = None

    def clear_target_cache(self):
        self._target_image = None
        self._target_features = None


    def match(self, img1, img2):
        """Return paired keypoint arrays (kpts1, kpts2), each (N, 2) float32."""
        im1_u8 = (np.asarray(img1) * 255.0).clip(0, 255).astype(np.uint8)

        if self.method == 'xfeat':
            return self._match_xfeat(im1_u8, img2)

        gray1 = cv2.cvtColor(im1_u8, cv2.COLOR_RGB2GRAY)
        kp1, des1 = self.extractor.detectAndCompute(gray1, None)

        if self._target_image is img2 and self._target_features is not None:
            kp2, des2 = self._target_features
        else:
            im2_u8 = (np.asarray(img2) * 255.0).clip(0, 255).astype(np.uint8)
            gray2 = cv2.cvtColor(im2_u8, cv2.COLOR_RGB2GRAY)
            kp2, des2 = self.extractor.detectAndCompute(gray2, None)
            self._target_image = img2
            self._target_features = (kp2, des2)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty.copy()

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            empty = np.zeros((0, 2), dtype=np.float32)
            return empty, empty.copy()

        distances = np.fromiter((m.distance for m in matches), dtype=np.float32, count=len(matches))
        order = np.argsort(distances, kind="stable")
        query_idx = np.fromiter((m.queryIdx for m in matches), dtype=np.int64, count=len(matches))[order]
        train_idx = np.fromiter((m.trainIdx for m in matches), dtype=np.int64, count=len(matches))[order]

        pts1_all = cv2.KeyPoint_convert(kp1)
        pts2_all = cv2.KeyPoint_convert(kp2)
        kpts1 = pts1_all[query_idx].astype(np.float32, copy=False)
        kpts2 = pts2_all[train_idx].astype(np.float32, copy=False)
        return kpts1, kpts2

    def _extract_xfeat(self, image_u8):
        parsed = self.extractor.parse_input(image_u8)
        return self.extractor.detectAndCompute(parsed)[0]

    def _match_xfeat(self, im1_u8, target):
        try:
            return self._match_xfeat_impl(im1_u8, target)
        except RuntimeError as exc:
            if not self._can_retry_xfeat_on_cpu(exc):
                raise
            self.clear_target_cache()
            self.extractor.dev = torch.device('cpu')
            self.extractor.net.to(self.extractor.dev)
            return self._match_xfeat_impl(im1_u8, target)

    def _match_xfeat_impl(self, im1_u8, target):
        out1 = self._extract_xfeat(im1_u8)
        if self._target_image is target and self._target_features is not None:
            out2 = self._target_features
        else:
            im2_u8 = (np.asarray(target) * 255.0).clip(0, 255).astype(np.uint8)
            out2 = self._extract_xfeat(im2_u8)
            self._target_image = target
            self._target_features = out2

        descriptors1 = out1['descriptors']
        descriptors2 = out2['descriptors']
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy()

        idxs0, idxs1 = self.extractor.match(descriptors1, descriptors2, min_cossim=-1)
        kpts1 = out1['keypoints'][idxs0].cpu().numpy()
        kpts2 = out2['keypoints'][idxs1].cpu().numpy()
        return kpts1.astype(np.float32, copy=False), kpts2.astype(np.float32, copy=False)

    def _can_retry_xfeat_on_cpu(self, exc):
        if getattr(self.extractor, 'dev', None) == torch.device('cpu'):
            return False
        message = str(exc).lower()
        return 'cuda' in message or 'cudnn' in message


def filter_matches(kpts1, kpts2, camera=None):
    """RANSAC homography filter on already-paired keypoint arrays.

    Returns a 6-tuple:
        (kpts1_f, kpts2_f, H_matrix, kpts1_n, kpts2_n, inliers_mask).
    inliers_mask is a (N,) bool array over the original kpts1/kpts2 marking
    which input pairs survived RANSAC; all-False on failure.
    If no camera is provided, the normalized coords are None.
    If fewer than 4 inliers survive RANSAC (or RANSAC fails), H_matrix is None
    and the filtered arrays are empty.
    """
    kpts1 = np.asarray(kpts1, dtype=np.float32)
    kpts2 = np.asarray(kpts2, dtype=np.float32)
    assert kpts1.shape == kpts2.shape and kpts1.ndim == 2 and kpts1.shape[1] == 2, \
        f"kpts1/kpts2 must be paired (N, 2) arrays, got {kpts1.shape} vs {kpts2.shape}"

    empty = np.zeros((0, 2), dtype=np.float32)
    fail_mask = np.zeros(kpts1.shape[0], dtype=bool)

    def _fail():
        if camera is None:
            return empty, empty.copy(), None, None, None, fail_mask
        return empty, empty.copy(), None, empty.copy(), empty.copy(), fail_mask

    if kpts1.shape[0] < 4:
        return _fail()

    pts1 = kpts1.reshape(-1, 1, 2)
    pts2 = kpts2.reshape(-1, 1, 2)
    H_matrix, mask = cv2.findHomography(
        pts1, pts2, cv2.RANSAC, ransacReprojThreshold=4.0
    )

    if H_matrix is None or mask is None:
        return _fail()

    inliers = mask.ravel().astype(bool)
    kpts1_f = kpts1[inliers]
    kpts2_f = kpts2[inliers]

    if kpts1_f.shape[0] < 4:
        return _fail()

    if camera is None:
        return kpts1_f, kpts2_f, H_matrix, None, None, inliers

    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    kpts1_n = np.empty_like(kpts1_f)
    kpts1_n[:, 0] = (kpts1_f[:, 0] - cx) / fx
    kpts1_n[:, 1] = (kpts1_f[:, 1] - cy) / fy
    kpts2_n = np.empty_like(kpts2_f)
    kpts2_n[:, 0] = (kpts2_f[:, 0] - cx) / fx
    kpts2_n[:, 1] = (kpts2_f[:, 1] - cy) / fy

    return kpts1_f, kpts2_f, H_matrix, kpts1_n, kpts2_n, inliers
