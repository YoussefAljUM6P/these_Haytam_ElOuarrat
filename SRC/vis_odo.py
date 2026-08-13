
import numpy as np
import cv2


def estimate_pose(frame1, frame2, K):
    frame1 = np.array(frame1)
    frame2 = np.array(frame2)

    sift = cv2.SIFT_create()

    kpts1, desc1 = sift.detectAndCompute(frame1, None)
    kpts2, desc2 = sift.detectAndCompute(frame2, None)

    flann = cv2.FlannBasedMatcher(indexParams=dict(algorithm=1, trees=5), searchParams=dict(checks=50))
    matches = flann.knnMatch(desc1, desc2, k=2)

    if len(matches) < 8:
        raise ValueError(f'Insuffucient kpts:{len(matches)}')
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) < 8:
        raise ValueError(f'Insuffucient filtered kpts:{len(good_matches)}')

    feats1 = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
    feats2 = np.float32([kpts2[m.trainIdx].pt for m in good_matches])

    E, mask = cv2.findEssentialMat(feats1, feats2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    _, R, t, mask = cv2.recoverPose(E, feats1, feats2, K)

    return R, t


def track_points(frame1, frame2, pts1):
    """
    frame1, frame2: grayscale np.ndarray
    pts1: np.ndarray of shape (N, 2), points to track in frame1

    Returns:
        pts1_tracked: (M, 2) — inlier points from frame1
        pts2_tracked: (M, 2) — corresponding tracked points in frame2
    """
    pts1 = pts1.reshape(-1, 1, 2).astype(np.float32)
    pts2, status, err = cv2.calcOpticalFlowPyrLK(
        frame1, frame2, pts1, None, winSize=(21, 21), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-2)
    )
    pts2r, status_r, _ = cv2.calcOpticalFlowPyrLK(
        frame2, frame1, pts2, None, winSize=(21, 21)
    )

    r_err = np.linalg.norm(pts2r - pts1, axis=1)

    status = status.ravel()
    err = err.ravel()

    good = (r_err < 1.0) & (err < 20) & (status == 1)

    return pts1[good].reshape(-1, 2), pts2[good].reshape(-1, 2)

def triangulate(R, t, pts1, pts2, K):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    pts_cam2 = (R @ pts3d.T + t).T
    good = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 100) & (pts_cam2[:, 2] > 0)

    # reprojection filter
    def reproject(P:np.ndarray, pts):
        h = np.hstack([pts, np.ones((len(pts), 1))])
        p = (P @ h.T).T
        return p[:, :2] / p[:, 2:3]

    err1 = np.linalg.norm(reproject(P1, pts3d) - pts1, axis=1)
    err2 = np.linalg.norm(reproject(P2, pts3d) - pts2, axis=1)
    good &= (err1 < 2.0) & (err2 < 2.0)

    return pts3d[good], good


def estimate_pose_pnp(pts3d, pts2d, K):
    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, K, None)

    if not success or inliers is None or len(inliers) < 10:
        raise ValueError("PnP failed or insufficient inliers")

    if len(inliers) / len(pts2d) < 0.5:
        raise ValueError(f"Too many outliers: {len(inliers)}/{len(pts2d)} inliers")

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, inliers

def vo_step(frame1, frame2, pts3d_map, K):
    """
    frame1, frame2: grayscale np.ndarray
    pts3d_map: (N, 3) 3D map points visible in frame1
    K: (3, 3) intrinsics

    Returns:
        R: camera rotation of frame2
        t: camera translation of frame2
        new_pts3d: newly triangulated points to add to the map
    """
    pts1, _ = cv2.projectPoints(pts3d_map, np.zeros((3,1)), np.zeros((3,1)), K, None)
    pts1 = pts1.reshape(-1, 2)

    pts1_tracked, pts2_tracked = track_points(frame1, frame2, pts1)

    # need to recover which pts3d_map survived tracking
    # track_points should return mask too — design gap
    R, t, inliers = estimate_pose_pnp(pts3d_map[:len    (pts1_tracked)], pts2_tracked, K)

    new_pts3d, _ = triangulate(R, t, pts1_tracked, pts2_tracked, K)

    return R, t, new_pts3d




