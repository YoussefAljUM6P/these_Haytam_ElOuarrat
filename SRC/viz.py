import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def save_side_by_side(img1, img2, path):
    out = np.concatenate([img1, img2], axis=1)
    out_uint8 = (out * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out_uint8).save(path)


def _to_uint8_rgb(image):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _to_gray01(image):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return np.clip(arr, 0.0, 1.0)
    return cv2.cvtColor(np.clip(arr, 0.0, 1.0), cv2.COLOR_RGB2GRAY)


def _panel_with_label(image, label):
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    label_strip = 22
    panel = np.zeros((height + label_strip, width, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        label,
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    panel[label_strip:, :, :] = image
    return panel


def save_current_desired_error_visualization(
    current,
    desired,
    path,
    *,
    current_label="current",
    desired_label="desired",
):
    current_rgb = _to_uint8_rgb(current)
    desired_rgb = _to_uint8_rgb(desired)
    if current_rgb.shape != desired_rgb.shape:
        raise ValueError(
            f"current and desired images must have same shape, got "
            f"{current_rgb.shape} vs {desired_rgb.shape}"
        )

    gray_current = _to_gray01(current)
    gray_desired = _to_gray01(desired)
    diff = np.abs(gray_current - gray_desired)
    diff_max = float(diff.max()) if diff.size else 0.0
    diff_mean = float(diff.mean()) if diff.size else 0.0
    diff_fixed = (np.clip(diff, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(diff_fixed, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    panels = [
        _panel_with_label(current_rgb, current_label),
        _panel_with_label(desired_rgb, desired_label),
        _panel_with_label(heat_rgb, f"photo error max={diff_max:.3f}"),
    ]
    canvas = np.concatenate(panels, axis=1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return {
        "visualization_path": str(path),
        "diff_max": diff_max,
        "diff_mean": diff_mean,
    }


def _to_int_pixels(kpts, x_offset=0):
    if len(kpts) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pts = np.rint(np.asarray(kpts, dtype=np.float32)).astype(np.int32)
    if x_offset:
        pts = pts.copy()
        pts[:, 0] += x_offset
    return pts


def save_match_visualization(
    img1,
    img2,
    kpts1_removed,
    kpts2_removed,
    matches_removed,
    kpts1_kept,
    kpts2_kept,
    matches_kept,
    path,
    draw_removed=True,
):
    W1 = img1.shape[1]
    canvas = np.concatenate([img1, img2], axis=1)
    canvas = (canvas * 255.0).clip(0, 255).astype(np.uint8)
    canvas = np.ascontiguousarray(canvas)

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    GREEN_ALPHA = 0.5
    RED_ALPHA = 0.25

    if draw_removed and matches_removed:
        p1_r = _to_int_pixels(kpts1_removed)
        p2_r = _to_int_pixels(kpts2_removed, x_offset=W1)
        red_overlay = canvas.copy()
        for i, j in matches_removed:
            cv2.line(red_overlay, tuple(p1_r[i]), tuple(p2_r[j]), RED, 1, lineType=cv2.LINE_AA)
        cv2.addWeighted(red_overlay, RED_ALPHA, canvas, 1.0 - RED_ALPHA, 0, canvas)

    p1_k = _to_int_pixels(kpts1_kept)
    p2_k = _to_int_pixels(kpts2_kept, x_offset=W1)
    green_overlay = canvas.copy()
    for i, j in matches_kept:
        a = tuple(p1_k[i])
        b = tuple(p2_k[j])
        cv2.line(green_overlay, a, b, GREEN, 1, lineType=cv2.LINE_AA)
        cv2.circle(green_overlay, a, 3, GREEN, 1, lineType=cv2.LINE_AA)
        cv2.circle(green_overlay, b, 3, GREEN, 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(green_overlay, GREEN_ALPHA, canvas, 1.0 - GREEN_ALPHA, 0, canvas)

    Image.fromarray(canvas).save(path)


def save_error_evolution(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    panel_height = 210
    margin_left = 95
    margin_right = 35
    margin_top = 42
    margin_bottom = 38
    gap = 18

    metrics = [
        (
            "feature error norm",
            lambda item: item.get("controller_info", {}).get("residual_norm"),
            (30, 110, 230),
        ),
        (
            "translation gap",
            lambda item: item.get("translation_gap"),
            (35, 150, 85),
        ),
        (
            "rotation distance (deg)",
            lambda item: item.get("rotation_error_deg"),
            (210, 95, 35),
        ),
    ]

    height = margin_top + margin_bottom + len(metrics) * panel_height + (len(metrics) - 1) * gap
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    title = "Servo error evolution"
    cv2.putText(
        canvas,
        title,
        (margin_left, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )

    def metric_points(get_value):
        points = []
        for item in history:
            value = get_value(item)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            points.append((int(item["iteration"]), value))
        return points

    plot_left = margin_left
    plot_right = width - margin_right
    plot_width = plot_right - plot_left

    for metric_index, (label, get_value, color) in enumerate(metrics):
        y0 = margin_top + metric_index * (panel_height + gap)
        y1 = y0 + panel_height
        plot_top = y0 + 28
        plot_bottom = y1 - 34
        plot_height = plot_bottom - plot_top

        cv2.putText(
            canvas,
            label,
            (plot_left, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 35, 35),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (plot_left, plot_top),
            (plot_right, plot_bottom),
            (210, 210, 210),
            1,
            lineType=cv2.LINE_AA,
        )

        points = metric_points(get_value)
        if not points:
            cv2.putText(
                canvas,
                "no data",
                (plot_left + 12, plot_top + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (120, 120, 120),
                1,
                lineType=cv2.LINE_AA,
            )
            continue

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = min(0.0, float(ys.min()))
        y_max = float(ys.max())
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        y_pad = 0.05 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad

        cv2.putText(
            canvas,
            f"{y_max:.4g}",
            (8, plot_top + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{y_min:.4g}",
            (8, plot_bottom + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"iter {int(x_min)}",
            (plot_left, plot_bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"iter {int(x_max)}",
            (plot_right - 75, plot_bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )

        pixel_points = []
        for x, y in points:
            px = plot_left + int(round(((x - x_min) / (x_max - x_min)) * plot_width))
            py = plot_bottom - int(round(((y - y_min) / (y_max - y_min)) * plot_height))
            pixel_points.append([px, py])
        pixel_points = np.asarray(pixel_points, dtype=np.int32).reshape(-1, 1, 2)

        if len(pixel_points) >= 2:
            cv2.polylines(
                canvas,
                [pixel_points],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        for point in pixel_points.reshape(-1, 2):
            cv2.circle(canvas, tuple(point), 3, color, -1, lineType=cv2.LINE_AA)

        first = points[0][1]
        last = points[-1][1]
        cv2.putText(
            canvas,
            f"{first:.5g} -> {last:.5g}",
            (plot_right - 170, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    Image.fromarray(canvas).save(path)
