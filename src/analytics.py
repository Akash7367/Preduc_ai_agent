"""
analytics.py
------------
Post-processing analytics for the tracking pipeline.

Generates:
  1. Movement heatmap image  (heatmap.png)
  2. Player count over time  (count_over_time.png)
  3. Trajectory overlay      (trajectories.png)
  4. Speed estimation        (pixels/frame → m/s with calibration factor)
  5. Summary statistics      printed to stdout / logger
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# Rough real-world calibration: pixels per metre for a standard football pitch
# (105 m long). Adjust for your specific video crop.
DEFAULT_PIXELS_PER_METRE = 8.0  # pixels / metre (very approximate)
DEFAULT_FPS = 30


def save_heatmap(
    heatmap: np.ndarray,
    output_path: str,
    reference_frame: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> None:
    """
    Save the accumulated heatmap as a colourised PNG.

    Parameters
    ----------
    heatmap : np.ndarray
        Float32 heat-accumulation array (H, W).
    output_path : str
        Destination file path.
    reference_frame : np.ndarray | None
        If provided, blend heatmap over this BGR frame for context.
    alpha : float
        Blend weight (0 = reference only, 1 = heatmap only).
    """
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    if reference_frame is not None:
        ref = cv2.resize(reference_frame, (coloured.shape[1], coloured.shape[0]))
        blended = cv2.addWeighted(ref, 1 - alpha, coloured, alpha, 0)
    else:
        blended = coloured

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, blended)
    logger.info(f"Heatmap saved → {output_path}")


def save_count_over_time(
    counts: List[int],
    output_path: str,
    fps: float = DEFAULT_FPS,
) -> None:
    """
    Save a line chart of player count per frame.

    Parameters
    ----------
    counts : list[int]
        Number of tracked players for each processed frame.
    output_path : str
        Destination file path (PNG).
    fps : float
        Video frame rate, used to convert frame indices to seconds.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping count chart.")
        return

    times = [i / fps for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, counts, color="#00C2FF", linewidth=1.5)
    ax.fill_between(times, counts, alpha=0.2, color="#00C2FF")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Tracked Players", fontsize=12)
    ax.set_title("Player Count Over Time", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Count chart saved → {output_path}")


def save_trajectory_image(
    trajectory_history: Dict[int, List[Tuple[int, int]]],
    frame_shape: Tuple[int, int],
    output_path: str,
    reference_frame: Optional[np.ndarray] = None,
) -> None:
    """
    Draw all stored trajectories on a single static image.

    Parameters
    ----------
    trajectory_history : dict[int, list[(cx, cy)]]
        Mapping from tracker_id → list of (cx, cy) centre points.
    frame_shape : (H, W)
        Canvas size.
    output_path : str
        Destination file path (PNG).
    reference_frame : np.ndarray | None
        If provided, draw trajectories over this BGR frame.
    """
    h, w = frame_shape
    if reference_frame is not None:
        canvas = reference_frame.copy()
        overlay = np.zeros_like(canvas)
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = canvas

    n_ids = max(trajectory_history.keys(), default=0) + 1

    for tid, pts in trajectory_history.items():
        if len(pts) < 2:
            continue
        hue = int(180 * (tid % 50) / 50)
        colour = cv2.cvtColor(
            np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR
        )[0][0]
        colour = tuple(int(c) for c in colour)

        for i in range(1, len(pts)):
            cv2.line(overlay, pts[i - 1], pts[i], colour, 2, cv2.LINE_AA)
        # Mark start and end
        cv2.circle(overlay, pts[0], 4, colour, -1)
        cv2.circle(overlay, pts[-1], 4, (255, 255, 255), -1)

    if reference_frame is not None:
        result = cv2.addWeighted(canvas, 0.6, overlay, 0.8, 0)
    else:
        result = overlay

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    logger.info(f"Trajectory image saved → {output_path}")


def estimate_speed(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    fps: float = DEFAULT_FPS,
    pixels_per_metre: float = DEFAULT_PIXELS_PER_METRE,
) -> float:
    """
    Estimate speed in km/h given two consecutive centre points.

    Parameters
    ----------
    p1, p2 : (int, int)
        Previous and current (cx, cy) in pixel coordinates.
    fps : float
        Video frame rate.
    pixels_per_metre : float
        Calibration factor (pixels per real-world metre).

    Returns
    -------
    float
        Estimated speed in km/h.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist_px = np.sqrt(dx * dx + dy * dy)
    dist_m = dist_px / pixels_per_metre
    speed_ms = dist_m * fps          # metres/second
    speed_kmh = speed_ms * 3.6       # km/h
    return round(speed_kmh, 1)


def print_summary(
    trajectory_history: Dict[int, List[Tuple[int, int]]],
    counts: List[int],
    fps: float = DEFAULT_FPS,
    pixels_per_metre: float = DEFAULT_PIXELS_PER_METRE,
) -> None:
    """
    Print a summary statistics table to the logger.
    """
    total_ids = len(trajectory_history)
    avg_count = np.mean(counts) if counts else 0
    max_count = max(counts) if counts else 0

    logger.info("=" * 50)
    logger.info("TRACKING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total unique IDs assigned : {total_ids}")
    logger.info(f"Average players per frame : {avg_count:.1f}")
    logger.info(f"Peak players in one frame : {max_count}")

    # Speed stats
    speeds = []
    for tid, pts in trajectory_history.items():
        for i in range(1, len(pts)):
            s = estimate_speed(pts[i - 1], pts[i], fps, pixels_per_metre)
            if s < 60:          # sanity filter: >60 km/h is a misdetection
                speeds.append(s)

    if speeds:
        logger.info(f"Estimated avg speed       : {np.mean(speeds):.1f} km/h")
        logger.info(f"Estimated max speed       : {np.max(speeds):.1f} km/h")
    logger.info("=" * 50)
