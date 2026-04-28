"""
pipeline.py
-----------
Main pipeline orchestration for multi-object detection and tracking.

Ties together:
  - PersonDetector     (YOLOv8)
  - SportsTracker      (ByteTrack)
  - TrackingAnnotator  (supervision annotators)
  - Analytics          (heatmap, count chart, trajectories)

Usage
-----
  from src.pipeline import run_pipeline

  run_pipeline(
      input_path="input_video.mp4",
      output_path="outputs/annotated.mp4",
      conf=0.4,
      frame_skip=1,
  )
"""

import logging
import os
import time
from collections import defaultdict
from typing import Optional

import cv2
import supervision as sv

from src.annotator import TrackingAnnotator, draw_frame_info
from src.analytics import (
    print_summary,
    save_count_over_time,
    save_heatmap,
    save_trajectory_image,
)
from src.detector import PersonDetector
from src.tracker import SportsTracker

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str,
    output_path: str = "outputs/annotated_output.mp4",
    model_path: str = "yolov8m.pt",
    conf: float = 0.35,
    iou: float = 0.45,
    frame_skip: int = 1,
    trace_length: int = 45,
    device: str = "cpu",
    save_analytics: bool = True,
    analytics_dir: Optional[str] = None,
) -> None:
    """
    Run the full detection-tracking-annotation pipeline on a video file.

    Parameters
    ----------
    input_path : str
        Path to the input video.
    output_path : str
        Path for the annotated output video.
    model_path : str
        YOLOv8 model identifier or path to weights.
    conf : float
        Detection confidence threshold.
    iou : float
        NMS IoU threshold.
    frame_skip : int
        Process every Nth frame (1 = every frame, 2 = every other, …).
        Skipped frames are written to output unchanged (preserving timing).
    trace_length : int
        Number of frames to retain in trajectory traces.
    device : str
        Inference device: 'cpu', 'cuda', or 'mps'.
    save_analytics : bool
        Whether to save heatmap, count chart, and trajectory images.
    analytics_dir : str | None
        Directory for analytics outputs. Defaults to same dir as output_path.
    """
    # ------------------------------------------------------------------ setup
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if analytics_dir is None:
        analytics_dir = os.path.dirname(output_path)
    os.makedirs(analytics_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # ---------------------------------------------------------------- init components
    detector = PersonDetector(model_path=model_path, confidence=conf, iou_threshold=iou, device=device)
    tracker = SportsTracker()
    annotator = TrackingAnnotator(trace_length=trace_length)

    # ---------------------------------------------------------------- video info
    video_info = sv.VideoInfo.from_video_path(input_path)
    total_frames = video_info.total_frames
    fps = video_info.fps
    logger.info(
        f"Video: {os.path.basename(input_path)} | "
        f"{video_info.width}×{video_info.height} | "
        f"{fps:.1f} fps | {total_frames} frames"
    )

    # ---------------------------------------------------------------- state
    counts: list[int] = []
    trajectory_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    reference_frame = None

    # ---------------------------------------------------------------- main loop
    frame_generator = sv.get_video_frames_generator(input_path)
    t0 = time.time()

    with sv.VideoSink(output_path, video_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            if frame_idx == 0:
                reference_frame = frame.copy()

            if frame_idx % frame_skip != 0:
                # Write unprocessed frame to keep audio-sync / timing correct
                sink.write_frame(frame)
                continue

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track
            tracked = tracker.update(detections)

            # 3. Store analytics data
            count = len(tracked) if tracked.tracker_id is not None else 0
            counts.append(count)

            if tracked.tracker_id is not None:
                for tid, xyxy in zip(tracked.tracker_id, tracked.xyxy):
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    trajectory_history[int(tid)].append((cx, cy))

            # 4. Annotate
            annotated_frame = annotator.annotate(frame, tracked)
            annotated_frame = draw_frame_info(
                annotated_frame, frame_idx, total_frames, count
            )

            sink.write_frame(annotated_frame)

            # Progress log every 100 frames
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                pct = (frame_idx / max(total_frames, 1)) * 100
                logger.info(
                    f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames} | "
                    f"tracked={count} | {elapsed:.0f}s elapsed"
                )

    elapsed_total = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed_total:.1f}s → {output_path}")

    # ---------------------------------------------------------------- analytics
    if save_analytics and annotator._heatmap is not None:
        h, w = video_info.height, video_info.width

        save_heatmap(
            annotator._heatmap,
            os.path.join(analytics_dir, "heatmap.png"),
            reference_frame=reference_frame,
        )
        save_count_over_time(
            counts,
            os.path.join(analytics_dir, "count_over_time.png"),
            fps=fps / frame_skip,
        )
        save_trajectory_image(
            trajectory_history,
            (h, w),
            os.path.join(analytics_dir, "trajectories.png"),
            reference_frame=reference_frame,
        )
        print_summary(trajectory_history, counts, fps=fps, pixels_per_metre=8.0)

    logger.info("All outputs saved.")
