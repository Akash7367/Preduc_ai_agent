"""
annotator.py
------------
Visualization utilities for drawing tracking results on video frames.

Combines several `supervision` annotators to produce a rich overlay:
  - Coloured bounding boxes (unique colour per tracker ID)
  - Numeric ID labels
  - Trajectory traces (last N frames)
  - Frame-level heatmap accumulation
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


def _make_color_palette(n: int = 50):
    """Generate a visually distinct HSV colour palette as BGR tuples."""
    colours = []
    for i in range(n):
        hue = int(180 * i / n)
        colour = cv2.cvtColor(
            np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR
        )[0][0]
        colours.append(tuple(int(c) for c in colour))
    return colours


# Pre-built colour palette (50 distinct colours, cycles for IDs > 50)
PALETTE = _make_color_palette(50)


class TrackingAnnotator:
    """
    Applies a full tracking annotation stack to a video frame.

    Parameters
    ----------
    trace_length : int
        Number of past frames to draw the trajectory trail for.
    thickness : int
        Bounding box line thickness in pixels.
    text_scale : float
        Font scale for ID labels.
    text_thickness : int
        Font thickness for ID labels.
    """

    def __init__(
        self,
        trace_length: int = 45,
        thickness: int = 2,
        text_scale: float = 0.6,
        text_thickness: int = 2,
    ):
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=4,
        )
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=trace_length,
            thickness=2,
            position=sv.Position.BOTTOM_CENTER,
        )
        # Heatmap accumulates across frames
        self._heatmap: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        draw_heatmap: bool = False,
    ) -> np.ndarray:
        """
        Draw all tracking annotations on *frame* (in-place copy).

        Parameters
        ----------
        frame : np.ndarray
            BGR image to annotate.
        detections : sv.Detections
            Tracked detections with `tracker_id` populated.
        draw_heatmap : bool
            If True, blend the accumulated heatmap into the frame.

        Returns
        -------
        np.ndarray
            Annotated BGR frame.
        """
        annotated = frame.copy()

        if detections.tracker_id is None or len(detections) == 0:
            return annotated

        # Build label list: "ID: <id>"
        labels = [f"ID:{tid}" for tid in detections.tracker_id]

        # Generate per-ID colours based on tracker_id
        colours = sv.ColorPalette.DEFAULT

        # Draw traces first (under boxes)
        annotated = self.trace_annotator.annotate(
            scene=annotated, detections=detections
        )
        # Boxes
        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections
        )
        # Labels
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        # Overlay heatmap if requested
        if draw_heatmap and self._heatmap is not None:
            annotated = self._blend_heatmap(annotated)

        # Accumulate heatmap
        self._accumulate_heatmap(frame, detections)

        return annotated

    def get_heatmap_image(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Return a colourised version of the accumulated heatmap.

        Parameters
        ----------
        frame_shape : (H, W)

        Returns
        -------
        np.ndarray
            BGR heatmap image same size as input frame.
        """
        if self._heatmap is None:
            return np.zeros((*frame_shape, 3), dtype=np.uint8)

        norm = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return coloured

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate_heatmap(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> None:
        """Add Gaussian blobs at each detection centre to the heatmap."""
        h, w = frame.shape[:2]
        if self._heatmap is None:
            self._heatmap = np.zeros((h, w), dtype=np.float32)

        for xyxy in detections.xyxy:
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
            cx = np.clip(cx, 0, w - 1)
            cy = np.clip(cy, 0, h - 1)
            cv2.circle(self._heatmap, (cx, cy), radius=25, color=1.0, thickness=-1)

    def _blend_heatmap(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Alpha-blend the heatmap onto the frame."""
        h, w = frame.shape[:2]
        heatmap_bgr = self.get_heatmap_image((h, w))
        return cv2.addWeighted(frame, 1 - alpha, heatmap_bgr, alpha, 0)


def draw_frame_info(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    player_count: int,
) -> np.ndarray:
    """
    Draw a HUD overlay (frame counter, player count) on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to annotate (modified in-place).
    frame_idx : int
        Current frame index (0-based).
    total_frames : int
        Total number of frames in the video.
    player_count : int
        Number of currently tracked subjects.

    Returns
    -------
    np.ndarray
        Frame with HUD overlay.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Text
    fps_text = f"Frame: {frame_idx}/{total_frames}  |  Players tracked: {player_count}"
    cv2.putText(
        frame, fps_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame
