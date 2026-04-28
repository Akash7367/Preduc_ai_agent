"""
tracker.py
----------
ByteTrack multi-object tracker wrapper.

Wraps `supervision.ByteTrack` with sports-tuned hyper-parameters.
ByteTrack assigns and maintains unique numeric IDs across frames,
handling occlusion by keeping low-confidence detections alive in a
secondary buffer before confirming or discarding them.

References
----------
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box
  Zhang et al., ECCV 2022.  https://arxiv.org/abs/2110.06864
- Supervision library: https://github.com/roboflow/supervision
"""

import logging

import supervision as sv

logger = logging.getLogger(__name__)


class SportsTracker:
    """
    ByteTrack-based multi-object tracker tuned for sports footage.

    Key parameter choices
    ---------------------
    track_activation_threshold : 0.35
        Lower than default (0.25) to confirm tracks only from reasonably
        confident detections, reducing ghost tracks on noise.
    lost_track_buffer : 60
        Keeps a track alive for ~2 seconds (60 frames @ 30 fps) when the
        subject is temporarily occluded or leaves the frame momentarily.
    minimum_matching_threshold : 0.80
        High IoU requirement ensures IDs are not swapped between nearby
        players who cross paths.
    frame_rate : 30
        Target FPS; scales the `lost_track_buffer` duration.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.35,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.80,
        frame_rate: int = 30,
    ):
        logger.info("Initialising ByteTrack with sports-tuned parameters.")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        logger.info("ByteTrack ready.")

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update the tracker with detections from the current frame.

        Parameters
        ----------
        detections : sv.Detections
            Raw detections from the detector for this frame.

        Returns
        -------
        sv.Detections
            Tracked detections with `tracker_id` field populated.
            Objects that were not detected but are still tracked (lost)
            are kept in the tracker's internal buffer and will reappear
            when re-detected.
        """
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state (call between independent video clips)."""
        self.tracker.reset()
        logger.info("Tracker state reset.")
