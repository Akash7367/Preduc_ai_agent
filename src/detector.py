"""
detector.py
-----------
YOLOv8-based object detector wrapper.

Loads a pre-trained YOLOv8 model and exposes a clean `detect(frame)`
interface that returns `supervision.Detections` objects.

By default only 'person' class (class ID 0) is returned, making it
suitable for sports footage with players/athletes.
"""

import logging
from typing import List, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class ID for 'person'
PERSON_CLASS_ID = 0


class PersonDetector:
    """
    Wraps YOLOv8 for person detection in sports footage.

    Parameters
    ----------
    model_path : str
        Path to YOLOv8 weights file, or a model name such as 'yolov8m.pt'.
        The model is downloaded automatically on first use.
    confidence : float
        Detection confidence threshold (0–1). Lower values catch more
        subjects but increase false positives.
    iou_threshold : float
        Non-maximum suppression IoU threshold.
    class_filter : list[int] | None
        List of COCO class IDs to keep. Defaults to [0] (person only).
        Pass None to keep all detected classes.
    device : str
        Inference device: 'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        class_filter: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter if class_filter is not None else [PERSON_CLASS_ID]
        self.device = device

        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        logger.info("Model loaded successfully.")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H, W, 3) as returned by OpenCV.

        Returns
        -------
        sv.Detections
            Supervision Detections object with filtered results.
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Filter to desired classes only
        if self.class_filter:
            mask = np.isin(detections.class_id, self.class_filter)
            detections = detections[mask]

        return detections
