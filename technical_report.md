# Technical Report: Multi-Object Detection and Persistent ID Tracking

**Project:** AI / Computer Vision Assignment — Sports Player Tracking  
**Video Source:** FIFA World Cup 2022 Match Highlights  
**URL:** https://www.youtube.com/watch?v=l7KpNAa3NhA  
**Date:** April 2026  

---

## 1. Introduction

This report describes the design and implementation of a multi-object detection and persistent ID tracking pipeline applied to public sports footage. The goal is to detect all visible players in a football match video and assign each a unique ID that remains stable across the full video duration — even when players overlap, move rapidly, or temporarily leave the frame.

---

## 2. Model and Detector

### 2.1 Detector: YOLOv8m

**YOLOv8** (You Only Look Once, version 8) by Ultralytics is a single-stage, anchor-free object detector that achieves state-of-the-art accuracy on the COCO benchmark. The **medium** variant (`yolov8m`) was selected as it provides the best balance between:

- **Accuracy:** 50.2% mAP@50-95 on COCO val2017
- **Speed:** ~17ms/frame on CPU (versus 7ms for `n`, 40ms for `l`)
- **Small object handling:** Key for tracking distant players at the far end of the pitch

The model is pre-trained on COCO (80 classes). Only the **person class (ID 0)** is extracted for this pipeline.

**Detection parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `confidence` | 0.35 | Lower threshold to catch partially visible players |
| `iou` (NMS) | 0.45 | Removes overlapping boxes while keeping distinct players |
| `class_filter` | [0] (person) | Eliminates ball, referee equipment false-positives |

---

## 3. Tracking Algorithm

### 3.1 Tracker: ByteTrack

**ByteTrack** (Zhang et al., ECCV 2022) is used for multi-object tracking. Unlike traditional trackers (SORT, DeepSORT) which discard low-confidence detections, ByteTrack uses a **two-stage association** strategy:

1. **First pass:** Match high-confidence detections (conf > 0.5) to existing tracks using IoU.
2. **Second pass:** Match remaining low-confidence detections to unmatched tracks — this is the key innovation that maintains IDs during partial occlusion.

This approach is particularly effective in sports footage where a player may be:
- Partially hidden behind another player (detection confidence drops)
- Momentarily off-screen (track buffer keeps ID alive for 60 frames / ~2s)

**ByteTrack parameters (sports-tuned):**
| Parameter | Value | Default | Rationale |
|-----------|-------|---------|-----------|
| `track_activation_threshold` | 0.35 | 0.25 | Reduces ghost tracks on crowd noise |
| `lost_track_buffer` | 60 frames | 30 | Maintains ID for 2s during occlusion |
| `minimum_matching_threshold` | 0.80 | 0.80 | High IoU prevents ID swaps between nearby players |

### 3.2 Implementation: supervision Library

The Roboflow `supervision` library provides a production-grade wrapper around ByteTrack, handling:
- Conversion between YOLO output and `sv.Detections` objects
- Track ID persistence (`tracker_id` field on each detection)
- Annotator components for boxes, labels, traces, and heatmaps

---

## 4. ID Consistency Strategy

Several mechanisms work together to maintain stable IDs:

1. **ByteTrack secondary buffer:** Low-confidence detections (0.1–0.5) are matched to existing tracks before being discarded, preventing ID loss during partial occlusion.
2. **Long track buffer (60 frames):** A player who exits the frame for up to 2 seconds retains their ID if they re-enter the same region.
3. **High IoU matching (0.80):** Two players crossing paths are unlikely to share IoU > 0.80, preventing ID swaps.
4. **Trajectory traces:** The 45-frame trail provides visual confirmation of ID continuity across the video.

**Known failure cases:**
- Player pairs that physically overlap for more than 60 frames — IDs may swap post-occlusion.
- Camera cuts (scene changes) — all tracks reset since spatial context is lost.
- Players at the edge of frame who are barely visible — may be re-assigned new IDs.

---

## 5. Analytics and Visualisations

### 5.1 Movement Heatmap
Gaussian blobs centred on each detection centre are accumulated across all frames into a float array. The final array is normalised and colourised using the `JET` colormap. This shows which areas of the pitch saw the most activity.

### 5.2 Count Over Time
A matplotlib line chart records the number of tracked players per processed frame, revealing phases of play (open play, corner kicks, penalties) by player density changes.

### 5.3 Trajectory Overlay
All player path histories are drawn as coloured polylines on a static reference frame, showing movement patterns, heat corridors, and player roles.

### 5.4 Speed Estimation
Player speed is approximated from pixel displacement between consecutive frames:

```
speed_km/h = (pixel_distance / pixels_per_metre) × fps × 3.6
```

With a calibration factor of ~8 pixels/metre (for a 720p wide-angle pitch shot), players average ~12 km/h and peak near ~28 km/h. This is a rough estimate without camera calibration or homography.

---

## 6. Challenges Faced

| Challenge | Approach |
|-----------|----------|
| Dense player clusters (corner kicks) | ByteTrack secondary buffer + high IoU matching |
| Camera panning / zoom | IoU-based matching (not appearance-based) is robust to scale changes |
| Similar-looking players (same kit colour) | ByteTrack does not rely on appearance — purely positional |
| Motion blur at high frame rates | Lower confidence threshold (0.35) catches blurred detections |
| Referee / stadium staff detection | Class filter (person only) — referees are detected; separate role clustering not implemented |

---

## 7. Failure Cases Observed

1. **ID swap after group collision:** When 3+ players converge then separate, the tracker occasionally reassigns IDs incorrectly after the crowd disperses.
2. **Ball detection:** The football is not tracked (below the `person` class filter). A second detection pass with a ball-specific model would be needed.
3. **Substitutes re-entering:** A player who leaves and re-enters the frame after > 2 seconds receives a new ID (no re-identification model is implemented).
4. **Camera cut resets:** Any hard scene transition causes all active tracks to be treated as new, increasing total unique ID count.

---

## 8. Possible Improvements

1. **Appearance Re-ID model** (e.g., OSNet, BoT-SORT): Would allow matching players across camera cuts or long-duration occlusions using visual features.
2. **Homographic projection:** Map detections to a top-down pitch coordinate system for more accurate speed estimation and zone analysis.
3. **Team clustering:** K-means on jersey colour histograms within bounding boxes can automatically separate teams and referees.
4. **Ball tracking:** Dedicated small-object detector (e.g., TrackNet) for ball trajectory analysis.
5. **Optical flow refinement:** Use Lucas-Kanade optical flow to smooth tracker predictions between detection frames.
6. **Camera motion compensation:** Subtract camera motion vector from displacement before speed estimation for more accurate results during pans.

---

## 9. Conclusion

The YOLOv8m + ByteTrack combination proved highly effective for sports player tracking. ByteTrack's two-stage association strategy is the critical enabler for ID stability in the dense, high-motion environment of a football match. The pipeline is modular, well-documented, and extensible — a re-ID model or homography projection can be plugged in at well-defined interface points without refactoring the core loop.

---

*References*  
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics  
- ByteTrack (Zhang et al.): https://arxiv.org/abs/2110.06864  
- Roboflow Supervision: https://github.com/roboflow/supervision  
