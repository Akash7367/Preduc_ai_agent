# Multi-Object Detection & Persistent ID Tracking

> **Assignment:** AI / Computer Vision / Data Science  
> **Live Demo:** [Streamlit App](https://akash7367-preduc-ai-agent-app-iqx3qt.streamlit.app/)  
> **Video Source:** [FIFA World Cup 2022 Match Highlights — YouTube](https://www.youtube.com/watch?v=l7KpNAa3NhA)  
> **Detector:** YOLOv8m (Ultralytics)  
> **Tracker:** ByteTrack (via `supervision`)

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [Outputs](#outputs)
6. [Configuration](#configuration)
7. [Assumptions & Limitations](#assumptions--limitations)
8. [Model & Tracker Choices](#model--tracker-choices)

---

## Overview

This pipeline:
- **Downloads** a publicly available sports video using `yt-dlp`
- **Detects** all players in every frame using **YOLOv8m**
- **Tracks** them with persistent, unique IDs using **ByteTrack**
- **Annotates** the output video with bounding boxes, IDs, and trajectory traces
- **Generates analytics**: movement heatmap, player count over time, trajectory overlay, speed estimates

---

## Project Structure

```
preduck_ai_Agent/
├── src/
│   ├── __init__.py
│   ├── downloader.py     # yt-dlp video downloader with optional trimming
│   ├── detector.py       # YOLOv8 person detection wrapper
│   ├── tracker.py        # ByteTrack tracker with sports-tuned params
│   ├── annotator.py      # Supervision-based annotation (boxes, IDs, traces, heatmap)
│   ├── analytics.py      # Heatmap, count chart, trajectory, speed estimation
│   └── pipeline.py       # Orchestrates the full detection → tracking → annotation loop
├── configs/
│   └── bytetrack.yaml    # Custom ByteTrack hyperparameters
├── notebooks/
│   └── demo.ipynb        # Interactive Jupyter walkthrough
├── outputs/              # Generated artifacts (created at runtime)
│   ├── annotated_output.mp4
│   ├── heatmap.png
│   ├── count_over_time.png
│   └── trajectories.png
├── main.py               # CLI entry point
├── requirements.txt
├── README.md
└── technical_report.md
```

---

## Installation

### Prerequisites
- **Python 3.10+**
- **ffmpeg** — required for video trimming after download  
  - Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Linux/macOS: `sudo apt install ffmpeg` / `brew install ffmpeg`

### Steps

```bash
# 1. Clone or navigate to project
cd preduck_ai_Agent

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

YOLOv8 model weights (`yolov8m.pt`) are downloaded **automatically** from Ultralytics on first run (~50 MB).

---

## How to Run

### Option A — Download public video and run full pipeline
```bash
python main.py --url "https://www.youtube.com/watch?v=l7KpNAa3NhA"
```

### Option B — Run on a local video file
```bash
python main.py --video path/to/your/video.mp4
```

### Option C — Full control
```bash
python main.py \
  --video input_video.mp4 \
  --output outputs/annotated.mp4 \
  --model yolov8m.pt \
  --conf 0.40 \
  --frame_skip 1 \
  --device cpu \
  --trace_length 45
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | — | Public video URL to download |
| `--video` | — | Local video path (skips download) |
| `--output` | `outputs/annotated_output.mp4` | Output annotated video |
| `--model` | `yolov8m.pt` | YOLOv8 model variant |
| `--conf` | `0.35` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--frame_skip` | `1` | Process every Nth frame |
| `--device` | `cpu` | `cpu` / `cuda` / `mps` |
| `--max_duration` | `180` | Max video seconds to download |
| `--trace_length` | `45` | Trajectory trail length (frames) |
| `--no_analytics` | off | Skip analytics image generation |

### Jupyter Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/annotated_output.mp4` | Annotated video with bounding boxes, IDs, trajectory traces |
| `outputs/heatmap.png` | Full-video player position heatmap blended over first frame |
| `outputs/count_over_time.png` | Line chart of tracked player count per frame |
| `outputs/trajectories.png` | All player paths overlaid on a static reference frame |
| `outputs/screenshots/screenshot_1.png` ... `screenshot_4.png` | Sample result screenshots for submission |

### Submission Deliverables Status

- `README.md`: present
- `technical_report.md` (1-2 page technical explanation): present
- `outputs/annotated_output.mp4`: present
- Original public video URL: present at top of this README
- Sample screenshots: present in `outputs/screenshots/`
- Demo guide for 3-5 min recording: `demo_video_script.md`

---

## Configuration

Edit `configs/bytetrack.yaml` to tune tracker behaviour:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `track_high_thresh` | `0.5` | Min confidence to start a new track |
| `track_low_thresh` | `0.1` | Low-confidence detection buffer |
| `track_buffer` | `60` | Frames to keep lost track alive (2s @ 30fps) |
| `match_thresh` | `0.80` | IoU required to match detection to track |

---

## Assumptions & Limitations

### Assumptions
- The video contains **people** as the primary subjects (uses COCO `person` class).
- Frame rate is approximately 30 fps (ByteTrack buffer is calibrated to this).
- The pitch/court is approximately the width of the frame (used for rough speed estimation).
- Lighting conditions are reasonable — heavy shadow / night footage reduces detection accuracy.

### Limitations
- **ID switches** may occur during heavy crowd collisions where multiple players overlap for >2 seconds.
- **Speed estimation** is a rough approximation (pixel displacement × calibration factor) without camera calibration.
- **GPU required** for real-time processing; CPU inference on 720p @ 30fps takes ~3–5 FPS.
- **Re-identification** (re-assigning the same ID after a player exits and re-enters frame) is not implemented — this would require an appearance model (e.g., OSNet / ReID).
- `yt-dlp` download depends on the availability and format of the target URL.

---

## Model & Tracker Choices

### YOLOv8m (Detector)
- Best accuracy/speed balance among YOLOv8 variants for the target hardware.
- Pre-trained on COCO — `person` class is extremely well-calibrated.
- Handles small, fast-moving subjects better than YOLOv5.

### ByteTrack (Tracker)
- State-of-the-art tracking without requiring an appearance/re-ID model.
- Key innovation: keeps **low-confidence** detections in a secondary buffer, drastically reducing ID loss during occlusion.
- Outperforms SORT and DeepSORT on MOT benchmarks with lower computational overhead.

### supervision Library
- Clean abstraction layer over both YOLO outputs and tracker updates.
- Provides production-ready `BoxAnnotator`, `TraceAnnotator`, and `HeatMapAnnotator`.
