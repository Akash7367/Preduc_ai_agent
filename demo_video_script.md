# 3-5 Minute Demo Video Script

Use this outline while recording your mandatory demo video.

## 0:00 - 0:30 | Problem and Objective

- Introduce the assignment: multi-object detection and persistent ID tracking in public sports footage.
- Mention the chosen detector and tracker: YOLOv8m + ByteTrack.
- State the input source URL from `README.md`.

## 0:30 - 1:30 | Project Structure

- Show key files:
  - `main.py`
  - `src/pipeline.py`
  - `src/detector.py`
  - `src/tracker.py`
  - `src/annotator.py`
  - `src/analytics.py`
- Explain modular design: download/input -> detect -> track -> annotate -> analytics.

## 1:30 - 2:30 | Run Command

- Run:
  - `python main.py --video input_video_raw.mp4 --output outputs/annotated_output.mp4 --conf 0.25`
- Briefly describe important arguments (`--video`, `--output`, `--conf`, `--frame_skip`, `--device`).

## 2:30 - 3:30 | Results Walkthrough

- Open and play `outputs/annotated_output.mp4`.
- Point out:
  - Per-person bounding boxes
  - Persistent `ID:<number>` labels
  - Trajectory traces
- Open and explain:
  - `outputs/heatmap.png`
  - `outputs/count_over_time.png`
  - `outputs/trajectories.png`
  - `outputs/screenshots/`

## 3:30 - 4:30 | Challenges, Limitations, Improvements

- Mention practical issues: occlusion, blur, similar appearance, camera motion.
- Mention known failure cases from `technical_report.md`.
- Mention next improvements: re-ID model, homography, ball tracking.

## 4:30 - 5:00 | Closing

- Reiterate that all mandatory deliverables are included.
- Show repository root and final artifact files.
