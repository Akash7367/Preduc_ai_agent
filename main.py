"""
main.py
-------
CLI entry point for the Multi-Object Detection & Persistent ID Tracking pipeline.

Usage examples
--------------
  # Download public video and run full pipeline
  python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

  # Run on an already-downloaded video
  python main.py --video input_video.mp4 --output outputs/annotated.mp4

  # Faster processing (every 2nd frame)
  python main.py --video input_video.mp4 --frame_skip 2

  # Use GPU
  python main.py --video input_video.mp4 --device cuda

  # Higher confidence threshold (fewer false positives)
  python main.py --video input_video.mp4 --conf 0.5
"""

import argparse
import logging
import os
import sys

# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Public sports video (Creative-Commons / publicly accessible) ───────────────
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # placeholder
PUBLIC_FOOTBALL_URL = (
    "https://www.youtube.com/watch?v=l7KpNAa3NhA"  # FIFA World Cup highlight reel
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Input ---------------------------------------------------------------
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--url",
        type=str,
        default=None,
        help="Public video URL to download (YouTube / Vimeo / etc.)",
    )
    group.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a local video file (skip download).",
    )

    # ---- Output --------------------------------------------------------------
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/annotated_output.mp4",
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--analytics_dir",
        type=str,
        default="outputs",
        help="Directory to save analytics images.",
    )

    # ---- Model / tracker -----------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLOv8 model name or path to weights file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (0–1).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (0–1).",
    )

    # ---- Performance ---------------------------------------------------------
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame, 2 = every other, …).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=180,
        help="Maximum video duration in seconds (0 = no limit).",
    )

    # ---- Misc ----------------------------------------------------------------
    parser.add_argument(
        "--trace_length",
        type=int,
        default=45,
        help="Number of past frames shown in trajectory traces.",
    )
    parser.add_argument(
        "--no_analytics",
        action="store_true",
        help="Skip saving analytics images.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Determine input video path ------------------------------------------
    video_path: str

    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
        logger.info(f"Using local video: {video_path}")

    else:
        url = args.url or PUBLIC_FOOTBALL_URL
        video_path = "input_video.mp4"
        logger.info(f"Downloading video: {url}")
        from src.downloader import download_video
        try:
            video_path = download_video(
                url=url,
                output_path=video_path,
                max_duration_sec=args.max_duration,
            )
        except Exception as exc:
            logger.error(f"Download failed: {exc}")
            logger.error(
                "Tip: Install yt-dlp and ffmpeg, or pass --video <local_file.mp4> directly."
            )
            sys.exit(1)

    # ---- Run pipeline --------------------------------------------------------
    from src.pipeline import run_pipeline

    run_pipeline(
        input_path=video_path,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        frame_skip=args.frame_skip,
        trace_length=args.trace_length,
        device=args.device,
        save_analytics=not args.no_analytics,
        analytics_dir=args.analytics_dir,
    )

    logger.info("Done! Check the 'outputs/' directory for all results.")


if __name__ == "__main__":
    main()
