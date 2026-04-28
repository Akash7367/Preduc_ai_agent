"""
downloader.py
-------------
Downloads a publicly available video using yt-dlp.
Supports optional trimming to a specified duration (in seconds) to keep
processing time manageable.
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def download_video(
    url: str,
    output_path: str = "input_video.mp4",
    max_duration_sec: int = 180,
) -> str:
    """
    Download a video from a public URL using yt-dlp.

    Parameters
    ----------
    url : str
        Public video URL (YouTube, Vimeo, etc.)
    output_path : str
        Local file path to save the downloaded video.
    max_duration_sec : int
        Maximum duration to keep (seconds). Video is trimmed using ffmpeg
        after download if it exceeds this limit. Set to 0 to skip trimming.

    Returns
    -------
    str
        Absolute path to the downloaded (and optionally trimmed) video.
    """
    output_path = os.path.abspath(output_path)
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    if os.path.exists(output_path):
        logger.info(f"Video already exists at {output_path}. Skipping download.")
        return output_path

    logger.info(f"Downloading video from: {url}")

    ydl_cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", raw_path,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(ydl_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    logger.info(f"Download complete: {raw_path}")

    if max_duration_sec > 0:
        logger.info(f"Trimming video to {max_duration_sec}s...")
        trim_cmd = [
            "ffmpeg", "-y",
            "-i", raw_path,
            "-t", str(max_duration_sec),
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(trim_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("ffmpeg trim failed, using full raw video instead.")
            os.rename(raw_path, output_path)
        else:
            os.remove(raw_path)
            logger.info(f"Trimmed video saved to: {output_path}")
    else:
        os.rename(raw_path, output_path)

    return output_path
