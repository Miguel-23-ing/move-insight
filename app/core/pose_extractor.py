"""
core/pose_extractor.py
YOLOv8-Pose processing pipeline.
Handles model loading, per-frame detection + EMA smoothing, delegates
drawing and metrics to the provided AnalysisModule, writes annotated video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from core.smoothing import EMAFilter
from analysis.base import AnalysisModule

# ── Model discovery ───────────────────────────────────────────────────────────
_CANDIDATE_PATHS = [
    Path(__file__).parent.parent.parent / "yolov8n-pose.pt",  # repo root (../Papá)
    Path(__file__).parent.parent / "yolov8n-pose.pt",          # app/
    Path("yolov8n-pose.pt"),                                    # cwd
]

_model_cache: dict[str, object] = {}


def _find_model_path() -> str:
    for p in _CANDIDATE_PATHS:
        if p.exists():
            return str(p)
    # Fallback – ultralytics will download it automatically
    return "yolov8n-pose.pt"


def get_model(model_path: str | None = None):
    """Return a cached YOLO model instance (loads once per process)."""
    from ultralytics import YOLO  # imported here to allow the rest of the module
    path = model_path or _find_model_path()
    if path not in _model_cache:
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


# ── Main processing function ──────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_path: str,
    module: AnalysisModule,
    model_path: str | None = None,
    alpha: float = 0.4,
    confidence_min: float = 0.35,
    visibility_threshold: float | None = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> tuple[list[dict], dict]:
    """Process a video frame-by-frame using the given analysis module.

    Parameters
    ----------
    video_path        : path to input MP4 video
    output_path       : path for the annotated output video
    module            : AnalysisModule to use for drawing & metrics
    model_path        : optional override for the .pt model path
    alpha             : EMA smoothing factor
    confidence_min    : minimum person-detection confidence (box-level)
    visibility_threshold : per-keypoint confidence threshold for module
    progress_callback : optional callable (fraction_done, frame_idx, total_frames)

    Returns
    -------
    metrics_history : list[dict]  – one metric dict per frame
    video_info      : dict        – fps, total_frames, duration_s, width, height
    """
    import subprocess
    from pathlib import Path as PathlibPath
    
    model   = get_model(model_path)
    smoother = EMAFilter(alpha=alpha)

    # Override module's visibility threshold if provided
    if visibility_threshold is not None:
        module.vis_thr = visibility_threshold

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prefer H.264/AVC for better browser compatibility. Fall back to MPEG-4
    # if the encoder is not available in the current OpenCV build.
    fourcc = None
    selected_codec = None
    for codec_name in ["mp4v", "avc1", "H264", "MJPG"]:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        test_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if test_writer.isOpened():
            test_writer.release()
            selected_codec = codec_name
            break
        test_writer.release()
    
    if selected_codec is None:
        raise RuntimeError(f"Cannot find a suitable video codec for {output_path}")
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    metrics_history: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            smoothed_pts = _detect_and_smooth(
                frame, model, smoother,
                confidence_min=confidence_min,
                n_kps=17,
            )

            if smoothed_pts is not None:
                frame_metrics = module.process_frame(frame, smoothed_pts, frame_idx, fps)
            else:
                frame_metrics = {}

            metrics_history.append(frame_metrics)
            writer.write(frame)
            frame_idx += 1

            if progress_callback is not None:
                frac = frame_idx / max(total, 1)
                progress_callback(frac, frame_idx, total)

    finally:
        cap.release()
        writer.release()

    # Re-encode with ffmpeg for web compatibility
    try:
        temp_path = str(PathlibPath(output_path).parent / f"temp_{PathlibPath(output_path).stem}.mp4")
        subprocess.run([
            "ffmpeg", "-i", output_path, "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-y", temp_path
        ], capture_output=True, timeout=120)
        
        if PathlibPath(temp_path).exists():
            PathlibPath(output_path).unlink()
            PathlibPath(temp_path).rename(output_path)
    except Exception:
        pass  # If ffmpeg fails, use the original file

    duration_s = frame_idx / fps if fps else 0.0
    video_info = {
        "fps":          fps,
        "total_frames": frame_idx,
        "duration_s":   duration_s,
        "width":        width,
        "height":       height,
    }
    return metrics_history, video_info


# ── Internal detection + smoothing ────────────────────────────────────────────

def _detect_and_smooth(
    frame: np.ndarray,
    model,
    smoother: EMAFilter,
    confidence_min: float,
    n_kps: int = 17,
) -> list[dict] | None:
    """Run YOLO on *frame*, select the best detection, apply EMA.

    Returns a list of 17 smoothed keypoint dicts, or None if no detection.
    """
    results = model(frame, verbose=False, conf=confidence_min)
    if not results:
        return None

    result = results[0]
    if (
        result.boxes is None
        or len(result.boxes) == 0
        or result.keypoints is None
    ):
        return None

    # Select person with highest bounding-box confidence
    confs_det = result.boxes.conf.detach().cpu().numpy()
    best      = int(np.argmax(confs_det))
    if float(confs_det[best]) < confidence_min:
        return None

    kps   = result.keypoints.xy[best].cpu().numpy()           # (N, 2)
    confs = (
        result.keypoints.conf[best].cpu().numpy()
        if result.keypoints.conf is not None
        else np.ones(n_kps, dtype=np.float32)
    )

    return smoother.update_batch(kps, confs)
