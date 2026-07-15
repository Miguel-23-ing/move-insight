"""
core/smoothing.py
Exponential Moving Average (EMA) filter for 2-D pose keypoint smoothing.
Refactored from process_all_videos.py – reusable across all analysis modules.
"""
from __future__ import annotations

import numpy as np


class EMAFilter:
    """Per-keypoint Exponential Moving Average smoothing filter.

    Parameters
    ----------
    alpha : float
        Smoothing factor ∈ (0, 1].  Higher → follows new data faster (less smooth).
        Recommended range: 0.25 (very smooth) – 0.6 (responsive).
    """

    def __init__(self, alpha: float = 0.4) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._history: dict[int, dict[str, float]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    def update(self, idx: int, x: float, y: float, conf: float) -> dict[str, float]:
        """Apply EMA to a single keypoint and return the smoothed state.

        Parameters
        ----------
        idx  : keypoint index (0–16 for YOLOv8-Pose)
        x, y : raw pixel coordinates
        conf : detection confidence

        Returns
        -------
        dict with keys 'x', 'y', 'conf'.
        """
        if idx not in self._history:
            self._history[idx] = {"x": float(x), "y": float(y), "conf": float(conf)}
        elif float(conf) > 0:
            h = self._history[idx]
            a = self.alpha
            h["x"]    = a * float(x)    + (1.0 - a) * h["x"]
            h["y"]    = a * float(y)    + (1.0 - a) * h["y"]
            h["conf"] = float(conf)
        return dict(self._history[idx])

    def update_batch(
        self, keypoints: np.ndarray, confidences: np.ndarray
    ) -> list[dict[str, float]]:
        """Batch-update all keypoints from YOLOv8 output arrays.

        Parameters
        ----------
        keypoints   : ndarray of shape (N, 2) – pixel xy coordinates
        confidences : ndarray of shape (N,)   – per-keypoint confidence

        Returns
        -------
        List of N smoothed keypoint dicts.
        """
        return [
            self.update(i, float(pt[0]), float(pt[1]), float(conf))
            for i, (pt, conf) in enumerate(zip(keypoints, confidences))
        ]

    def reset(self) -> None:
        """Clear all stored history (call between different videos)."""
        self._history.clear()

    def get(self, idx: int) -> dict[str, float] | None:
        """Return current smoothed state for keypoint *idx*, or None."""
        return dict(self._history[idx]) if idx in self._history else None


# ── Scalar EMA helper ─────────────────────────────────────────────────────────

def smooth_scalar(
    prev: float | None, new: float | None, alpha: float = 0.4
) -> float | None:
    """EMA for a single scalar value (e.g. a computed angle).

    Mirrors the ``_smooth_value`` helper from process_all_videos.py.
    Returns *prev* when *new* is None (retain last valid value).
    """
    if new is None:
        return prev
    if prev is None:
        return new
    return alpha * float(new) + (1.0 - alpha) * float(prev)
