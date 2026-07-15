"""
analysis/base.py
Abstract base class + biomechanical helper library.
Includes cycle detection, symmetry index, CV, angular velocity,
cadence estimation, stride regularity and oscillation frequency.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """Container for one computed metric with all UI metadata."""
    label: str                    # Display name
    value: float | None           # Numeric result; None if not calculable
    unit: str                     # "°", "%", "pasos/min", "°/s", etc.
    ref_range: str                # Orientative reference string
    status: str                   # "normal" | "atencion" | "alerta" | "sin_datos"
    description: str              # Clinical interpretation sentence
    calculable: bool = True       # False when insufficient data for a real measurement
    coverage_pct: float | None = None  # % of video frames with valid keypoints
    sample_count: int = 0         # Number of valid frame samples used
    deviation: str | None = None  # "high" | "low" when outside reference range


# ── Geometric helpers ─────────────────────────────────────────────────────────

def angle_three_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float | None:
    """Angle in degrees at vertex *b* (vectors b→a and b→c).  None if degenerate."""
    ba, bc = a - b, c - b
    na, nc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if na == 0.0 or nc == 0.0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc)) / (na * nc)))))


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float | None:
    """Angle in degrees between two 2-D vectors.  None if degenerate."""
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))))


def point_line_distance_norm(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float | None:
    """Perpendicular distance of *p* from line *a*→*b*, normalised by |ab| (0–1 ratio).

    Proxy for genu valgo/varo in vista frontal. En vista sagital 2D la rodilla suele
    proyectarse sobre la línea cadera-tobillo, por lo que este índice puede ser ~0
    aunque exista valgo/varo real (limitación documentada en el reporte).
    """
    ab = b - a
    ab_len = float(np.linalg.norm(ab))
    if ab_len < 1.0:
        return None
    ap = p - a
    cross = float(ab[0] * ap[1] - ab[1] * ap[0])
    return abs(cross) / (ab_len * ab_len)


def knee_alignment_pct(
    knee: np.ndarray, hip: np.ndarray, ankle: np.ndarray
) -> float | None:
    """Lateral knee deviation as % of hip–ankle segment length (frontal-view proxy)."""
    ratio = point_line_distance_norm(knee, hip, ankle)
    return round(ratio * 100.0, 2) if ratio is not None else None


def hip_span_ratio(
    l_hip: np.ndarray, r_hip: np.ndarray,
    l_shoulder: np.ndarray, r_shoulder: np.ndarray,
) -> float | None:
    """Horizontal hip span / torso height — proxy for frontal vs sagittal camera view.

    Values > ~0.18 suggest frontal/dorsal view; < ~0.12 suggests sagittal/profile view
    where knee valgus/varo cannot be measured reliably in 2D.
    """
    hip_w = abs(float(r_hip[0] - l_hip[0]))
    torso_h = abs(float((l_shoulder[1] + r_shoulder[1]) / 2 - (l_hip[1] + r_hip[1]) / 2))
    if torso_h < 1.0:
        return None
    return hip_w / torso_h


# ── Abstract base class ───────────────────────────────────────────────────────

class AnalysisModule(ABC):
    """
    Abstract base for all body-zone analysis modules.

    Subclasses define class attributes:
      NAME, DESCRIPTION, THRESHOLDS, REFERENCE_RANGES, METRIC_LABELS, METRIC_UNITS

    Threshold schema per key:
      {"better": "lower"|"higher"|"range",
       "normal_max": float, "warn_max": float,   # for "lower"
       "normal_min": float, "warn_min": float,   # for "higher"
       "normal_min","normal_max","warn_min","warn_max" }   # for "range"
    """

    NAME: str              = "Análisis"
    DESCRIPTION: str       = ""
    THRESHOLDS: dict       = {}
    REFERENCE_RANGES: dict = {}
    METRIC_LABELS: dict    = {}
    METRIC_UNITS: dict     = {}
    DEFAULT_VISIBILITY_THRESHOLD: float = 0.15

    def __init__(self, visibility_threshold: float | None = None) -> None:
        self.vis_thr = (
            visibility_threshold
            if visibility_threshold is not None
            else self.DEFAULT_VISIBILITY_THRESHOLD
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def get_relevant_keypoints(self) -> list[int]: ...

    @abstractmethod
    def get_connections(self) -> list[tuple[int, int]]: ...

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, smoothed_pts: list[dict],
        frame_idx: int, fps: float,
    ) -> dict[str, float | None]: ...

    @abstractmethod
    def compute_summary(
        self, metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]: ...

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, key: str, value: float | None) -> str:
        if value is None:
            return "sin_datos"
        if key not in self.THRESHOLDS:
            return "normal"
        t    = self.THRESHOLDS[key]
        mode = t.get("better", "lower")
        if mode == "lower":
            if value <= t.get("normal_max", float("inf")):   return "normal"
            if value <= t.get("warn_max",   float("inf")):   return "atencion"
            return "alerta"
        if mode == "higher":
            if value >= t.get("normal_min", float("-inf")):  return "normal"
            if value >= t.get("warn_min",   float("-inf")):  return "atencion"
            return "alerta"
        if mode == "range":
            nlo, nhi = t.get("normal_min", 0.0), t.get("normal_max", 9999.0)
            wlo, whi = t.get("warn_min",   0.0), t.get("warn_max",   9999.0)
            if nlo <= value <= nhi: return "normal"
            if wlo <= value <= whi: return "atencion"
            return "alerta"
        return "normal"

    def _deviation_direction(self, key: str, value: float | None, status: str) -> str | None:
        if value is None or status in ("normal", "sin_datos") or key not in self.THRESHOLDS:
            return None
        t = self.THRESHOLDS[key]
        mode = t.get("better", "lower")
        if mode == "lower":
            return "high"
        if mode == "higher":
            return "low"
        if mode == "range":
            nlo, nhi = t.get("normal_min", 0.0), t.get("normal_max", 9999.0)
            if value < nlo:
                return "low"
            if value > nhi:
                return "high"
        return "high"

    # ── Keypoint helpers ──────────────────────────────────────────────────────

    def _pt(self, pts: list[dict], idx: int) -> tuple[np.ndarray, float]:
        p = pts[idx]
        return np.array([p["x"], p["y"]], dtype=np.float32), p["conf"]

    def _valid(self, *confs: float) -> bool:
        return all(c >= self.vis_thr for c in confs)

    def _valid_majority(self, *confs: float, min_count: int = 2) -> bool:
        """At least *min_count* keypoints above visibility threshold."""
        return sum(1 for c in confs if c >= self.vis_thr) >= min_count

    def _series(
        self, history: list[dict[str, float | None]], key: str
    ) -> list[float]:
        return [v for frame in history if (v := frame.get(key)) is not None]

    def _coverage(
        self, history: list[dict[str, float | None]], key: str
    ) -> tuple[float, int]:
        """Return (coverage_pct, sample_count) for a metric key across the video."""
        if not history:
            return 0.0, 0
        count = sum(1 for frame in history if frame.get(key) is not None)
        return count / len(history) * 100.0, count

    def _frontal_view_pct(
        self,
        history: list[dict[str, float | None]],
        key: str = "hip_span",
        min_ratio: float = 0.16,
    ) -> float:
        """Percentage of frames where hip span suggests frontal/dorsal camera view."""
        spans = [v for frame in history if (v := frame.get(key)) is not None]
        if not spans:
            return 0.0
        frontal = sum(1 for v in spans if v >= min_ratio)
        return frontal / len(spans) * 100.0

    @staticmethod
    def _alignment_is_measurable(
        series: list[float],
        frontal_pct: float,
        min_frontal_pct: float = 25.0,
    ) -> tuple[bool, str]:
        """Return whether knee valgus/varo proxy is valid for this video."""
        if not series:
            return False, "sin muestras válidas de alineación"
        if frontal_pct < min_frontal_pct:
            return False, (
                "vista sagital/lateral detectada — este índice requiere "
                "vista frontal o posterior del miembro inferior"
            )
        if max(series) < 1.5 and float(np.std(series)) < 0.6:
            return False, (
                "desviación lateral no detectable en 2D — probable vista sagital "
                "o proyección que colapsa el valgo/varo"
            )
        return True, ""

    # ── Basic statistics ──────────────────────────────────────────────────────

    @staticmethod
    def _rom(series: list[float]) -> float | None:
        return (max(series) - min(series)) if len(series) >= 5 else None

    @staticmethod
    def _mean(series: list[float]) -> float | None:
        return float(np.mean(series)) if series else None

    @staticmethod
    def _std(series: list[float]) -> float | None:
        return float(np.std(series)) if len(series) >= 3 else None

    # ── Cycle detection ───────────────────────────────────────────────────────

    @staticmethod
    def _find_peaks(series: list[float], min_distance: int = 8) -> list[int]:
        """Indices of local maxima with a minimum-distance constraint."""
        peaks: list[int] = []
        for i in range(1, len(series) - 1):
            if series[i] >= series[i - 1] and series[i] >= series[i + 1]:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        return peaks

    @staticmethod
    def _find_valleys(series: list[float], min_distance: int = 8) -> list[int]:
        """Indices of local minima with a minimum-distance constraint."""
        valleys: list[int] = []
        for i in range(1, len(series) - 1):
            if series[i] <= series[i - 1] and series[i] <= series[i + 1]:
                if not valleys or i - valleys[-1] >= min_distance:
                    valleys.append(i)
        return valleys

    @staticmethod
    def _per_cycle_rom(
        series: list[float], fps: float = 30.0, min_cycle_s: float = 0.4
    ) -> float | None:
        """Mean ROM across detected movement cycles.
        More robust than global max-min because it is not biased by outliers."""
        if len(series) < 15:
            return AnalysisModule._rom(series)
        min_d  = max(4, int(fps * min_cycle_s * 0.5))
        peaks  = AnalysisModule._find_peaks(series, min_d)
        valleys= AnalysisModule._find_valleys(series, min_d)
        if len(peaks) < 2 or not valleys:
            return AnalysisModule._rom(series)
        roms: list[float] = []
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            vs = [series[v] for v in valleys if p1 <= v <= p2]
            if vs:
                roms.append(series[p1] - min(vs))
        return float(np.mean(roms)) if roms else AnalysisModule._rom(series)

    @staticmethod
    def _count_cycles(series: list[float], fps: float = 30.0, min_cycle_s: float = 0.4) -> int:
        """Number of detected movement cycles (peak-to-peak)."""
        min_d = max(4, int(fps * min_cycle_s * 0.5))
        peaks = AnalysisModule._find_peaks(series, min_d)
        return max(0, len(peaks) - 1)

    # ── Symmetry & variability ────────────────────────────────────────────────

    @staticmethod
    def _symmetry_index(left: float | None, right: float | None) -> float | None:
        """Symmetry Index: 2·|L−R|/(L+R)·100%.
        0% = perfect symmetry.  Clinically meaningful: <10% normal, >20% significant."""
        if left is None or right is None:
            return None
        denom = abs(left) + abs(right)
        if denom < 0.001:
            return 0.0
        return min(200.0, 2.0 * abs(left - right) / denom * 100.0)

    @staticmethod
    def _cv(series: list[float]) -> float | None:
        """Coefficient of Variation (std/|mean|·100%).
        Measures movement consistency: <10% = very consistent, >20% = highly variable."""
        if len(series) < 5:
            return None
        m = abs(float(np.mean(series)))
        if m < 0.5:
            return None
        return float(np.std(series)) / m * 100.0

    # ── Angular kinematics ────────────────────────────────────────────────────

    @staticmethod
    def _angular_velocity(series: list[float], fps: float) -> float | None:
        """Mean absolute angular velocity in degrees/second."""
        if len(series) < 2 or fps <= 0:
            return None
        diffs = [abs(series[i + 1] - series[i]) * fps for i in range(len(series) - 1)]
        return float(np.mean(diffs)) if diffs else None

    @staticmethod
    def _peak_velocity(series: list[float], fps: float) -> float | None:
        """95th percentile angular velocity (deg/s) — captures peak movement speed."""
        if len(series) < 10 or fps <= 0:
            return None
        diffs = [abs(series[i + 1] - series[i]) * fps for i in range(len(series) - 1)]
        return float(np.percentile(diffs, 95))

    # ── Gait-specific ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_cadence(
        series: list[float], fps: float, min_cycle_s: float = 0.6
    ) -> float | None:
        """Estimate walking cadence (steps/min) from an oscillating angle series.
        Uses peak-to-peak intervals; each full cycle ≈ 2 steps.
        Reference: normal adults 100-120 steps/min (Perry & Burnfield 2010)."""
        min_d = max(5, int(fps * min_cycle_s))
        peaks = AnalysisModule._find_peaks(series, min_d)
        if len(peaks) < 2:
            return None
        intervals   = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        avg_interval = float(np.mean(intervals))
        if avg_interval < 1:
            return None
        cycle_time_s = avg_interval / fps
        return (2.0 / cycle_time_s) * 60.0           # 2 steps per cycle

    @staticmethod
    def _stride_regularity(series: list[float]) -> float | None:
        """Stride-to-stride regularity via autocorrelation (Moe-Nilssen 1998).
        Returns dominant autocorrelation peak: 1.0 = perfectly regular, 0 = random."""
        if len(series) < 30:
            return None
        arr  = np.array(series, dtype=float) - float(np.mean(series))
        norm = float(np.dot(arr, arr))
        if norm < 0.001:
            return None
        corr = np.correlate(arr, arr, mode="full")
        corr = corr[len(corr) // 2:]                 # positive lags only
        corr = corr / norm
        for i in range(2, len(corr) - 1):
            if corr[i] > corr[i - 1] and corr[i] > corr[i + 1]:
                return float(np.clip(corr[i], 0.0, 1.0))
        return None

    @staticmethod
    def _oscillation_frequency(series: list[float], fps: float) -> float | None:
        """Dominant oscillation frequency (Hz) via FFT."""
        if len(series) < 30 or fps <= 0:
            return None
        arr     = np.array(series, dtype=float) - float(np.mean(series))
        mag     = np.abs(np.fft.rfft(arr))
        freqs   = np.fft.rfftfreq(len(arr), d=1.0 / fps)
        if len(mag) < 2:
            return None
        peak_i  = int(np.argmax(mag[1:])) + 1
        return float(freqs[peak_i])

    # ── MetricResult factory ──────────────────────────────────────────────────

    def _make_metric_from_series(
        self,
        key: str,
        value: float | None,
        history: list[dict[str, float | None]],
        raw_key: str,
        description: str = "",
        min_coverage_pct: float = 8.0,
        min_samples: int = 5,
    ) -> MetricResult:
        """Build MetricResult with automatic coverage from frame history."""
        cov, n = self._coverage(history, raw_key)
        return self._make_metric(
            key,
            value,
            description=description,
            coverage_pct=cov,
            sample_count=n,
            min_coverage_pct=min_coverage_pct,
            min_samples=min_samples,
        )

    def _make_metric(
        self,
        key: str,
        value: float | None,
        description: str = "",
        override_status: Optional[str] = None,
        coverage_pct: float | None = None,
        sample_count: int = 0,
        min_coverage_pct: float = 8.0,
        min_samples: int = 5,
    ) -> MetricResult:
        label = self.METRIC_LABELS.get(key, key)
        unit  = self.METRIC_UNITS.get(key, "°")
        ref   = self.REFERENCE_RANGES.get(key, "N/D")

        calculable = value is not None
        if calculable and sample_count > 0 and sample_count < min_samples:
            calculable = False
        if calculable and coverage_pct is not None and coverage_pct < min_coverage_pct:
            calculable = False

        if not calculable:
            return MetricResult(
                label=label,
                value=None,
                unit=unit,
                ref_range=ref,
                status="sin_datos",
                description=(
                    description or
                    f"{label}: no calculable con la información disponible "
                    f"(cobertura insuficiente de keypoints o ángulo de cámara no compatible)."
                ),
                calculable=False,
                coverage_pct=coverage_pct,
                sample_count=sample_count,
                deviation=None,
            )

        status = override_status if override_status else self.classify(key, value)
        deviation = self._deviation_direction(key, value, status)
        rounded = round(float(value), 2) if unit in ("", "%") and abs(value) < 2 else round(float(value), 1)
        return MetricResult(
            label=label,
            value=rounded,
            unit=unit,
            ref_range=ref,
            status=status,
            description=description or self._auto_describe(key, value, status),
            calculable=True,
            coverage_pct=coverage_pct,
            sample_count=sample_count,
            deviation=deviation,
        )

    def _make_alignment_metric(
        self,
        key: str,
        series: list[float],
        history: list[dict[str, float | None]],
        raw_key: str,
        describe_fn=None,
        min_coverage_pct: float = 5.0,
    ) -> MetricResult:
        """Build knee alignment metric; marks sin_datos when frontal view is insufficient."""
        label = self.METRIC_LABELS.get(key, key)
        unit = self.METRIC_UNITS.get(key, "%")
        ref = self.REFERENCE_RANGES.get(key, "N/D")
        cov, n = self._coverage(history, raw_key)
        frontal_pct = self._frontal_view_pct(history)
        measurable, reason = self._alignment_is_measurable(series, frontal_pct)

        if not series or n < 5 or cov < min_coverage_pct or not measurable:
            detail = reason or "cobertura insuficiente de keypoints"
            return MetricResult(
                label=label,
                value=None,
                unit=unit,
                ref_range=ref,
                status="sin_datos",
                description=(
                    f"{label}: no calculable con la información disponible ({detail})."
                ),
                calculable=False,
                coverage_pct=cov,
                sample_count=n,
                deviation=None,
            )

        val = self._mean(series)
        st = self.classify(key, val)
        desc = (
            describe_fn(key, val, st)
            if describe_fn
            else self._auto_describe(key, val, st)
        )
        return self._make_metric(
            key, val,
            description=desc,
            coverage_pct=cov,
            sample_count=n,
            min_coverage_pct=min_coverage_pct,
        )

    def _auto_describe(self, key: str, value: float | None, status: str) -> str:
        label = self.METRIC_LABELS.get(key, key)
        unit  = self.METRIC_UNITS.get(key, "°")
        ref   = self.REFERENCE_RANGES.get(key, "N/D")
        if status == "sin_datos" or value is None:
            return (
                f"{label}: no calculable con la información disponible "
                f"(datos insuficientes o visibilidad de articulaciones limitada)."
            )
        val_s = f"{value:.1f}{unit}"
        if status == "normal":
            return f"{label}: {val_s}. Dentro del rango de referencia ({ref})."
        if status == "atencion":
            return f"{label}: {val_s}. Levemente fuera del rango orientativo ({ref}). Se sugiere monitoreo."
        return f"{label}: {val_s}. Fuera del rango de referencia ({ref}). Se recomienda evaluación profesional."
