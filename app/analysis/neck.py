"""
analysis/neck.py  —  Cuello

Clinical references:
  - AAOS: lateral flexion 0-45°, forward flexion 0-80°, extension 0-50°, rotation 0-80°
  - Drerup & Hierholzer (1992): head tilt assessment
  - Griegel-Morris et al. (1992): postural deviations of the head/neck

Keypoints: 0=nose  3=L_ear  4=R_ear  5=L_shoulder  6=R_shoulder
"""
from __future__ import annotations
import math
import numpy as np
from analysis.base import AnalysisModule, MetricResult, vector_angle
from core.drawing import (
    draw_skeleton, draw_metric_panel,
    COLOR_TEAL, COLOR_WHITE,
)

VERTICAL = np.array([0.0, -1.0], dtype=np.float32)
HORIZ    = np.array([1.0,  0.0], dtype=np.float32)


class NeckModule(AnalysisModule):
    NAME        = "Cuello"
    DESCRIPTION = (
        "Análisis de cabeza y cuello: inclinación lateral, ROM de inclinación, "
        "proyección anterior (forward head posture), asimetría de orejas, "
        "variabilidad y consistencia postural."
    )

    THRESHOLDS = {
        "head_tilt_mean":    {"better": "lower", "normal_max":  5.0, "warn_max": 10.0},
        "head_tilt_rom":     {"better": "lower", "normal_max":  8.0, "warn_max": 15.0},
        "ear_asymmetry":     {"better": "lower", "normal_max":  5.0, "warn_max": 10.0},
        "nose_dev_mean":     {"better": "lower", "normal_max":  5.0, "warn_max": 10.0},
        "fwd_head_mean":     {"better": "lower", "normal_max": 10.0, "warn_max": 20.0},
        "fwd_head_max":      {"better": "lower", "normal_max": 15.0, "warn_max": 30.0},
        "neck_axis_mean":    {"better": "lower", "normal_max":  5.0, "warn_max": 10.0},
        "cv_tilt":           {"better": "lower", "normal_max": 25.0, "warn_max": 45.0},
    }

    REFERENCE_RANGES = {
        "head_tilt_mean":  "< 5°",
        "head_tilt_rom":   "< 8°",
        "ear_asymmetry":   "< 5°",
        "nose_dev_mean":   "< 5%",
        "fwd_head_mean":   "< 10%",
        "fwd_head_max":    "< 15%",
        "neck_axis_mean":  "< 5°",
        "cv_tilt":         "< 25%",
    }

    METRIC_LABELS = {
        "head_tilt_mean":  "Inclinación Lateral Cabeza (media)",
        "head_tilt_rom":   "ROM Inclinación Lateral Cabeza",
        "ear_asymmetry":   "Asimetría de Orejas (Δ altura)",
        "nose_dev_mean":   "Desviación Lateral de Nariz",
        "fwd_head_mean":   "Proyección Anterior de Cabeza (media)",
        "fwd_head_max":    "Proyección Anterior Máxima",
        "neck_axis_mean":  "Inclinación Eje Cabeza–Cuello",
        "cv_tilt":         "Variabilidad de Inclinación (CV)",
    }

    METRIC_UNITS = {
        "head_tilt_mean":"°","head_tilt_rom":"°",
        "ear_asymmetry":"°",
        "nose_dev_mean":"%","fwd_head_mean":"%","fwd_head_max":"%",
        "neck_axis_mean":"°","cv_tilt":"%",
    }

    _DESC = {
        "head_tilt_mean": {
            "normal":   "Inclinación lateral de cabeza {v}° — dentro del rango normal (<5°). Postura cefálica equilibrada.",
            "atencion": "Inclinación lateral {v}° — moderada. Evaluar hábitos posturales, tensión muscular cervical o dismetría.",
            "alerta":   "Inclinación lateral {v}° — significativa. Evaluar causas musculoesqueléticas o neurológicas.",
        },
        "fwd_head_mean": {
            "normal":   "Proyección anterior de cabeza {v}% — normal. Sin evidencia de postura en 'cabeza adelantada'.",
            "atencion": "Proyección anterior {v}% — moderada. Patrón típico de postura sedentaria o tensión cervical crónica.",
            "alerta":   "Proyección anterior {v}% — significativa (Forward Head Posture). Puede asociarse a cervicalgia, cefalea y tensión muscular cervical elevada.",
        },
        "ear_asymmetry": {
            "normal":   "Asimetría de orejas {v}° — simétrica. Buena alineación lateral de la cabeza.",
            "atencion": "Asimetría de orejas {v}° — moderada. Posible inclinación lateral habitual.",
            "alerta":   "Asimetría de orejas {v}° — significativa. Evaluar tortícolis, hábito postural o causa estructural.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.1f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [0, 3, 4, 5, 6]
    def get_connections(self) -> list[tuple[int, int]]: return [(5, 6)]

    def process_frame(
        self, frame: np.ndarray, smoothed_pts: list[dict],
        frame_idx: int, fps: float,
    ) -> dict[str, float | None]:
        h, w = frame.shape[:2]
        draw_skeleton(frame, smoothed_pts,
                      highlight_kps=self.get_relevant_keypoints(),
                      visibility_threshold=self.vis_thr)
        nose,  nose_c  = self._pt(smoothed_pts, 0)
        l_ear, l_ear_c = self._pt(smoothed_pts, 3)
        r_ear, r_ear_c = self._pt(smoothed_pts, 4)
        l_sh,  l_sh_c  = self._pt(smoothed_pts, 5)
        r_sh,  r_sh_c  = self._pt(smoothed_pts, 6)

        m: dict[str, float | None] = {}

        # Ear asymmetry (height diff, mapped to degrees)
        ea = None
        if self._valid(l_ear_c, r_ear_c, l_sh_c, r_sh_c):
            sh_w = abs(float(r_sh[0] - l_sh[0]))
            if sh_w > 1.0:
                ea = math.degrees(math.atan2(abs(float(l_ear[1] - r_ear[1])), sh_w))
        m["ear_asym"] = ea

        # Head lateral tilt: mid_ear → nose vs vertical
        tilt = None
        if self._valid(l_ear_c, r_ear_c, nose_c):
            mid = (l_ear + r_ear) / 2
            tilt = vector_angle(nose - mid, VERTICAL)
        m["tilt"] = tilt

        # Nose lateral deviation (% shoulder width)
        nd = None
        if self._valid(nose_c, l_sh_c, r_sh_c):
            mid_x = (l_sh[0] + r_sh[0]) / 2
            sh_w  = abs(float(r_sh[0] - l_sh[0]))
            if sh_w > 1.0:
                nd = abs(float(nose[0]) - mid_x) / sh_w * 100
        m["nose_dev"] = nd

        # Forward head: nose anterior to shoulder mid (y-axis %)
        fh = None
        if self._valid(nose_c, l_sh_c, r_sh_c, l_ear_c, r_ear_c):
            mid_sh_y  = float((l_sh[1] + r_sh[1]) / 2)
            mid_ear_y = float((l_ear[1] + r_ear[1]) / 2)
            head_h    = abs(mid_ear_y - mid_sh_y)
            if head_h > 1.0:
                fh = max(0.0, (mid_sh_y - float(nose[1])) / head_h * 100)
        m["fwd_head"] = fh

        # Neck axis tilt: l_ear → r_ear vs horizontal
        na = None
        if self._valid(l_ear_c, r_ear_c):
            na = vector_angle(r_ear - l_ear, HORIZ)
        m["neck_axis"] = na

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("CUELLO", COLOR_TEAL),
            (f"Inclin:{_f(m['tilt'])}  Eje:{_f(m['neck_axis'])}", COLOR_WHITE),
            (f"Orejas:{_f(m['ear_asym'])}  FwdHead:{_f(m['fwd_head'])}%", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        tilt = self._series(metrics_history, "tilt")
        ea   = self._series(metrics_history, "ear_asym")
        nd   = self._series(metrics_history, "nose_dev")
        fh   = self._series(metrics_history, "fwd_head")
        na   = self._series(metrics_history, "neck_axis")
        R:   dict[str, MetricResult] = {}

        # Head tilt
        tm = self._mean(tilt); st = self.classify("head_tilt_mean", tm)
        R["head_tilt_mean"] = self._make_metric("head_tilt_mean", tm, description=self._describe("head_tilt_mean", tm, st))
        R["head_tilt_rom"]  = self._make_metric("head_tilt_rom",  self._per_cycle_rom(tilt, fps) if tilt else None)

        # Ear asymmetry
        eam = self._mean(ea); st = self.classify("ear_asymmetry", eam)
        R["ear_asymmetry"]  = self._make_metric("ear_asymmetry",  eam, description=self._describe("ear_asymmetry", eam, st))

        # Nose deviation
        R["nose_dev_mean"]  = self._make_metric("nose_dev_mean",  self._mean(nd))

        # Forward head
        fhm = self._mean(fh); fhx = max(fh) if fh else None
        st  = self.classify("fwd_head_mean", fhm)
        R["fwd_head_mean"]  = self._make_metric("fwd_head_mean",  fhm, description=self._describe("fwd_head_mean", fhm, st))
        R["fwd_head_max"]   = self._make_metric("fwd_head_max",   fhx)

        # Neck axis
        R["neck_axis_mean"] = self._make_metric("neck_axis_mean", self._mean(na))

        # CV of tilt
        R["cv_tilt"] = self._make_metric("cv_tilt", self._cv(tilt))

        return R
