"""
analysis/hip.py  —  Cadera

Clinical references:
  - Perry & Burnfield (2010): gait hip ROM 40-50°
  - AAOS: hip flexion 0-120°, extension 0-30°, abduction 0-45°
  - Trendelenburg sign: pelvic obliquity >5° during single-leg stance

Keypoints: 5=L_shoulder  6=R_shoulder  11=L_hip  12=R_hip  13=L_knee  14=R_knee
"""
from __future__ import annotations
import numpy as np
from analysis.base import (
    AnalysisModule, MetricResult,
    angle_three_points, vector_angle,
)
from core.drawing import (
    draw_skeleton, draw_angle_arc, draw_metric_panel, draw_joint_tag,
    COLOR_TEAL, COLOR_WHITE,
)

VERTICAL = np.array([0.0, -1.0], dtype=np.float32)
HORIZ    = np.array([1.0,  0.0], dtype=np.float32)


class HipModule(AnalysisModule):
    NAME        = "Cadera"
    DESCRIPTION = (
        "Análisis de cadera: ROM por ciclo (flexo-extensión), peak de flexión y extensión, "
        "oblicuidad pélvica y su ROM, índice de simetría, velocidad angular, "
        "desplazamiento lateral y variabilidad del movimiento."
    )

    THRESHOLDS = {
        # Hip ROM in gait: 40-50° normal (Perry)
        "hip_rom_L":           {"better": "higher", "normal_min": 30.0, "warn_min": 15.0},
        "hip_rom_R":           {"better": "higher", "normal_min": 30.0, "warn_min": 15.0},
        # Peak flexion: ~30° at initial contact (gait); >90° functional
        "peak_flex_L":         {"better": "higher", "normal_min": 20.0, "warn_min": 10.0},
        "peak_flex_R":         {"better": "higher", "normal_min": 20.0, "warn_min": 10.0},
        # Symmetry index
        "si_rom":              {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "si_peak_flex":        {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        # Pelvic obliquity
        "pelvis_obliq_mean":   {"better": "lower",  "normal_max":  5.0, "warn_max":  8.0},
        "pelvis_obliq_rom":    {"better": "lower",  "normal_max":  6.0, "warn_max": 10.0},
        # Shoulder obliquity
        "shoulder_obliq":      {"better": "lower",  "normal_max":  5.0, "warn_max":  8.0},
        # Lateral shift
        "lateral_shift":       {"better": "lower",  "normal_max":  5.0, "warn_max": 10.0},
        # Angular velocity
        "av_L":                {"better": "range",  "normal_min": 30,   "normal_max": 200,
                                                     "warn_min":   15,   "warn_max":  300},
        "av_R":                {"better": "range",  "normal_min": 30,   "normal_max": 200,
                                                     "warn_min":   15,   "warn_max":  300},
        # CV
        "cv_L":                {"better": "lower",  "normal_max": 15.0, "warn_max": 25.0},
        "cv_R":                {"better": "lower",  "normal_max": 15.0, "warn_max": 25.0},
    }

    REFERENCE_RANGES = {
        "hip_rom_L":         "> 30° (marcha)",
        "hip_rom_R":         "> 30° (marcha)",
        "peak_flex_L":       "> 20°",
        "peak_flex_R":       "> 20°",
        "si_rom":            "< 10%",
        "si_peak_flex":      "< 10%",
        "pelvis_obliq_mean": "< 5°",
        "pelvis_obliq_rom":  "< 6°",
        "shoulder_obliq":    "< 5°",
        "lateral_shift":     "< 5%",
        "av_L":              "30–200 °/s",
        "av_R":              "30–200 °/s",
        "cv_L":              "< 15%",
        "cv_R":              "< 15%",
    }

    METRIC_LABELS = {
        "hip_rom_L":         "ROM Cadera Izquierda (por ciclo)",
        "hip_rom_R":         "ROM Cadera Derecha (por ciclo)",
        "peak_flex_L":       "Peak Flexión Cadera Izq",
        "peak_flex_R":       "Peak Flexión Cadera Der",
        "si_rom":            "Índice de Simetría — ROM Cadera",
        "si_peak_flex":      "Índice de Simetría — Peak Flexión",
        "pelvis_obliq_mean": "Oblicuidad Pélvica (media)",
        "pelvis_obliq_rom":  "ROM Oblicuidad Pélvica",
        "shoulder_obliq":    "Oblicuidad de Hombros",
        "lateral_shift":     "Desplazamiento Lateral de Pelvis",
        "av_L":              "Velocidad Angular Cadera Izq",
        "av_R":              "Velocidad Angular Cadera Der",
        "cv_L":              "Variabilidad Cadera Izq (CV)",
        "cv_R":              "Variabilidad Cadera Der (CV)",
    }

    METRIC_UNITS = {
        "hip_rom_L":"°","hip_rom_R":"°",
        "peak_flex_L":"°","peak_flex_R":"°",
        "si_rom":"%","si_peak_flex":"%",
        "pelvis_obliq_mean":"°","pelvis_obliq_rom":"°",
        "shoulder_obliq":"°","lateral_shift":"%",
        "av_L":"°/s","av_R":"°/s",
        "cv_L":"%","cv_R":"%",
    }

    _DESC = {
        "pelvis_obliq_mean": {
            "normal":   "Oblicuidad pélvica media {v}° — dentro del rango normal (<5°). Pelvis estable.",
            "atencion": "Oblicuidad pélvica {v}° — levemente elevada. Posible signo de Trendelenburg incipiente o debilidad de abductores.",
            "alerta":   "Oblicuidad pélvica {v}° — elevada. Evaluar debilidad glútea, dismetría o alteración neurológica.",
        },
        "si_rom": {
            "normal":   "Simetría de ROM de cadera {v}% — distribución bilateral equilibrada.",
            "atencion": "Simetría de ROM de cadera {v}% — asimetría moderada. Evaluar compensaciones o dolor unilateral.",
            "alerta":   "Simetría de ROM de cadera {v}% — asimetría significativa. Evaluar causa biomecánica o clínica.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.1f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [5, 6, 11, 12, 13, 14]
    def get_connections(self) -> list[tuple[int, int]]:
        return [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14)]

    def process_frame(
        self, frame: np.ndarray, smoothed_pts: list[dict],
        frame_idx: int, fps: float,
    ) -> dict[str, float | None]:
        h, w = frame.shape[:2]
        draw_skeleton(frame, smoothed_pts,
                      highlight_kps=self.get_relevant_keypoints(),
                      visibility_threshold=self.vis_thr)
        l_sh, l_sh_c = self._pt(smoothed_pts, 5)
        r_sh, r_sh_c = self._pt(smoothed_pts, 6)
        l_hi, l_hi_c = self._pt(smoothed_pts, 11)
        r_hi, r_hi_c = self._pt(smoothed_pts, 12)
        l_kn, l_kn_c = self._pt(smoothed_pts, 13)
        r_kn, r_kn_c = self._pt(smoothed_pts, 14)

        m: dict[str, float | None] = {}
        m["hL"] = angle_three_points(l_sh, l_hi, l_kn) if self._valid(l_sh_c, l_hi_c, l_kn_c) else None
        m["hR"] = angle_three_points(r_sh, r_hi, r_kn) if self._valid(r_sh_c, r_hi_c, r_kn_c) else None
        m["pv"] = vector_angle(r_hi - l_hi, HORIZ) if self._valid(l_hi_c, r_hi_c) else None
        m["so"] = vector_angle(r_sh - l_sh, HORIZ) if self._valid(l_sh_c, r_sh_c) else None
        if self._valid(l_hi_c, r_hi_c):
            mid_x = float((l_hi[0] + r_hi[0]) / 2)
            m["ls"] = abs((mid_x - w / 2) / w * 100)
        else:
            m["ls"] = None

        occupied: list = []
        if m["hL"] is not None:
            draw_angle_arc(frame, l_hi, l_sh, l_kn, m["hL"], COLOR_TEAL, 26)
            draw_joint_tag(frame, smoothed_pts[11], f"C I{m['hL']:.0f}°", "L", occupied, self.vis_thr)
        if m["hR"] is not None:
            draw_angle_arc(frame, r_hi, r_sh, r_kn, m["hR"], COLOR_TEAL, 26)
            draw_joint_tag(frame, smoothed_pts[12], f"C D{m['hR']:.0f}°", "R", occupied, self.vis_thr)
        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("CADERA", COLOR_TEAL),
            (f"Cadera  I:{_f(m['hL'])}  D:{_f(m['hR'])}", COLOR_WHITE),
            (f"Pelvis:{_f(m['pv'])}  LShift:{_f(m['ls'])}%", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        hL = self._series(metrics_history, "hL")
        hR = self._series(metrics_history, "hR")
        pv = self._series(metrics_history, "pv")
        so = self._series(metrics_history, "so")
        ls = self._series(metrics_history, "ls")
        R:  dict[str, MetricResult] = {}

        # Per-cycle ROM
        romL = self._per_cycle_rom(hL, fps) if hL else None
        romR = self._per_cycle_rom(hR, fps) if hR else None
        for key, val in [("hip_rom_L", romL), ("hip_rom_R", romR)]:
            R[key] = self._make_metric(key, val)

        # Peak flexion (hip angle minimum ≈ most flexed)
        # Convention: shoulder-hip-knee angle 180° = straight, smaller = more flexed
        pfL = (180.0 - min(hL)) if hL else None
        pfR = (180.0 - min(hR)) if hR else None
        for key, val in [("peak_flex_L", pfL), ("peak_flex_R", pfR)]:
            R[key] = self._make_metric(key, val)

        # Symmetry
        si_r = self._symmetry_index(romL, romR)
        si_p = self._symmetry_index(pfL, pfR)
        for key, val in [("si_rom", si_r), ("si_peak_flex", si_p)]:
            st = self.classify(key, val)
            R[key] = self._make_metric(key, val, description=self._describe(key, val, st))

        # Pelvic obliquity
        pv_m   = self._mean(pv)
        pv_rom = self._per_cycle_rom(pv, fps) if pv else None
        st_pv  = self.classify("pelvis_obliq_mean", pv_m)
        R["pelvis_obliq_mean"] = self._make_metric("pelvis_obliq_mean", pv_m,
            description=self._describe("pelvis_obliq_mean", pv_m, st_pv))
        R["pelvis_obliq_rom"]  = self._make_metric("pelvis_obliq_rom",  pv_rom)

        # Shoulder obliquity, lateral shift
        R["shoulder_obliq"] = self._make_metric("shoulder_obliq", self._mean(so))
        R["lateral_shift"]  = self._make_metric("lateral_shift",  self._mean(ls))

        # Angular velocity
        R["av_L"] = self._make_metric("av_L", self._angular_velocity(hL, fps))
        R["av_R"] = self._make_metric("av_R", self._angular_velocity(hR, fps))

        # CV
        R["cv_L"] = self._make_metric("cv_L", self._cv(hL))
        R["cv_R"] = self._make_metric("cv_R", self._cv(hR))

        return R
