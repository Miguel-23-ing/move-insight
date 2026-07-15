"""
analysis/shoulder.py  —  Hombro

Clinical references:
  - AAOS: flexion 0-180°, extension 0-60°, abduction 0-180°
  - Murray et al. (1967): arm swing in gait ~30-40° total ROM
  - Handford & Srinivasan (2014): arm swing contributes to gait stability

Keypoints: 5=L_shoulder  6=R_shoulder  7=L_elbow  8=R_elbow  11=L_hip  12=R_hip
"""
from __future__ import annotations
import numpy as np
from analysis.base import AnalysisModule, MetricResult, angle_three_points, vector_angle
from core.drawing import (
    draw_skeleton, draw_angle_arc, draw_metric_panel, draw_joint_tag,
    COLOR_TEAL, COLOR_WHITE,
)

VERTICAL = np.array([0.0, -1.0], dtype=np.float32)
HORIZ    = np.array([1.0,  0.0], dtype=np.float32)


class ShoulderModule(AnalysisModule):
    NAME        = "Hombro"
    DESCRIPTION = (
        "Análisis de hombro: ROM por ciclo (flexo-abducción), peak de flexión/extensión, "
        "índice de simetría, frecuencia de balanceo, velocidad angular, "
        "diferencia de altura y variabilidad del movimiento."
    )

    THRESHOLDS = {
        "rom_L":          {"better": "higher", "normal_min": 20.0, "warn_min": 10.0},
        "rom_R":          {"better": "higher", "normal_min": 20.0, "warn_min": 10.0},
        "si_rom":         {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "peak_flex_L":    {"better": "higher", "normal_min": 15.0, "warn_min":  8.0},
        "peak_flex_R":    {"better": "higher", "normal_min": 15.0, "warn_min":  8.0},
        "swing_freq_L":   {"better": "range",  "normal_min":  0.5, "normal_max": 1.5,
                                                "warn_min":    0.2, "warn_max":   2.5},
        "swing_freq_R":   {"better": "range",  "normal_min":  0.5, "normal_max": 1.5,
                                                "warn_min":    0.2, "warn_max":   2.5},
        "height_diff":    {"better": "lower",  "normal_max":  5.0, "warn_max": 10.0},
        "av_L":           {"better": "range",  "normal_min": 20,   "normal_max": 150,
                                                "warn_min":   10,   "warn_max":  250},
        "av_R":           {"better": "range",  "normal_min": 20,   "normal_max": 150,
                                                "warn_min":   10,   "warn_max":  250},
        "cv_L":           {"better": "lower",  "normal_max": 20.0, "warn_max": 35.0},
        "cv_R":           {"better": "lower",  "normal_max": 20.0, "warn_max": 35.0},
    }

    REFERENCE_RANGES = {
        "rom_L":       "> 20° (marcha/actividad)",
        "rom_R":       "> 20° (marcha/actividad)",
        "si_rom":      "< 10%",
        "peak_flex_L": "> 15°",
        "peak_flex_R": "> 15°",
        "swing_freq_L":"0.5–1.5 Hz",
        "swing_freq_R":"0.5–1.5 Hz",
        "height_diff": "< 5°",
        "av_L":        "20–150 °/s",
        "av_R":        "20–150 °/s",
        "cv_L":        "< 20%",
        "cv_R":        "< 20%",
    }

    METRIC_LABELS = {
        "rom_L":       "ROM Hombro Izquierdo (por ciclo)",
        "rom_R":       "ROM Hombro Derecho (por ciclo)",
        "si_rom":      "Índice de Simetría — ROM Hombro",
        "peak_flex_L": "Peak Balanceo Anterior Izq",
        "peak_flex_R": "Peak Balanceo Anterior Der",
        "swing_freq_L":"Frecuencia de Balanceo Izq",
        "swing_freq_R":"Frecuencia de Balanceo Der",
        "height_diff": "Diferencia de Altura de Hombros",
        "av_L":        "Velocidad Angular Hombro Izq",
        "av_R":        "Velocidad Angular Hombro Der",
        "cv_L":        "Variabilidad Hombro Izq (CV)",
        "cv_R":        "Variabilidad Hombro Der (CV)",
    }

    METRIC_UNITS = {
        "rom_L":"°","rom_R":"°",
        "si_rom":"%",
        "peak_flex_L":"°","peak_flex_R":"°",
        "swing_freq_L":"Hz","swing_freq_R":"Hz",
        "height_diff":"°",
        "av_L":"°/s","av_R":"°/s",
        "cv_L":"%","cv_R":"%",
    }

    _DESC = {
        "si_rom": {
            "normal":   "Simetría de balanceo de hombros {v}% — patrón recíproco equilibrado.",
            "atencion": "Simetría de hombros {v}% — asimetría moderada en el balanceo. Puede indicar reducción unilateral por dolor o rigidez.",
            "alerta":   "Simetría de hombros {v}% — asimetría significativa. Evaluar dolor, restricción articular o patología neurológica.",
        },
        "height_diff": {
            "normal":   "Diferencia de altura entre hombros {v}° — dentro del rango normal (<5°).",
            "atencion": "Diferencia de altura {v}° — moderada. Posible postura asimétrica o escoliosis funcional.",
            "alerta":   "Diferencia de altura {v}° — significativa. Evaluar deformidad postural o causa estructural.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.1f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [5, 6, 7, 8, 11, 12]
    def get_connections(self) -> list[tuple[int, int]]:
        return [(5, 6), (5, 7), (6, 8), (5, 11), (6, 12), (11, 12)]

    def process_frame(
        self, frame: np.ndarray, smoothed_pts: list[dict],
        frame_idx: int, fps: float,
    ) -> dict[str, float | None]:
        draw_skeleton(frame, smoothed_pts,
                      highlight_kps=self.get_relevant_keypoints(),
                      visibility_threshold=self.vis_thr)
        l_sh, l_sh_c = self._pt(smoothed_pts, 5)
        r_sh, r_sh_c = self._pt(smoothed_pts, 6)
        l_el, l_el_c = self._pt(smoothed_pts, 7)
        r_el, r_el_c = self._pt(smoothed_pts, 8)
        l_hi, l_hi_c = self._pt(smoothed_pts, 11)
        r_hi, r_hi_c = self._pt(smoothed_pts, 12)

        m: dict[str, float | None] = {}
        m["shL"] = angle_three_points(l_hi, l_sh, l_el) if self._valid(l_hi_c, l_sh_c, l_el_c) else None
        m["shR"] = angle_three_points(r_hi, r_sh, r_el) if self._valid(r_hi_c, r_sh_c, r_el_c) else None

        # Height diff (normalised to torso)
        hd = None
        if self._valid(l_sh_c, r_sh_c, l_hi_c, r_hi_c):
            import math
            dy = abs(float(l_sh[1] - r_sh[1]))
            th = abs(float((l_sh[1]+r_sh[1])/2 - (l_hi[1]+r_hi[1])/2))
            if th > 1.0:
                hd = math.degrees(math.atan2(dy, th))
        m["hd"] = hd

        occupied: list = []
        if m["shL"] is not None:
            draw_angle_arc(frame, l_sh, l_hi, l_el, m["shL"], COLOR_TEAL, 24)
            draw_joint_tag(frame, smoothed_pts[5], f"H I{m['shL']:.0f}°", "L", occupied, self.vis_thr)
        if m["shR"] is not None:
            draw_angle_arc(frame, r_sh, r_hi, r_el, m["shR"], COLOR_TEAL, 24)
            draw_joint_tag(frame, smoothed_pts[6], f"H D{m['shR']:.0f}°", "R", occupied, self.vis_thr)

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("HOMBRO", COLOR_TEAL),
            (f"Flex  Izq:{_f(m['shL'])}  Der:{_f(m['shR'])}", COLOR_WHITE),
            (f"Alt dif:{_f(m['hd'])}", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        shL = self._series(metrics_history, "shL")
        shR = self._series(metrics_history, "shR")
        hd  = self._series(metrics_history, "hd")
        R:  dict[str, MetricResult] = {}

        romL = self._per_cycle_rom(shL, fps) if shL else None
        romR = self._per_cycle_rom(shR, fps) if shR else None
        R["rom_L"] = self._make_metric("rom_L", romL)
        R["rom_R"] = self._make_metric("rom_R", romR)

        si = self._symmetry_index(romL, romR)
        st = self.classify("si_rom", si)
        R["si_rom"] = self._make_metric("si_rom", si, description=self._describe("si_rom", si, st))

        # Peak forward flexion: minimum angle (most flexed) → 180 - min
        pfL = (180.0 - min(shL)) if shL else None
        pfR = (180.0 - min(shR)) if shR else None
        R["peak_flex_L"] = self._make_metric("peak_flex_L", pfL)
        R["peak_flex_R"] = self._make_metric("peak_flex_R", pfR)

        # Swing frequency
        R["swing_freq_L"] = self._make_metric("swing_freq_L", self._oscillation_frequency(shL, fps) if shL else None)
        R["swing_freq_R"] = self._make_metric("swing_freq_R", self._oscillation_frequency(shR, fps) if shR else None)

        # Height diff mean
        hdm = self._mean(hd); st = self.classify("height_diff", hdm)
        R["height_diff"] = self._make_metric("height_diff", hdm, description=self._describe("height_diff", hdm, st))

        # Angular velocity
        R["av_L"] = self._make_metric("av_L", self._angular_velocity(shL, fps))
        R["av_R"] = self._make_metric("av_R", self._angular_velocity(shR, fps))

        # CV
        R["cv_L"] = self._make_metric("cv_L", self._cv(shL))
        R["cv_R"] = self._make_metric("cv_R", self._cv(shR))

        return R
