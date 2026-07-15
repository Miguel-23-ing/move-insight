"""
analysis/arm.py  —  Brazo (Balanceo)

Clinical references:
  - Murray et al. (1967): arm swing contributes to gait stability
  - Pontzer et al. (2009): arm swing reduces trunk rotation and metabolic cost
  - Romkes et al. (2006): arm swing symmetry in normal vs. pathological gait

Keypoints: 5=L_sh  6=R_sh  7=L_el  8=R_el  9=L_wr  10=R_wr  11=L_hi  12=R_hi
"""
from __future__ import annotations
import numpy as np
from analysis.base import AnalysisModule, MetricResult, angle_three_points
from core.drawing import (
    draw_skeleton, draw_angle_arc, draw_metric_panel, draw_joint_tag,
    COLOR_TEAL, COLOR_WHITE,
)


class ArmModule(AnalysisModule):
    NAME        = "Brazo"
    DESCRIPTION = (
        "Análisis del balanceo de brazo durante la marcha u otras actividades: "
        "ROM por ciclo, peak anterior y posterior, índice de simetría, "
        "frecuencia de oscilación, velocidad angular, ángulo de codo y variabilidad."
    )

    THRESHOLDS = {
        "rom_L":          {"better": "higher", "normal_min": 20.0, "warn_min": 10.0},
        "rom_R":          {"better": "higher", "normal_min": 20.0, "warn_min": 10.0},
        "si_rom":         {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "swing_freq_L":   {"better": "range",  "normal_min":  0.5, "normal_max": 1.5,
                                                "warn_min":    0.2, "warn_max":   2.5},
        "swing_freq_R":   {"better": "range",  "normal_min":  0.5, "normal_max": 1.5,
                                                "warn_min":    0.2, "warn_max":   2.5},
        "elbow_mean_L":   {"better": "range",  "normal_min": 70.0, "normal_max": 140.0,
                                                "warn_min":   50.0, "warn_max":  160.0},
        "elbow_mean_R":   {"better": "range",  "normal_min": 70.0, "normal_max": 140.0,
                                                "warn_min":   50.0, "warn_max":  160.0},
        "si_elbow":       {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "av_L":           {"better": "range",  "normal_min": 20,   "normal_max": 150,
                                                "warn_min":   10,   "warn_max":  250},
        "av_R":           {"better": "range",  "normal_min": 20,   "normal_max": 150,
                                                "warn_min":   10,   "warn_max":  250},
        "cv_L":           {"better": "lower",  "normal_max": 20.0, "warn_max": 35.0},
        "cv_R":           {"better": "lower",  "normal_max": 20.0, "warn_max": 35.0},
    }

    REFERENCE_RANGES = {
        "rom_L":        "> 20° (marcha)",
        "rom_R":        "> 20° (marcha)",
        "si_rom":       "< 10%",
        "swing_freq_L": "0.5–1.5 Hz",
        "swing_freq_R": "0.5–1.5 Hz",
        "elbow_mean_L": "70–140°",
        "elbow_mean_R": "70–140°",
        "si_elbow":     "< 10%",
        "av_L":         "20–150 °/s",
        "av_R":         "20–150 °/s",
        "cv_L":         "< 20%",
        "cv_R":         "< 20%",
    }

    METRIC_LABELS = {
        "rom_L":        "ROM Balanceo Brazo Izquierdo (por ciclo)",
        "rom_R":        "ROM Balanceo Brazo Derecho (por ciclo)",
        "si_rom":       "Índice de Simetría — Balanceo",
        "swing_freq_L": "Frecuencia de Balanceo Izq",
        "swing_freq_R": "Frecuencia de Balanceo Der",
        "elbow_mean_L": "Ángulo Codo Izquierdo (media)",
        "elbow_mean_R": "Ángulo Codo Derecho (media)",
        "si_elbow":     "Simetría de Posición de Codo",
        "av_L":         "Velocidad Angular Brazo Izq",
        "av_R":         "Velocidad Angular Brazo Der",
        "cv_L":         "Variabilidad Balanceo Izq (CV)",
        "cv_R":         "Variabilidad Balanceo Der (CV)",
    }

    METRIC_UNITS = {
        "rom_L":"°","rom_R":"°","si_rom":"%",
        "swing_freq_L":"Hz","swing_freq_R":"Hz",
        "elbow_mean_L":"°","elbow_mean_R":"°","si_elbow":"%",
        "av_L":"°/s","av_R":"°/s",
        "cv_L":"%","cv_R":"%",
    }

    _DESC = {
        "si_rom": {
            "normal":   "Simetría del balanceo de brazos {v}% — patrón recíproco equilibrado. Favorece la estabilidad del tronco durante la marcha.",
            "atencion": "Simetría del balanceo {v}% — asimetría moderada. Puede indicar compensación por dolor, rigidez o déficit neurológico unilateral.",
            "alerta":   "Simetría del balanceo {v}% — asimetría significativa. Evaluar causa neurológica, ortopédica o dolor unilateral.",
        },
        "rom_L": {
            "normal":   "ROM de balanceo izq {v}° — amplitud adecuada para la actividad registrada.",
            "atencion": "ROM de balanceo izq {v}° — reducido. Puede indicar rigidez de hombro, dolor o reducción voluntaria.",
            "alerta":   "ROM de balanceo izq {v}° — significativamente reducido. Evaluar causa.",
        },
        "rom_R": {
            "normal":   "ROM de balanceo der {v}° — amplitud adecuada.",
            "atencion": "ROM de balanceo der {v}° — reducido. Posible rigidez o compensación.",
            "alerta":   "ROM de balanceo der {v}° — significativamente reducido. Evaluar causa.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.1f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [5, 6, 7, 8, 9, 10, 11, 12]
    def get_connections(self) -> list[tuple[int, int]]:
        return [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12)]

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
        l_wr, l_wr_c = self._pt(smoothed_pts, 9)
        r_wr, r_wr_c = self._pt(smoothed_pts, 10)
        l_hi, l_hi_c = self._pt(smoothed_pts, 11)
        r_hi, r_hi_c = self._pt(smoothed_pts, 12)

        m: dict[str, float | None] = {}
        m["armL"]   = angle_three_points(l_hi, l_sh, l_el) if self._valid(l_hi_c, l_sh_c, l_el_c) else None
        m["armR"]   = angle_three_points(r_hi, r_sh, r_el) if self._valid(r_hi_c, r_sh_c, r_el_c) else None
        m["elbowL"] = angle_three_points(l_sh, l_el, l_wr) if self._valid(l_sh_c, l_el_c, l_wr_c) else None
        m["elbowR"] = angle_three_points(r_sh, r_el, r_wr) if self._valid(r_sh_c, r_el_c, r_wr_c) else None

        occupied: list = []
        if m["elbowL"] is not None:
            draw_angle_arc(frame, l_el, l_sh, l_wr, m["elbowL"], COLOR_TEAL, 22)
            draw_joint_tag(frame, smoothed_pts[7], f"Co I{m['elbowL']:.0f}°", "L", occupied, self.vis_thr)
        if m["elbowR"] is not None:
            draw_angle_arc(frame, r_el, r_sh, r_wr, m["elbowR"], COLOR_TEAL, 22)
            draw_joint_tag(frame, smoothed_pts[8], f"Co D{m['elbowR']:.0f}°", "R", occupied, self.vis_thr)
        if m["armL"] is not None:
            draw_joint_tag(frame, smoothed_pts[5], f"B I{m['armL']:.0f}°", "L", occupied, self.vis_thr)
        if m["armR"] is not None:
            draw_joint_tag(frame, smoothed_pts[6], f"B D{m['armR']:.0f}°", "R", occupied, self.vis_thr)

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("BRAZO", COLOR_TEAL),
            (f"Swing I:{_f(m['armL'])}  D:{_f(m['armR'])}", COLOR_WHITE),
            (f"Codo  I:{_f(m['elbowL'])}  D:{_f(m['elbowR'])}", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        aL = self._series(metrics_history, "armL")
        aR = self._series(metrics_history, "armR")
        eL = self._series(metrics_history, "elbowL")
        eR = self._series(metrics_history, "elbowR")
        R:  dict[str, MetricResult] = {}

        romL = self._per_cycle_rom(aL, fps) if aL else None
        romR = self._per_cycle_rom(aR, fps) if aR else None
        for key, val in [("rom_L", romL), ("rom_R", romR)]:
            st = self.classify(key, val)
            R[key] = self._make_metric(key, val, description=self._describe(key, val, st))

        si = self._symmetry_index(romL, romR)
        st = self.classify("si_rom", si)
        R["si_rom"] = self._make_metric("si_rom", si, description=self._describe("si_rom", si, st))

        R["swing_freq_L"] = self._make_metric("swing_freq_L", self._oscillation_frequency(aL, fps) if aL else None)
        R["swing_freq_R"] = self._make_metric("swing_freq_R", self._oscillation_frequency(aR, fps) if aR else None)

        emL = self._mean(eL); emR = self._mean(eR)
        R["elbow_mean_L"] = self._make_metric("elbow_mean_L", emL)
        R["elbow_mean_R"] = self._make_metric("elbow_mean_R", emR)
        R["si_elbow"]     = self._make_metric("si_elbow",     self._symmetry_index(emL, emR))

        R["av_L"] = self._make_metric("av_L", self._angular_velocity(aL, fps))
        R["av_R"] = self._make_metric("av_R", self._angular_velocity(aR, fps))
        R["cv_L"] = self._make_metric("cv_L", self._cv(aL))
        R["cv_R"] = self._make_metric("cv_R", self._cv(aR))

        return R
