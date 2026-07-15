"""
analysis/elbow.py  —  Codo

Clinical references:
  - AAOS: elbow ROM 0-145° (full ext to full flex)
  - Morrey et al. (1981): functional elbow ROM 30-130°
  - Werner & An (1994): extension deficit >10° clinically significant

Keypoints: 5=L_sh  6=R_sh  7=L_el  8=R_el  9=L_wr  10=R_wr
"""
from __future__ import annotations
import numpy as np
from analysis.base import AnalysisModule, MetricResult, angle_three_points
from core.drawing import (
    draw_skeleton, draw_angle_arc, draw_metric_panel, draw_joint_tag,
    COLOR_TEAL, COLOR_WHITE,
)


class ElbowModule(AnalysisModule):
    NAME        = "Codo"
    DESCRIPTION = (
        "Análisis de codo: flexo-extensión, ROM por ciclo, flexión máxima, "
        "déficit de extensión, índice de simetría, velocidad angular "
        "y variabilidad del movimiento."
    )

    THRESHOLDS = {
        # Morrey 1981: functional ROM 30-130° (total 100°); full ROM to 145°
        "rom_L":          {"better": "higher", "normal_min": 90.0,  "warn_min": 45.0},
        "rom_R":          {"better": "higher", "normal_min": 90.0,  "warn_min": 45.0},
        # Peak flexion: ~145° AAOS max; functional >130°
        "peak_flex_L":    {"better": "higher", "normal_min": 120.0, "warn_min": 90.0},
        "peak_flex_R":    {"better": "higher", "normal_min": 120.0, "warn_min": 90.0},
        # Extension deficit: Werner & An – >10° significant
        "ext_deficit_L":  {"better": "lower",  "normal_max": 10.0,  "warn_max": 20.0},
        "ext_deficit_R":  {"better": "lower",  "normal_max": 10.0,  "warn_max": 20.0},
        # Symmetry
        "si_rom":         {"better": "lower",  "normal_max": 10.0,  "warn_max": 20.0},
        "si_peak_flex":   {"better": "lower",  "normal_max": 10.0,  "warn_max": 20.0},
        # Angular velocity
        "av_L":           {"better": "range",  "normal_min": 30,    "normal_max": 300,
                                                "warn_min":   10,    "warn_max":  500},
        "av_R":           {"better": "range",  "normal_min": 30,    "normal_max": 300,
                                                "warn_min":   10,    "warn_max":  500},
        # CV
        "cv_L":           {"better": "lower",  "normal_max": 12.0,  "warn_max": 22.0},
        "cv_R":           {"better": "lower",  "normal_max": 12.0,  "warn_max": 22.0},
    }

    REFERENCE_RANGES = {
        "rom_L":         "> 90° (funcional >100°)",
        "rom_R":         "> 90° (funcional >100°)",
        "peak_flex_L":   "> 120° (AAOS máx 145°)",
        "peak_flex_R":   "> 120°",
        "ext_deficit_L": "< 10°",
        "ext_deficit_R": "< 10°",
        "si_rom":        "< 10%",
        "si_peak_flex":  "< 10%",
        "av_L":          "30–300 °/s",
        "av_R":          "30–300 °/s",
        "cv_L":          "< 12%",
        "cv_R":          "< 12%",
    }

    METRIC_LABELS = {
        "rom_L":         "ROM Codo Izquierdo (por ciclo)",
        "rom_R":         "ROM Codo Derecho (por ciclo)",
        "peak_flex_L":   "Flexión Máxima Izquierda",
        "peak_flex_R":   "Flexión Máxima Derecha",
        "ext_deficit_L": "Déficit de Extensión Izquierda",
        "ext_deficit_R": "Déficit de Extensión Derecha",
        "si_rom":        "Índice de Simetría — ROM Codo",
        "si_peak_flex":  "Índice de Simetría — Flexión Máx.",
        "av_L":          "Velocidad Angular Codo Izq",
        "av_R":          "Velocidad Angular Codo Der",
        "cv_L":          "Variabilidad Codo Izq (CV)",
        "cv_R":          "Variabilidad Codo Der (CV)",
    }

    METRIC_UNITS = {
        "rom_L":"°","rom_R":"°",
        "peak_flex_L":"°","peak_flex_R":"°",
        "ext_deficit_L":"°","ext_deficit_R":"°",
        "si_rom":"%","si_peak_flex":"%",
        "av_L":"°/s","av_R":"°/s",
        "cv_L":"%","cv_R":"%",
    }

    _DESC = {
        "rom_L": {
            "normal":   "ROM de codo izq {v}° por ciclo — dentro del rango funcional (ref >90°).",
            "atencion": "ROM de codo izq {v}° — reducido (ref >90°). Posible rigidez articular o limitación funcional.",
            "alerta":   "ROM de codo izq {v}° — significativamente reducido. Evaluar artrosis, contractura o daño articular.",
        },
        "rom_R": {
            "normal":   "ROM de codo der {v}° — dentro del rango funcional.",
            "atencion": "ROM de codo der {v}° — reducido. Posible rigidez o limitación.",
            "alerta":   "ROM de codo der {v}° — significativamente reducido. Evaluar causa.",
        },
        "ext_deficit_L": {
            "normal":   "Déficit de extensión izq {v}° — extensión casi completa. Sin restricción significativa.",
            "atencion": "Déficit de extensión izq {v}° — extensión levemente limitada. Verificar contractura en flexión.",
            "alerta":   "Déficit de extensión izq {v}° — déficit clínicamente significativo. Evaluar contractura, calcificación o adherencias.",
        },
        "ext_deficit_R": {
            "normal":   "Déficit de extensión der {v}° — extensión casi completa.",
            "atencion": "Déficit de extensión der {v}° — levemente limitada.",
            "alerta":   "Déficit de extensión der {v}° — déficit clínicamente significativo. Evaluar causa.",
        },
        "si_rom": {
            "normal":   "Simetría de ROM de codo {v}% — ambos codos con amplitud de movimiento similar.",
            "atencion": "Simetría de ROM {v}% — asimetría moderada. Evaluar restricción o compensación unilateral.",
            "alerta":   "Simetría de ROM {v}% — asimetría significativa. Se recomienda evaluación clínica bilateral.",
        },
        "peak_flex_L": {
            "normal":   "Flexión máxima codo izq {v}° — adecuada (ref >120°, funcional).",
            "atencion": "Flexión máxima codo izq {v}° — reducida (ref >120°). Posible limitación articular.",
            "alerta":   "Flexión máxima codo izq {v}° — significativamente reducida. Evaluar articulación.",
        },
        "peak_flex_R": {
            "normal":   "Flexión máxima codo der {v}° — adecuada (ref >120°).",
            "atencion": "Flexión máxima codo der {v}° — reducida.",
            "alerta":   "Flexión máxima codo der {v}° — significativamente reducida.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.1f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [5, 6, 7, 8, 9, 10]
    def get_connections(self) -> list[tuple[int, int]]:
        return [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]

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

        m: dict[str, float | None] = {}
        # angle_three_points → 180°=full extension, smaller=more flexed
        m["eL"] = angle_three_points(l_sh, l_el, l_wr) if self._valid(l_sh_c, l_el_c, l_wr_c) else None
        m["eR"] = angle_three_points(r_sh, r_el, r_wr) if self._valid(r_sh_c, r_el_c, r_wr_c) else None

        occupied: list = []
        if m["eL"] is not None:
            draw_angle_arc(frame, l_el, l_sh, l_wr, m["eL"], COLOR_TEAL, 28)
            draw_joint_tag(frame, smoothed_pts[7], f"Izq {m['eL']:.0f}°", "L", occupied, self.vis_thr)
        if m["eR"] is not None:
            draw_angle_arc(frame, r_el, r_sh, r_wr, m["eR"], COLOR_TEAL, 28)
            draw_joint_tag(frame, smoothed_pts[8], f"Der {m['eR']:.0f}°", "R", occupied, self.vis_thr)

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("CODO", COLOR_TEAL),
            (f"Flex  Izq:{_f(m['eL'])}  Der:{_f(m['eR'])}", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        eL = self._series(metrics_history, "eL")
        eR = self._series(metrics_history, "eR")
        R:  dict[str, MetricResult] = {}

        # Per-cycle ROM
        romL = self._per_cycle_rom(eL, fps) if eL else None
        romR = self._per_cycle_rom(eR, fps) if eR else None
        for key, val in [("rom_L", romL), ("rom_R", romR)]:
            st = self.classify(key, val)
            R[key] = self._make_metric(key, val, description=self._describe(key, val, st))

        # Peak flexion: convention angle_three_points 180°=extended → 180-min=peak flexion degrees
        pfL = (180.0 - min(eL)) if eL else None
        pfR = (180.0 - min(eR)) if eR else None
        for key, val in [("peak_flex_L", pfL), ("peak_flex_R", pfR)]:
            st = self.classify(key, val)
            R[key] = self._make_metric(key, val, description=self._describe(key, val, st))

        # Extension deficit
        edL = max(0.0, 180.0 - max(eL)) if eL else None
        edR = max(0.0, 180.0 - max(eR)) if eR else None
        for key, val in [("ext_deficit_L", edL), ("ext_deficit_R", edR)]:
            st = self.classify(key, val)
            R[key] = self._make_metric(key, val, description=self._describe(key, val, st))

        # Symmetry
        si_r = self._symmetry_index(romL, romR)
        si_p = self._symmetry_index(pfL, pfR)
        for key, val in [("si_rom", si_r), ("si_peak_flex", si_p)]:
            st = self.classify(key, val)
            R[key] = self._make_metric(key, val, description=self._describe(key, val, st))

        # Angular velocity
        R["av_L"] = self._make_metric("av_L", self._angular_velocity(eL, fps))
        R["av_R"] = self._make_metric("av_R", self._angular_velocity(eR, fps))

        # CV
        R["cv_L"] = self._make_metric("cv_L", self._cv(eL))
        R["cv_R"] = self._make_metric("cv_R", self._cv(eR))

        return R
