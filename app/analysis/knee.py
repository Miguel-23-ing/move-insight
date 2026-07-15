"""
analysis/knee.py  —  Rodilla

Clinical references:
  - Perry & Burnfield (2010): gait knee ROM 55-65°
  - AAOS: functional ROM 0-135°
  - Rowe et al.: loading-response ~15-20°, peak swing ~60°

Keypoints: 11=left_hip  12=right_hip  13=left_knee  14=right_knee
           15=left_ankle  16=right_ankle
"""
from __future__ import annotations
import numpy as np
from analysis.base import (
    AnalysisModule, MetricResult,
    angle_three_points, knee_alignment_pct, hip_span_ratio,
)
from core.drawing import (
    draw_skeleton, draw_angle_arc, draw_metric_panel, draw_joint_tag,
    COLOR_TEAL, COLOR_WHITE,
)


class KneeModule(AnalysisModule):
    NAME        = "Rodilla"
    DESCRIPTION = (
        "Análisis de rodilla: flexo-extensión, ROM por ciclo, peak de flexión, "
        "déficit de extensión, índice de simetría, velocidad angular, "
        "variabilidad del movimiento y alineación (genu valgo/varo)."
    )

    THRESHOLDS = {
        # ROM: Perry & Burnfield – 55-65° normal gait; >40° functional gait
        "knee_rom_L":          {"better": "higher", "normal_min": 45.0, "warn_min": 25.0},
        "knee_rom_R":          {"better": "higher", "normal_min": 45.0, "warn_min": 25.0},
        # Peak flexion: >60° normal in gait; AAOS functional >90°
        "peak_flexion_L":      {"better": "higher", "normal_min": 55.0, "warn_min": 30.0},
        "peak_flexion_R":      {"better": "higher", "normal_min": 55.0, "warn_min": 30.0},
        # Extension deficit: should be ≤10° from 180°
        "ext_deficit_L":       {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "ext_deficit_R":       {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        # Symmetry Index
        "si_rom":              {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "si_peak_flex":        {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        # Angular velocity: for gait ~100-250 °/s mean absolute
        "av_L":                {"better": "range",  "normal_min":  60,  "normal_max": 280,
                                                     "warn_min":    30,  "warn_max":  380},
        "av_R":                {"better": "range",  "normal_min":  60,  "normal_max": 280,
                                                     "warn_min":    30,  "warn_max":  380},
        # CV: movement consistency
        "cv_L":                {"better": "lower",  "normal_max": 12.0, "warn_max": 22.0},
        "cv_R":                {"better": "lower",  "normal_max": 12.0, "warn_max": 22.0},
        # Knee alignment (valgus/varus): % lateral deviation, frontal view
        "align_L":             {"better": "lower",  "normal_max": 8.0,  "warn_max": 14.0},
        "align_R":             {"better": "lower",  "normal_max": 8.0,  "warn_max": 14.0},
    }

    REFERENCE_RANGES = {
        "knee_rom_L":      "> 45° (marcha)",
        "knee_rom_R":      "> 45° (marcha)",
        "peak_flexion_L":  "> 55°",
        "peak_flexion_R":  "> 55°",
        "ext_deficit_L":   "< 10°",
        "ext_deficit_R":   "< 10°",
        "si_rom":          "< 10%",
        "si_peak_flex":    "< 10%",
        "av_L":            "60–280 °/s",
        "av_R":            "60–280 °/s",
        "cv_L":            "< 12%",
        "cv_R":            "< 12%",
        "align_L":         "< 8% (eje neutro)",
        "align_R":         "< 8% (eje neutro)",
    }

    METRIC_LABELS = {
        "knee_rom_L":      "ROM Rodilla Izquierda (por ciclo)",
        "knee_rom_R":      "ROM Rodilla Derecha (por ciclo)",
        "peak_flexion_L":  "Flexión Máxima Izquierda",
        "peak_flexion_R":  "Flexión Máxima Derecha",
        "ext_deficit_L":   "Déficit de Extensión Izquierda",
        "ext_deficit_R":   "Déficit de Extensión Derecha",
        "si_rom":          "Índice de Simetría — ROM",
        "si_peak_flex":    "Índice de Simetría — Flexión Máx.",
        "av_L":            "Velocidad Angular Rodilla Izq",
        "av_R":            "Velocidad Angular Rodilla Der",
        "cv_L":            "Variabilidad Rodilla Izq (CV)",
        "cv_R":            "Variabilidad Rodilla Der (CV)",
        "align_L":         "Alineación Rodilla Izq (valgo/varo)",
        "align_R":         "Alineación Rodilla Der (valgo/varo)",
    }

    METRIC_UNITS = {
        "knee_rom_L":"°","knee_rom_R":"°",
        "peak_flexion_L":"°","peak_flexion_R":"°",
        "ext_deficit_L":"°","ext_deficit_R":"°",
        "si_rom":"%","si_peak_flex":"%",
        "av_L":"°/s","av_R":"°/s",
        "cv_L":"%","cv_R":"%",
        "align_L": "%","align_R": "%",
    }

    _DESC = {
        "knee_rom_L": {
            "normal":   "ROM de rodilla izq {v}° por ciclo. Dentro del rango esperado en marcha (>45°). Buena movilidad articular.",
            "atencion": "ROM de rodilla izq {v}° — reducido (ref >45°). Puede indicar rigidez, dolor o estrategia de marcha cautelosa.",
            "alerta":   "ROM de rodilla izq {v}° — significativamente reducido. Evaluar rigidez articular, dolor u otras causas de restricción.",
        },
        "knee_rom_R": {
            "normal":   "ROM de rodilla der {v}° por ciclo. Dentro del rango esperado (>45°).",
            "atencion": "ROM de rodilla der {v}° — reducido (ref >45°). Posible rigidez o marcha cautelosa.",
            "alerta":   "ROM de rodilla der {v}° — significativamente reducido. Se recomienda evaluación clínica.",
        },
        "si_rom": {
            "normal":   "Simetría de ROM {v}% — excelente equilibrio bilateral. Buena distribución de carga.",
            "atencion": "Simetría de ROM {v}% — asimetría moderada. Evaluar compensaciones posturales o dolor unilateral.",
            "alerta":   "Simetría de ROM {v}% — asimetría significativa. Posible dolor, debilidad o alteración biomecánica unilateral.",
        },
        "ext_deficit_L": {
            "normal":   "Déficit de extensión izq {v}° — extensión casi completa. Sin restricción significativa.",
            "atencion": "Déficit de extensión izq {v}° — extensión ligeramente limitada. Verificar contractura en flexión.",
            "alerta":   "Déficit de extensión izq {v}° — extensión significativamente limitada. Evaluar contractura, artrosis o dolor.",
        },
        "ext_deficit_R": {
            "normal":   "Déficit de extensión der {v}° — extensión casi completa.",
            "atencion": "Déficit de extensión der {v}° — extensión ligeramente limitada.",
            "alerta":   "Déficit de extensión der {v}° — extensión significativamente limitada. Evaluar contractura o dolor.",
        },
        "align_L": {
            "normal":   "Alineación rodilla izq — dentro del eje cadera-tobillo. Sin evidencia de valgo/varo.",
            "atencion": "Alineación rodilla izq — desviación moderada del eje. Posible genu valgo o varo incipiente.",
            "alerta":   "Alineación rodilla izq — desviación significativa del eje mecánico. Evaluar genu valgo/varo.",
        },
        "align_R": {
            "normal":   "Alineación rodilla der — dentro del eje cadera-tobillo.",
            "atencion": "Alineación rodilla der — desviación moderada. Posible genu valgo o varo.",
            "alerta":   "Alineación rodilla der — desviación significativa. Evaluar eje mecánico.",
        },
        "cv_L": {
            "normal":   "Variabilidad rodilla izq (CV) {v}% — movimiento muy consistente ciclo a ciclo.",
            "atencion": "Variabilidad rodilla izq (CV) {v}% — variabilidad moderada. Puede reflejar adaptación o fatiga.",
            "alerta":   "Variabilidad rodilla izq (CV) {v}% — alta variabilidad. Posible inestabilidad articular.",
        },
        "cv_R": {
            "normal":   "Variabilidad rodilla der (CV) {v}% — movimiento consistente.",
            "atencion": "Variabilidad rodilla der (CV) {v}% — variabilidad moderada.",
            "alerta":   "Variabilidad rodilla der (CV) {v}% — alta variabilidad. Evaluar estabilidad.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.1f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [5, 6, 11, 12, 13, 14, 15, 16]
    def get_connections(self) -> list[tuple[int, int]]:
        return [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]

    def process_frame(
        self, frame: np.ndarray, smoothed_pts: list[dict],
        frame_idx: int, fps: float,
    ) -> dict[str, float | None]:
        draw_skeleton(frame, smoothed_pts,
                      highlight_kps=self.get_relevant_keypoints(),
                      visibility_threshold=self.vis_thr)
        l_hi, l_hi_c = self._pt(smoothed_pts, 11)
        r_hi, r_hi_c = self._pt(smoothed_pts, 12)
        l_kn, l_kn_c = self._pt(smoothed_pts, 13)
        r_kn, r_kn_c = self._pt(smoothed_pts, 14)
        l_an, l_an_c = self._pt(smoothed_pts, 15)
        r_an, r_an_c = self._pt(smoothed_pts, 16)
        l_sh, l_sh_c = self._pt(smoothed_pts, 5)
        r_sh, r_sh_c = self._pt(smoothed_pts, 6)

        m: dict[str, float | None] = {}
        m["kL"] = angle_three_points(l_hi, l_kn, l_an) if self._valid(l_hi_c, l_kn_c, l_an_c) else None
        m["kR"] = angle_three_points(r_hi, r_kn, r_an) if self._valid(r_hi_c, r_kn_c, r_an_c) else None
        if self._valid_majority(l_hi_c, l_kn_c, l_an_c):
            m["aL"] = knee_alignment_pct(l_kn, l_hi, l_an)
        else:
            m["aL"] = None
        if self._valid_majority(r_hi_c, r_kn_c, r_an_c):
            m["aR"] = knee_alignment_pct(r_kn, r_hi, r_an)
        else:
            m["aR"] = None
        if self._valid_majority(l_hi_c, r_hi_c, l_sh_c, r_sh_c):
            m["hip_span"] = hip_span_ratio(l_hi, r_hi, l_sh, r_sh)
        else:
            m["hip_span"] = None

        occupied: list = []
        if m["kL"] is not None:
            draw_angle_arc(frame, l_kn, l_hi, l_an, m["kL"], COLOR_TEAL, 28)
            draw_joint_tag(frame, smoothed_pts[13], f"Izq {m['kL']:.0f}°", "L", occupied, self.vis_thr)
        if m["kR"] is not None:
            draw_angle_arc(frame, r_kn, r_hi, r_an, m["kR"], COLOR_TEAL, 28)
            draw_joint_tag(frame, smoothed_pts[14], f"Der {m['kR']:.0f}°", "R", occupied, self.vis_thr)

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("RODILLA", COLOR_TEAL),
            (f"Flex Izq:{_f(m['kL'])}  Der:{_f(m['kR'])}", COLOR_WHITE),
            (f"Alin Izq:{_f(m['aL'])}  Der:{_f(m['aR'])}", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        kL = self._series(metrics_history, "kL")
        kR = self._series(metrics_history, "kR")
        aL = self._series(metrics_history, "aL")
        aR = self._series(metrics_history, "aR")
        R:  dict[str, MetricResult] = {}

        # Per-cycle ROM
        for key, series, raw_key in [("knee_rom_L", kL, "kL"), ("knee_rom_R", kR, "kR")]:
            val = self._per_cycle_rom(series, fps) if series else None
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
            )

        # Peak flexion (convention: angle_three_points → 180°=full ext, lower=more flexed)
        min_kL = min(kL) if kL else None
        min_kR = min(kR) if kR else None
        pfL = (180.0 - min_kL) if min_kL is not None else None
        pfR = (180.0 - min_kR) if min_kR is not None else None
        for key, val, raw_key in [
            ("peak_flexion_L", pfL, "kL"), ("peak_flexion_R", pfR, "kR"),
        ]:
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
            )

        # Extension deficit (180 - max_angle); max_angle = most extended
        max_kL = max(kL) if kL else None
        max_kR = max(kR) if kR else None
        edL = max(0.0, 180.0 - max_kL) if max_kL is not None else None
        edR = max(0.0, 180.0 - max_kR) if max_kR is not None else None
        for key, val, raw_key in [
            ("ext_deficit_L", edL, "kL"), ("ext_deficit_R", edR, "kR"),
        ]:
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
            )

        # Symmetry indices
        si_r = self._symmetry_index(romL := self._per_cycle_rom(kL, fps) if kL else None,
                                    romR := self._per_cycle_rom(kR, fps) if kR else None)
        si_p = self._symmetry_index(pfL, pfR)
        for key, val, raw_key in [
            ("si_rom", si_r, "kL"), ("si_peak_flex", si_p, "kL"),
        ]:
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
                min_coverage_pct=10.0,
            )

        # Angular velocity
        R["av_L"] = self._make_metric_from_series("av_L", self._angular_velocity(kL, fps), metrics_history, "kL")
        R["av_R"] = self._make_metric_from_series("av_R", self._angular_velocity(kR, fps), metrics_history, "kR")

        # CV
        for key, series, raw_key in [("cv_L", kL, "kL"), ("cv_R", kR, "kR")]:
            val = self._cv(series)
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
            )

        # Alignment (requires frontal/dorsal view — not valid in sagittal 2D)
        for key, series, raw_key in [("align_L", aL, "aL"), ("align_R", aR, "aR")]:
            R[key] = self._make_alignment_metric(
                key, series, metrics_history, raw_key,
                describe_fn=self._describe,
            )

        return R
