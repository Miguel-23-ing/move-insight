"""
analysis/gait.py  —  Marcha Completa (Full Gait Analysis)

Clinical references:
  - Perry & Burnfield, "Gait Analysis: Normal and Pathological Function" (2010)
  - Whittle, "Gait Analysis: An Introduction" (4th ed.)
  - Winter, "Biomechanics and Motor Control of Human Movement" (2009)
  - Moe-Nilssen (1998) – stride regularity via autocorrelation

Keypoints (YOLOv8-Pose, 17-point COCO):
  5=left_shoulder   6=right_shoulder
  11=left_hip       12=right_hip
  13=left_knee      14=right_knee
  15=left_ankle     16=right_ankle
"""
from __future__ import annotations
from collections import deque
import numpy as np
from analysis.base import (
    AnalysisModule, MetricResult,
    angle_three_points, vector_angle, knee_alignment_pct, hip_span_ratio,
)
from core.drawing import (
    draw_skeleton, draw_angle_arc, draw_metric_panel, draw_joint_tag,
    COLOR_TEAL, COLOR_NORMAL, COLOR_WARN, COLOR_ALERT, COLOR_WHITE,
)

VERTICAL = np.array([0.0, -1.0], dtype=np.float32)
HORIZ    = np.array([1.0,  0.0], dtype=np.float32)


class GaitModule(AnalysisModule):
    """Full walking-gait biomechanical analysis (frontal or dorsal view)."""

    NAME        = "Marcha Completa"
    DESCRIPTION = (
        "Análisis biomecánico completo de la marcha: cadencia, ROM de cadera y rodilla "
        "por ciclo, índice de simetría bilateral, velocidad angular, regularidad de zancada, "
        "oblicuidad pélvica, inclinación de tronco y alineación de rodilla."
    )

    # ── Thresholds (all evidence-based) ──────────────────────────────────────
    THRESHOLDS = {
        # Cadence: Perry & Burnfield 2010 – normal adults 100-120 steps/min
        "cadence":             {"better": "range",  "normal_min": 100,  "normal_max": 120,
                                                     "warn_min":    80,  "warn_max":  140},
        # Knee ROM: Perry & Burnfield – 55-65° in normal gait cycle
        "knee_rom_L":          {"better": "higher", "normal_min": 45.0, "warn_min": 25.0},
        "knee_rom_R":          {"better": "higher", "normal_min": 45.0, "warn_min": 25.0},
        # Hip ROM: ~40-50° in normal gait
        "hip_rom_L":           {"better": "higher", "normal_min": 30.0, "warn_min": 15.0},
        "hip_rom_R":           {"better": "higher", "normal_min": 30.0, "warn_min": 15.0},
        # Symmetry Index: <10% normal, 10-20% mild asymmetry, >20% significant
        "si_knee":             {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        "si_hip":              {"better": "lower",  "normal_max": 10.0, "warn_max": 20.0},
        # Stride regularity: Moe-Nilssen 1998 – >0.90 highly regular
        "stride_reg_L":        {"better": "higher", "normal_min": 0.90, "warn_min": 0.70},
        "stride_reg_R":        {"better": "higher", "normal_min": 0.90, "warn_min": 0.70},
        # Pelvis obliquity: <5° normal in gait
        "pelvis_obliquity":    {"better": "lower",  "normal_max":  5.0, "warn_max":  8.0},
        "pelvis_obliquity_rom":{"better": "lower",  "normal_max":  6.0, "warn_max": 10.0},
        # Trunk lean: <5° normal
        "trunk_lean":          {"better": "lower",  "normal_max":  5.0, "warn_max": 10.0},
        "trunk_oscillation":   {"better": "lower",  "normal_max":  3.0, "warn_max":  6.0},
        # Angular velocity knee: rough ref ~80-200 °/s mean absolute
        "av_knee_L":           {"better": "range",  "normal_min":  60,  "normal_max": 250,
                                                     "warn_min":    30,  "warn_max":  350},
        "av_knee_R":           {"better": "range",  "normal_min":  60,  "normal_max": 250,
                                                     "warn_min":    30,  "warn_max":  350},
        # CV of knee angle: <15% normal, >25% high variability
        "cv_knee_L":           {"better": "lower",  "normal_max": 15.0, "warn_max": 25.0},
        "cv_knee_R":           {"better": "lower",  "normal_max": 15.0, "warn_max": 25.0},
        # Step width ratio (ankle dist / hip width): 0.6-1.4 normal
        "step_width_ratio":    {"better": "range",  "normal_min":  0.6, "normal_max":  1.4,
                                                     "warn_min":    0.3, "warn_max":   1.8},
        # Knee alignment deviation (genu valgo/varo proxy) — % of segment length
        "knee_align_L":        {"better": "lower",  "normal_max": 8.0,  "warn_max": 14.0},
        "knee_align_R":        {"better": "lower",  "normal_max": 8.0,  "warn_max": 14.0},
    }

    REFERENCE_RANGES = {
        "cadence":              "100–120 pasos/min",
        "knee_rom_L":           "> 45° (marcha normal)",
        "knee_rom_R":           "> 45° (marcha normal)",
        "hip_rom_L":            "> 30°",
        "hip_rom_R":            "> 30°",
        "si_knee":              "< 10%",
        "si_hip":               "< 10%",
        "stride_reg_L":         "> 0.90",
        "stride_reg_R":         "> 0.90",
        "pelvis_obliquity":     "< 5°",
        "pelvis_obliquity_rom": "< 6°",
        "trunk_lean":           "< 5°",
        "trunk_oscillation":    "< 3°",
        "av_knee_L":            "60–250 °/s",
        "av_knee_R":            "60–250 °/s",
        "cv_knee_L":            "< 15%",
        "cv_knee_R":            "< 15%",
        "step_width_ratio":     "0.6–1.4",
        "knee_align_L":         "< 8%",
        "knee_align_R":         "< 8%",
    }

    METRIC_LABELS = {
        "cadence":              "Cadencia (pasos/min)",
        "knee_rom_L":           "ROM Rodilla Izquierda (por ciclo)",
        "knee_rom_R":           "ROM Rodilla Derecha (por ciclo)",
        "hip_rom_L":            "ROM Cadera Izquierda (por ciclo)",
        "hip_rom_R":            "ROM Cadera Derecha (por ciclo)",
        "si_knee":              "Índice de Simetría — Rodilla",
        "si_hip":               "Índice de Simetría — Cadera",
        "stride_reg_L":         "Regularidad de Zancada Izquierda",
        "stride_reg_R":         "Regularidad de Zancada Derecha",
        "pelvis_obliquity":     "Oblicuidad Pélvica (media)",
        "pelvis_obliquity_rom": "ROM Oblicuidad Pélvica",
        "trunk_lean":           "Inclinación de Tronco (media)",
        "trunk_oscillation":    "Oscilación Lateral de Tronco (DE)",
        "av_knee_L":            "Velocidad Angular Rodilla Izq",
        "av_knee_R":            "Velocidad Angular Rodilla Der",
        "cv_knee_L":            "Variabilidad Rodilla Izq (CV)",
        "cv_knee_R":            "Variabilidad Rodilla Der (CV)",
        "step_width_ratio":     "Ancho de Paso / Ancho de Cadera",
        "knee_align_L":         "Alineación Rodilla Izq (valgo/varo)",
        "knee_align_R":         "Alineación Rodilla Der (valgo/varo)",
    }

    METRIC_UNITS = {
        "cadence":              "pasos/min",
        "knee_rom_L":           "°", "knee_rom_R": "°",
        "hip_rom_L":            "°", "hip_rom_R":  "°",
        "si_knee":              "%", "si_hip":      "%",
        "stride_reg_L":         "",  "stride_reg_R":"",
        "pelvis_obliquity":     "°", "pelvis_obliquity_rom": "°",
        "trunk_lean":           "°", "trunk_oscillation":    "°",
        "av_knee_L":            "°/s","av_knee_R":  "°/s",
        "cv_knee_L":            "%", "cv_knee_R":   "%",
        "step_width_ratio":     "",
        "knee_align_L":         "%",  "knee_align_R": "%",
    }

    # ── Specific clinical descriptions ────────────────────────────────────────
    _DESC = {
        "cadence": {
            "normal":   "Cadencia de {v} pasos/min — dentro del rango normal para adultos (100–120 pasos/min). Sugiere velocidad de marcha adecuada.",
            "atencion": "Cadencia de {v} pasos/min — fuera del rango típico (100–120). Puede indicar marcha cautelosa, fatiga o alteración del equilibrio.",
            "alerta":   "Cadencia de {v} pasos/min — significativamente alterada. Se recomienda evaluación funcional de la marcha.",
        },
        "si_knee": {
            "normal":   "Simetría de rodilla {v}% — excelente equilibrio bilateral. El sistema locomotor distribuye la carga simétricamente.",
            "atencion": "Simetría de rodilla {v}% — asimetría moderada. Puede indicar compensación por dolor o debilidad unilateral.",
            "alerta":   "Simetría de rodilla {v}% — asimetría significativa. Evaluar dolor, debilidad muscular o diferencia de longitud de miembro.",
        },
        "si_hip": {
            "normal":   "Simetría de cadera {v}% — patrón simétrico de oscilación pélvica.",
            "atencion": "Simetría de cadera {v}% — asimetría moderada en el balanceo de cadera. Revisar compensaciones posturales.",
            "alerta":   "Simetría de cadera {v}% — asimetría significativa. Se recomienda evaluación clínica de cadera y pelvis.",
        },
        "stride_reg_L": {
            "normal":   "Regularidad de zancada izq {v} — marcha muy consistente ciclo a ciclo. Buen control motor.",
            "atencion": "Regularidad de zancada izq {v} — variabilidad moderada entre ciclos. Puede indicar adaptación postural o terreno irregular.",
            "alerta":   "Regularidad de zancada izq {v} — baja consistencia entre ciclos. Evaluar control de equilibrio y estabilidad.",
        },
        "stride_reg_R": {
            "normal":   "Regularidad de zancada der {v} — marcha muy consistente ciclo a ciclo.",
            "atencion": "Regularidad de zancada der {v} — variabilidad moderada entre ciclos.",
            "alerta":   "Regularidad de zancada der {v} — baja consistencia. Evaluar control motor y equilibrio.",
        },
        "knee_rom_L": {
            "normal":   "ROM rodilla izq {v}° — adecuado para marcha normal (ref >45°). Buena movilidad articular.",
            "atencion": "ROM rodilla izq {v}° — reducido (ref >45°). Puede indicar rigidez, dolor o estrategia de marcha cautelosa.",
            "alerta":   "ROM rodilla izq {v}° — significativamente reducido. Evaluar rigidez articular, dolor u otras causas de restricción.",
        },
        "knee_rom_R": {
            "normal":   "ROM rodilla der {v}° — adecuado para marcha normal (ref >45°).",
            "atencion": "ROM rodilla der {v}° — reducido (ref >45°). Posible rigidez o marcha cautelosa.",
            "alerta":   "ROM rodilla der {v}° — significativamente reducido. Se recomienda evaluación clínica.",
        },
        "pelvis_obliquity": {
            "normal":   "Oblicuidad pélvica media {v}° — dentro del rango normal (<5°). Pelvis estable durante la marcha.",
            "atencion": "Oblicuidad pélvica {v}° — levemente elevada. Puede indicar debilidad de abductores de cadera (signo de Trendelenburg incipiente).",
            "alerta":   "Oblicuidad pélvica {v}° — elevada. Evaluar debilidad glútea, dismetría de miembros o alteración neurológica.",
        },
        "cv_knee_L": {
            "normal":   "Variabilidad rodilla izq (CV) {v}% — marcha muy consistente ciclo a ciclo.",
            "atencion": "Variabilidad rodilla izq (CV) {v}% — variabilidad moderada. Puede reflejar adaptación o fatiga.",
            "alerta":   "Variabilidad rodilla izq (CV) {v}% — alta variabilidad. Posible inestabilidad articular o control motor deficiente.",
        },
        "cv_knee_R": {
            "normal":   "Variabilidad rodilla der (CV) {v}% — marcha muy consistente ciclo a ciclo.",
            "atencion": "Variabilidad rodilla der (CV) {v}% — variabilidad moderada.",
            "alerta":   "Variabilidad rodilla der (CV) {v}% — alta variabilidad. Evaluar estabilidad articular.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                unit = self.METRIC_UNITS.get(key, "°")
                return tmpl.replace("{v}", f"{value:.1f}{unit}")
        return self._auto_describe(key, value, status)

    # ── AnalysisModule interface ──────────────────────────────────────────────

    def get_relevant_keypoints(self) -> list[int]:
        return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def get_connections(self) -> list[tuple[int, int]]:
        return [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)]

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
        l_an, l_an_c = self._pt(smoothed_pts, 15)
        r_an, r_an_c = self._pt(smoothed_pts, 16)

        m: dict[str, float | None] = {}

        # Hip angles (shoulder–hip–knee)
        m["hip_L"] = angle_three_points(l_sh, l_hi, l_kn) if self._valid(l_sh_c, l_hi_c, l_kn_c) else None
        m["hip_R"] = angle_three_points(r_sh, r_hi, r_kn) if self._valid(r_sh_c, r_hi_c, r_kn_c) else None

        # Knee angles (hip–knee–ankle)
        m["knee_L"] = angle_three_points(l_hi, l_kn, l_an) if self._valid(l_hi_c, l_kn_c, l_an_c) else None
        m["knee_R"] = angle_three_points(r_hi, r_kn, r_an) if self._valid(r_hi_c, r_kn_c, r_an_c) else None

        # Tibia inclination vs vertical
        m["tibia_L"] = vector_angle(l_kn - l_an, VERTICAL) if self._valid(l_kn_c, l_an_c) else None
        m["tibia_R"] = vector_angle(r_kn - r_an, VERTICAL) if self._valid(r_kn_c, r_an_c) else None

        # Pelvis obliquity
        m["pelvis"] = vector_angle(r_hi - l_hi, HORIZ) if self._valid(l_hi_c, r_hi_c) else None

        # Trunk lean
        trunk = None
        if self._valid(l_hi_c, r_hi_c, l_sh_c, r_sh_c):
            trunk = vector_angle((l_sh + r_sh) / 2 - (l_hi + r_hi) / 2, VERTICAL)
        m["trunk"] = trunk

        # Step width ratio
        swr = None
        if self._valid(l_hi_c, r_hi_c):
            hw = abs(float(r_hi[0] - l_hi[0]))
            if hw > 1.0 and self._valid(l_an_c, r_an_c):
                swr = abs(float(r_an[0] - l_an[0])) / hw
        m["swr"] = swr

        # Knee alignment (relaxed visibility: 2 of 3 keypoints)
        if self._valid_majority(l_hi_c, l_kn_c, l_an_c):
            m["ka_L"] = knee_alignment_pct(l_kn, l_hi, l_an)
        else:
            m["ka_L"] = None
        if self._valid_majority(r_hi_c, r_kn_c, r_an_c):
            m["ka_R"] = knee_alignment_pct(r_kn, r_hi, r_an)
        else:
            m["ka_R"] = None
        if self._valid(l_hi_c, r_hi_c, l_sh_c, r_sh_c):
            m["hip_span"] = hip_span_ratio(l_hi, r_hi, l_sh, r_sh)
        else:
            m["hip_span"] = None

        # Draw overlays
        occupied: list = []
        for side, kn, hi, an, idx_kn, idx_hi, lab in [
            ("L", l_kn, l_hi, l_an, 13, 11, "R I"),
            ("R", r_kn, r_hi, r_an, 14, 12, "R D"),
        ]:
            ang = m[f"knee_{side}"]
            if ang is not None:
                draw_angle_arc(frame, kn, hi, an, ang, COLOR_TEAL, 22)
                draw_joint_tag(frame, smoothed_pts[idx_kn], f"{lab}{ang:.0f}°", side, occupied, self.vis_thr)
            hang = m[f"hip_{side}"]
            if hang is not None:
                draw_joint_tag(frame, smoothed_pts[idx_hi], f"C{side}{hang:.0f}°", side, occupied, self.vis_thr)

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [
            ("MARCHA COMPLETA", COLOR_TEAL),
            (f"Rodilla  I:{_f(m['knee_L'])}  D:{_f(m['knee_R'])}", COLOR_WHITE),
            (f"Cadera   I:{_f(m['hip_L'])}  D:{_f(m['hip_R'])}", COLOR_WHITE),
            (f"Pelvis:{_f(m['pelvis'])}  Tronco:{_f(m['trunk'])}", COLOR_WHITE),
        ])
        return m

    # ── Summary (all clinical metrics) ────────────────────────────────────────

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float,
        video_duration_s: float,
    ) -> dict[str, MetricResult]:

        kL = self._series(metrics_history, "knee_L")
        kR = self._series(metrics_history, "knee_R")
        hL = self._series(metrics_history, "hip_L")
        hR = self._series(metrics_history, "hip_R")
        pv = self._series(metrics_history, "pelvis")
        tr = self._series(metrics_history, "trunk")
        sw = self._series(metrics_history, "swr")
        aL = self._series(metrics_history, "ka_L")
        aR = self._series(metrics_history, "ka_R")

        R: dict[str, MetricResult] = {}

        # ── Cadence (from hip oscillation, more reliable than knee) ───────────
        cad_series = hL or hR or kL or kR
        cad = self._estimate_cadence(cad_series, fps) if cad_series else None
        cad_key = "hip_L" if hL else ("hip_R" if hR else ("knee_L" if kL else "knee_R"))
        st  = self.classify("cadence", cad)
        R["cadence"] = self._make_metric_from_series(
            "cadence", cad, metrics_history, cad_key,
            description=self._describe("cadence", cad, st),
            min_coverage_pct=15.0, min_samples=20,
        )

        # ── Per-cycle ROM ─────────────────────────────────────────────────────
        for key, series, raw_key in [
            ("knee_rom_L", kL, "knee_L"), ("knee_rom_R", kR, "knee_R"),
            ("hip_rom_L",  hL, "hip_L"),  ("hip_rom_R",  hR, "hip_R"),
        ]:
            val = self._per_cycle_rom(series, fps) if series else None
            st  = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
            )

        # ── Symmetry Index ────────────────────────────────────────────────────
        si_kn = self._symmetry_index(self._per_cycle_rom(kL, fps), self._per_cycle_rom(kR, fps))
        si_hi = self._symmetry_index(self._per_cycle_rom(hL, fps), self._per_cycle_rom(hR, fps))
        for key, val, raw_key in [
            ("si_knee", si_kn, "knee_L"),
            ("si_hip", si_hi, "hip_L"),
        ]:
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
                min_coverage_pct=10.0,
            )

        # ── Stride regularity ─────────────────────────────────────────────────
        sr_L = self._stride_regularity(kL)
        sr_R = self._stride_regularity(kR)
        for key, val, raw_key in [
            ("stride_reg_L", sr_L, "knee_L"),
            ("stride_reg_R", sr_R, "knee_R"),
        ]:
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
                min_coverage_pct=20.0, min_samples=30,
            )

        # ── Pelvis ────────────────────────────────────────────────────────────
        pv_m   = self._mean(pv)
        pv_rom = self._per_cycle_rom(pv, fps) if pv else None
        st_pv  = self.classify("pelvis_obliquity", pv_m)
        R["pelvis_obliquity"] = self._make_metric_from_series(
            "pelvis_obliquity", pv_m, metrics_history, "pelvis",
            description=self._describe("pelvis_obliquity", pv_m, st_pv),
        )
        R["pelvis_obliquity_rom"] = self._make_metric_from_series(
            "pelvis_obliquity_rom", pv_rom, metrics_history, "pelvis",
        )

        # ── Trunk ─────────────────────────────────────────────────────────────
        R["trunk_lean"] = self._make_metric_from_series(
            "trunk_lean", self._mean(tr), metrics_history, "trunk",
        )
        R["trunk_oscillation"] = self._make_metric_from_series(
            "trunk_oscillation", self._std(tr), metrics_history, "trunk",
        )

        # ── Angular velocity ──────────────────────────────────────────────────
        R["av_knee_L"] = self._make_metric_from_series("av_knee_L", self._angular_velocity(kL, fps), metrics_history, "knee_L")
        R["av_knee_R"] = self._make_metric_from_series("av_knee_R", self._angular_velocity(kR, fps), metrics_history, "knee_R")

        # ── Coefficient of Variation ──────────────────────────────────────────
        for key, series, raw_key in [("cv_knee_L", kL, "knee_L"), ("cv_knee_R", kR, "knee_R")]:
            val = self._cv(series)
            st = self.classify(key, val)
            R[key] = self._make_metric_from_series(
                key, val, metrics_history, raw_key,
                description=self._describe(key, val, st),
            )

        # ── Step width ratio ──────────────────────────────────────────────────
        R["step_width_ratio"] = self._make_metric_from_series(
            "step_width_ratio", self._mean(sw), metrics_history, "swr",
        )

        # ── Knee alignment (frontal view required) ────────────────────────────────
        for key, series, raw_key in [
            ("knee_align_L", aL, "ka_L"),
            ("knee_align_R", aR, "ka_R"),
        ]:
            R[key] = self._make_alignment_metric(
                key, series, metrics_history, raw_key,
                describe_fn=self._describe,
            )

        return R
