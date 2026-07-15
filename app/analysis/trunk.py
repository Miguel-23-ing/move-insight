"""
analysis/trunk.py  —  Tronco

Clinical references:
  - Whittle (2007): trunk lateral lean <5° in normal gait; ROM ~3-5°
  - Krebs et al. (1992): trunk rotation 8-10° ROM in normal gait
  - Saunders et al. (1953): "determinants of gait" – trunk motion

Keypoints: 5=L_shoulder  6=R_shoulder  11=L_hip  12=R_hip
"""
from __future__ import annotations
import numpy as np
from analysis.base import AnalysisModule, MetricResult, vector_angle
from core.drawing import (
    draw_skeleton, draw_metric_panel,
    COLOR_TEAL, COLOR_WHITE,
)

VERTICAL = np.array([0.0, -1.0], dtype=np.float32)
HORIZ    = np.array([1.0,  0.0], dtype=np.float32)


class TrunkModule(AnalysisModule):
    NAME        = "Tronco"
    DESCRIPTION = (
        "Análisis de tronco: inclinación sagital y lateral, ROM de ambas, "
        "rotación estimada, oblicuidad de hombros y pelvis, frecuencia de oscilación, "
        "variabilidad y consistencia del movimiento."
    )

    THRESHOLDS = {
        "trunk_lean_mean":   {"better": "lower",  "normal_max":  5.0, "warn_max": 10.0},
        "trunk_lean_rom":    {"better": "lower",  "normal_max":  8.0, "warn_max": 15.0},
        "lat_tilt_mean":     {"better": "lower",  "normal_max":  5.0, "warn_max":  8.0},
        "lat_tilt_rom":      {"better": "lower",  "normal_max":  6.0, "warn_max": 10.0},
        "trunk_rotation":    {"better": "lower",  "normal_max": 12.0, "warn_max": 20.0},
        "shoulder_obliq":    {"better": "lower",  "normal_max":  5.0, "warn_max":  8.0},
        "pelvic_obliq":      {"better": "lower",  "normal_max":  5.0, "warn_max":  8.0},
        "oscillation_freq":  {"better": "range",  "normal_min":  0.5, "normal_max": 1.5,
                                                   "warn_min":    0.2, "warn_max":   2.5},
        "cv_lateral":        {"better": "lower",  "normal_max": 20.0, "warn_max": 35.0},
        "cv_sagittal":       {"better": "lower",  "normal_max": 20.0, "warn_max": 35.0},
    }

    REFERENCE_RANGES = {
        "trunk_lean_mean":  "< 5°",
        "trunk_lean_rom":   "< 8°",
        "lat_tilt_mean":    "< 5°",
        "lat_tilt_rom":     "< 6°",
        "trunk_rotation":   "< 12°",
        "shoulder_obliq":   "< 5°",
        "pelvic_obliq":     "< 5°",
        "oscillation_freq": "0.5–1.5 Hz",
        "cv_lateral":       "< 20%",
        "cv_sagittal":      "< 20%",
    }

    METRIC_LABELS = {
        "trunk_lean_mean":  "Inclinación Sagital de Tronco (media)",
        "trunk_lean_rom":   "ROM Inclinación Sagital",
        "lat_tilt_mean":    "Inclinación Lateral de Tronco (media)",
        "lat_tilt_rom":     "ROM Inclinación Lateral",
        "trunk_rotation":   "Rotación de Tronco (proxy)",
        "shoulder_obliq":   "Oblicuidad de Hombros",
        "pelvic_obliq":     "Oblicuidad Pélvica",
        "oscillation_freq": "Frecuencia de Oscilación Lateral",
        "cv_lateral":       "Variabilidad Oscilación Lateral (CV)",
        "cv_sagittal":      "Variabilidad Inclinación Sagital (CV)",
    }

    METRIC_UNITS = {
        "trunk_lean_mean":"°","trunk_lean_rom":"°",
        "lat_tilt_mean":"°",  "lat_tilt_rom":"°",
        "trunk_rotation":"°", "shoulder_obliq":"°","pelvic_obliq":"°",
        "oscillation_freq":"Hz",
        "cv_lateral":"%","cv_sagittal":"%",
    }

    _DESC = {
        "trunk_lean_mean": {
            "normal":   "Inclinación sagital de tronco {v}° — dentro del rango normal (<5°).",
            "atencion": "Inclinación sagital {v}° — levemente elevada. Puede indicar compensación postural o debilidad de extensores.",
            "alerta":   "Inclinación sagital {v}° — significativa. Evaluar postura, dolor lumbar o debilidad extensora.",
        },
        "lat_tilt_mean": {
            "normal":   "Inclinación lateral de tronco {v}° — normal (<5°).",
            "atencion": "Inclinación lateral {v}° — moderada. Posible asimetría postural o compensación.",
            "alerta":   "Inclinación lateral {v}° — significativa. Evaluar escoliosis funcional, dolor o dismetría.",
        },
        "trunk_rotation": {
            "normal":   "Rotación de tronco estimada {v}° — dentro del rango normal en movimiento (<12°).",
            "atencion": "Rotación de tronco {v}° — moderadamente elevada. Posible compensación o asimetría.",
            "alerta":   "Rotación de tronco {v}° — significativa. Evaluar causa biomecánica o musculoesquelética.",
        },
        "oscillation_freq": {
            "normal":   "Frecuencia de oscilación {v} Hz — concordante con patrón de movimiento normal.",
            "atencion": "Frecuencia de oscilación {v} Hz — fuera del rango esperado. Puede indicar cadencia alterada.",
            "alerta":   "Frecuencia de oscilación {v} Hz — significativamente alterada. Evaluar patrón de movimiento.",
        },
    }

    def _describe(self, key: str, value: float | None, status: str) -> str:
        if key in self._DESC and value is not None:
            tmpl = self._DESC[key].get(status, "")
            if tmpl:
                return tmpl.replace("{v}", f"{value:.2f}{self.METRIC_UNITS.get(key,'°')}")
        return self._auto_describe(key, value, status)

    def get_relevant_keypoints(self) -> list[int]: return [5, 6, 11, 12]
    def get_connections(self) -> list[tuple[int, int]]:
        return [(5, 6), (5, 11), (6, 12), (11, 12)]

    def process_frame(
        self, frame: np.ndarray, smoothed_pts: list[dict],
        frame_idx: int, fps: float,
    ) -> dict[str, float | None]:
        draw_skeleton(frame, smoothed_pts,
                      highlight_kps=self.get_relevant_keypoints(),
                      visibility_threshold=self.vis_thr)
        l_sh, l_sh_c = self._pt(smoothed_pts, 5)
        r_sh, r_sh_c = self._pt(smoothed_pts, 6)
        l_hi, l_hi_c = self._pt(smoothed_pts, 11)
        r_hi, r_hi_c = self._pt(smoothed_pts, 12)

        m: dict[str, float | None] = {}

        # Trunk sagittal lean (mid_hip→mid_shoulder vs vertical)
        trunk = None
        if self._valid(l_hi_c, r_hi_c, l_sh_c, r_sh_c):
            mid_hip = (l_hi + r_hi) / 2
            mid_sh  = (l_sh + r_sh) / 2
            trunk   = vector_angle(mid_sh - mid_hip, VERTICAL)
        m["trunk"] = trunk

        # Shoulder obliquity vs horizontal
        m["sh_obl"] = vector_angle(r_sh - l_sh, HORIZ) if self._valid(l_sh_c, r_sh_c) else None

        # Pelvic obliquity
        m["pv_obl"] = vector_angle(r_hi - l_hi, HORIZ) if self._valid(l_hi_c, r_hi_c) else None

        # Lateral tilt: height diff shoulders normalised by torso height (≈degrees)
        lat_tilt = None
        if self._valid(l_sh_c, r_sh_c, l_hi_c, r_hi_c):
            dy       = abs(float(l_sh[1] - r_sh[1]))
            torso_h  = abs(float((l_sh[1]+r_sh[1])/2 - (l_hi[1]+r_hi[1])/2))
            if torso_h > 1.0:
                import math
                lat_tilt = math.degrees(math.atan2(dy, torso_h))
        m["lat_tilt"] = lat_tilt

        # Trunk rotation proxy (shoulder width / hip width ratio deviation)
        rot = None
        if self._valid(l_sh_c, r_sh_c, l_hi_c, r_hi_c):
            sh_w = abs(float(r_sh[0] - l_sh[0]))
            hi_w = abs(float(r_hi[0] - l_hi[0]))
            if hi_w > 1.0:
                import math
                rot = abs(1.0 - sh_w / hi_w) * 60.0   # rough mapping: ±1 ratio diff ≈ 60°
        m["rot"] = rot

        def _f(v): return "--" if v is None else f"{v:.0f}"
        draw_metric_panel(frame, [("TRONCO", COLOR_TEAL),
            (f"Sagital:{_f(m['trunk'])}  Lateral:{_f(m['lat_tilt'])}", COLOR_WHITE),
            (f"Hombros:{_f(m['sh_obl'])}  Pelvis:{_f(m['pv_obl'])}", COLOR_WHITE)])
        return m

    def compute_summary(
        self,
        metrics_history: list[dict[str, float | None]],
        fps: float, video_duration_s: float,
    ) -> dict[str, MetricResult]:
        tr  = self._series(metrics_history, "trunk")
        lt  = self._series(metrics_history, "lat_tilt")
        so  = self._series(metrics_history, "sh_obl")
        pv  = self._series(metrics_history, "pv_obl")
        rot = self._series(metrics_history, "rot")
        R:  dict[str, MetricResult] = {}

        # Sagittal lean
        tm = self._mean(tr); st = self.classify("trunk_lean_mean", tm)
        R["trunk_lean_mean"] = self._make_metric("trunk_lean_mean", tm, description=self._describe("trunk_lean_mean", tm, st))
        R["trunk_lean_rom"]  = self._make_metric("trunk_lean_rom",  self._per_cycle_rom(tr, fps) if tr else None)

        # Lateral tilt
        lm = self._mean(lt); st = self.classify("lat_tilt_mean", lm)
        R["lat_tilt_mean"]   = self._make_metric("lat_tilt_mean",  lm, description=self._describe("lat_tilt_mean", lm, st))
        R["lat_tilt_rom"]    = self._make_metric("lat_tilt_rom",   self._per_cycle_rom(lt, fps) if lt else None)

        # Rotation proxy
        rm = self._mean(rot); st = self.classify("trunk_rotation", rm)
        R["trunk_rotation"]  = self._make_metric("trunk_rotation", rm, description=self._describe("trunk_rotation", rm, st))

        # Obliquities
        R["shoulder_obliq"]  = self._make_metric("shoulder_obliq", self._mean(so))
        R["pelvic_obliq"]    = self._make_metric("pelvic_obliq",   self._mean(pv))

        # Oscillation frequency (lateral)
        freq = self._oscillation_frequency(lt, fps) if lt else None
        st   = self.classify("oscillation_freq", freq)
        R["oscillation_freq"]= self._make_metric("oscillation_freq", freq, description=self._describe("oscillation_freq", freq, st))

        # CV
        R["cv_lateral"]      = self._make_metric("cv_lateral",  self._cv(lt))
        R["cv_sagittal"]     = self._make_metric("cv_sagittal", self._cv(tr))

        return R
