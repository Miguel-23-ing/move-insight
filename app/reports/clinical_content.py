"""
reports/clinical_content.py
Contenido clínico detallado por índice biomecánico — MoveInsight.
Referencias orientativas: Perry & Burnfield (2010), Whittle (2007), AAOS ROM,
Moe-Nilssen (1998), Winter (2009).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.base import MetricResult

LIMITATIONS_2D = (
    "Estimación 2D basada en visión por computador (YOLOv8-Pose). No sustituye "
    "captura de movimiento 3D certificada. Los resultados dependen de la calidad "
    "del video, resolución, ángulo de cámara, ropa y oclusión de articulaciones."
)


@dataclass
class ClinicalDetail:
    definition: str
    clinical_relevance: str
    interpretation_normal: str
    interpretation_high: str
    interpretation_low: str
    differential_causes: list[str] = field(default_factory=list)
    management_considerations: list[str] = field(default_factory=list)
    limitations: str = LIMITATIONS_2D


_GENERIC = ClinicalDetail(
    definition="Índice biomecánico derivado de la estimación de pose en 2D.",
    clinical_relevance="Apoya la evaluación cuantitativa del movimiento en consulta.",
    interpretation_normal="Valor dentro del rango de referencia orientativo.",
    interpretation_high="Valor por encima del rango — requiere correlación clínica.",
    interpretation_low="Valor por debajo del rango — requiere correlación clínica.",
    differential_causes=["Variabilidad interindividual", "Condición aguda o crónica según contexto clínico"],
    management_considerations=["Evaluación clínica presencial complementaria"],
)


def _side(label: str) -> str:
    ll = label.lower()
    if "izq" in ll or "izquierd" in ll:
        return "izquierdo"
    if "der" in ll or "derech" in ll:
        return "derecho"
    return "bilateral"


def _severity_prefix(status: str) -> str:
    if status == "alerta":
        return "**Severidad: alteración marcada (alerta).** "
    if status == "atencion":
        return "**Severidad: desviación leve a moderada (atención).** "
    return ""


def _interpret_for_metric(m: "MetricResult", detail: ClinicalDetail) -> str:
    if not m.calculable or m.value is None:
        return (
            "No calculable con la información disponible. "
            "Revise visibilidad de articulaciones, duración del video y ángulo de cámara."
        )
    if m.status == "normal":
        return detail.interpretation_normal
    low_na = "no aplica" in detail.interpretation_low.lower()
    high_na = "no aplica" in detail.interpretation_high.lower()
    if m.deviation == "low" and not low_na:
        base = detail.interpretation_low
    elif m.deviation == "high" and not high_na:
        base = detail.interpretation_high
    elif low_na and not high_na:
        base = detail.interpretation_high
    elif high_na and not low_na:
        base = detail.interpretation_low
    else:
        base = detail.interpretation_high if m.status == "alerta" else detail.interpretation_low
    return _severity_prefix(m.status) + base


def _is_high_deviation(m: "MetricResult") -> bool:
    """Heuristic fallback when deviation direction is unavailable."""
    return m.deviation != "low"


def _fmt_val(m: "MetricResult") -> str:
    if m.value is None:
        return "N/D"
    u = m.unit
    if u in ("", "%"):
        return f"{m.value:.2f}{u}"
    return f"{m.value:.1f}{u}"


# ── Generic builders ──────────────────────────────────────────────────────────

def _rom_detail(joint: str, side: str, ref: str) -> ClinicalDetail:
    return ClinicalDetail(
        definition=(
            f"Rango de movimiento (ROM) de {joint} {side} por ciclo de marcha/movimiento, "
            f"calculado como la diferencia entre flexión máxima y extensión máxima detectadas "
            f"en cada ciclo (detección de picos/valles). Unidad: grados (°)."
        ),
        clinical_relevance=(
            f"El ROM de {joint} refleja la movilidad articular funcional durante la actividad "
            f"registrada. En marcha, un ROM reducido puede indicar estrategia antálgica, rigidez "
            f"capsular, dolor o debilidad muscular que limita la flexión en fase de balanceo."
        ),
        interpretation_normal=(
            f"ROM de {{val}} dentro del rango orientativo ({ref}). "
            f"Movilidad articular funcional adecuada para la actividad analizada."
        ),
        interpretation_high=(
            f"ROM elevado por encima del rango esperado. Puede observarse en hiperlaxitud, "
            f"patrón de marcha acelerada o artefacto de detección 2D."
        ),
        interpretation_low=(
            f"ROM reducido ({ref}). Sugiere posible rigidez articular, dolor (claudicación "
            f"antálgica), contractura en flexión, edema intraarticular, artrosis, secuelas "
            f"post-quirúrgicas o déficit neuromuscular."
        ),
        differential_causes=[
            "Gonartrosis / artrosis de rodilla" if "rodilla" in joint else f"Artrosis / capsulitis de {joint}",
            "Rigidez post-quirúrgica o inmovilización prolongada",
            "Dolor musculoesquelético (marcha antálgica)",
            "Contractura muscular (isquiotibiales, flexores de cadera)",
            "Sincinesia o espasticidad (origen neurológico)",
            "Edema o derrame articular agudo",
        ],
        management_considerations=[
            "Evaluación clínica articular: dolor, bloqueos, crepitación, estabilidad ligamentaria",
            "Medición goniométrica estática en consultorio para comparar con ROM dinámico",
            "Valorar estudio de imagen (RX, RM) si hay sospecha estructural",
            "Fisioterapia dirigida: movilización, fortalecimiento excéntrico, reeducación de marcha",
        ],
    )


def _symmetry_detail(region: str) -> ClinicalDetail:
    return ClinicalDetail(
        definition=(
            f"Índice de simetría (SI) de {region}: SI = 2·|L−R|/(L+R)·100%. "
            f"0% = simetría perfecta; >10% = asimetría clínicamente relevante; >20% = significativa."
        ),
        clinical_relevance=(
            f"Evalúa el equilibrio bilateral del patrón de movimiento de {region}. "
            f"La asimetría es un marcador sensible de compensación por dolor, debilidad unilateral, "
            f"dismetría de miembros o alteración neurológica."
        ),
        interpretation_normal=(
            "Simetría bilateral adecuada (<10%). Distribución equilibrada de la carga "
            "y amplitud de movimiento entre lados."
        ),
        interpretation_high=(
            "Asimetría moderada a severa. Evaluar patrón compensatorio unilateral: "
            "posible claudicación, insuficiencia de estabilizadores, dismetría o déficit neurológico."
        ),
        interpretation_low="No aplica — el índice solo tiene límite superior.",
        differential_causes=[
            "Dolor unilateral (marcha antálgica)",
            "Debilidad muscular unilateral (glúteo medio, cuádriceps)",
            "Dismetría de miembros inferiores",
            "Secuelas de fractura, artroplastia o meniscectomía",
            "Hemiparesia o neuropatía periférica",
            "Pie equino, pie caído o deformidad estructural unilateral",
        ],
        management_considerations=[
            "Correlacionar con hallazgos del mismo lado en otras métricas del reporte",
            "Evaluación de fuerza muscular manual (MRC) bilateral",
            "Test funcional unilateral (sentadilla, step-down, Trendelenburg)",
            "Considerar interconsulta ortopédica o neurología según contexto clínico",
        ],
    )


_FMT = "{val}"


def _personalize(detail: ClinicalDetail, m: "MetricResult") -> ClinicalDetail:
    val = _fmt_val(m)
    repl = lambda s: s.replace(_FMT, val) if _FMT in s else s.replace("{val}", val)
    return ClinicalDetail(
        definition=detail.definition,
        clinical_relevance=detail.clinical_relevance,
        interpretation_normal=repl(detail.interpretation_normal),
        interpretation_high=repl(detail.interpretation_high),
        interpretation_low=repl(detail.interpretation_low),
        differential_causes=detail.differential_causes,
        management_considerations=detail.management_considerations,
        limitations=detail.limitations,
    )


# ── Metric registry: (analysis_name, metric_key) → ClinicalDetail ─────────────

_REGISTRY: dict[tuple[str, str], ClinicalDetail] = {}


def _reg(analysis: str, key: str, detail: ClinicalDetail) -> None:
    _REGISTRY[(analysis, key)] = detail
    # Aliases for modules sharing keys
    if analysis == "Rodilla":
        _REGISTRY[("Piernas / Marcha inferior", key)] = detail


# ── GAIT / Cuerpo Completo ────────────────────────────────────────────────────

_GAIT = "Cuerpo Completo / Marcha"

_reg(_GAIT, "cadence", ClinicalDetail(
    definition=(
        "Cadencia: número de pasos por minuto, estimada a partir de los ciclos de "
        "oscilación de cadera/rodilla (intervalos pico-a-pico × 2 pasos/ciclo). "
        "Unidad: pasos/min."
    ),
    clinical_relevance=(
        "La cadencia es un determinante fundamental de la velocidad de marcha y la "
        "eficiencia energética. Valores bajos sugieren marcha cautelosa; valores muy "
        "altos pueden indicar inestabilidad o patrón compensatorio."
    ),
    interpretation_normal=(
        "Cadencia dentro del rango orientativo para adultos (100–120 pasos/min). "
        "Compatible con velocidad de marcha funcional normal."
    ),
    interpretation_high=(
        "Cadencia elevada (>120–140 pasos/min). Puede asociarse a pasos cortos, "
        "marcha apresurada, inestabilidad o intento de minimizar tiempo de apoyo unipodal."
    ),
    interpretation_low=(
        "Cadencia reducida (<100 pasos/min). Sugiere marcha lenta/cautelosa por dolor, "
        "miedo a caer, debilidad generalizada, enfermedad parkinsoniana, o secuelas de ACV."
    ),
    differential_causes=[
        "Parkinsonismo (marcha festinante con pasos cortos)",
        "Miedo a caer / inestabilidad postural",
        "Dolor generalizado o articular",
        "Debilidad muscular global (sarcopenia, miopatía)",
        "Secuelas de ACV o enfermedad neurológica",
        "Obstrucción vascular claudicante (marcha intermitente)",
    ],
    management_considerations=[
        "Evaluar velocidad de marcha (m/s) y longitud de zancada en consultorio",
        "Test de equilibrio (TUG, Berg) si hay sospecha de inestabilidad",
        "Valorar causa neurológica si cadencia muy baja con pasos cortos",
        "Reeducación de marcha con metrónomo en fisioterapia",
    ],
))

for side_key, side_label in [("knee_rom_L", "izquierda"), ("knee_rom_R", "derecha")]:
    d = _rom_detail("rodilla", side_label, ">45° en marcha")
    d.interpretation_normal = f"ROM de {{val}} por ciclo — adecuado para marcha (ref >45°)."
    d.interpretation_low = f"ROM de {{val}} — reducido. Evaluar rigidez, dolor o marcha cautelosa."
    _reg(_GAIT, side_key, d)

for side_key, side_label in [("hip_rom_L", "izquierda"), ("hip_rom_R", "derecha")]:
    d = _rom_detail("cadera", side_label, ">30° en marcha")
    _reg(_GAIT, side_key, d)

_reg(_GAIT, "si_knee", _symmetry_detail("rodilla (ROM por ciclo)"))
_reg(_GAIT, "si_hip", _symmetry_detail("cadera (ROM por ciclo)"))

_reg(_GAIT, "stride_reg_L", ClinicalDetail(
    definition=(
        "Regularidad de zancada izquierda: autocorrelación de la serie temporal del ángulo "
        "de rodilla (Moe-Nilssen, 1998). Escala 0–1; >0.90 = marcha muy regular."
    ),
    clinical_relevance=(
        "Refleja la consistencia ciclo a ciclo del patrón de marcha. Baja regularidad "
        "indica variabilidad motora elevada, posible inestabilidad o adaptación al terreno."
    ),
    interpretation_normal="Regularidad >0.90 — patrón de marcha consistente y predecible.",
    interpretation_high="No aplica — solo límite inferior.",
    interpretation_low=(
        "Regularidad <0.70–0.90 — variabilidad elevada entre ciclos. Evaluar equilibrio, "
        "fatiga, neuropatía o superficie irregular."
    ),
    differential_causes=["Neuropatía periférica", "Ataxia", "Fatiga muscular", "Dolor fluctuante", "Terreno irregular"],
    management_considerations=["Test de equilibrio dinámico", "Evaluación neurológica si variabilidad persistente"],
))
_reg(_GAIT, "stride_reg_R", ClinicalDetail(
    definition="Regularidad de zancada derecha (misma metodología que lado izquierdo).",
    clinical_relevance="Evalúa consistencia del patrón de marcha del miembro derecho.",
    interpretation_normal="Regularidad >0.90 — marcha consistente.",
    interpretation_high="No aplica.",
    interpretation_low="Regularidad reducida — evaluar control motor y equilibrio.",
    differential_causes=["Neuropatía", "Ataxia", "Fatiga", "Dolor unilateral derecho"],
    management_considerations=["Correlacionar con stride_reg_L y simetría global"],
))

_reg(_GAIT, "pelvis_obliquity", ClinicalDetail(
    definition=(
        "Oblicuidad pélvica media: ángulo entre la línea intercristal (cadera L–R) "
        "y la horizontal, promediado en el video. Unidad: grados (°)."
    ),
    clinical_relevance=(
        "Durante la marcha, la pelvis oscila en el plano frontal. Oblicuidad media "
        "elevada sugiere inestabilidad pélvica o signo de Trendelenburg."
    ),
    interpretation_normal="Oblicuidad <5° — estabilidad pélvica adecuada en marcha.",
    interpretation_high=(
        "Oblicuidad >5–8° — posible insuficiencia de glúteo medio, dismetría de miembros "
        "o compensación por dolor de cadera/rodilla."
    ),
    interpretation_low="No aplica — solo límite superior.",
    differential_causes=[
        "Debilidad de glúteo medio (Trendelenburg)",
        "Dismetría de miembros inferiores",
        "Artrosis de cadera unilateral",
        "Luxación congénita de cadera (adulto)",
        "Secuelas de fractura de pelvis o cadera",
    ],
    management_considerations=[
        "Test de Trendelenburg en exploración clínica",
        "Evaluación de fuerza de abductores de cadera (MRC)",
        "RX de pelvis en bipedestación si sospecha de dismetría",
    ],
))

_reg(_GAIT, "pelvis_obliquity_rom", ClinicalDetail(
    definition="ROM de oblicuidad pélvica: amplitud de oscilación pélvica frontal por ciclo.",
    clinical_relevance="Oscilación pélvica excesiva indica inestabilidad o compensación durante el apoyo unipodal.",
    interpretation_normal="ROM <6° — oscilación pélvica dentro de rango normal.",
    interpretation_high="ROM >6–10° — oscilación excesiva. Evaluar estabilizadores de cadera.",
    interpretation_low="No aplica.",
    differential_causes=["Insuficiencia de glúteo medio", "Pie equino", "Debilidad de core"],
    management_considerations=["Fortalecimiento de glúteo medio y core", "Evaluación de calzado ortésico"],
))

_reg(_GAIT, "trunk_lean", ClinicalDetail(
    definition="Inclinación sagital media del tronco (eje cadera→hombro vs vertical). Unidad: °.",
    clinical_relevance="Inclinación anterior excesiva aumenta carga lumbar; posterior puede indicar compensación.",
    interpretation_normal="Inclinación <5° — postura de tronco neutra durante la marcha.",
    interpretation_high="Inclinación >5–10° — posible compensación por dolor lumbar, debilidad de extensores o rigidez de cadera.",
    interpretation_low="No aplica.",
    differential_causes=["Dolor lumbar crónico", "Contractura de flexores de cadera", "Debilidad de extensores de tronco", "Obesidad abdominal"],
    management_considerations=["Evaluación postural y lumbar", "Fortalecimiento de core y glúteos"],
))

_reg(_GAIT, "trunk_oscillation", ClinicalDetail(
    definition="Oscilación lateral del tronco (desviación estándar de inclinación). Unidad: °.",
    clinical_relevance="Oscilación excesiva del tronco indica inestabilidad o estrategia compensatoria de equilibrio.",
    interpretation_normal="Oscilación <3° — control postural adecuado.",
    interpretation_high="Oscilación >3–6° — inestabilidad lateral. Evaluar equilibrio y fuerza de cadera.",
    interpretation_low="No aplica.",
    differential_causes=["Debilidad de estabilizadores de cadera", "Ataxia", "Miedo a caer"],
    management_considerations=["Entrenamiento de equilibrio", "Evaluación neurológica si persiste"],
))

for sk in ["av_knee_L", "av_knee_R"]:
    _reg(_GAIT, sk, ClinicalDetail(
        definition="Velocidad angular media de rodilla (|Δθ|·fps). Unidad: °/s.",
        clinical_relevance="Refleja la velocidad de flexo-extensión durante la marcha; alterada en patologías neuromusculares.",
        interpretation_normal="Velocidad angular dentro de 60–250 °/s — patrón cinemático normal.",
        interpretation_high="Velocidad muy elevada — posible marcha acelerada o artefacto de detección.",
        interpretation_low="Velocidad reducida — posible marcha lenta, rigidez o dolor.",
        differential_causes=["Parkinsonismo", "Rigidez articular", "Dolor", "Debilidad muscular"],
        management_considerations=["Correlacionar con cadencia y ROM"],
    ))

for sk in ["cv_knee_L", "cv_knee_R"]:
    _reg(_GAIT, sk, ClinicalDetail(
        definition="Coeficiente de variación (CV = σ/|μ|·100%) del ángulo de rodilla. Unidad: %.",
        clinical_relevance="Mide consistencia del movimiento entre ciclos; CV alto = alta variabilidad motora.",
        interpretation_normal="CV <15% — movimiento consistente ciclo a ciclo.",
        interpretation_high="CV >15–25% — variabilidad elevada. Evaluar inestabilidad o fatiga.",
        interpretation_low="No aplica.",
        differential_causes=["Inestabilidad articular", "Fatiga", "Neuropatía", "Dolor intermitente"],
        management_considerations=["Evaluar estabilidad ligamentaria", "Control de fatiga en actividad"],
    ))

_reg(_GAIT, "step_width_ratio", ClinicalDetail(
    definition=(
        "Relación ancho de paso / ancho de cadera: distancia horizontal entre tobillos "
        "dividida por distancia intercristal. Adimensional."
    ),
    clinical_relevance=(
        "El ancho de base modifica estabilidad. Base muy estrecha aumenta riesgo de caída; "
        "muy amplia sugiere inestabilidad compensatoria."
    ),
    interpretation_normal="Ratio 0.6–1.4 — base de sustentación funcional.",
    interpretation_high="Base muy amplia (>1.4) — posible marcha en tijera compensatoria o inestabilidad.",
    interpretation_low="Base muy estrecha (<0.6) — posible inestabilidad o marcha sobre línea.",
    differential_causes=["Ataxia", "Espasticidad", "Miedo a caer", "Dismetría", "Neuropatía"],
    management_considerations=["Evaluación de equilibrio", "Considerar ayuda técnica si inestabilidad"],
))

for sk in ["knee_align_L", "knee_align_R"]:
    _reg(_GAIT, sk, ClinicalDetail(
        definition=(
            "Desviación lateral de la rodilla respecto al eje cadera–tobillo, expresada "
            "como % de la longitud del segmento. Proxy 2D de genu valgo/varo (vista frontal)."
        ),
        clinical_relevance=(
            "La alineación del eje mecánico de la rodilla condiciona la distribución de "
            "cargas articulares. Valgo/varo excesivo predispone a artrosis compartimental."
        ),
        interpretation_normal="Desviación <8% — alineación dentro de rango orientativo.",
        interpretation_high=(
            "Desviación >8–14% — posible genu valgo o varo. Correlacionar con dolor "
            "retropatelar, artrosis medial/lateral o inestabilidad ligamentaria."
        ),
        interpretation_low="No aplica — solo límite superior.",
        differential_causes=[
            "Genu valgo (exceso de Q-angle)",
            "Genu varo",
            "Artrosis compartimental unilateral",
            "Lesión de LCA/LCP con inestabilidad",
            "Displasia de rótula",
            "Enfermedad de Blount (pediátrico)",
        ],
        management_considerations=[
            "Requiere vista frontal del video para validez; en vista sagital el valor tiende a ~0",
            "Evaluación clínica de eje mecánico en bipedestación",
            "RX en carga (teleradiografía) para cuantificación precisa",
            "Derivación ortopédica si desviación significativa con dolor",
        ],
        limitations=(
            LIMITATIONS_2D + " Este índice requiere vista frontal o posterior; "
            "en vista sagital/lateral la proyección 2D colapsa la desviación lateral."
        ),
    ))

# ── KNEE module ─────────────────────────────────────────────────────────────

_KNEE = "Rodilla"
for k in ["knee_rom_L", "knee_rom_R"]:
    _reg(_KNEE, k, _rom_detail("rodilla", _side(k), ">45°"))
for k in ["peak_flexion_L", "peak_flexion_R"]:
    _reg(_KNEE, k, ClinicalDetail(
        definition="Flexión máxima de rodilla = 180° − ángulo mínimo cadera–rodilla–tobillo.",
        clinical_relevance="Peak de flexión en swing phase; reducido en rigidez o dolor.",
        interpretation_normal="Flexión máxima >55° — adecuada para marcha.",
        interpretation_high="Flexión excesiva — hiperlaxitud o artefacto.",
        interpretation_low="Flexión reducida — rigidez, dolor, bloqueo articular.",
        differential_causes=["Artrosis", "Meniscopatía", "Rigidez post-quirúrgica", "Derrame articular"],
        management_considerations=["Goniometría estática", "Evaluación meniscal y ligamentaria"],
    ))
for k in ["ext_deficit_L", "ext_deficit_R"]:
    _reg(_KNEE, k, ClinicalDetail(
        definition="Déficit de extensión = 180° − ángulo máximo de extensión. Unidad: °.",
        clinical_relevance="Extensión incompleta altera la marcha (fase de apoyo) y aumenta consumo energético.",
        interpretation_normal="Déficit <10° — extensión funcional completa.",
        interpretation_high="Déficit >10–20° — extensión limitada clínicamente significativa.",
        interpretation_low="No aplica.",
        differential_causes=["Contractura en flexión", "Artrosis", "Cuerpo libre articular", "Espasticidad"],
        management_considerations=["Evaluar bloqueo articular", "Considerar movilización y estiramiento"],
    ))
for k in ["si_rom", "si_peak_flex"]:
    _reg(_KNEE, k, _symmetry_detail("rodilla"))
for k in ["av_L", "av_R"]:
    _reg(_KNEE, k, _REGISTRY[(_GAIT, k.replace("av_", "av_knee_"))])
for k in ["cv_L", "cv_R"]:
    _reg(_KNEE, k, _REGISTRY[(_GAIT, k.replace("cv_", "cv_knee_"))])
_reg(_KNEE, "align_L", _REGISTRY[(_GAIT, "knee_align_L")])
_reg(_KNEE, "align_R", _REGISTRY[(_GAIT, "knee_align_R")])

# ── HIP ───────────────────────────────────────────────────────────────────────

_HIP = "Cadera"
for sk in ["hip_rom_L", "hip_rom_R"]:
    _reg(_HIP, sk, _rom_detail("cadera", _side(sk), ">30°"))
_reg(_HIP, "si_rom", _symmetry_detail("cadera"))
_reg(_HIP, "si_peak_flex", _symmetry_detail("cadera (flexión máxima)"))
_reg(_HIP, "pelvis_obliq_mean", _REGISTRY[(_GAIT, "pelvis_obliquity")])
_reg(_HIP, "pelvis_obliq_rom", _REGISTRY[(_GAIT, "pelvis_obliquity_rom")])
_reg(_HIP, "shoulder_obliq", ClinicalDetail(
    definition="Oblicuidad de hombros: ángulo línea hombro L–R vs horizontal.",
    clinical_relevance="Asimetría escapular/pélvica en plano frontal; puede indicar escoliosis funcional.",
    interpretation_normal="Oblicuidad <5°.",
    interpretation_high="Oblicuidad >5° — evaluar escoliosis, dismetría o postura habitual.",
    interpretation_low="No aplica.",
    differential_causes=["Escoliosis", "Dismetría", "Postura habitual", "Debilidad unilateral"],
    management_considerations=["Evaluación postural completa", "RX columna si sospecha estructural"],
))
_reg(_HIP, "lateral_shift", ClinicalDetail(
    definition="Desplazamiento lateral de la pelvis respecto al centro del encuadre (% ancho).",
    clinical_relevance="Shift lateral puede indicar compensación por debilidad unilateral o dolor.",
    interpretation_normal="Desplazamiento <5% — centrado funcional.",
    interpretation_high="Desplazamiento >5–10% — posible compensación unilateral.",
    interpretation_low="No aplica.",
    differential_causes=["Trendelenburg", "Dolor unilateral", "Dismetría"],
    management_considerations=["Test de Trendelenburg", "Evaluación de glúteo medio"],
))

# ── TRUNK ─────────────────────────────────────────────────────────────────────

_TRUNK = "Tronco"
for k, gk in [("trunk_lean_mean", "trunk_lean"), ("pelvic_obliq", "pelvis_obliquity")]:
    if gk in [x[1] for x in _REGISTRY if x[0] == _GAIT]:
        src = _REGISTRY.get((_GAIT, gk))
        if src:
            _reg(_TRUNK, k, src)
_reg(_TRUNK, "lat_tilt_mean", ClinicalDetail(
    definition="Inclinación lateral del tronco estimada por diferencia de altura de hombros.",
    clinical_relevance="Inclinación lateral excesiva sugiere compensación o escoliosis funcional.",
    interpretation_normal="Inclinación <5°.",
    interpretation_high="Inclinación >5° — evaluar asimetría postural.",
    interpretation_low="No aplica.",
    differential_causes=["Escoliosis funcional", "Dismetría", "Dolor unilateral"],
    management_considerations=["Evaluación postural global"],
))
_reg(_TRUNK, "trunk_rotation", ClinicalDetail(
    definition="Proxy de rotación de tronco basado en ratio ancho hombros/cadera.",
    clinical_relevance="Rotación excesiva en marcha puede indicar compensación por rigidez de cadera.",
    interpretation_normal="Rotación <12° — dentro de rango.",
    interpretation_high="Rotación >12° — posible compensación rotacional.",
    interpretation_low="No aplica.",
    differential_causes=["Rigidez de cadera", "Debilidad de core", "Patología lumbar"],
    management_considerations=["Evaluación de movilidad de cadera y columna"],
))

# ── SHOULDER, ARM, ELBOW, NECK — register key metrics ─────────────────────────

def _shoulder_arm_common(analysis: str) -> None:
    _reg(analysis, "si_rom", _symmetry_detail("hombro/brazo"))
    for sk in ["rom_L", "rom_R"]:
        _reg(analysis, sk, _rom_detail("hombro/brazo", _side(sk), ">20°"))
    _reg(analysis, "height_diff", _REGISTRY.get((_HIP, "shoulder_obliq"), ClinicalDetail(
        definition="Diferencia de altura entre hombros.", clinical_relevance="Asimetría escapular.",
        interpretation_normal="Diferencia <5°.", interpretation_high="Diferencia >5°.",
        interpretation_low="No aplica.", differential_causes=["Escoliosis"], management_considerations=["Evaluación postural"],
    )))

_shoulder_arm_common("Hombro")
_shoulder_arm_common("Brazo")
_reg("Codo", "si_rom", _symmetry_detail("codo"))
_reg("Codo", "rom_L", _rom_detail("codo", "izquierdo", ">90°"))
_reg("Codo", "rom_R", _rom_detail("codo", "derecho", ">90°"))

_reg("Cuello", "head_tilt_mean", ClinicalDetail(
    definition="Inclinación lateral media de la cabeza (nariz vs eje orejas).",
    clinical_relevance="Evalúa postura cefálica y posible tortícolis funcional.",
    interpretation_normal="Inclinación <5° — postura cefálica equilibrada.",
    interpretation_high="Inclinación >5° — posible tortícolis, contractura SCM o hábito postural.",
    interpretation_low="No aplica.",
    differential_causes=["Tortícolis", "Contractura de trapecio/SCM", "Neuropatía craneal XI", "Postura sedentaria"],
    management_considerations=["Evaluación cervical manual", "RM cervical si déficit neurológico"],
))
_reg("Cuello", "fwd_head_mean", ClinicalDetail(
    definition="Proyección anterior de la cabeza respecto al eje hombros (% altura cefálica).",
    clinical_relevance="Forward Head Posture (FHP) asociada a cervicalgia, cefalea tensional y disfunción ATM.",
    interpretation_normal="Proyección <10% — alineación cefálica adecuada.",
    interpretation_high="Proyección >10–20% — FHP moderada a severa.",
    interpretation_low="No aplica — solo límite superior.",
    differential_causes=["Postura sedentaria prolongada", "Cifosis torácica", "Débil flexores cervicales profundos", "Visión deficiente"],
    management_considerations=["Reeducación postural", "Fortalecimiento flexores profundos", "Ergonomía laboral"],
))
_reg("Cuello", "fwd_head_max", ClinicalDetail(
    definition="Máxima proyección anterior de cabeza registrada en el video (%).",
    clinical_relevance="Captura el peor momento de FHP durante el movimiento.",
    interpretation_normal="Máximo <15% — picos de proyección dentro de rango.",
    interpretation_high="Máximo >15–30% — picos significativos de FHP.",
    interpretation_low="No aplica.",
    differential_causes=["Movimientos bruscos cervicales", "Postura dinámica alterada"],
    management_considerations=["Correlacionar con head_tilt_mean y exploración cervical"],
))
_reg("Cuello", "head_tilt_rom", ClinicalDetail(
    definition="ROM de inclinación lateral de cabeza por ciclo de movimiento.",
    clinical_relevance="Evalúa amplitud funcional de inclinación cervical lateral.",
    interpretation_normal="ROM <8° — movilidad lateral conservada.",
    interpretation_high="ROM >8–15° — oscilación lateral excesiva.",
    interpretation_low="No aplica.",
    differential_causes=["Inestabilidad cervical", "Tortícolis funcional", "Hiperlaxitud"],
    management_considerations=["Evaluación de movilidad cervical activa/pasiva"],
))
_reg("Cuello", "ear_asymmetry", ClinicalDetail(
    definition="Asimetría de altura entre orejas expresada en grados.",
    clinical_relevance="Indicador de inclinación cefálica y posible tortícolis.",
    interpretation_normal="Asimetría <5° — alineación lateral simétrica.",
    interpretation_high="Asimetría >5–10° — inclinación lateral significativa.",
    interpretation_low="No aplica.",
    differential_causes=["Tortícolis", "Contractura SCM", "Escoliosis cervical", "Neuropatía XI"],
    management_considerations=["Exploración de SCM y trapecio", "RM cervical si déficit neurológico"],
))
_reg("Cuello", "nose_dev_mean", ClinicalDetail(
    definition="Desviación lateral de la nariz respecto al eje mediano (% ancho de hombros).",
    clinical_relevance="Complementa la evaluación de alineación cefálica en plano frontal.",
    interpretation_normal="Desviación <5% — nariz centrada.",
    interpretation_high="Desviación >5–10% — posible desviación postural o tortícolis.",
    interpretation_low="No aplica.",
    differential_causes=["Tortícolis", "Hábito postural", "Desviación septal (no distinguible en 2D)"],
    management_considerations=["Correlacionar con head_tilt_mean y ear_asymmetry"],
))
_reg("Cuello", "neck_axis_mean", ClinicalDetail(
    definition="Inclinación del eje oreja-oreja respecto a la horizontal.",
    clinical_relevance="Evalúa alineación del eje cefálico en plano frontal.",
    interpretation_normal="Inclinación <5° — eje cefálico horizontal.",
    interpretation_high="Inclinación >5° — cabeza lateralizada.",
    interpretation_low="No aplica.",
    differential_causes=["Tortícolis", "Postura habitual asimétrica"],
    management_considerations=["Evaluación postural y muscular cervical"],
))
_reg("Cuello", "cv_tilt", ClinicalDetail(
    definition="Coeficiente de variación de la inclinación lateral de cabeza.",
    clinical_relevance="Alta variabilidad sugiere control motor cervical inconsistente.",
    interpretation_normal="CV <25% — postura cervical estable.",
    interpretation_high="CV >25–45% — variabilidad postural elevada.",
    interpretation_low="No aplica.",
    differential_causes=["Fatiga cervical", "Dolor intermitente", "Inestabilidad"],
    management_considerations=["Evaluar resistencia de flexores cervicales profundos"],
))

# ── HIP module (extended) ─────────────────────────────────────────────────────

for k in ["peak_flex_L", "peak_flex_R"]:
    _reg(_HIP, k, ClinicalDetail(
        definition="Flexión máxima de cadera = 180° − ángulo mínimo hombro–cadera–rodilla.",
        clinical_relevance="Peak de flexión en fase de balanceo; reducido en rigidez o contractura de flexores.",
        interpretation_normal="Flexión máxima >20° — adecuada para marcha.",
        interpretation_high="Flexión excesiva — hiperlaxitud o artefacto de detección.",
        interpretation_low="Flexión reducida — contractura de flexores, artrosis, dolor.",
        differential_causes=["Artrosis de cadera", "Contractura de iliopsoas", "Capsulitis", "Prótesis rígida"],
        management_considerations=["Test de Thomas para flexores", "Evaluación articular de cadera"],
    ))
for k in ["av_L", "av_R"]:
    _reg(_HIP, k, ClinicalDetail(
        definition="Velocidad angular media de cadera (|Δθ|·fps). Unidad: °/s.",
        clinical_relevance="Refleja velocidad de flexo-extensión coxofemoral durante la actividad.",
        interpretation_normal="Velocidad dentro de 30–200 °/s.",
        interpretation_high="Velocidad muy elevada — marcha acelerada o artefacto.",
        interpretation_low="Velocidad reducida — rigidez, dolor o debilidad.",
        differential_causes=["Parkinsonismo", "Artrosis", "Dolor de cadera", "Debilidad glútea"],
        management_considerations=["Correlacionar con ROM y cadencia"],
    ))
for k in ["cv_L", "cv_R"]:
    _reg(_HIP, k, ClinicalDetail(
        definition="Coeficiente de variación del ángulo de cadera.",
        clinical_relevance="Consistencia del patrón de movimiento coxofemoral.",
        interpretation_normal="CV <15% — movimiento consistente.",
        interpretation_high="CV >15–25% — variabilidad elevada.",
        interpretation_low="No aplica.",
        differential_causes=["Inestabilidad", "Fatiga", "Dolor intermitente"],
        management_considerations=["Evaluar control motor y fatiga"],
    ))

# ── TRUNK module (extended) ───────────────────────────────────────────────────

for k, title in [
    ("trunk_lean_rom", "ROM inclinación sagital"),
    ("lat_tilt_rom", "ROM inclinación lateral"),
]:
    _reg(_TRUNK, k, ClinicalDetail(
        definition=f"{title} de tronco por ciclo de movimiento.",
        clinical_relevance="Oscilación excesiva del tronco indica inestabilidad o compensación.",
        interpretation_normal="ROM dentro de rango orientativo (<6–8°).",
        interpretation_high="ROM elevado — oscilación excesiva del tronco.",
        interpretation_low="No aplica.",
        differential_causes=["Debilidad de core", "Inestabilidad de cadera", "Ataxia"],
        management_considerations=["Fortalecimiento de core y estabilizadores"],
    ))
_reg(_TRUNK, "shoulder_obliq", _REGISTRY.get((_HIP, "shoulder_obliq"), _GENERIC))
_reg(_TRUNK, "pelvic_obliq", _REGISTRY.get((_GAIT, "pelvis_obliquity"), _GENERIC))
_reg(_TRUNK, "oscillation_freq", ClinicalDetail(
    definition="Frecuencia dominante de oscilación lateral del tronco (FFT). Unidad: Hz.",
    clinical_relevance="Concordancia con cadencia de marcha (~0.8–1.2 Hz en adultos).",
    interpretation_normal="Frecuencia 0.5–1.5 Hz — patrón concordante con marcha.",
    interpretation_high="Frecuencia >1.5–2.5 Hz — posible artefacto o movimiento acelerado.",
    interpretation_low="Frecuencia <0.5 Hz — movimiento muy lento o datos insuficientes.",
    differential_causes=["Cadencia alterada", "Video corto", "Movimiento no cíclico"],
    management_considerations=["Correlacionar con cadencia si análisis de marcha disponible"],
))
for k in ["cv_lateral", "cv_sagittal"]:
    _reg(_TRUNK, k, ClinicalDetail(
        definition="Coeficiente de variación de inclinación de tronco.",
        clinical_relevance="Variabilidad postural durante el movimiento.",
        interpretation_normal="CV <20% — control postural consistente.",
        interpretation_high="CV >20–35% — variabilidad postural elevada.",
        interpretation_low="No aplica.",
        differential_causes=["Inestabilidad", "Fatiga", "Dolor lumbar intermitente"],
        management_considerations=["Evaluación de control motor y resistencia de core"],
    ))

# ── SHOULDER module (extended) ────────────────────────────────────────────────

for k in ["peak_flex_L", "peak_flex_R"]:
    _reg("Hombro", k, ClinicalDetail(
        definition="Peak de balanceo anterior del hombro (180° − ángulo mínimo).",
        clinical_relevance="Amplitud de balanceo en marcha; contribuye a estabilidad del tronco.",
        interpretation_normal="Peak >15° — balanceo funcional adecuado.",
        interpretation_high="Balanceo excesivo — posible hipermobilidad.",
        interpretation_low="Balanceo reducido — rigidez, dolor o parálisis unilateral.",
        differential_causes=["Capsulitis adhesiva", "Lesión de manguito rotador", "Hemiparesia"],
        management_considerations=["Evaluación de manguito rotador", "Test de Neer/Hawkins"],
    ))
for k in ["swing_freq_L", "swing_freq_R"]:
    _reg("Hombro", k, ClinicalDetail(
        definition="Frecuencia de oscilación del balanceo de hombro (Hz).",
        clinical_relevance="Debe ser ~2× la cadencia de marcha (reciprocidad cruzada).",
        interpretation_normal="Frecuencia 0.5–1.5 Hz — concordante con marcha.",
        interpretation_high="Frecuencia elevada — posible marcha acelerada.",
        interpretation_low="Frecuencia baja — movimiento lento o no cíclico.",
        differential_causes=["Cadencia alterada", "Actividad no locomotora"],
        management_considerations=["Correlacionar con cadencia en análisis de marcha"],
    ))
for k in ["av_L", "av_R", "cv_L", "cv_R"]:
    _reg("Hombro", k, ClinicalDetail(
        definition="Velocidad angular / variabilidad del balanceo de hombro.",
        clinical_relevance="Cinemática del balanceo durante la actividad registrada.",
        interpretation_normal="Dentro de rango orientativo para la actividad.",
        interpretation_high="Valor elevado — movimiento rápido o variable.",
        interpretation_low="Valor reducido — movimiento lento o restringido.",
        differential_causes=["Rigidez articular", "Dolor", "Debilidad neuromuscular"],
        management_considerations=["Evaluación articular y neuromuscular de hombro"],
    ))

# ── ARM module (extended) ─────────────────────────────────────────────────────

for k in ["swing_freq_L", "swing_freq_R"]:
    _reg("Brazo", k, _REGISTRY.get(("Hombro", k), _GENERIC))
for k in ["elbow_mean_L", "elbow_mean_R"]:
    _reg("Brazo", k, ClinicalDetail(
        definition="Ángulo medio de codo (hombro–codo–muñeca) durante la actividad.",
        clinical_relevance="En marcha normal el codo se mantiene ~70–140° de flexión.",
        interpretation_normal="Ángulo 70–140° — flexión funcional de codo en marcha.",
        interpretation_high="Codo muy extendido (>140°) — posible hiperextensión.",
        interpretation_low="Codo muy flexionado (<70°) — posible contractura o postura defensiva.",
        differential_causes=["Contractura de flexores", "Espasticidad", "Dolor de codo"],
        management_considerations=["Evaluación de flexo-extensión de codo"],
    ))
_reg("Brazo", "si_elbow", _symmetry_detail("posición de codo"))
for k in ["av_L", "av_R", "cv_L", "cv_R"]:
    _reg("Brazo", k, _REGISTRY.get(("Hombro", k), _GENERIC))

# ── ELBOW module (extended) ───────────────────────────────────────────────────

for k in ["peak_flex_L", "peak_flex_R"]:
    _reg("Codo", k, ClinicalDetail(
        definition="Flexión máxima de codo alcanzada en el video.",
        clinical_relevance="ROM funcional de flexión; AAOS ref >120° funcional.",
        interpretation_normal="Flexión >120° — rango funcional adecuado.",
        interpretation_high="Flexión excesiva — hiperlaxitud.",
        interpretation_low="Flexión limitada — rigidez, artrosis, contractura.",
        differential_causes=["Artrosis postraumática", "Contractura capsular", "Cuerpo libre"],
        management_considerations=["Goniometría estática", "Evaluación articular"],
    ))
for k in ["ext_deficit_L", "ext_deficit_R"]:
    _reg("Codo", k, ClinicalDetail(
        definition="Déficit de extensión = 180° − ángulo máximo de extensión. Unidad: °.",
        clinical_relevance="Extensión incompleta limita funcionalidad de miembro superior.",
        interpretation_normal="Déficit <10° — extensión funcional completa.",
        interpretation_high="Déficit >10–20° — extensión limitada clínicamente significativa.",
        interpretation_low="No aplica — solo límite superior.",
        differential_causes=["Contractura en flexión", "Artrosis", "Cuerpo libre articular", "Espasticidad"],
        management_considerations=["Evaluar bloqueo articular", "Movilización y estiramiento"],
    ))
_reg("Codo", "si_peak_flex", _symmetry_detail("codo (flexión máxima)"))
for k in ["av_L", "av_R", "cv_L", "cv_R"]:
    _reg("Codo", k, ClinicalDetail(
        definition="Velocidad angular / variabilidad del ángulo de codo.",
        clinical_relevance="Cinemática de flexo-extensión durante la actividad.",
        interpretation_normal="Dentro de rango orientativo (30–300 °/s; CV <12%).",
        interpretation_high="Velocidad/variabilidad elevada.",
        interpretation_low="Velocidad/variabilidad reducida — movimiento lento o rígido.",
        differential_causes=["Artrosis", "Rigidez post-quirúrgica", "Dolor"],
        management_considerations=["Evaluación clínica de codo"],
    ))


# ── Public API ────────────────────────────────────────────────────────────────


def get_clinical_detail(analysis_name: str, metric_key: str, m: "MetricResult") -> ClinicalDetail:
    detail = _REGISTRY.get((analysis_name, metric_key), _GENERIC)
    return _personalize(detail, m)


def get_interpretation(analysis_name: str, metric_key: str, m: "MetricResult") -> str:
    detail = get_clinical_detail(analysis_name, metric_key, m)
    return _interpret_for_metric(m, detail)
