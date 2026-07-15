"""
reports/report_builder.py
Informe clínico detallado, resumen narrativo y capa de datos para UI/PDF — MoveInsight.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import streamlit as st

from reports.clinical_content import ClinicalDetail, get_clinical_detail, get_interpretation

if TYPE_CHECKING:
    from analysis.base import MetricResult

# ── Paleta MoveInsight ────────────────────────────────────────────────────────
_STATUS = {
    "normal": {
        "bg": "#FFFFFF", "border": "#E2E8F0",
        "badge_bg": "#E6F4EA", "badge_txt": "#1E8E3E",
        "badge_title": "Normal", "badge_sub": "Dentro de rango",
        "label": "Normal",
    },
    "atencion": {
        "bg": "#FFFFFF", "border": "#E2E8F0",
        "badge_bg": "#FEF7E0", "badge_txt": "#B06000",
        "badge_title": "Atención", "badge_sub": "Leve desviación",
        "label": "Atención",
    },
    "alerta": {
        "bg": "#FFFFFF", "border": "#E2E8F0",
        "badge_bg": "#FCE8E6", "badge_txt": "#C5221F",
        "badge_title": "Alerta", "badge_sub": "Fuera de rango",
        "label": "Alerta",
    },
    "sin_datos": {
        "bg": "#F8FAFC", "border": "#CBD5E1",
        "badge_bg": "#F1F5F9", "badge_txt": "#475569",
        "badge_title": "Sin datos", "badge_sub": "No calculable",
        "label": "No calculable",
    },
}
_DEFAULT = _STATUS["sin_datos"]

DISCLAIMER = (
    "MoveInsight es una herramienta de apoyo cuantitativo al análisis de movimiento. "
    "Los valores, rangos de referencia e interpretaciones son orientativos y no constituyen "
    "diagnóstico médico ni plan de tratamiento. Toda decisión clínica es responsabilidad "
    "exclusiva del profesional de salud a cargo."
)


@dataclass
class MetricReportEntry:
    """Structured clinical payload for UI and PDF (one metric)."""
    key: str
    metric: "MetricResult"
    detail: ClinicalDetail
    interpretation: str
    deviation_text: str


@dataclass
class ReportData:
    analysis_name: str
    duration_s: float
    metrics: dict[str, "MetricResult"]
    generated_at: datetime
    avg_coverage: float
    calculable_count: int
    alert_count: int
    attention_count: int
    normal_count: int
    no_data_count: int
    thresholds: dict | None = None


def format_metric_value(m: "MetricResult") -> str:
    if not m.calculable or m.value is None:
        return "No calculable con la información disponible"
    u = m.unit
    if u in ("", "%"):
        return f"{m.value:.2f}{u}"
    return f"{m.value:.1f}{u}"


def build_report_data(
    analysis_name: str,
    metrics: dict[str, "MetricResult"],
    duration_s: float,
    thresholds: dict | None = None,
) -> ReportData:
    calculable = [m for m in metrics.values() if m.calculable]
    coverages = [m.coverage_pct for m in calculable if m.coverage_pct is not None]
    return ReportData(
        analysis_name=analysis_name,
        duration_s=duration_s,
        metrics=metrics,
        generated_at=datetime.now(),
        avg_coverage=sum(coverages) / len(coverages) if coverages else 0.0,
        calculable_count=len(calculable),
        alert_count=sum(1 for m in metrics.values() if m.status == "alerta"),
        attention_count=sum(1 for m in metrics.values() if m.status == "atencion"),
        normal_count=sum(1 for m in metrics.values() if m.status == "normal"),
        no_data_count=sum(1 for m in metrics.values() if m.status == "sin_datos"),
        thresholds=thresholds,
    )


def _metric_side(key: str, label: str) -> str | None:
    ll = f"{key} {label}".lower()
    if any(t in ll for t in ("_l", " izq", "izquierd", "left")):
        return "izquierdo"
    if any(t in ll for t in ("_r", " der", "derech", "right")):
        return "derecho"
    return None


def _deviation_text(
    key: str,
    m: "MetricResult",
    thresholds: dict | None,
) -> str:
    if not m.calculable or m.value is None or not thresholds or key not in thresholds:
        return ""
    t = thresholds[key]
    v = float(m.value)
    mode = t.get("better", "lower")
    unit = m.unit

    def _fmt(delta: float) -> str:
        if unit in ("", "%"):
            return f"{delta:.2f}{unit}"
        return f"{delta:.1f}{unit}"

    if mode == "lower":
        ref = t.get("normal_max")
        if ref is not None and v > ref:
            return f"Por encima del límite superior de referencia ({ref}{unit}) en {_fmt(v - ref)}."
    elif mode == "higher":
        ref = t.get("normal_min")
        if ref is not None and v < ref:
            return f"Por debajo del límite inferior de referencia ({ref}{unit}) en {_fmt(ref - v)}."
    elif mode == "range":
        nlo, nhi = t.get("normal_min"), t.get("normal_max")
        if nlo is not None and v < nlo:
            return f"Por debajo del rango de referencia ({nlo}–{nhi}{unit}) en {_fmt(nlo - v)}."
        if nhi is not None and v > nhi:
            return f"Por encima del rango de referencia ({nlo}–{nhi}{unit}) en {_fmt(v - nhi)}."
    return ""


def build_metric_entries(report: ReportData) -> list[MetricReportEntry]:
    """Unified structured data for screen rendering and PDF export."""
    entries: list[MetricReportEntry] = []
    for key, m in report.metrics.items():
        detail = get_clinical_detail(report.analysis_name, key, m)
        interp = get_interpretation(report.analysis_name, key, m)
        dev = _deviation_text(key, m, report.thresholds)
        entries.append(MetricReportEntry(
            key=key, metric=m, detail=detail,
            interpretation=interp, deviation_text=dev,
        ))
    return entries


def _duration_str(duration_s: float) -> str:
    mins, secs = int(duration_s // 60), duration_s % 60
    return f"{mins}m {secs:.0f}s" if mins else f"{secs:.0f}s"


def _quality_label(avg_cov: float) -> str:
    if avg_cov >= 60:
        return "Alta"
    if avg_cov >= 30:
        return "Moderada"
    if avg_cov > 0:
        return "Limitada"
    return "No disponible"


def _related_patterns(metrics: dict[str, "MetricResult"]) -> list[str]:
    """Detect cross-metric compensatory patterns (heuristic)."""
    patterns: list[str] = []
    altered = {k: m for k, m in metrics.items() if m.status in ("alerta", "atencion") and m.calculable}

    left_keys = [k for k in altered if "_L" in k or "Izq" in altered[k].label or "izq" in k.lower()]
    right_keys = [k for k in altered if "_R" in k or "Der" in altered[k].label or "der" in k.lower()]

    if left_keys and len(left_keys) >= 2:
        labels = ", ".join(altered[k].label for k in left_keys[:3])
        patterns.append(
            f"Convergencia de hallazgos en **miembro izquierdo** ({labels}), "
            f"sugiriendo posible patrón compensatorio unilateral (hipótesis orientativa)."
        )
    if right_keys and len(right_keys) >= 2:
        labels = ", ".join(altered[k].label for k in right_keys[:3])
        patterns.append(
            f"Convergencia de hallazgos en **miembro derecho** ({labels}), "
            f"sugiriendo posible patrón compensatorio unilateral."
        )

    si_keys = [k for k in altered if k.startswith("si_")]
    rom_keys = [k for k in altered if "rom" in k.lower()]
    if si_keys and rom_keys:
        patterns.append(
            "Coexistencia de **asimetría bilateral** y **ROM alterado**, compatible con "
            "marcha antálgica o compensación por dolor/debilidad unilateral."
        )

    pelvis_keys = [k for k in altered if "pelvis" in k.lower() or "pelvic" in k.lower()]
    hip_keys = [k for k in altered if "hip" in k.lower() or "cadera" in altered[k].label.lower()]
    if pelvis_keys and hip_keys:
        patterns.append(
            "Alteraciones de **cadera/pelvis** concurrentes — considerar evaluación de "
            "estabilizadores (glúteo medio) y signo de Trendelenburg."
        )

    cadence = metrics.get("cadence")
    if cadence and cadence.calculable and cadence.status != "normal":
        slow = cadence.value is not None and cadence.value < 100
        if slow and (si_keys or rom_keys):
            patterns.append(
                "**Cadencia reducida** junto con asimetría o ROM limitado sugiere "
                "posible marcha cautelosa o claudicación funcional."
            )

    return patterns


def _related_findings_for_metric(
    key: str,
    m: "MetricResult",
    altered: list[tuple[str, "MetricResult"]],
) -> str:
    """Link a finding to other altered metrics on the same side or region."""
    side = _metric_side(key, m.label)
    if not side:
        return ""
    related: list[str] = []
    for ok, om in altered:
        if ok == key:
            continue
        if _metric_side(ok, om.label) == side:
            related.append(om.label)
    if not related:
        return ""
    names = ", ".join(related[:3])
    return (
        f" En el mismo miembro ({side}) también se observan alteraciones en: {names}. "
        f"Este patrón sugiere una posible compensación unilateral (hipótesis orientativa)."
    )


def build_narrative_summary(report: ReportData) -> dict[str, str]:
    """Structured narrative sections for UI and PDF."""
    total = len(report.metrics)
    dur = _duration_str(report.duration_s)
    quality = _quality_label(report.avg_coverage)

    # 1. General data
    general = (
        f"**Tipo de análisis:** {report.analysis_name}\n\n"
        f"**Duración del video:** {dur}\n\n"
        f"**Parámetros evaluados:** {total} "
        f"({report.normal_count} dentro de rango, "
        f"{report.attention_count} en atención, "
        f"{report.alert_count} en alerta, "
        f"{report.no_data_count} no calculables)\n\n"
        f"**Calidad de detección:** {quality} "
        f"(cobertura media de keypoints: {report.avg_coverage:.0f}% en métricas calculables)"
    )

    # 2. Main findings
    order = {"alerta": 0, "atencion": 1, "normal": 2, "sin_datos": 3}
    altered = sorted(
        [(k, m) for k, m in report.metrics.items() if m.status in ("alerta", "atencion")],
        key=lambda x: order.get(x[1].status, 9),
    )

    findings_parts: list[str] = []
    if not altered:
        findings_parts.append(
            "No se identificaron parámetros fuera del rango de referencia orientativo "
            "entre las métricas calculables. El patrón global sugiere simetría y "
            "amplitud de movimiento conservadas dentro de los umbrales utilizados."
        )
    else:
        for key, m in altered:
            detail = get_clinical_detail(report.analysis_name, key, m)
            interp = get_interpretation(report.analysis_name, key, m)
            val_s = format_metric_value(m)
            dev = _deviation_text(key, m, report.thresholds)
            sev = "severa (alerta)" if m.status == "alerta" else "leve a moderada (atención)"
            cross = _related_findings_for_metric(key, m, altered)
            dev_part = f" {dev}" if dev else ""
            findings_parts.append(
                f"**{m.label}** — Valor obtenido: {val_s} (referencia: {m.ref_range}). "
                f"Desviación {sev}.{dev_part} {m.description} {interp} "
                f"Posibles causas asociadas: {'; '.join(detail.differential_causes[:4])}.{cross}"
            )

    findings = "\n\n".join(findings_parts)

    # 3. Global pattern
    patterns = _related_patterns(report.metrics)
    if not patterns and not altered:
        global_pattern = (
            "El conjunto de hallazgos no sugiere un patrón patológico reconocible "
            "más allá de la variabilidad fisiológica esperada."
        )
    elif patterns:
        global_pattern = " ".join(patterns)
    else:
        global_pattern = (
            "Los hallazgos aislados no configuran un patrón compensatorio claro. "
            "Se recomienda correlacionar con la exploración clínica presencial."
        )

    # 4. Recommendations
    recs: list[str] = []
    if report.no_data_count > 0:
        recs.append(
            "Repetir la captura con **vista frontal o sagital completa**, buena iluminación "
            "y articulaciones visibles durante todo el ciclo de movimiento."
        )
    if report.alert_count > 0:
        recs.append(
            "Realizar **evaluación clínica presencial** de las estructuras con parámetros en alerta."
        )
    if any(k.startswith("si_") and report.metrics[k].status != "normal"
           for k in report.metrics if k.startswith("si_")):
        recs.append(
            "Evaluar **fuerza muscular bilateral** (MRC) y tests funcionales unilaterales."
        )
    if any("align" in k and report.metrics[k].status != "normal" for k in report.metrics):
        recs.append(
            "Para alineación de rodilla: confirmar con **vista frontal** y considerar "
            "teleradiografía en carga si hay dolor o sospecha de valgo/varo."
        )
    recs.append(
        "Programar **seguimiento comparativo** con nuevo video en 4–8 semanas "
        "si se instaura intervención terapéutica."
    )
    if not recs:
        recs.append("Mantener seguimiento rutinario según criterio clínico.")

    recommendations = "\n\n".join(f"• {r}" for r in recs)

    return {
        "general": general,
        "findings": findings,
        "global_pattern": global_pattern,
        "recommendations": recommendations,
        "disclaimer": DISCLAIMER,
    }


def _md_to_html(text: str) -> str:
    import re
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return html.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")


def build_summary_text(
    analysis_name: str,
    metrics: dict[str, "MetricResult"],
    duration_s: float,
    thresholds: dict | None = None,
) -> str:
    """Returns HTML narrative for dashboard summary panel."""
    report = build_report_data(analysis_name, metrics, duration_s, thresholds=thresholds)
    sections = build_narrative_summary(report)
    html_parts = [
        f"<h4 style='color:#0F4C81;margin-top:0;'>1. Datos generales</h4><p>{_md_to_html(sections['general'])}</p>",
        f"<h4 style='color:#0F4C81;'>2. Hallazgos principales</h4><p>{_md_to_html(sections['findings'])}</p>",
        f"<h4 style='color:#0F4C81;'>3. Patrón global</h4><p>{_md_to_html(sections['global_pattern'])}</p>",
        f"<h4 style='color:#0F4C81;'>4. Recomendaciones sugeridas</h4><p>{_md_to_html(sections['recommendations'])}</p>",
        f"<h4 style='color:#0F4C81;'>5. Aviso clínico</h4><p style='font-size:0.9rem;'>{sections['disclaimer']}</p>",
    ]
    return "".join(html_parts)


def render_narrative_summary(sections: dict[str, str]) -> None:
    """Streamlit-native structured narrative (alternative to HTML block)."""
    st.markdown("#### 1. Datos generales del análisis")
    st.markdown(sections["general"])
    st.markdown("#### 2. Hallazgos principales")
    st.markdown(sections["findings"] or "_Sin hallazgos fuera de rango._")
    st.markdown("#### 3. Patrón global observado")
    st.markdown(sections["global_pattern"])
    st.markdown("#### 4. Recomendaciones sugeridas")
    st.markdown(sections["recommendations"])
    st.markdown("#### 5. Aviso clínico formal")
    st.caption(sections["disclaimer"])


def metric_card_html(key: str, m: "MetricResult", thresholds: dict | None = None) -> str:
    cfg = _STATUS.get(m.status, _DEFAULT)
    val_display = format_metric_value(m)
    cov = f" · Cobertura: {m.coverage_pct:.0f}%" if m.coverage_pct is not None and m.calculable else ""

    return f"""
<div style="background:{cfg['bg']};border:1px solid {cfg['border']};border-radius:12px;
padding:16px 20px;margin-bottom:12px;display:flex;align-items:center;
justify-content:space-between;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
  <div style="flex:1;">
    <div style="color:#0F4C81;font-size:0.9rem;font-weight:700;margin-bottom:4px;">{m.label}</div>
    <div style="font-size:1.4rem;font-weight:800;color:#0F172A;">{val_display}</div>
    <div style="color:#64748B;font-size:0.8rem;margin-top:2px;">Rango ref. {m.ref_range}{cov}</div>
  </div>
  <div style="background:{cfg['badge_bg']};border:1px solid {cfg['badge_txt']}33;
border-radius:8px;padding:8px 16px;text-align:center;min-width:120px;">
    <div style="color:{cfg['badge_txt']};font-weight:700;font-size:0.9rem;">{cfg['badge_title']}</div>
    <div style="color:{cfg['badge_txt']};font-size:0.7rem;opacity:0.85;">{cfg['badge_sub']}</div>
  </div>
</div>"""


def render_metric_expander(
    analysis_name: str,
    key: str,
    m: "MetricResult",
    thresholds: dict | None = None,
) -> None:
    """Streamlit expander with full clinical detail per metric."""
    cfg = _STATUS.get(m.status, _DEFAULT)
    detail = get_clinical_detail(analysis_name, key, m)
    interp = get_interpretation(analysis_name, key, m)
    val_s = format_metric_value(m)
    dev = _deviation_text(key, m, thresholds)
    title = f"{m.label} — {val_s} · {cfg['label']}"

    with st.expander(title, expanded=(m.status == "alerta")):
        if not m.calculable:
            st.warning(
                "**No calculable con la información disponible.** "
                "Revise ángulo de cámara (vista frontal para alineación de rodilla), "
                "iluminación, duración del video y visibilidad de articulaciones."
            )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Valor obtenido:** {val_s}")
            st.markdown(f"**Rango de referencia:** {m.ref_range}")
            st.markdown(f"**Estado:** {cfg['label']}")
            if m.coverage_pct is not None:
                st.markdown(f"**Cobertura de detección:** {m.coverage_pct:.0f}% ({m.sample_count} muestras)")
        with c2:
            st.markdown(f"**Resumen:** {m.description}")
            if dev:
                st.markdown(f"**Desviación respecto a referencia:** {dev}")

        st.markdown("---")
        st.markdown("#### Definición del índice")
        st.markdown(detail.definition)

        st.markdown("#### Relevancia clínica")
        st.markdown(detail.clinical_relevance)

        st.markdown("#### Interpretación según el valor obtenido")
        st.markdown(f"- **Dentro de rango:** {detail.interpretation_normal}")
        st.markdown(f"- **Por encima del rango:** {detail.interpretation_high}")
        st.markdown(f"- **Por debajo del rango:** {detail.interpretation_low}")
        st.info(f"**Interpretación de este caso:** {interp}")

        st.markdown("#### Posibles causas / diagnósticos diferenciales")
        for cause in detail.differential_causes:
            st.markdown(f"- {cause}")

        st.markdown("#### Consideraciones para el manejo clínico")
        for item in detail.management_considerations:
            st.markdown(f"- {item}")
        st.caption(
            "*Orientativo — no constituye plan de tratamiento definitivo.*"
        )

        st.markdown("#### Limitaciones de la medición")
        st.markdown(detail.limitations)


def export_pdf(
    analysis_name: str,
    metrics: dict[str, "MetricResult"],
    duration_s: float,
    logo_path: str | None = None,
    thresholds: dict | None = None,
) -> bytes:
    from reports.pdf_exporter import generate_clinical_pdf
    report = build_report_data(analysis_name, metrics, duration_s, thresholds=thresholds)
    sections = build_narrative_summary(report)
    entries = build_metric_entries(report)
    return generate_clinical_pdf(report, sections, entries=entries, logo_path=logo_path)
