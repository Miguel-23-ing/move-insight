"""
reports/pdf_exporter.py
Generación de informe clínico en PDF — reportlab.

Se eligió reportlab por: control preciso de layout multipágina, tablas, estilos
tipográficos, encabezados/pies de página y colores corporativos sin dependencias
del sistema (WeasyPrint requiere GTK; fpdf2 es más limitado para informes largos).
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from reports.report_builder import DISCLAIMER, format_metric_value

if TYPE_CHECKING:
    from reports.report_builder import MetricReportEntry, ReportData

# MoveInsight palette
C_PRIMARY   = colors.HexColor("#0C8E8C")
C_DARK      = colors.HexColor("#0F4C81")
C_TEXT      = colors.HexColor("#334155")
C_MUTED     = colors.HexColor("#64748B")
C_NORMAL    = colors.HexColor("#1E8E3E")
C_WARN      = colors.HexColor("#B06000")
C_ALERT     = colors.HexColor("#C5221F")
C_NODATA    = colors.HexColor("#475569")
C_BG_LIGHT  = colors.HexColor("#F0F4F8")
C_BORDER    = colors.HexColor("#E2E8F0")


def _status_color(status: str):
    return {
        "normal": C_NORMAL, "atencion": C_WARN,
        "alerta": C_ALERT, "sin_datos": C_NODATA,
    }.get(status, C_MUTED)


def _status_hex(status: str) -> str:
    return {
        "normal": "#1E8E3E", "atencion": "#B06000",
        "alerta": "#C5221F", "sin_datos": "#475569",
    }.get(status, "#64748B")


def _status_label(status: str) -> str:
    return {
        "normal": "Normal", "atencion": "Atención",
        "alerta": "Alerta", "sin_datos": "No calculable",
    }.get(status, status)


def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "MITitle", parent=base["Title"],
            fontName="Helvetica-Bold", fontSize=18,
            textColor=C_DARK, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "MISub", parent=base["Normal"],
            fontName="Helvetica", fontSize=10,
            textColor=C_MUTED, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "MIH2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=13,
            textColor=C_PRIMARY, spaceBefore=14, spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "MIH3", parent=base["Heading3"],
            fontName="Helvetica-Bold", fontSize=11,
            textColor=C_DARK, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "MIBody", parent=base["Normal"],
            fontName="Helvetica", fontSize=9.5,
            textColor=C_TEXT, leading=14, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "MIBullet", parent=base["Normal"],
            fontName="Helvetica", fontSize=9,
            textColor=C_TEXT, leading=13,
            leftIndent=12, bulletIndent=0,
        ),
        "footer": ParagraphStyle(
            "MIFooter", parent=base["Normal"],
            fontName="Helvetica", fontSize=7.5,
            textColor=C_MUTED, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "MITCell", parent=base["Normal"],
            fontName="Helvetica", fontSize=8.5,
            textColor=C_TEXT, leading=11,
        ),
        "table_header": ParagraphStyle(
            "MITHead", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8.5,
            textColor=colors.white, leading=11,
        ),
    }
    return styles


class _ClinicalDoc(BaseDocTemplate):
    def __init__(self, buf, report: "ReportData", logo_path: str | None = None, **kw):
        self._report = report
        self._logo_path = logo_path
        self._page_num = 0
        super().__init__(buf, **kw)

    def handle_pageBegin(self):
        self._page_num += 1
        super().handle_pageBegin()

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        # Header bar
        canvas.setFillColor(C_PRIMARY)
        canvas.rect(0, h - 18 * mm, w, 18 * mm, fill=1, stroke=0)
        if self._logo_path and Path(self._logo_path).exists():
            try:
                canvas.drawImage(
                    self._logo_path, 8 * mm, h - 16 * mm,
                    width=12 * mm, height=12 * mm,
                    preserveAspectRatio=True, mask="auto",
                )
                title_x = 22 * mm
            except Exception:
                title_x = 20 * mm
        else:
            title_x = 20 * mm
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(title_x, h - 12 * mm, "MoveInsight — Informe Clínico de Análisis de Movimiento")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 20 * mm, h - 12 * mm, self._report.analysis_name)
        # Footer
        canvas.setStrokeColor(C_BORDER)
        canvas.line(20 * mm, 15 * mm, w - 20 * mm, 15 * mm)
        canvas.setFillColor(C_MUTED)
        canvas.setFont("Helvetica", 6.5)
        # Full disclaimer wrapped in footer area
        disclaimer_lines = [
            DISCLAIMER[i:i + 95] for i in range(0, len(DISCLAIMER), 95)
        ]
        y = 12 * mm
        for line in disclaimer_lines[:2]:
            canvas.drawString(20 * mm, y, line)
            y -= 3 * mm
        canvas.drawRightString(w - 20 * mm, 10 * mm, f"Página {self._page_num}")
        canvas.restoreState()


def generate_clinical_pdf(
    report: "ReportData",
    sections: dict[str, str],
    entries: list["MetricReportEntry"] | None = None,
    logo_path: str | None = None,
) -> bytes:
    import re
    buf = io.BytesIO()
    styles = _build_styles()

    if entries is None:
        from reports.report_builder import build_metric_entries
        entries = build_metric_entries(report)

    doc = _ClinicalDoc(
        buf,
        report,
        logo_path=logo_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=28 * mm,
        bottomMargin=22 * mm,
        title="MoveInsight Informe Clínico",
    )

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main",
    )
    doc.addPageTemplates([
        PageTemplate(id="All", frames=[frame], onPage=doc._header_footer),
    ])

    story: list = []
    gen_date = report.generated_at.strftime("%d/%m/%Y %H:%M")
    dur_m, dur_s = int(report.duration_s // 60), report.duration_s % 60
    dur_str = f"{dur_m}m {dur_s:.0f}s" if dur_m else f"{dur_s:.0f}s"

    # ── Cover info ────────────────────────────────────────────────────────────
    story.append(Paragraph("Informe de Análisis Biomecánico", styles["title"]))
    story.append(Paragraph(
        f"Fecha: {gen_date} &nbsp;|&nbsp; Análisis: <b>{report.analysis_name}</b> "
        f"&nbsp;|&nbsp; Duración video: {dur_str}",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 6))

    # ── Section 1: General ────────────────────────────────────────────────────
    story.append(Paragraph("1. Datos generales del análisis", styles["h2"]))
    general_text = sections["general"].replace("**", "").replace("\n\n", "<br/><br/>")
    story.append(Paragraph(general_text, styles["body"]))
    story.append(Spacer(1, 10))

    # ── Section 2: Findings ───────────────────────────────────────────────────
    story.append(Paragraph("2. Hallazgos principales", styles["h2"]))
    findings_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", sections["findings"])
    findings_html = findings_html.replace("\n\n", "<br/><br/>")
    story.append(Paragraph(findings_html, styles["body"]))
    story.append(Spacer(1, 10))

    # ── Section 3: Global pattern ─────────────────────────────────────────────
    story.append(Paragraph("3. Patrón global observado", styles["h2"]))
    pat_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", sections["global_pattern"])
    story.append(Paragraph(pat_html, styles["body"]))
    story.append(Spacer(1, 10))

    # ── Section 4: Recommendations ────────────────────────────────────────────
    story.append(Paragraph("4. Recomendaciones sugeridas", styles["h2"]))
    rec_html = sections["recommendations"].replace("\n\n", "<br/>")
    story.append(Paragraph(rec_html, styles["body"]))
    story.append(Spacer(1, 14))

    # ── Metrics table ─────────────────────────────────────────────────────────
    story.append(Paragraph("5. Tabla de métricas", styles["h2"]))
    story.append(Spacer(1, 4))

    order = {"alerta": 0, "atencion": 1, "normal": 2, "sin_datos": 3}
    sorted_entries = sorted(
        entries,
        key=lambda e: order.get(e.metric.status, 9),
    )

    table_data = [[
        Paragraph("Métrica", styles["table_header"]),
        Paragraph("Valor", styles["table_header"]),
        Paragraph("Referencia", styles["table_header"]),
        Paragraph("Estado", styles["table_header"]),
        Paragraph("Interpretación breve", styles["table_header"]),
    ]]

    status_row_colors: list[tuple] = []
    for entry in sorted_entries:
        m = entry.metric
        brief = entry.interpretation
        if len(brief) > 160:
            brief = brief[:157] + "…"
        if not m.calculable:
            brief = "No calculable con la información disponible."
        table_data.append([
            Paragraph(m.label, styles["table_cell"]),
            Paragraph(format_metric_value(m), styles["table_cell"]),
            Paragraph(m.ref_range, styles["table_cell"]),
            Paragraph(
                f'<font color="{_status_hex(m.status)}">'
                f"<b>{_status_label(m.status)}</b></font>",
                styles["table_cell"],
            ),
            Paragraph(brief, styles["table_cell"]),
        ])

    col_widths = [55 * mm, 25 * mm, 28 * mm, 22 * mm, 50 * mm]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # ── Annex: full clinical detail ───────────────────────────────────────────
    story.append(Paragraph("Anexo: Detalle clínico por índice", styles["h2"]))
    story.append(Spacer(1, 6))

    for entry in sorted_entries:
        m, detail, interp = entry.metric, entry.detail, entry.interpretation
        story.append(Paragraph(
            f"{m.label} — {format_metric_value(m)} ({_status_label(m.status)})",
            styles["h3"],
        ))
        if entry.deviation_text:
            story.append(Paragraph(
                f"<b>Desviación respecto a referencia:</b> {entry.deviation_text}",
                styles["body"],
            ))
        story.append(Paragraph(f"<b>Definición:</b> {detail.definition}", styles["body"]))
        story.append(Paragraph(f"<b>Relevancia clínica:</b> {detail.clinical_relevance}", styles["body"]))
        story.append(Paragraph(f"<b>Interpretación (este caso):</b> {interp}", styles["body"]))
        causes = "; ".join(detail.differential_causes[:5])
        story.append(Paragraph(f"<b>Causas posibles:</b> {causes}", styles["body"]))
        mgmt = "; ".join(detail.management_considerations[:3])
        story.append(Paragraph(f"<b>Manejo clínico orientativo:</b> {mgmt}", styles["body"]))
        story.append(Paragraph(f"<b>Limitaciones:</b> {detail.limitations}", styles["body"]))
        story.append(Spacer(1, 8))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("Aviso clínico formal", styles["h2"]))
    story.append(Paragraph(sections["disclaimer"], styles["body"]))

    doc.build(story)
    return buf.getvalue()
