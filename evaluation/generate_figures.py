import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError as e:
    print(f"\n  Dependencia no encontrada: {e}")
    print("    Instalar con: pip install matplotlib numpy")
    sys.exit(1)


BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / "results"
FIGURES_DIR  = BASE_DIR / "figures_v2"
FIGURES_DIR.mkdir(exist_ok=True)

COMPARISON_JSON = RESULTS_DIR / "comparison_1034_summary.json"
ERROR_JSON      = RESULTS_DIR / "error_analysis_1034.json"
FULL_RESULTS    = RESULTS_DIR / "full_1034.json"


COLOR_AGENT    = "#1565C0"
COLOR_BASE     = "#90A4AE"
COLOR_MATCH    = "#43A047"
COLOR_MISMATCH = "#E53935"
COLOR_FAILED   = "#FB8C00"

ERROR_COLORS = {
    "A": "#E53935",
    "D": "#7B1FA2",
    "C": "#FB8C00",
    "E": "#1E88E5",
    "B": "#B0BEC5",
}

ERROR_LABELS = {
    "A": "Selección de columnas\n(SELECT * vs específicas)",
    "D": "Error de tipo/conversión\n(TEXT fallback, car_1/wta_1)",
    "C": "Semántica de agregación\n(estrategia de agregación distinta)",
    "E": "Error de esquema/JOIN\n(tabla equivocada)",
    "B": "Ordenación/LIMIT\n(no observado)",
}


def setup_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   11,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#E0E0E0",
        "grid.linewidth":    0.8,
        "axes.axisbelow":    True,
    })


def fig1_ea_por_nivel(comparison: dict) -> Path:
    DIFF_ORDER = ["easy", "medium", "hard", "extra"]

    DIFF_LABELS = [
        "Easy\n(n=365)",
        "Medium\n(n=405)",
        "Hard\n(n=157)",
        "Extra\n(n=107)",
    ]

    by_diff = comparison["by_difficulty"]
    agent_ea = [by_diff[d]["agent_correct"] / by_diff[d]["total"] * 100
                for d in DIFF_ORDER]
    base_ea  = [by_diff[d]["base_correct"]  / by_diff[d]["total"] * 100
                for d in DIFF_ORDER]
    deltas   = [a - b for a, b in zip(agent_ea, base_ea)]

    all_labels = DIFF_LABELS + ["Total\n(n=1034)"]
    agent_ea.append(comparison["agent_ea_pct"])
    base_ea.append(comparison["base_ea_pct"])
    deltas.append(comparison["delta_ea_pct"])

    x     = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_a = ax.bar(x - width/2, agent_ea, width,
                    label="Agente (max_attempts=3)",
                    color=COLOR_AGENT, alpha=0.92, zorder=3)
    bars_b = ax.bar(x + width/2, base_ea,  width,
                    label="Baseline (max_attempts=1)",
                    color=COLOR_BASE,  alpha=0.92, zorder=3)

    ax.axhline(comparison["agent_ea_pct"], color=COLOR_AGENT,
               linestyle="--", linewidth=1.2, alpha=0.5, zorder=2)
    ax.axhline(comparison["base_ea_pct"],  color=COLOR_BASE,
               linestyle="--", linewidth=1.2, alpha=0.5, zorder=2)

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                color=COLOR_AGENT, fontweight="bold")

    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                color="#546E7A")

    for i, (xa, xb, delta) in enumerate(zip(
            [b.get_x() + b.get_width()/2 for b in bars_a],
            [b.get_x() + b.get_width()/2 for b in bars_b],
            deltas)):
        x_mid = (xa + xb) / 2
        y_max = max(agent_ea[i], base_ea[i]) + 5.5
        if abs(delta) > 0.05:
            color = COLOR_MATCH if delta > 0 else COLOR_MISMATCH
            ax.text(x_mid, y_max, f"Δ={delta:+.1f}%",
                    ha="center", va="bottom", fontsize=9,
                    color=color, fontweight="bold")

    ax.axvline(x=len(DIFF_ORDER) - 0.5, color="#BDBDBD",
               linestyle=":", linewidth=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Execution Accuracy (%)")
    ax.set_title(
        "Execution Accuracy por nivel de dificultad: Agente (retry loop) vs Baseline",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.legend(loc="lower right", framealpha=0.9)

    ax.text(0.01, 0.02,
            "Spider dev set completo (n=1034) | modelo: claude-sonnet-4-5",
            transform=ax.transAxes, fontsize=8.5, color="#757575",
            va="bottom")

    out = FIGURES_DIR / "fig1_ea_por_nivel.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_tipos_error(error_analysis: dict) -> Path:
    by_cat = error_analysis["by_category"]
    total  = error_analysis["total_mismatches"]
    n_q    = error_analysis.get("total_queries", 1034)
    ea_pct = error_analysis.get("ea_pct", round((n_q - total) / n_q * 100, 2))

    d_infra = by_cat.get("D", 0)
    ea_adj  = round((n_q - total + d_infra) / n_q * 100, 2) if n_q else 0

    order = sorted(
        [c for c in "ABCDE" if by_cat.get(c, 0) > 0],
        key=lambda c: -by_cat.get(c, 0)
    )
    for c in "ABCDE":
        if c not in order:
            order.append(c)

    labels = [ERROR_LABELS[c] for c in order]
    counts = [by_cat.get(c, 0) for c in order]
    colors = [ERROR_COLORS[c] for c in order]
    pcts   = [n / total * 100 if total else 0 for n in counts]

    y   = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(y, counts, color=colors, alpha=0.88, height=0.6, zorder=3)

    for i, (bar, n, pct, cat) in enumerate(zip(bars, counts, pcts, order)):
        w = bar.get_width()
        if n > 0:
            ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{n}  ({pct:.1f}%)",
                    va="center", ha="left", fontsize=10.5, fontweight="bold",
                    color=colors[i])
        else:
            ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                    "0  (no observado)",
                    va="center", ha="left", fontsize=10.5, color="#9E9E9E")

        if n > 0:
            ax.text(w / 2, bar.get_y() + bar.get_height()/2,
                    cat, va="center", ha="center",
                    fontsize=11, fontweight="bold", color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Número de errores")
    ax.set_xlim(0, max(counts) + max(counts) * 0.25 if counts else 10)

    ax.set_title(
        f"Distribución de tipos de error — {total} errores (EA = {ea_pct:.2f}%)",
        fontsize=14, fontweight="bold", pad=12
    )

    legend_handles = [
        mpatches.Patch(color=ERROR_COLORS["A"], label="A — Selección de columnas (error del agente)"),
        mpatches.Patch(color=ERROR_COLORS["D"], label="D — Error de tipo/conversión (limitación infraestructura)"),
        mpatches.Patch(color=ERROR_COLORS["C"], label="C — Semántica de agregación (error del agente)"),
        mpatches.Patch(color=ERROR_COLORS["E"], label="E — Error de esquema/JOIN"),
        mpatches.Patch(color=ERROR_COLORS["B"], label="B — Ordenación/LIMIT (no observado)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.9, title="Categoría de error",
              title_fontsize=9.5)

    fig.text(0.01, 0.01,
             f"Nota: los {d_infra} errores Tipo D (car_1, wta_1) son limitación de la "
             f"evaluación (TEXT fallback SQLite→PG),\n"
             f"no errores del agente. EA ajustada sin Tipo D: {ea_adj:.2f}%",
             fontsize=8.5, color="#757575", va="bottom")

    fig.subplots_adjust(bottom=0.14)

    out = FIGURES_DIR / "fig2_tipos_error.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_intentos(full_results: list) -> Path:
    counts = defaultdict(Counter)
    for rec in full_results:
        att  = rec.get("attempts", 1)
        note = rec.get("ea_note", "unknown")
        if note in ("match", "mismatch"):
            counts[att][note] += 1

    att_values      = sorted(counts.keys())
    match_counts    = [counts[a]["match"]    for a in att_values]
    mismatch_counts = [counts[a]["mismatch"] for a in att_values]
    totals          = [counts[a]["match"] + counts[a]["mismatch"] for a in att_values]

    x     = np.arange(len(att_values))
    width = 0.55

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_match    = ax.bar(x, match_counts,    width,
                           label="Acierto (correcto)", color=COLOR_MATCH, alpha=0.90, zorder=3)
    bars_mismatch = ax.bar(x, mismatch_counts, width,
                           bottom=match_counts,
                           label="Error (incorrecto)", color=COLOR_MISMATCH, alpha=0.88, zorder=3)

    for i, (n_m, n_mm, total_q) in enumerate(zip(match_counts, mismatch_counts, totals)):
        if n_m > 0:
            pct = n_m / total_q * 100
            ax.text(i, n_m / 2,
                    f"{n_m}\n({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        if n_mm > 0:
            pct = n_mm / total_q * 100
            ax.text(i, n_m + n_mm / 2,
                    f"{n_mm}\n({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        ax.text(i, total_q + 0.8, f"n={total_q}",
                ha="center", va="bottom", fontsize=10, color="#424242")

    for i, (n_m, total_q) in enumerate(zip(match_counts, totals)):
        ea = n_m / total_q * 100 if total_q else 0
        ax.text(i + width/2 + 0.05, n_m - 1,
                f"EA={ea:.0f}%",
                ha="left", va="top", fontsize=9, color=COLOR_MATCH,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Intento {a}\n({'primer intento' if a==1 else 'retry activado'})"
         for a in att_values],
        fontsize=11
    )
    ax.set_ylabel("Número de queries")
    ax.set_ylim(0, max(totals) + max(totals) * 0.12 if totals else 10)
    ax.set_title(
        "Distribución de resultados por número de intentos\n"
        "(Agente con retry loop, max_attempts=3)",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.legend(loc="upper right", framealpha=0.9)

    if len(att_values) >= 2:
        ax.annotate(
            "El retry loop activa\ncuando hay error\nde ejecución",
            xy=(1, match_counts[1] / 2 if len(match_counts) > 1 else 1),
            xytext=(1.35, (match_counts[1] + mismatch_counts[1]) * 0.8 if len(match_counts) > 1 else 2),
            fontsize=9, color="#424242",
            arrowprops=dict(arrowstyle="->", color="#757575", lw=1.2),
            ha="center"
        )

    avg_att = sum(r.get("attempts", 1) for r in full_results) / len(full_results) if full_results else 0
    n_with_retry = sum(totals[1:])
    fig.text(0.01, 0.01,
             f"Media de intentos: {avg_att:.2f}  |  "
             f"Queries con retry activado: {n_with_retry}  |  "
             f"Queries sin retry: {totals[0] if totals else 0}",
             fontsize=8.5, color="#757575", va="bottom")

    fig.subplots_adjust(bottom=0.12)

    out = FIGURES_DIR / "fig3_intentos.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def load_all():
    missing = []
    for p in [COMPARISON_JSON, ERROR_JSON, FULL_RESULTS]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        print(f"  Archivos no encontrados: {missing}")
        print("    Ejecuta los pasos anteriores en orden:")
        print("      1. python paso6_analyze_errors.py")
        print("      2. python paso8_compare.py")
        sys.exit(1)

    with open(COMPARISON_JSON, encoding="utf-8") as f:
        comparison = json.load(f)
    with open(ERROR_JSON, encoding="utf-8") as f:
        error_analysis = json.load(f)
    with open(FULL_RESULTS, encoding="utf-8") as f:
        full_results = json.load(f)

    print(f"  {COMPARISON_JSON.name}      cargado")
    print(f"  {ERROR_JSON.name}      cargado")
    print(f"  {FULL_RESULTS.name}          cargado ({len(full_results)} registros)")
    return comparison, error_analysis, full_results


def main():
    print("=" * 66)
    print("  SEMANA 14 — FASE 8.3: Visualizaciones (paso9_visualizations.py)")
    print("=" * 66)
    print()

    setup_style()
    comparison, error_analysis, full_results = load_all()
    print()

    print("⚙️   Generando Fig. 1 — EA por nivel (agente vs baseline)...")
    out1 = fig1_ea_por_nivel(comparison)
    print(f"  {out1}")

    print("⚙️   Generando Fig. 2 — Distribución de tipos de error...")
    out2 = fig2_tipos_error(error_analysis)
    print(f"  {out2}")

    print("⚙️   Generando Fig. 3 — Distribución de intentos...")
    out3 = fig3_intentos(full_results)
    print(f"  {out3}")

    print()
    print("=" * 66)
    print("  FIGURAS GENERADAS (n=1034, dev set completo)")
    print("=" * 66)
    print(f"""
  fig1_ea_por_nivel.png
    Grouped bar chart: EA del agente vs baseline por nivel de dificultad.
    Labels actualizados: easy/n=365, medium/n=405, hard/n=157, extra/n=107.
    → Sección 4.3 de la memoria (Resultados principales)

  fig2_tipos_error.png
    Horizontal bar chart: mismatches clasificados por tipo de error (A–E).
    Título y nota Tipo D calculados dinámicamente desde error_analysis_1034.json.
    → Sección 4.4 de la memoria (Análisis de errores)

  fig3_intentos.png
    Stacked bar chart: distribución de match/mismatch por número de intentos.
    Ilustra el mecanismo del retry loop y su efectividad diferencial.
    → Sección 4.5 de la memoria (Contribución del retry loop)

  Directorio: {FIGURES_DIR}
""")
    print("  → Siguiente: Actualizar capitulo4_evaluacion.md (Fase 9)\n")


if __name__ == "__main__":
    main()
