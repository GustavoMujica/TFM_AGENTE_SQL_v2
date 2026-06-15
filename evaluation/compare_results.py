import json
from collections import defaultdict
from pathlib import Path


BASE_DIR      = Path(__file__).parent
RESULTS_DIR   = BASE_DIR / "results"
AGENT_JSON    = RESULTS_DIR / "full_1034.json"
BASELINE_JSON = RESULTS_DIR / "baseline_1034.json"
OUT_JSON      = RESULTS_DIR / "comparison_1034_summary.json"
OUT_REPORT    = RESULTS_DIR / "comparison_1034_report.txt"

ERROR_ANALYSIS_JSON = RESULTS_DIR / "error_analysis_1034.json"

DIFF_ORDER = ["easy", "medium", "hard", "extra"]

SUBSET_TOTALS = {"easy": 365, "medium": 405, "hard": 157, "extra": 107}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    for r in records:
        if "id" not in r and "idx" in r:
            r["id"] = r["idx"]

    return {r["id"]: r for r in records}


def classify_pair(agent_rec: dict, base_rec: dict) -> str:
    a_note = agent_rec.get("ea_note", "")
    b_note = base_rec.get("ea_note", "")

    a_ok = (a_note == "match")
    b_ok = (b_note == "match")

    if a_ok and b_ok:
        return "A"
    if a_ok and b_note == "mismatch":
        return "B"
    if a_ok and b_note == "pred_failed":
        return "C"
    if not a_ok and b_ok:
        return "D"
    if a_note == "mismatch" and b_note == "mismatch":
        return "E"
    if a_note == "mismatch" and b_note == "pred_failed":
        return "F"
    if a_note == "pred_failed":
        return "E"
    return "?"


def analyze(agent: dict, baseline: dict) -> dict:
    cells   = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": [], "?": []}
    by_diff = {d: {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
               for d in DIFF_ORDER}
    by_db   = defaultdict(lambda: {"A": 0, "B": 0, "C": 0,
                                   "D": 0, "E": 0, "F": 0})

    common_ids = set(agent.keys()) & set(baseline.keys())

    for qid in sorted(common_ids):
        a_rec = agent[qid]
        b_rec = baseline[qid]
        cell  = classify_pair(a_rec, b_rec)
        diff  = a_rec.get("difficulty", "unknown")
        db    = a_rec.get("db_id", "unknown")

        entry = {
            "id":         qid,
            "db_id":      db,
            "difficulty": diff,
            "question":   a_rec.get("question", ""),
            "agent_note": a_rec.get("ea_note", ""),
            "base_note":  b_rec.get("ea_note", ""),
            "agent_sql":  a_rec.get("pred_sql", ""),
            "base_sql":   b_rec.get("pred_sql", ""),
            "gold_sql":   a_rec.get("gold_sql", ""),
            "agent_att":  a_rec.get("attempts", 0),
            "base_att":   b_rec.get("attempts", 0),
        }

        cells[cell].append(entry)
        if diff in by_diff:
            by_diff[diff][cell].append(entry)
        by_db[db][cell] += 1

    counts = {c: len(cells[c]) for c in "ABCDEF"}
    n_excluded = len(cells.get("?", []))

    agent_correct   = counts["A"] + counts["B"] + counts["C"]
    base_correct    = counts["A"] + counts["D"]
    total           = len(common_ids)
    net_improvement = counts["C"] - counts["D"]

    c_by_diff = {d: len(by_diff[d]["C"]) for d in DIFF_ORDER}
    f_by_diff = {d: len(by_diff[d]["F"]) for d in DIFF_ORDER}

    diff_totals = {
        d: sum(1 for qid in common_ids if agent[qid].get("difficulty") == d)
        for d in DIFF_ORDER
    }

    return {
        "total":           total,
        "agent_correct":   agent_correct,
        "base_correct":    base_correct,
        "agent_ea_pct":    round(agent_correct / total * 100, 2),
        "base_ea_pct":     round(base_correct  / total * 100, 2),
        "delta_ea_pct":    round((agent_correct - base_correct) / total * 100, 2),
        "cells":           counts,
        "cells_detail":    {c: cells[c] for c in "ABCDEF"},
        "c_by_difficulty": c_by_diff,
        "f_by_difficulty": f_by_diff,
        "net_improvement": net_improvement,
        "n_excluded":      n_excluded,
        "by_difficulty":   {
            d: {
                "total":         diff_totals.get(d, 0),
                "agent_correct": sum(1 for qid in common_ids
                                     if agent[qid].get("difficulty") == d
                                     and classify_pair(agent[qid], baseline[qid]) in ("A","B","C")),
                "base_correct":  sum(1 for qid in common_ids
                                     if agent[qid].get("difficulty") == d
                                     and classify_pair(agent[qid], baseline[qid]) in ("A","D")),
                "cells": {c: len(by_diff[d][c]) for c in "ABCDEF"},
            }
            for d in DIFF_ORDER
        },
        "by_db": dict(by_db),
    }


CELL_DESC = {
    "A": "Ambos correctos (baseline suficiente)",
    "B": "Agente correcto, baseline mismatch (estocástico)",
    "C": "Agente correcto, baseline pred_failed (RETRY RESCATÓ)",
    "D": "Baseline correcto, agente mismatch (degradación estocástica)",
    "E": "Ambos incorrectos — fallo semántico, retry sin señal",
    "F": "Agente mismatch, baseline pred_failed (retry parcial, car_1/wta_1)",
}


def _load_d_infra() -> int:
    if not ERROR_ANALYSIS_JSON.exists():
        return 0
    try:
        with open(ERROR_ANALYSIS_JSON, encoding="utf-8") as f:
            ea = json.load(f)
        return ea.get("by_category", {}).get("D", 0)
    except Exception:
        return 0


def print_report(analysis: dict, file=None):
    lines = []
    SEP   = "=" * 68
    sep   = "─" * 68

    def emit(s=""):
        lines.append(s)
        print(s)

    total = analysis["total"]
    c     = analysis["cells"]

    emit(SEP)
    emit("  SEMANA 14 — FASE 8.2: Comparación Agente vs Baseline")
    emit(SEP)

    emit()
    emit(f"  EA Agente   (max_attempts=3): {analysis['agent_ea_pct']:>6.2f}%"
         f"  ({analysis['agent_correct']}/{total})")
    emit(f"  EA Baseline (max_attempts=1): {analysis['base_ea_pct']:>6.2f}%"
         f"  ({analysis['base_correct']}/{total})")
    emit(f"  Delta EA:                     {analysis['delta_ea_pct']:>+6.2f}%")
    emit()

    emit(f"  {sep}")
    emit("  MATRIZ DE TRANSICIÓN (agente × baseline)")
    emit(f"  {sep}")
    emit()
    emit("                         B A S E L I N E")
    emit("                    match    mismatch  pred_failed")
    emit("  ┌─────────────┬──────────┬──────────┬──────────┐")
    emit(f"  │   match     │  A={c['A']:>4}  │  B={c['B']:>4}  │  C={c['C']:>4}  │")
    emit("  │ A G E N T E ├──────────┼──────────┼──────────┤")
    emit(f"  │  mismatch   │  D={c['D']:>4}  │  E={c['E']:>4}  │  F={c['F']:>4}  │")
    emit("  └─────────────┴──────────┴──────────┴──────────┘")
    emit()

    emit(f"  {sep}")
    emit("  INTERPRETACIÓN DE CADA CELDA")
    emit(f"  {sep}")
    for cell, n in sorted(c.items()):
        pct  = n / total * 100
        desc = CELL_DESC.get(cell, "")
        emit(f"  {cell} ({n:>4} = {pct:5.1f}%)  {desc}")
    emit()

    emit(f"  {sep}")
    emit("  APORTE NETO DEL RETRY LOOP")
    emit(f"  {sep}")
    emit(f"""
  Queries rescatadas bruto (C):      {c['C']:>4}  (pred_failed → match)
  Queries con degradación (D):      -{c['D']:>3}  (match → mismatch, estocástico)
  ─────────────────────────────────────────────
  Mejora NETA:                      +{analysis['net_improvement']:>3}  ({analysis['net_improvement']/total*100:.2f}% del total)

  Reducción pred_failed por retry:  {c['C'] + c['F']:>4}  (pred_failed → ejecutadas)
    de las cuales: {c['C']} correctas (celda C) + {c['F']} mismatch (celda F)

  El +{analysis['delta_ea_pct']:.2f}% de EA se descompone en:
    +{c['C']} queries rescatadas por retry   → +{c['C']/total*100:.2f}%
    -{c['D']} query con degradación estocást → -{c['D']/total*100:.2f}%
    Resultado neto                           → +{analysis['net_improvement']/total*100:.2f}%
""")

    emit(f"  {sep}")
    emit(f"  QUERIES RESCATADAS POR EL RETRY LOOP (C = {c['C']})")
    emit(f"  {sep}")
    emit(f"  {'ID':>5}  {'Nivel':<7}  {'DB':<30}  {'att agente':>10}")
    emit(f"  {sep[:57]}")
    for entry in analysis["cells_detail"]["C"]:
        emit(f"  {str(entry['id']):>5}  {entry['difficulty']:<7}  "
             f"{entry['db_id']:<30}  att={entry['agent_att']}")
    emit()

    emit("  Rescatadas por nivel:")
    by_diff = analysis["by_difficulty"]
    for d in DIFF_ORDER:
        n_c = analysis["c_by_difficulty"][d]
        n_total_d = by_diff[d]["total"]
        if n_c > 0 and n_total_d > 0:
            emit(f"    {d:<8}: {n_c} queries  ({n_c/n_total_d*100:.1f}% del nivel)")
    emit()

    emit(f"  {sep}")
    emit(f"  DEGRADACIÓN ESTOCÁSTICA (D = {c['D']})")
    emit(f"  {sep}")
    for entry in analysis["cells_detail"]["D"]:
        emit(f"  ID={entry['id']}  {entry['db_id']} ({entry['difficulty']})")
        emit(f"  Q: {str(entry['question'])[:80]}")
        emit(f"  Gold:     {str(entry['gold_sql'] or '')[:75]}")
        emit(f"  Baseline: {str(entry['base_sql'] or '')[:75]}  ← CORRECTO")
        emit(f"  Agente:   {str(entry['agent_sql'] or '')[:75]}  ← INCORRECTO")
        emit(f"  Nota: el agente generó SQL distinto en su ejecución (variabilidad LLM).")
    if not analysis["cells_detail"]["D"]:
        emit("  (ninguna degradación estocástica observada)")
    emit()

    emit(f"  {sep}")
    emit(f"  RETRY PARCIAL — ejecución corregida, semántica incorrecta (F = {c['F']})")
    emit(f"  {sep}")
    emit(f"  {'ID':>5}  {'Nivel':<7}  {'DB':<30}  {'agent_att':>9}")
    emit(f"  {sep[:57]}")
    for entry in analysis["cells_detail"]["F"]:
        emit(f"  {str(entry['id']):>5}  {entry['difficulty']:<7}  "
             f"{entry['db_id']:<30}  att={entry['agent_att']}")
    emit()
    f_dbs = {e["db_id"] for e in analysis["cells_detail"]["F"]}
    if f_dbs:
        emit(f"  DBs afectadas: {sorted(f_dbs)}")
        emit("  Causa predominante: TEXT fallback — el retry corrige el error de")
        emit("  PostgreSQL pero la comparación SQLite vs PostgreSQL sigue fallando.")
    emit()

    emit(f"  {sep}")
    emit("  EA POR NIVEL — AGENTE vs BASELINE")
    emit(f"  {sep}")
    emit(f"  {'Nivel':<8}  {'Total':>6}  {'Agente':>8}  {'Baseline':>9}  {'Delta':>7}  Celda C")
    emit(f"  {sep[:65]}")
    for d in DIFF_ORDER:
        dd    = analysis["by_difficulty"][d]
        n     = dd["total"]
        a_ea  = round(dd["agent_correct"] / n * 100, 1) if n else 0
        b_ea  = round(dd["base_correct"]  / n * 100, 1) if n else 0
        delta = round(a_ea - b_ea, 1)
        n_c   = analysis["c_by_difficulty"][d]
        emit(f"  {d:<8}  {n:>6}  {a_ea:>7.1f}%  {b_ea:>8.1f}%  {delta:>+6.1f}%  C={n_c}")
    emit(f"  {sep[:65]}")
    a_total = analysis["agent_correct"]
    b_total = analysis["base_correct"]
    emit(f"  {'TOTAL':<8}  {total:>6}  {a_total/total*100:>7.1f}%  "
         f"{b_total/total*100:>8.1f}%  {(a_total-b_total)/total*100:>+6.1f}%  C={c['C']}")
    emit()

    D_INFRA = _load_d_infra()
    if D_INFRA > 0:
        adj_a = round((analysis["agent_correct"] + D_INFRA) / total * 100, 2)
        adj_b = round((analysis["base_correct"]  + D_INFRA) / total * 100, 2)
        emit(f"  {sep}")
        emit(f"  EA AJUSTADA (excluyendo {D_INFRA} mismatches Tipo D — infraestructura)")
        emit(f"  {sep}")
        emit(f"  Agente   ajustado: {adj_a:.2f}%")
        emit(f"  Baseline ajustado: {adj_b:.2f}%")
        emit(f"  Delta    ajustado: {adj_a - adj_b:+.2f}%  (invariante al ajuste)")
        emit()
    else:
        emit(f"  {sep}")
        emit("  EA ajustada no disponible: ejecuta paso6_analyze_errors.py primero")
        emit(f"      (para cargar D_INFRA desde {ERROR_ANALYSIS_JSON.name})")
        emit()

    emit(f"  {sep}")
    emit("  CONCLUSIONES PARA LA MEMORIA (sección 4.5)")
    emit(f"  {sep}")
    emit(f"""
  1. APORTE DEL RETRY LOOP: +{analysis['delta_ea_pct']:.2f}% EA neto ({analysis['net_improvement']} queries).
     El mecanismo es: pred_failed (error de PostgreSQL) → retry inyecta
     el error en el prompt → LLM corrige la sintaxis o referencia de tabla.

  2. APORTE DUAL:
     - EA:           +{analysis['delta_ea_pct']:.2f} pp   (C={c['C']} queries rescatadas, D={c['D']} degradaciones)
     - Robustez:     −{round(c['C']+c['F'],0)} pred_failed eliminados ({round((c['C']+c['F'])/(c['C']+c['F']+len(analysis['cells_detail']['E'])+c['D'])*100 if (c['C']+c['F'])>0 else 0,0):.0f}% reducción aprox.)

  3. IMPACTO ASIMÉTRICO POR DIFICULTAD:
     - easy  (C={analysis['c_by_difficulty']['easy']}):  retry sin efecto. Easy queries fallan
       semánticamente (mismatch), no con errores de ejecución.
     - hard  (C={analysis['c_by_difficulty']['hard']}):  máximo beneficio relativo. JOINs complejos
       frecuentemente producen errores de ejecución que el retry puede corregir.
     - extra (C={analysis['c_by_difficulty']['extra']}):  beneficio moderado.

  4. LIMITACIÓN DEL RETRY LOOP (celda E = {c['E']}):
     El retry loop es CIEGO ante errores semánticos. Las {c['E']} queries de
     la celda E fallan en ambas configuraciones (SQL ejecuta pero resultado
     incorrecto). Sin señal de error, el retry no puede activarse.

  5. RETRY PARCIAL (celda F = {c['F']}):
     {c['F']} queries donde el retry corrige el error de ejecución pero el
     resultado sigue siendo incorrecto (TEXT fallback, car_1/wta_1).
""")
    emit(SEP)

    if file:
        fp = Path(file)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(lines))
        print(f"\n  Informe guardado en: {fp}")


def main():
    print("=" * 68)
    print("  SEMANA 14 — FASE 8.2: Comparación Agente vs Baseline")
    print("=" * 68)
    print()

    for p in [AGENT_JSON, BASELINE_JSON]:
        if not p.exists():
            print(f"  Archivo no encontrado: {p}")
            if "full_1034" in str(p):
                print("    Ejecuta paso5_run_eval_v2.py --full primero.")
            else:
                print("    Ejecuta paso7_baseline_v2.py --full primero.")
            return

    print(f"  Cargando {AGENT_JSON.name}    ...", end="  ")
    agent = load_json(AGENT_JSON)
    print(f"{len(agent)} registros")

    print(f"  Cargando {BASELINE_JSON.name} ...", end="  ")
    baseline = load_json(BASELINE_JSON)
    print(f"{len(baseline)} registros")
    print()

    print("⚙️   Calculando matriz de transición...")
    analysis = analyze(agent, baseline)
    print()

    print_report(analysis, file=OUT_REPORT)

    summary_out = {k: v for k, v in analysis.items() if k != "cells_detail"}
    summary_out["rescued_ids"]  = [e["id"] for e in analysis["cells_detail"]["C"]]
    summary_out["degraded_ids"] = [e["id"] for e in analysis["cells_detail"]["D"]]
    summary_out["partial_ids"]  = [e["id"] for e in analysis["cells_detail"]["F"]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print(f"  JSON guardado en:    {OUT_JSON}")

    print()
    print("  → Siguiente: python paso9_visualizations.py")
    print()


if __name__ == "__main__":
    main()
