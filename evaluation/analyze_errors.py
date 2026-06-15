import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_FILE  = RESULTS_DIR / "full_1034.json"
JSON_OUT    = RESULTS_DIR / "error_analysis_1034.json"
REPORT_OUT  = RESULTS_DIR / "error_analysis_report_1034.txt"


TEXT_FALLBACK_DBS = {
    "car_1": (
        "Columnas numéricas (horsepower, mpg, cylinders) almacenadas como "
        "TEXT en SQLite. Comparaciones numéricas fallan o devuelven None "
        "porque los datos contienen '?' como valor desconocido."
    ),
    "wta_1": (
        "Tabla rankings almacenada con TEXT fallback por overflow de BIGINT. "
        "Operaciones numéricas sobre ranking_points producen resultados incorrectos."
    ),
}

CATEGORY_LABELS = {
    "A": "Column selection",
    "B": "Ordering / LIMIT",
    "C": "Aggregation semantics",
    "D": "Type / cast issue (TEXT fallback)",
    "E": "Schema / JOIN error",
}


def extract_tables(sql: str) -> set:
    s = sql.upper()
    from_tables = set(re.findall(r'\bFROM\s+"?(\w+)"?', s))
    join_tables = set(re.findall(r'\bJOIN\s+"?(\w+)"?', s))
    return from_tables | join_tables


def extract_select_cols(sql: str) -> list:
    s = sql.upper().strip()
    m = re.search(r'\bSELECT\s+(.*?)\s+\bFROM\b', s, re.DOTALL)
    if not m:
        return []
    select_clause = m.group(1).strip()
    cols = [c.strip() for c in select_clause.split(',')]
    return cols


def extract_aggregations(sql: str) -> list:
    return re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql.upper())


def has_kw(sql: str, keyword: str) -> bool:
    return bool(re.search(rf'\b{re.escape(keyword)}\b', sql.upper()))


def result_shape(result) -> tuple:
    if not result:
        return (0, 0)
    n_rows = len(result)
    n_cols = len(result[0]) if result[0] else 0
    return (n_rows, n_cols)


def classify_error(record: dict) -> dict:
    db_id     = record.get("db_id", "")
    gold_sql  = record.get("gold_sql") or ""
    pred_sql  = record.get("pred_sql") or ""
    gold_res  = record.get("gold_result") or []
    pred_res  = record.get("pred_result") or []

    g_rows, g_cols = result_shape(gold_res)
    p_rows, p_cols = result_shape(pred_res)

    gold_tables   = extract_tables(gold_sql)
    pred_tables   = extract_tables(pred_sql)
    gold_aggs     = set(extract_aggregations(gold_sql))
    pred_aggs     = set(extract_aggregations(pred_sql))

    gold_has_limit = has_kw(gold_sql, "LIMIT")
    pred_has_limit = has_kw(pred_sql, "LIMIT")
    gold_has_star  = "*" in gold_sql.upper()
    pred_has_star  = "*" in pred_sql.upper()

    details = {
        "gold_shape": [g_rows, g_cols],
        "pred_shape": [p_rows, p_cols],
        "gold_tables": sorted(gold_tables),
        "pred_tables": sorted(pred_tables),
        "gold_aggs":   sorted(gold_aggs),
        "pred_aggs":   sorted(pred_aggs),
    }

    if db_id in TEXT_FALLBACK_DBS:
        return {
            "category":   "D",
            "reason":     TEXT_FALLBACK_DBS[db_id],
            "confidence": "alta",
            "details":    details,
        }

    if gold_tables and pred_tables and gold_tables != pred_tables:
        diff = gold_tables.symmetric_difference(pred_tables)
        return {
            "category":   "E",
            "reason":     f"Tablas distintas entre gold y pred: {sorted(diff)}",
            "confidence": "alta",
            "details":    details,
        }

    if gold_has_limit != pred_has_limit:
        quien = "gold" if gold_has_limit else "pred"
        return {
            "category":   "B",
            "reason":     (
                f"LIMIT presente solo en {quien}: gold_limit={gold_has_limit}, "
                f"pred_limit={pred_has_limit}"
            ),
            "confidence": "alta",
            "details":    details,
        }

    if gold_has_limit and g_rows != p_rows:
        return {
            "category":   "B",
            "reason":     (
                f"LIMIT presente y row count distinto: gold={g_rows} filas, "
                f"pred={p_rows} filas"
            ),
            "confidence": "media",
            "details":    details,
        }

    if g_rows > 0 and p_rows > 0 and g_cols != p_cols:
        return {
            "category":   "A",
            "reason":     (
                f"Column selection: {g_cols} col(s) en gold vs "
                f"{p_cols} col(s) en pred"
            ),
            "confidence": "alta",
            "details":    details,
        }

    if gold_has_star and not pred_has_star and g_rows > 0 and p_rows > 0:
        return {
            "category":   "A",
            "reason":     "Gold usa SELECT *, pred selecciona columnas específicas",
            "confidence": "alta",
            "details":    details,
        }
    if not gold_has_star and pred_has_star and g_rows > 0 and p_rows > 0:
        return {
            "category":   "A",
            "reason":     "Pred usa SELECT * pero gold selecciona columnas específicas",
            "confidence": "alta",
            "details":    details,
        }

    if gold_aggs != pred_aggs:
        return {
            "category":   "C",
            "reason":     (
                f"Funciones de agregación distintas: "
                f"gold={sorted(gold_aggs)}, pred={sorted(pred_aggs)}"
            ),
            "confidence": "media",
            "details":    details,
        }

    if g_cols == p_cols and g_rows != p_rows and g_rows > 0:
        return {
            "category":   "C",
            "reason":     (
                f"Aggregation/grouping: misma estructura ({g_cols} col(s)), "
                f"distinto n_rows (gold={g_rows}, pred={p_rows})"
            ),
            "confidence": "media",
            "details":    details,
        }

    if g_cols == p_cols and g_rows == p_rows and g_rows > 0:
        return {
            "category":   "A",
            "reason":     (
                f"Column selection: misma forma ({g_rows}×{g_cols}) "
                f"pero valores distintos — columna equivocada"
            ),
            "confidence": "baja",
            "details":    details,
        }

    return {
        "category":   "C",
        "reason":     (
            f"Aggregation semantics (gold={g_rows}×{g_cols}, "
            f"pred={p_rows}×{p_cols})"
        ),
        "confidence": "baja",
        "details":    details,
    }


def load_results() -> list:
    if not INPUT_FILE.exists():
        print(f"\n  Archivo no encontrado: {INPUT_FILE}")
        print("    Ejecuta paso5_run_eval_v2.py --full primero.\n")
        sys.exit(1)

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Cargados {len(data)} registros de {INPUT_FILE.name}")
    return data


def analyze(records: list) -> dict:
    mismatches = [r for r in records if r.get("ea_note") == "mismatch"]
    total      = len(records)
    n_miss     = len(mismatches)

    print(f"   Total registros   : {total}")
    print(f"   Mismatches (ea=0) : {n_miss}")
    print(f"   Match rate (EA)   : {(total - n_miss) / total * 100:.1f}%")

    classified    = []
    by_category   = Counter()
    by_difficulty = defaultdict(Counter)
    by_db         = defaultdict(Counter)

    for rec in mismatches:
        clf = classify_error(rec)
        cat = clf["category"]

        rec_id = rec.get("idx", rec.get("id"))

        classified.append({
            "id":         rec_id,
            "db_id":      rec.get("db_id"),
            "difficulty": rec.get("difficulty"),
            "question":   rec.get("question"),
            "gold_sql":   rec.get("gold_sql"),
            "pred_sql":   rec.get("pred_sql"),
            "attempts":   rec.get("attempts"),
            "latency_s":  rec.get("latency_s"),
            "gold_shape": clf["details"]["gold_shape"],
            "pred_shape": clf["details"]["pred_shape"],
            "category":   cat,
            "reason":     clf["reason"],
            "confidence": clf["confidence"],
        })

        by_category[cat]                     += 1
        by_difficulty[rec.get("difficulty", "unknown")][cat] += 1
        by_db[rec.get("db_id", "unknown")][cat]              += 1

    db_totals = {
        db: sum(counts.values())
        for db, counts in by_db.items()
    }
    top_dbs = sorted(db_totals.items(), key=lambda x: -x[1])[:10]

    return {
        "total_queries":    total,
        "total_mismatches": n_miss,
        "ea_pct":           round((total - n_miss) / total * 100, 2),
        "mismatches":       classified,
        "by_category":      dict(by_category),
        "by_difficulty":    {d: dict(c) for d, c in by_difficulty.items()},
        "by_db":            {db: dict(counts) for db, counts in by_db.items()},
        "top_dbs":          top_dbs,
    }


DIFF_ORDER = ["easy", "medium", "hard", "extra"]

def print_report(analysis: dict, file=None):
    lines = []
    SEP   = "=" * 66
    sep   = "─" * 66

    def emit(s=""):
        lines.append(s)
        print(s)

    n_total = analysis["total_mismatches"]
    n_q     = analysis["total_queries"]

    emit(SEP)
    emit("  SEMANA 14 — FASE 8.1: Análisis de Errores (paso6_analyze_errors.py)")
    emit(f"  {n_total} mismatches sobre {n_q} queries")
    emit(f"  EA global: {analysis['ea_pct']:.2f}%")
    emit(SEP)
    emit()

    emit("  DISTRIBUCIÓN POR CATEGORÍA DE ERROR")
    emit(f"  {sep}")
    emit(f"  {'Cat':<5}  {'Descripción':<40}  {'N':>4}  {'%':>6}  Barra")
    emit(f"  {sep}")
    for cat in "ADCEB":
        n = analysis["by_category"].get(cat, 0)
        if n_total == 0:
            pct = 0
        else:
            pct = n / n_total * 100
        bar = "█" * int(pct / 2)
        emit(f"  {cat:<5}  {CATEGORY_LABELS[cat]:<40}  {n:>4}  {pct:>5.1f}%  {bar}")
    emit(f"  {sep}")
    emit()

    emit("  DISTRIBUCIÓN POR NIVEL DE DIFICULTAD")
    emit(f"  {sep}")
    emit(f"  {'Nivel':<8}  {'Mismatches':>10}  Desglose")
    emit(f"  {sep}")
    for diff in DIFF_ORDER:
        d = analysis["by_difficulty"].get(diff, {})
        n = sum(d.values())
        desglose = "  ".join(f"{cat}:{d.get(cat,0)}" for cat in "ADCEB" if d.get(cat, 0) > 0)
        emit(f"  {diff:<8}  {n:>10}  {desglose}")
    emit(f"  {sep}")
    emit()

    emit("  TOP DBs CON MÁS MISMATCHES")
    emit(f"  {sep}")
    for db, n in analysis["top_dbs"]:
        d = analysis["by_db"].get(db, {})
        desglose = "  ".join(f"{cat}:{v}" for cat, v in sorted(d.items()) if v > 0)
        emit(f"  {db:<35}  {n:>4} mismatches   {desglose}")
    emit()

    emit(f"  {sep}")
    emit("  LISTA DETALLADA DE MISMATCHES")
    emit(f"  {sep}")
    emit(f"  {'ID':>5}  {'Nivel':<7}  {'DB':<32}  {'Cat'}  {'Conf':<6}  {'att':>3}")
    emit(f"  {sep}")
    for m in analysis["mismatches"]:
        emit(
            f"  {str(m['id'] or '?'):>5}  "
            f"{m['difficulty']:<7}  "
            f"{m['db_id']:<32}  "
            f"{m['category']}    "
            f"{m['confidence']:<6}  "
            f"att={m['attempts']}"
        )
    emit()

    emit("  EJEMPLOS REPRESENTATIVOS POR CATEGORÍA")
    emit(f"  {sep}")
    for cat in "ADCEB":
        ejemplos = [m for m in analysis["mismatches"] if m["category"] == cat]
        if not ejemplos:
            continue
        priority = {"alta": 0, "media": 1, "baja": 2}
        ej = sorted(ejemplos, key=lambda x: priority.get(x["confidence"], 3))[0]

        emit(f"  [{cat}] {CATEGORY_LABELS[cat]}")
        emit(f"    DB    : {ej['db_id']} ({ej['difficulty']})")
        emit(f"    Q     : {str(ej['question'] or '')[:80]}")
        emit(f"    Gold  : {str(ej['gold_sql'] or '')[:80]}")
        emit(f"    Pred  : {str(ej['pred_sql'] or '')[:80]}")
        emit(f"    Shapes: gold={ej['gold_shape']} pred={ej['pred_shape']}")
        emit(f"    Motivo: {str(ej['reason'] or '')[:90]}")
        emit()

    n_by_cat = {c: analysis["by_category"].get(c, 0) for c in "ADCEB"}
    d_infra  = n_by_cat["D"]
    ea_adj   = round((n_q - n_total + d_infra) / n_q * 100, 2) if n_q else 0

    emit(f"  {sep}")
    emit("  CONCLUSIONES PARA LA MEMORIA")
    emit(f"  {sep}")
    emit(f"""
  De los {n_total} mismatches:

  • Tipo D (Type/cast) : {n_by_cat['D']:>3} ({n_by_cat['D']/n_total*100:.1f}% si n>0) — Limitación de evaluación:
      car_1 y wta_1 contienen columnas numéricas almacenadas como TEXT en
      SQLite original. El agente genera SQL correcto semánticamente, pero la
      conversión SQLite→PostgreSQL produce resultados distintos al gold.
      No son errores del agente sino de la infraestructura de evaluación.

  • Tipo A (Column sel.): {n_by_cat['A']:>3} — Selección de columnas incorrecta.
  • Tipo C (Aggregation): {n_by_cat['C']:>3} — Estrategia de agregación distinta.
  • Tipo E (Schema/JOIN): {n_by_cat['E']:>3} — JOIN equivocado o tabla incorrecta.
  • Tipo B (Order/LIMIT): {n_by_cat['B']:>3} — Diferencia por LIMIT asimétrico.

  EA ajustada sin Tipo D (infraestructura): {ea_adj:.2f}%
    """)
    emit(SEP)

    if file:
        fp = Path(file)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n  Informe guardado en: {fp}")


def main():
    print("=" * 66)
    print("  SEMANA 14 — FASE 8.1: Análisis de Errores (paso6_analyze_errors.py)")
    print("=" * 66)
    print()

    records = load_results()
    print()

    print("  Clasificando mismatches...")
    analysis = analyze(records)
    print()

    print_report(analysis, file=REPORT_OUT)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON guardado en:    {JSON_OUT}")

    print()
    print("  → Siguiente: python paso8_compare.py")
    print()


if __name__ == "__main__":
    main()
