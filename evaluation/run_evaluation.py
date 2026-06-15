import os
import signal
import sys
import json
import time
import sqlite3
import argparse
from pathlib import Path
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv()
os.environ["DB_NAME"] = "spider_eval"
os.environ["DB_HOST"] = "localhost"

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.agent_v3_cached import build_agent, schema_cache
import anthropic


EVAL_DIR      = Path(__file__).parent
SPIDER_DIR    = EVAL_DIR / "spider"
RESULTS_DIR   = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_PREFIX      = "full_1034"
CHECKPOINT_PATH = RESULTS_DIR / f"{OUT_PREFIX}_partial.json"
FINAL_PATH      = RESULTS_DIR / f"{OUT_PREFIX}.json"
SUMMARY_PATH    = RESULTS_DIR / f"{OUT_PREFIX}_summary.json"
DEV_JSON        = SPIDER_DIR / "dev_enriched.json"
SQLITE_DB_DIR   = SPIDER_DIR / "database"


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            f = float(obj)
            return int(f) if f == int(f) else round(f, 6)
        return super().default(obj)


def normalize_value(v):
    if v is None:
        return None

    if isinstance(v, bool):
        return int(v)

    if isinstance(v, Decimal):
        try:
            f = float(v)
            return int(f) if f == int(f) else round(f, 6)
        except (ValueError, OverflowError):
            return str(v).strip()

    if isinstance(v, (int, float)):
        try:
            f = float(v)
            return int(f) if f == int(f) else round(f, 6)
        except (ValueError, OverflowError):
            return str(v).strip()

    if isinstance(v, str):
        v_stripped = v.strip()
        if v_stripped.lower() in ("none", "null", ""):
            return None
        try:
            f = float(v_stripped)
            return int(f) if f == int(f) else round(f, 6)
        except ValueError:
            return v_stripped

    return str(v).strip()


def normalize_resultset(results) -> list | None:
    if results is None:
        return None

    normalized = []
    for row in results:
        if isinstance(row, (list, tuple)):
            norm_row = tuple(normalize_value(v) for v in row)
        else:
            norm_row = (normalize_value(row),)
        normalized.append(norm_row)

    try:
        normalized.sort(key=lambda r: [str(v) for v in r])
    except TypeError:
        normalized.sort(key=str)

    return normalized


def compare_results(r_gold, r_pred) -> bool:
    return normalize_resultset(r_gold) == normalize_resultset(r_pred)


def run_gold_sqlite(gold_sql: str, db_id: str) -> tuple[list | None, str | None]:
    sqlite_path = SQLITE_DB_DIR / db_id / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        return None, f"SQLite no encontrado: {sqlite_path}"
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        cur.execute(gold_sql)
        rows = cur.fetchall()
        conn.close()
        return list(rows), None
    except Exception as e:
        return None, str(e)


_agent = build_agent()


def run_agent_spider(question: str, db_id: str) -> dict:
    return _agent.invoke({
        "question":      question,
        "db_id":         db_id,
        "schema":        None,
        "sql_generated": None,
        "is_valid":      None,
        "results":       None,
        "error":         None,
        "answer":        None,
        "attempts":      0,
    })


def run_agent_with_backoff(question: str, db_id: str, max_retries: int = 4) -> dict:
    for intento in range(max_retries):
        try:
            return run_agent_spider(question, db_id)
        except anthropic.RateLimitError:
            wait = 60 * (2 ** intento)
            print(f"\n  Rate limit (RateLimitError). Esperando {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code == 429:
                wait = 60 * (2 ** intento)
                print(f"\n  HTTP 429 (APIStatusError). Esperando {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Rate limit persistente tras {max_retries} reintentos. "
        "Revisa tu cuota en console.anthropic.com."
    )


def load_checkpoint() -> tuple[list, set]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            records = json.load(f)
        done = {r["idx"] for r in records}
        print(f"  Checkpoint cargado: {len(done)} queries ya evaluadas. Reanudando...")
        return records, done
    return [], set()


def save_checkpoint(records: list):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, cls=SafeEncoder)


def evaluate_one(agent_result: dict, example: dict, idx: int) -> dict:
    db_id    = example["db_id"]
    gold_sql = example["gold_sql"]
    diff     = example["difficulty"]

    pred_sql     = agent_result.get("sql_generated")
    pred_results = agent_result.get("results")
    attempts     = agent_result.get("attempts", 0)
    latency      = agent_result.get("_latency", 0.0)

    gold_rows, gold_error = run_gold_sqlite(gold_sql, db_id)

    base = {
        "idx":        idx,
        "db_id":      db_id,
        "difficulty": diff,
        "question":   example["question"],
        "gold_sql":   gold_sql,
        "pred_sql":   pred_sql,
        "attempts":   attempts,
        "latency_s":  latency,
    }

    if gold_error:
        return {
            **base,
            "ea":          None,
            "ea_note":     "gold_error",
            "gold_result": None,
            "pred_result": pred_results,
            "gold_error":  gold_error,
        }

    if pred_results is None:
        return {
            **base,
            "ea":          0,
            "ea_note":     "pred_failed",
            "gold_result": [list(r) for r in gold_rows],
            "pred_result": None,
            "gold_error":  None,
        }

    match = compare_results(gold_rows, pred_results)
    return {
        **base,
        "ea":          1 if match else 0,
        "ea_note":     "match" if match else "mismatch",
        "gold_result": [list(r) for r in gold_rows],
        "pred_result": pred_results,
        "gold_shape":  [len(gold_rows), len(gold_rows[0]) if gold_rows else 0],
        "pred_shape":  [len(pred_results), len(pred_results[0]) if pred_results else 0],
        "gold_error":  None,
    }


def compute_metrics(records: list) -> dict:
    by_diff: dict[str, dict] = {}
    for r in records:
        d = r["difficulty"]
        if d not in by_diff:
            by_diff[d] = {
                "total": 0, "correct": 0, "failed": 0,
                "latencies": [], "attempts": [],
            }
        by_diff[d]["total"] += 1
        if r["ea"] == 1:
            by_diff[d]["correct"] += 1
        if r.get("ea_note") == "pred_failed":
            by_diff[d]["failed"] += 1
        by_diff[d]["latencies"].append(r["latency_s"])
        by_diff[d]["attempts"].append(r["attempts"])

    evaluable = [r for r in records if r["ea"] is not None]
    correct   = sum(1 for r in evaluable if r["ea"] == 1)
    n         = len(evaluable)

    return {
        "ea_global":    round(correct / n * 100, 2) if n else 0.0,
        "correct":      correct,
        "total":        n,
        "pred_failed":  sum(1 for r in records if r.get("ea_note") == "pred_failed"),
        "gold_errors":  sum(1 for r in records if r.get("ea_note") == "gold_error"),
        "avg_latency":  round(
            sum(r["latency_s"] for r in records) / len(records), 2
        ) if records else 0.0,
        "avg_attempts": round(
            sum(r["attempts"] for r in records) / len(records), 2
        ) if records else 0.0,
        "by_difficulty": {
            d: {
                "ea":          round(v["correct"] / v["total"] * 100, 2),
                "correct":     v["correct"],
                "total":       v["total"],
                "pred_failed": v["failed"],
                "avg_latency": round(
                    sum(v["latencies"]) / len(v["latencies"]), 2
                ),
                "avg_attempts": round(
                    sum(v["attempts"]) / len(v["attempts"]), 2
                ),
            }
            for d, v in by_diff.items()
        },
    }


def _graceful_exit(records_ref: list):
    def handler(sig, frame):
        print("\n\n  Ctrl+C recibido — guardando estado parcial...")
        save_checkpoint(records_ref)
        if records_ref:
            metrics = compute_metrics(records_ref)
            elapsed = -1
            with open(FINAL_PATH, "w", encoding="utf-8") as f:
                json.dump(records_ref, f, indent=2,
                          ensure_ascii=False, cls=SafeEncoder)
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                json.dump({"elapsed_s": elapsed, "partial": True, **metrics},
                          f, indent=2)
            evaluable = [r for r in records_ref if r["ea"] is not None]
            ea = sum(r["ea"] for r in evaluable) / len(evaluable) * 100
            print(f"  EA parcial: {ea:.2f}%  ({len(records_ref)} queries guardadas)")
            print(f"  {FINAL_PATH}")
        sys.exit(0)
    return handler


def main():
    parser = argparse.ArgumentParser(
        description="Evaluacion Text-to-SQL sobre Spider dev set completo."
    )
    parser.add_argument(
        "--full",  action="store_true",
        help="Evaluacion completa (1034 queries, ~40-50 min, ~$4-5 USD)"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot mode: 2 queries por nivel + 2 de student_transcripts_tracking (~10 queries)"
    )
    args = parser.parse_args()

    if not DEV_JSON.exists():
        print(f"\n  No se encontro {DEV_JSON}")
        print("    Ejecuta primero: python evaluation/paso1_explore.py\n")
        sys.exit(1)

    with open(DEV_JSON, encoding="utf-8") as f:
        all_examples = json.load(f)

    if args.pilot:
        DB20 = "student_transcripts_tracking"
        pilot: list[dict] = []
        for diff in ["easy", "medium", "hard", "extra"]:
            bucket = [ex for ex in all_examples if ex["difficulty"] == diff]
            pilot.extend(bucket[:2])
        db20_examples = [ex for ex in all_examples if ex["db_id"] == DB20]
        pilot.extend(db20_examples[:2])
        all_examples = pilot
        print(f"\n  MODO PILOT: {len(all_examples)} queries")
        print(f"  (2 por nivel de dificultad + 2 de {DB20})\n")

    elif args.full:
        print(f"\n  MODO FULL: {len(all_examples)} queries")
        print(f"  Estimacion: ~40-50 min, ~$4-5 USD\n")

    else:
        print("\n  Especifica --pilot o --full")
        print("  Ejemplo: python evaluation/paso5_run_eval_v2.py --pilot\n")
        parser.print_help()
        return

    n = len(all_examples)
    records, done_indices = load_checkpoint()
    signal.signal(signal.SIGINT, _graceful_exit(records))
    t_total = time.time()

    print(f"  {'IDX':>4}  {'DIFF':<6}  {'DB_ID':<30}  PREGUNTA")
    print(f"  {'─'*4}  {'─'*6}  {'─'*30}  {'─'*50}")

    for idx, example in enumerate(all_examples):
        if idx in done_indices:
            continue

        db_id   = example["db_id"]
        diff    = example["difficulty"]
        q_short = example["question"][:52] + (
            "..." if len(example["question"]) > 52 else ""
        )
        print(f"  [{idx + 1:>4}/{n}]  [{diff:<6}]  {db_id:<30}  {q_short}")

        t0 = time.time()
        agent_result = run_agent_with_backoff(example["question"], db_id)
        agent_result["_latency"] = round(time.time() - t0, 2)

        record = evaluate_one(agent_result, example, idx)
        records.append(record)

        sym = "OK  " if record["ea"] == 1 else (
            "??  " if record["ea"] is None else "FAIL"
        )
        print(
            f"           {sym}  EA={record['ea']}  "
            f"att={record['attempts']}  "
            f"lat={record['latency_s']}s  "
            f"[{record['ea_note']}]"
        )

        time.sleep(0.5)

        if (idx + 1) % 10 == 0 or idx == n - 1:
            save_checkpoint(records)
            evaluable_so_far = [r for r in records if r["ea"] is not None]
            if evaluable_so_far:
                rolling_ea = sum(
                    r["ea"] for r in evaluable_so_far
                ) / len(evaluable_so_far) * 100
                print(
                    f"\n  ── Checkpoint guardado "
                    f"({len(records)}/{n})  EA rolling={rolling_ea:.1f}% ──\n"
                )

    metrics = compute_metrics(records)
    elapsed = round(time.time() - t_total, 1)

    print(f"\n{'='*62}")
    print(
        f"  EA global : {metrics['ea_global']}%  "
        f"({metrics['correct']}/{metrics['total']})"
    )
    print(f"  pred_failed : {metrics['pred_failed']}")
    print(f"  gold_errors : {metrics['gold_errors']}")
    print(f"  Latencia media : {metrics['avg_latency']}s/query")
    print(f"  Intentos medios: {metrics['avg_attempts']}")
    print()
    for d in ["easy", "medium", "hard", "extra"]:
        if d in metrics["by_difficulty"]:
            m = metrics["by_difficulty"][d]
            print(
                f"  {d:<8}: {m['ea']:>6.2f}%  "
                f"({m['correct']:>3}/{m['total']:>3})  "
                f"lat={m['avg_latency']}s  "
                f"att={m['avg_attempts']}"
            )
    print(f"\n  Tiempo total: {elapsed}s  ({elapsed/60:.1f} min)")
    print(f"{'='*62}\n")

    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, cls=SafeEncoder)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"elapsed_s": elapsed, **metrics},
            f, indent=2, ensure_ascii=False,
        )

    print(f"  {FINAL_PATH}")
    print(f"  {SUMMARY_PATH}\n")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"  Checkpoint eliminado: {CHECKPOINT_PATH.name}\n")


if __name__ == "__main__":
    main()
