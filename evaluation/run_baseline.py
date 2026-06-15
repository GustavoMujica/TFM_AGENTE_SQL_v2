import os
import sys
import json
import time
import signal
import sqlite3
import argparse
from pathlib import Path
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()
os.environ["DB_NAME"] = "spider_eval"
os.environ["DB_HOST"] = "localhost"

import agent.agent_v3_cached as _agent_module
_agent_module.MAX_ATTEMPTS = 1

from agent.agent_v3_cached import build_agent, schema_cache
import anthropic

EVAL_DIR    = Path(__file__).parent
SPIDER_DIR  = EVAL_DIR / "spider"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_PREFIX      = "baseline_1034"
CHECKPOINT_PATH = RESULTS_DIR / f"{OUT_PREFIX}_partial.json"
FINAL_PATH      = RESULTS_DIR / f"{OUT_PREFIX}.json"
SUMMARY_PATH    = RESULTS_DIR / f"{OUT_PREFIX}_summary.json"
DEV_JSON        = SPIDER_DIR / "dev_enriched.json"
SQLITE_DB_DIR   = SPIDER_DIR / "database"

DB20 = "student_transcripts_tracking"

_graceful_exit = False

def _handle_sigint(sig, frame):
    global _graceful_exit
    print("\n\n  Ctrl+C recibido. Guardando checkpoint y saliendo limpiamente...")
    _graceful_exit = True

signal.signal(signal.SIGINT, _handle_sigint)


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


def run_gold_sqlite(gold_sql: str, db_id: str):
    sqlite_path = SQLITE_DB_DIR / db_id / f"{db_id}.sqlite"
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
            print(f"\n  RateLimitError (429). Esperando {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code == 429:
                wait = 60 * (2 ** intento)
                print(f"\n  HTTP 429. Esperando {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Rate limit persistente tras {max_retries} reintentos")


def load_checkpoint() -> tuple[list, set]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            records = json.load(f)
        done = {r["idx"] for r in records}
        print(f"  Checkpoint: {len(done)} queries ya evaluadas. Reanudando...")
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
        return {**base, "ea": None, "ea_note": "gold_error",
                "gold_result": None, "pred_result": pred_results}

    if pred_results is None:
        return {**base, "ea": 0, "ea_note": "pred_failed",
                "gold_result": [list(r) for r in gold_rows],
                "pred_result": None}

    match = compare_results(gold_rows, pred_results)
    return {
        **base,
        "ea":          1 if match else 0,
        "ea_note":     "match" if match else "mismatch",
        "gold_result": [list(r) for r in gold_rows],
        "pred_result": pred_results,
        "gold_shape":  [len(gold_rows), len(gold_rows[0]) if gold_rows else 0],
        "pred_shape":  [len(pred_results), len(pred_results[0]) if pred_results else 0],
    }


def compute_metrics(records: list) -> dict:
    by_diff: dict = {}
    for r in records:
        d = r["difficulty"]
        if d not in by_diff:
            by_diff[d] = {"total": 0, "correct": 0, "failed": 0,
                          "latencies": [], "attempts": []}
        by_diff[d]["total"] += 1
        if r["ea"] == 1:
            by_diff[d]["correct"] += 1
        if r["ea_note"] == "pred_failed":
            by_diff[d]["failed"] += 1
        by_diff[d]["latencies"].append(r["latency_s"])
        by_diff[d]["attempts"].append(r["attempts"])

    evaluable = [r for r in records if r["ea"] is not None]
    correct   = sum(1 for r in evaluable if r["ea"] == 1)
    n         = len(evaluable)

    result = {
        "ea_global":    round(correct / n * 100, 2) if n else 0,
        "correct":      correct,
        "total":        n,
        "pred_failed":  sum(1 for r in records if r["ea_note"] == "pred_failed"),
        "gold_errors":  sum(1 for r in records if r["ea_note"] == "gold_error"),
        "avg_latency":  round(sum(r["latency_s"] for r in records) / len(records), 2),
        "avg_attempts": round(sum(r["attempts"]  for r in records) / len(records), 2),
        "by_difficulty": {
            d: {
                "ea":          round(v["correct"] / v["total"] * 100, 2),
                "correct":     v["correct"],
                "total":       v["total"],
                "pred_failed": v["failed"],
                "avg_latency": round(sum(v["latencies"]) / len(v["latencies"]), 2),
                "avg_attempts":round(sum(v["attempts"])  / len(v["attempts"]),  2),
            }
            for d, v in by_diff.items()
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Spider eval (MAX_ATTEMPTS=1)"
    )
    parser.add_argument("--full",  action="store_true",
                        help="Evaluación completa (1034 ejemplos)")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot: 2 por nivel + 2 de student_transcripts_tracking (~10)")
    args = parser.parse_args()

    if not args.full and not args.pilot:
        print("  Uso: --pilot  o  --full")
        return

    assert _agent_module.MAX_ATTEMPTS == 1, (
        f"MAX_ATTEMPTS={_agent_module.MAX_ATTEMPTS}. "
        "Debe ser 1 para el baseline. Algo reseteó el valor antes de tiempo."
    )
    print(f"  Verificado: MAX_ATTEMPTS={_agent_module.MAX_ATTEMPTS} (baseline correcto)")

    if not DEV_JSON.exists():
        print(f"No se encuentra {DEV_JSON}")
        sys.exit(1)
    with open(DEV_JSON, encoding="utf-8") as f:
        all_examples = json.load(f)

    if args.pilot:
        pilot = []
        for diff in ["easy", "medium", "hard", "extra"]:
            bucket = [ex for ex in all_examples if ex["difficulty"] == diff]
            pilot.extend(bucket[:2])
        pilot.extend([ex for ex in all_examples if ex["db_id"] == DB20][:2])
        all_examples = pilot
        print(f"  MODO PILOT: {len(all_examples)} queries")
        print(f"  (2 por nivel + 2 de {DB20})")
    else:
        print(f"  MODO FULL BASELINE: {len(all_examples)} queries")
        print(f"  MAX_ATTEMPTS={_agent_module.MAX_ATTEMPTS} — sin retry loop")

    n = len(all_examples)
    records, done_indices = load_checkpoint()
    t_total = time.time()

    for idx, example in enumerate(all_examples):
        if _graceful_exit:
            print(f"\n  Guardando checkpoint parcial ({len(records)} queries)...")
            save_checkpoint(records)
            break

        if idx in done_indices:
            continue

        db_id  = example["db_id"]
        diff   = example["difficulty"]
        q_short = example["question"][:55] + ("..." if len(example["question"]) > 55 else "")
        print(f"  [{idx+1:>4}/{n}] [{diff:<6}] {db_id:<35} {q_short}")

        t0 = time.time()
        try:
            agent_result = run_agent_with_backoff(example["question"], db_id)
        except Exception as e:
            print(f"          Error inesperado: {e}")
            agent_result = {
                "sql_generated": None, "results": None,
                "attempts": 0, "error": str(e),
            }
        agent_result["_latency"] = round(time.time() - t0, 2)

        record = evaluate_one(agent_result, example, idx)
        records.append(record)

        sym = "OK" if record["ea"] == 1 else ("??" if record["ea"] is None else "FAIL")
        print(f"          {sym} EA={record['ea']}  att={record['attempts']}  "
              f"lat={record['latency_s']}s  [{record['ea_note']}]")

        if (idx + 1) % 10 == 0 or idx == n - 1:
            save_checkpoint(records)

        time.sleep(0.5)

    metrics = compute_metrics(records)
    elapsed = round(time.time() - t_total, 1)

    print(f"\n{'='*62}")
    print(f"  BASELINE n={len(records)} | MAX_ATTEMPTS=1")
    print(f"{'='*62}")
    print(f"  EA global : {metrics['ea_global']}%  ({metrics['correct']}/{metrics['total']})")
    for d, m in metrics["by_difficulty"].items():
        print(f"  {d:<8} : {m['ea']}%  ({m['correct']}/{m['total']})  "
              f"pred_failed={m['pred_failed']}")
    print(f"  pred_failed total : {metrics['pred_failed']}")
    print(f"  gold_errors       : {metrics['gold_errors']}")
    print(f"  avg_latency       : {metrics['avg_latency']}s")
    print(f"  avg_attempts      : {metrics['avg_attempts']}")
    print(f"  Tiempo total      : {elapsed}s  ({elapsed/max(len(records),1):.1f}s/query)")
    print(f"{'='*62}")

    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, cls=SafeEncoder)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({"elapsed_s": elapsed, "max_attempts_used": 1, **metrics},
                  f, indent=2, ensure_ascii=False)

    print(f"\n{FINAL_PATH}")
    print(f"{SUMMARY_PATH}")

    if not _graceful_exit and len(records) == n:
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            print(f"Checkpoint eliminado: {CHECKPOINT_PATH.name}")

    _agent_module.MAX_ATTEMPTS = 3
    print(f"  MAX_ATTEMPTS restaurado a {_agent_module.MAX_ATTEMPTS}")


if __name__ == "__main__":
    main()
