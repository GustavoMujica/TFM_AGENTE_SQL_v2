import time
import threading
from dataclasses import dataclass, field
from typing import Optional
import psycopg2

import sys
import os
from .logger_config import get_logger

log = get_logger("tfm_agent.schema_cache")


@dataclass
class _CacheEntry:
    schema:     str
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.monotonic() - self.created_at) > ttl_seconds


class SchemaCache:

    def __init__(self, ttl_seconds: float = 300.0):
        self._ttl      = ttl_seconds
        self._store:   dict[str, _CacheEntry] = {}
        self._lock     = threading.Lock()
        self._hits     = 0
        self._misses   = 0
        self._refreshes = 0

    def get(
        self,
        conn: psycopg2.extensions.connection,
        db_name:     str = "default",
        schema_name: str = "public",
    ) -> str:
        key = f"{db_name}::{schema_name}"

        with self._lock:
            entry = self._store.get(key)

            if entry and not entry.is_expired(self._ttl):
                self._hits += 1
                age = time.monotonic() - entry.created_at
                log.debug(
                    "Cache HIT  [%s] — age=%.1fs, TTL=%.0fs",
                    key, age, self._ttl
                )
                return entry.schema

            if entry:
                self._refreshes += 1
                log.info("Cache REFRESH [%s] — expirado tras %.0fs", key, self._ttl)
            else:
                self._misses += 1
                log.info("Cache MISS [%s] — primera consulta", key)

            schema = self._fetch_from_db(conn, schema_name)
            self._store[key] = _CacheEntry(schema=schema)
            log.debug("Schema almacenado en cache [%s] — %d chars", key, len(schema))
            return schema

    def invalidate(self, db_name: str = "default", schema_name: str = "public") -> None:
        key = f"{db_name}::{schema_name}"
        with self._lock:
            if key in self._store:
                del self._store[key]
                log.info("Cache invalidado explícitamente [%s]", key)
            else:
                log.debug("Cache invalidate: clave no encontrada [%s]", key)

    def invalidate_all(self) -> None:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            log.info("Cache vaciado completamente (%d entradas eliminadas)", count)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses + self._refreshes
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "hits":      self._hits,
                "misses":    self._misses,
                "refreshes": self._refreshes,
                "total":     total,
                "hit_rate":  round(hit_rate, 1),
                "entries":   len(self._store),
            }

    def _fetch_from_db(
        self,
        conn: psycopg2.extensions.connection,
        schema_name: str,
    ) -> str:
        log.debug("Consultando information_schema (schema='%s')", schema_name)
        t0 = time.monotonic()

        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
        """, (schema_name,))
        rows = cur.fetchall()
        cur.close()

        elapsed = time.monotonic() - t0

        tables: dict[str, list[str]] = {}
        for table, column, dtype in rows:
            tables.setdefault(table, []).append(f"{column}({dtype})")

        lines = [f"{t}: {', '.join(cols)}" for t, cols in tables.items()]
        schema_str = "\n".join(lines)

        log.info(
            "information_schema consultada: %d tablas, %d chars, %.0fms",
            len(tables), len(schema_str), elapsed * 1000,
        )
        return schema_str


schema_cache = SchemaCache(ttl_seconds=300)


def _run_tests(conn: psycopg2.extensions.connection):
    log.info("=" * 60)
    log.info("Tests de SchemaCache")
    log.info("=" * 60)

    cache = SchemaCache(ttl_seconds=2.0)
    passed = 0
    failed = 0

    def check(label: str, condition: bool):
        nonlocal passed, failed
        if condition:
            log.info("%s", label)
            passed += 1
        else:
            log.warning("%s", label)
            failed += 1

    schema1 = cache.get(conn, db_name="test")
    stats = cache.stats()
    check("T1: primera llamada → miss (misses=1)", stats["misses"] == 1)
    check("T1: schema no vacío", bool(schema1))

    schema2 = cache.get(conn, db_name="test")
    stats = cache.stats()
    check("T2: segunda llamada → hit (hits=1)", stats["hits"] == 1)
    check("T2: schema idéntico al anterior", schema1 == schema2)

    log.info("Esperando %ds para que expire el TTL...", 3)
    time.sleep(3)
    schema3 = cache.get(conn, db_name="test")
    stats = cache.stats()
    check("T3: tras TTL → refresh (refreshes=1)", stats["refreshes"] == 1)
    check("T3: schema sigue siendo válido tras refresh", bool(schema3))

    cache.get(conn, db_name="test")
    cache.invalidate(db_name="test")
    schema4 = cache.get(conn, db_name="test")
    stats = cache.stats()
    check("T4: tras invalidate → miss adicional", stats["misses"] == 2)
    check("T4: schema sigue siendo válido", bool(schema4))

    stats = cache.stats()
    check("T5: hit_rate > 0%", stats["hit_rate"] > 0)
    log.info("    Estadísticas finales: %s", stats)

    cache.invalidate_all()
    stats_after = cache.stats()
    check("T6: invalidate_all → entries=0", stats_after["entries"] == 0)

    log.info("=" * 60)
    log.info("Resultado: %d/%d tests pasados", passed, passed + failed)
    if failed == 0:
        log.info("SchemaCache verificado correctamente")
    else:
        log.warning("%d test(s) fallaron", failed)
    log.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "tfm_ecommerce"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )

    ok = _run_tests(conn)
    conn.close()
    sys.exit(0 if ok else 1)
