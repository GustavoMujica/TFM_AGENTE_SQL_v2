import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.agent_v3_cached import build_agent, AgentState
from .models import QueryRequest, QueryResponse, ErrorResponse
from .db_pool import init_pool, close_pool, get_pool

log = logging.getLogger("tfm_agent.api")

_agent = build_agent()

def run_agent(question: str) -> dict:
    estado: AgentState = {
        "question":      question,
        "schema":        None,
        "sql_generated": None,
        "is_valid":      None,
        "results":       None,
        "error":         None,
        "answer":        None,
        "attempts":      0,
    }
    return _agent.invoke(estado)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== TFM Agente SQL API v%s — startup ===", app.version)
    try:
        init_pool()
        log.info("DB pool inicializado (minconn=1, maxconn=5)")
    except Exception as e:
        log.warning("Pool no disponible en startup: %s", e)

    yield

    log.info("=== Shutdown — cerrando pool ===")
    close_pool()


app = FastAPI(
    title="TFM — Agente SQL",
    description=(
        "API REST para el agente conversacional que convierte preguntas "
        "en lenguaje natural a consultas SQL sobre PostgreSQL."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

_started_at = datetime.now(timezone.utc)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tfm-agente-sql.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Excepción no manejada en %s %s: %s",
              request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor."},
    )


def _serialize_results(results: list | None) -> list[list[Any]]:
    if not results:
        return []
    output = []
    for row in results:
        serialized_row = []
        for val in row:
            if isinstance(val, Decimal):
                serialized_row.append(float(val))
            elif hasattr(val, 'isoformat'):
                serialized_row.append(val.isoformat())
            else:
                serialized_row.append(val)
        output.append(serialized_row)
    return output


@app.get(
    "/health",
    tags=["Sistema"],
    summary="Verificación de disponibilidad",
)
def health() -> dict:
    uptime = (datetime.now(timezone.utc) - _started_at).total_seconds()
    db_status = "ok"

    try:
        pool = get_pool()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        finally:
            pool.putconn(conn)
    except Exception as e:
        log.warning("DB ping fallido en /health: %s", e)
        db_status = "error"

    return {
        "status":         "ok" if db_status == "ok" else "degraded",
        "version":        app.version,
        "uptime_seconds": round(uptime, 1),
        "db":             db_status,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Agente"],
    summary="Procesar pregunta en lenguaje natural",
    responses={
        422: {"model": ErrorResponse, "description": "Pregunta vacía o supera 500 caracteres"},
        503: {"model": ErrorResponse, "description": "Agente agotó reintentos sin respuesta válida"},
    },
)
def query(request: QueryRequest) -> QueryResponse:
    log.info("Query recibida: %s", request.question[:80])
    state = run_agent(request.question)

    if not state.get("answer"):
        raise HTTPException(
            status_code=503,
            detail="El agente no pudo generar una respuesta. Intente reformular la pregunta.",
        )

    log.info("Query resuelta en %d intento(s)", state.get("attempts", 0))
    return QueryResponse(
        answer=state["answer"],
        sql=state.get("sql_generated") or "",
        results=_serialize_results(state.get("results")),
        attempts=state.get("attempts", 0),
    )
