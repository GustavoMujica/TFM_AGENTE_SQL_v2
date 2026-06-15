import os
import time
import psycopg2
import sqlparse
import anthropic

from dotenv import load_dotenv
from typing import Optional
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .logger_config import setup_logging, get_logger
from .schema_cache import schema_cache   

load_dotenv()
setup_logging()

log_schema     = get_logger("tfm_agent.schema")
log_sql        = get_logger("tfm_agent.sql")
log_validation = get_logger("tfm_agent.validation")
log_execution  = get_logger("tfm_agent.execution")
log_format     = get_logger("tfm_agent.format")
log_router     = get_logger("tfm_agent.router")
log_db         = get_logger("tfm_agent.db")


MAX_ATTEMPTS = 3

SQL_BLACKLIST = {"drop", "delete", "truncate", "alter", "create",
                 "insert", "update", "grant", "revoke"}

FEW_SHOT = """Ejemplos:

Pregunta: ¿Cuántos registros hay en users?
SQL: SELECT COUNT(*) FROM users;

Pregunta: ¿Cuál es el valor máximo de price en products?
SQL: SELECT MAX(price) FROM products;

Pregunta: ¿Cuántos pedidos ha hecho cada usuario?
SQL: SELECT u.name, COUNT(o.id) AS pedidos FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name;
"""

FAILURE_ANSWER = (
    "Lo siento, no he podido generar una consulta válida para tu pregunta "
    "tras {attempts} intentos. Por favor, reformula la pregunta o contacta "
    "con el administrador del sistema."
)


class AgentState(TypedDict):
    question:      str
    db_id:         None
    schema:        Optional[str]
    sql_generated: Optional[str]
    is_valid:      Optional[bool]
    results:       Optional[list]
    error:         Optional[str]
    answer:        Optional[str]
    attempts:      int


llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0, max_tokens=1024)

_conexion: Optional[psycopg2.extensions.connection] = None


def get_connection() -> psycopg2.extensions.connection:
    global _conexion
    needs_reconnect = _conexion is None or _conexion.closed != 0
    if needs_reconnect:
        if _conexion is not None:
            log_db.warning("Conexión PostgreSQL cerrada — reconectando")
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            _conexion = psycopg2.connect(database_url, connect_timeout=10)
        else:
            _conexion = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "tfm_ecommerce"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD"),
                connect_timeout=10,
            )
        log_db.info("Conexión a PostgreSQL establecida")
    return _conexion


def _clean_sql_output(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
    if text.endswith("```"):
        text = text[:text.rfind("```")]
    return text.strip()


def build_prompt(schema: str, question: str, error: Optional[str] = None) -> str:
    error_block = ""
    if error:
        error_block = (
            f"\n  ATENCIÓN — El SQL anterior produjo este error:\n"
            f"{error}\n\n"
            f"Analiza el error cuidadosamente y genera un SQL corregido "
            f"que evite ese problema.\n"
        )
    return (
        f"Eres un experto en SQL y PostgreSQL.\n\n"
        f"Dado este esquema de base de datos:\n{schema}\n\n"
        f"{FEW_SHOT}\n"
        f"Reglas:\n"
        f"- Responde SOLO con el SQL, sin ningún texto adicional\n"
        f"- Sin markdown, sin comillas, sin bloques de código\n"
        f"- Usa únicamente las tablas y columnas del esquema\n"
        f"- Para comparar con un agregado, usa subquery o CTE\n"
        f"{error_block}"
        f"Pregunta: {question}\n"
        f"SQL:"
    )




def node_get_schema(state: AgentState) -> dict:
    log_schema.info("Obteniendo schema (vía cache)")
    try:
        conn   = get_connection()
        db_id  = state.get("db_id")          

        if db_id:                             
            cache_key   = db_id
            schema_name = db_id
        else:                                 
            cache_key   = os.getenv("DB_NAME", "tfm_ecommerce")
            schema_name = "public"

        schema = schema_cache.get(conn, db_name=cache_key, schema_name=schema_name)

        stats = schema_cache.stats()
        log_schema.info(
            "Schema listo — %d chars | cache stats: hits=%d misses=%d hit_rate=%.0f%%",
            len(schema), stats["hits"], stats["misses"], stats["hit_rate"],
        )
        log_schema.debug("Schema completo:\n%s", schema)
        return {"schema": schema}


    except psycopg2.OperationalError as e:
        log_schema.error("PostgreSQL no disponible al extraer schema: %s", e)
        return {"schema": "", "error": f"Base de datos no disponible: {e}"}




def node_generate_sql(state: AgentState) -> dict:
    attempt = state["attempts"] + 1
    error   = state.get("error")
    if error:
        log_sql.warning("Intento %d/%d — reintento. Error previo: %s",
                        attempt, MAX_ATTEMPTS, error[:120])
    else:
        log_sql.info("Intento %d/%d — primera pasada", attempt, MAX_ATTEMPTS)

    prompt = build_prompt(state["schema"], state["question"], error)
    log_sql.debug("Prompt (%d chars):\n%s", len(prompt), prompt)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        sql = _clean_sql_output(response.content)
        log_sql.info("SQL generado: %s", sql[:120])
        return {"sql_generated": sql, "attempts": attempt, "error": None}

    except anthropic.APITimeoutError as e:
        msg = f"Timeout Claude API (intento {attempt}): {e}"
        log_sql.error(msg)
        return {"sql_generated": None, "attempts": attempt, "error": msg, "is_valid": False}

    except anthropic.APIStatusError as e:
        msg = f"Error HTTP Claude API — status {e.status_code}: {e.message}"
        log_sql.error(msg)
        return {"sql_generated": None, "attempts": attempt, "error": msg, "is_valid": False}

    except anthropic.APIError as e:
        msg = f"Error Claude API (intento {attempt}): {e}"
        log_sql.error(msg)
        return {"sql_generated": None, "attempts": attempt, "error": msg, "is_valid": False}


def node_validate_sql(state: AgentState) -> dict:
    sql = state.get("sql_generated")
    if not sql:
        log_validation.warning("SQL nulo — posible fallo de API")
        return {"is_valid": False}

    log_validation.info("Validando SQL (intento %d): %s", state["attempts"], sql[:80])

    if not sql.strip():
        msg = "SQL vacío"
        log_validation.warning("Validación fallida — %s", msg)
        return {"is_valid": False, "error": msg}

    normalized = sql.strip().lower()
    parsed = sqlparse.parse(normalized)
    if not parsed or parsed[0].get_type() != "SELECT":
        first_word = normalized.split()[0] if normalized.split() else ""
        msg = f"Solo SELECT permitido. Encontrado: '{first_word.upper()}'"
        log_validation.warning("Validación fallida — %s", msg)
        return {"is_valid": False, "error": msg}

    tokens = set(normalized.split())
    found = tokens & SQL_BLACKLIST
    if found:
        msg = f"Palabras peligrosas detectadas: {found}"
        log_validation.warning("Validación fallida — %s", msg)
        return {"is_valid": False, "error": msg}

    if len(sql) > 2000:
        msg = f"SQL excede longitud máxima: {len(sql)} chars"
        log_validation.warning("Validación fallida — %s", msg)
        return {"is_valid": False, "error": msg}

    log_validation.info("SQL válido ")
    return {"is_valid": True, "error": None}


def node_execute_sql(state: AgentState) -> dict:
    sql = state["sql_generated"]
    db_id = state.get("db_id") 
    log_execution.info("Ejecutando SQL contra PostgreSQL")
    log_execution.debug("SQL: %s", sql)

    try:
        conn = get_connection()
        cur = conn.cursor()

        if db_id:                            
            cur.execute(f'SET search_path TO "{db_id}"')

        cur.execute(sql)
        raw_rows = cur.fetchall()
        cur.close()
        results = [[str(cell) for cell in row] for row in raw_rows]
        log_execution.info("Ejecución exitosa: %d fila(s)", len(results))
        return {"results": results, "error": None}

    except psycopg2.OperationalError as e:
        log_execution.error("OperationalError: %s", e)
        try: get_connection().rollback()
        except Exception: pass
        return {"results": None, "error": f"Error de conexión a PostgreSQL: {e}"}

    except psycopg2.ProgrammingError as e:
        log_execution.warning("ProgrammingError (tabla/columna inválida): %s", e)
        try: get_connection().rollback()
        except Exception: pass
        return {"results": None, "error": f"Error PostgreSQL: {e}"}

    except psycopg2.Error as e:
        log_execution.error("Error PostgreSQL inesperado: %s", e)
        try: get_connection().rollback()
        except Exception: pass
        return {"results": None, "error": f"Error PostgreSQL: {e}"}


def node_format_answer(state: AgentState) -> dict:
    log_format.info("Formateando respuesta en lenguaje natural")
    format_prompt = (
        f"Pregunta del usuario: {state['question']}\n"
        f"Resultados de la base de datos: {state['results']}\n\n"
        f"Responde en español, de forma clara y concisa, sin mencionar SQL."
    )
    try:
        response = llm.invoke([HumanMessage(content=format_prompt)])
        answer = response.content.strip()
        log_format.info("Respuesta generada: %d chars", len(answer))
        return {"answer": answer}
    except anthropic.APIError as e:
        log_format.error("Error al formatear: %s — usando fallback", e)
        return {"answer": f"Resultados obtenidos: {state['results']}"}


def node_handle_failure(state: AgentState) -> dict:
    log_router.error(
        "Fallo controlado tras %d intentos. Último error: %s",
        state["attempts"], state.get("error", "desconocido")
    )
    return {"answer": FAILURE_ANSWER.format(attempts=state["attempts"])}


def decide_after_validation(state: AgentState) -> str:
    if state.get("is_valid"):
        log_router.info("Validación OK → execute_sql")
        return "execute_sql"
    if state["attempts"] >= MAX_ATTEMPTS:
        log_router.critical("MAX_ATTEMPTS agotado (validación)")
        return "handle_failure"
    log_router.warning("SQL inválido → retry. Error: %s", state.get("error"))
    return "generate_sql"


def decide_after_execution(state: AgentState) -> str:
    if state["results"] is not None:
        log_router.info("Ejecución OK → format_answer")
        return "format_answer"
    if state["attempts"] >= MAX_ATTEMPTS:
        log_router.critical("MAX_ATTEMPTS agotado (ejecución)")
        return "handle_failure"
    log_router.warning("Error ejecución → retry. Error: %s", state.get("error"))
    return "generate_sql"


def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("get_schema",     node_get_schema)
    graph.add_node("generate_sql",   node_generate_sql)
    graph.add_node("validate_sql",   node_validate_sql)
    graph.add_node("execute_sql",    node_execute_sql)
    graph.add_node("format_answer",  node_format_answer)
    graph.add_node("handle_failure", node_handle_failure)

    graph.set_entry_point("get_schema")
    graph.add_edge("get_schema",     "generate_sql")
    graph.add_edge("generate_sql",   "validate_sql")
    graph.add_edge("handle_failure", END)
    graph.add_edge("format_answer",  END)

    graph.add_conditional_edges("validate_sql", decide_after_validation,
        {"execute_sql": "execute_sql", "generate_sql": "generate_sql",
         "handle_failure": "handle_failure"})
    graph.add_conditional_edges("execute_sql", decide_after_execution,
        {"format_answer": "format_answer", "generate_sql": "generate_sql",
         "handle_failure": "handle_failure"})

    return graph.compile()


def run_benchmark():
    agente = build_agent()
    log_bm = get_logger("tfm_agent.benchmark")

    casos = [
        ("H1", "¿Cuántos usuarios hay en total?"),
        ("H2", "¿Cuál es el precio máximo de los productos?"),
        ("H3", "¿Cuánto ha gastado en total cada usuario? Ordena de mayor a menor."),
        ("H4", "¿Qué productos han sido pedidos más veces?"),
        ("H5", "¿Qué usuarios han gastado más que el promedio de todos los usuarios?"),
    ]

    log_bm.info("=" * 70)
    log_bm.info("AGENT v3 — Benchmark semana 8, Paso 3 (schema cache)")
    log_bm.info("Cada invocación loguea si el schema vino de cache o de DB")
    log_bm.info("=" * 70)

    resultados = []

    for caso_id, pregunta in casos:
        log_bm.info("─" * 70)
        log_bm.info("Caso %s: %s", caso_id, pregunta)

        estado_inicial: AgentState = {
            "question": pregunta, "schema": None, "sql_generated": None,
            "is_valid": None, "results": None, "error": None,
            "answer": None, "attempts": 0,
        }

        t0 = time.time()
        estado_final = agente.invoke(estado_inicial)
        elapsed = time.time() - t0

        exito = bool(estado_final.get("answer")) and "Lo siento" not in (estado_final.get("answer") or "")
        log_bm.info("Resultado: %s | %.1fs", "ÉXITO" if exito else "FALLO", elapsed)
        resultados.append((caso_id, exito, elapsed))

    stats = schema_cache.stats()
    latencias = [t for _, _, t in resultados]
    exitos = sum(1 for _, ok, _ in resultados if ok)

    log_bm.info("=" * 70)
    log_bm.info("RESUMEN")
    log_bm.info("=" * 70)
    for caso_id, ok, elapsed in resultados:
        log_bm.info("  %s %s — %.1fs", "OK" if ok else "FAIL", caso_id, elapsed)

    log_bm.info("")
    log_bm.info("Score          : %d/%d", exitos, len(resultados))
    log_bm.info("Latencia media : %.1fs", sum(latencias) / len(latencias))
    log_bm.info("Latencia máx.  : %.1fs", max(latencias))
    log_bm.info("")
    log_bm.info("── Cache stats ──────────────────────────────────")
    log_bm.info("  Hits          : %d", stats["hits"])
    log_bm.info("  Misses        : %d", stats["misses"])
    log_bm.info("  Refreshes     : %d", stats["refreshes"])
    log_bm.info("  Hit rate      : %.0f%%", stats["hit_rate"])
    log_bm.info("  (1 miss esperado = primera invocación)")
    log_bm.info("  (%d hits esperados = invocaciones 2-5)", len(casos) - 1)

    if exitos == len(resultados):
        log_bm.info(" TODOS LOS CASOS PASARON")
    else:
        log_bm.warning("  %d caso(s) fallaron", len(resultados) - exitos)


if __name__ == "__main__":
    run_benchmark()