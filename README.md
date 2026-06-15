# TFM_AGENTE_SQL_v2
# Agente Conversacional para Consultas sobre Bases de Datos

**Trabajo de Fin de Máster — Máster en Deep Learning**

Sistema de traducción automática de lenguaje natural a SQL (*Text-to-SQL*) basado en un agente LangGraph con mecanismo de autocorrección, desplegado en producción sobre arquitectura cloud.

**Demo en producción:** [https://tfm-agente-sql.vercel.app](https://tfm-agente-sql.vercel.app)

---



## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| LLM | `claude-sonnet-4-5` (Anthropic), `temperature=0` |
| Orquestación del agente | LangGraph `StateGraph` |
| Framework LLM | LangChain Anthropic |
| Backend | FastAPI + Uvicorn |
| Base de datos (producción) | PostgreSQL — Neon (serverless) |
| Base de datos (evaluación) | PostgreSQL — Neon (20 schemas Spider) |
| Frontend | HTML / CSS / JavaScript (vanilla) |
| Despliegue backend | Render (Docker) |
| Despliegue frontend | Vercel |
| Cache de schema | TTL=300 s (implementación propia) |
| Validación SQL | `sqlparse` |

---

## Estructura del Repositorio

```
TFM_AGENTE_SQL/
│
├── agent/
│   ├── agent_v3_cached.py      # Agente principal — StateGraph 6 nodos
│   └── schema_cache.py         # Cache de schema con TTL=300 s
│
├── api/
│   ├── main.py                 # API REST — FastAPI + endpoints
│   ├── models.py               # Modelos Pydantic (request/response)
│   ├── dependencies.py         # Dependencias FastAPI
│   └── db_pool.py              # Pool de conexiones PostgreSQL
│
├── frontend/
│   └── index.html              # Interfaz web
│
├── evaluation/
│   ├── run_evaluation.py       # Evaluación agente sobre Spider (n=1.034)
│   ├── run_baseline.py         # Evaluación baseline (MAX_ATTEMPTS=1)
│   ├── analyze_errors.py       # Clasificación de errores (categorías A–E)
│   ├── compare_results.py      # Matriz de transición agente vs baseline (A–F)
│   ├── generate_figures.py     # Generación de figuras PNG para la memoria
│   │
│   ├── figures/
│   │   ├── fig1_ea_por_nivel.png   # EA agente vs baseline por nivel
│   │   ├── fig2_tipos_error.png    # Distribución de tipos de error
│   │   └── fig3_intentos.png       # Distribución de intentos
│   │
│   └── results/
│       ├── full_1034.json              # Resultados agente — 1.034 queries
│       ├── full_1034_summary.json      # Métricas agregadas agente
│       ├── baseline_1034.json          # Resultados baseline — 1.034 queries
│       ├── baseline_1034_summary.json  # Métricas agregadas baseline
│       ├── error_analysis_1034.json    # 263 mismatches clasificados
│       ├── error_analysis_report_1034.txt
│       ├── comparison_1034_summary.json    # Matriz A–F + métricas
│       └── comparison_1034_report.txt
│
├── Dockerfile
├── requirements.txt
├── logger_config.py
└── .gitignore
```

---

## Instalación y Ejecución Local

### Requisitos

- Python 3.11+
- PostgreSQL accesible (local o Neon)
- API key de Anthropic

### Configuración

```bash
git clone https://github.com/GustavoMujica/TFM_AGENTE_SQL_v2.git
cd TFM_AGENTE_SQL_v2
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz del proyecto:

```env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://usuario:password@host/nombre_db
```

### Ejecutar el backend

```bash

uvicorn api.main:app --reload --port 8000


```

Verificar /health con DB ping:
```

powershellInvoke-RestMethod http://localhost:8000/health

```

Respuesta esperada:
json{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3.1,
  "db": "ok"
}

La API queda disponible en `http://localhost:8000`. Endpoint principal: `POST /query`.

### Ejecutar el frontend

Abre `frontend/index.html` directamente en el navegador o sirve la carpeta con cualquier servidor estático.

---

## Reproducir la Evaluación

> **Importante:** los scripts de evaluación deben ejecutarse **desde la raíz del repositorio**, ya que importan el agente mediante `from agent.agent_v3_cached import build_agent`.

La evaluación requiere acceso a las 20 bases de datos de Spider cargadas en PostgreSQL y al dataset `dev.json` de Spider v1.0 (no incluido en este repositorio por tamaño).

```bash
# Evaluación completa del agente (n=1.034, ~$3-5 en API calls)
python evaluation/run_evaluation.py --full

# Evaluación baseline (MAX_ATTEMPTS=1)
python evaluation/run_baseline.py --full

# Análisis de errores sobre los resultados guardados
python evaluation/analyze_errors.py

# Comparación agente vs baseline (matriz A–F)
python evaluation/compare_results.py

# Generar figuras PNG
python evaluation/generate_figures.py
```

Los resultados ya calculados están disponibles en `evaluation/results/` para reproducir las figuras sin necesidad de re-ejecutar la evaluación.

---

## Autor

**Gustavo Mujica**
Máster en Deep Learning 