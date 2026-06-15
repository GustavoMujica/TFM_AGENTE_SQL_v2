from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pregunta en lenguaje natural sobre la base de datos",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "¿Cuántos usuarios hay en total?"},
                {"question": "¿Cuánto ha gastado en total cada usuario?"},
                {"question": "¿Qué usuarios han gastado más que el promedio?"},
            ]
        }
    }


class QueryResponse(BaseModel):
    answer:   str            = Field(..., description="Respuesta en lenguaje natural generada por el agente")
    sql:      str            = Field(..., description="SQL generado y ejecutado contra PostgreSQL")
    results:  list[list[Any]]= Field(..., description="Filas devueltas por PostgreSQL, serializadas como listas")
    attempts: int            = Field(..., description="Número de intentos de generación SQL (1 = sin reintento)")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Descripción del error")