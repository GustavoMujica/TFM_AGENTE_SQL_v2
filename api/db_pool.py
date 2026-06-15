import os
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv()

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool() -> psycopg2.pool.ThreadedConnectionPool:    
    global _pool
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=database_url)
    else:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "tfm_ecommerce"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
        )
    return _pool


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    if _pool is None:
        raise RuntimeError("Pool no inicializado — verifica el lifespan.")
    return _pool


def close_pool() -> None:
    if _pool is not None:
        _pool.closeall()