from typing import Generator
import psycopg2.extensions
from .db_pool import get_pool


def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)   