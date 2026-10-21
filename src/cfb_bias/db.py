"""Postgres access: engine, schema creation, idempotent upsert, read helper."""

from importlib import resources

import pandas as pd
from sqlalchemy import Connection, Engine, create_engine, text


def get_engine(url: str) -> Engine:
    return create_engine(url)


def init_schema(engine: Engine) -> None:
    """Execute schema.sql (every statement is CREATE TABLE IF NOT EXISTS)."""
    ddl = resources.files("cfb_bias").joinpath("schema.sql").read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)


def upsert(engine: Engine, table: str, df: pd.DataFrame, key: list[str],
           conn: Connection | None = None) -> int:
    """INSERT ... ON CONFLICT (key) DO UPDATE. Returns number of rows written.

    NaN/NaT/None become NULL. An empty DataFrame writes nothing and returns 0.
    With `conn`, the write joins that open transaction instead of committing its own.
    """
    if df.empty:
        return 0
    quote = engine.dialect.identifier_preparer.quote
    cols = list(df.columns)
    non_key = [c for c in cols if c not in key]
    action = (
        "DO UPDATE SET " + ", ".join(f"{quote(c)} = EXCLUDED.{quote(c)}" for c in non_key)
        if non_key
        else "DO NOTHING"
    )
    sql = (
        f"INSERT INTO {quote(table)} ({', '.join(quote(c) for c in cols)}) "
        f"VALUES ({', '.join(f':p{i}' for i in range(len(cols)))}) "
        f"ON CONFLICT ({', '.join(quote(c) for c in key)}) {action}"
    )
    clean = df.astype(object).where(df.notna(), None)
    rows = [
        {f"p{i}": v for i, v in enumerate(rec)}
        for rec in clean.itertuples(index=False, name=None)
    ]
    if conn is not None:
        conn.execute(text(sql), rows)
    else:
        with engine.begin() as own:
            own.execute(text(sql), rows)
    return len(rows)


def read_sql(engine: Engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)
