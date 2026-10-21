import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

from cfb_bias.db import init_schema

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://cfb:cfb@localhost:5432/cfb_test",
)


def require_test_database(url: URL) -> None:
    """Refuse to run the destructive schema reset against a non-test database."""
    if not (url.database or "").endswith("_test"):
        raise ValueError(
            f"Refusing to reset database {url.database!r}: TEST_DATABASE_URL must point "
            "at a database whose name ends with '_test'."
        )


@pytest.fixture(scope="session")
def pg_engine():
    engine = create_engine(TEST_DATABASE_URL)
    try:
        require_test_database(engine.url)
    except ValueError as exc:
        engine.dispose()
        pytest.exit(str(exc), returncode=2)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        engine.dispose()
        pytest.skip("Postgres not running")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    init_schema(engine)
    yield engine
    engine.dispose()
