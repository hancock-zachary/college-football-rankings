import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from cfb_bias.db import init_schema

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://cfb:cfb@localhost:5432/cfb_test",
)


@pytest.fixture(scope="session")
def pg_engine():
    engine = create_engine(TEST_DATABASE_URL)
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
