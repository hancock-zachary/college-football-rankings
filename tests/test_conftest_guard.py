import pytest
from conftest import require_test_database
from sqlalchemy.engine import make_url


@pytest.mark.parametrize("url", [
    "postgresql+psycopg://cfb:cfb@localhost:5432/cfb_test",
    "postgresql+psycopg://u:p@h/other_test",
])
def test_guard_accepts_test_databases(url):
    require_test_database(make_url(url))


@pytest.mark.parametrize("url", [
    "postgresql+psycopg://cfb:cfb@localhost:5432/cfb",
    "postgresql+psycopg://cfb:cfb@localhost:5432/test_cfb",
    "postgresql+psycopg://cfb:cfb@localhost:5432/",
])
def test_guard_refuses_non_test_databases(url):
    with pytest.raises(ValueError, match="_test"):
        require_test_database(make_url(url))
