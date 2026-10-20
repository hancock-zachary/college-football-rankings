import pandas as pd
import pytest

from cfb_bias.ratings.bootstrap import bootstrap_ratings


def mean_rate(games, teams):
    return pd.Series(float(len(games)), index=pd.Index(teams, name="team"))


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        bootstrap_ratings(mean_rate, pd.DataFrame({"x": [1]}), ["A"], n=0)


def test_sorted_and_reset_index():
    out = bootstrap_ratings(mean_rate, pd.DataFrame({"x": [1, 2]}), ["B", "A"], n=5)
    assert list(out["team"]) == ["A", "B"]
    assert list(out.index) == [0, 1]
