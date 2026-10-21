import numpy as np
import pandas as pd

from cfb_bias.analysis.decay import half_life, preseason_decay

COLS = ["poll", "poll_week", "coef", "se", "ci_lo", "ci_hi", "n_obs"]


def _synthetic(weight_fn, weeks=range(2, 11), n=300, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k in weeks:
        team = np.arange(n)
        colley = rng.normal(size=n)
        pre = rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {
                    "poll_week": k,
                    "team": [f"T{t}" for t in team],
                    "season": 2020,
                    "colley": colley,
                    "colley_sos": rng.normal(size=n),
                    "wins": rng.integers(0, 10, n).astype(float),
                    "losses": rng.integers(0, 10, n).astype(float),
                    "preseason_ap_score": pre,
                    "y": 0.5 * colley + weight_fn(k) * pre + rng.normal(scale=0.05, size=n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_decay_curve_decreases_and_half_life_is_five():
    df = _synthetic(lambda k: 0.8 * 0.5 ** ((k - 2) / 2.9))  # just under the exact 3-week boundary
    out = preseason_decay(df, "y", "AP Top 25")
    assert list(out.columns) == COLS
    assert out["poll"].eq("AP Top 25").all()
    assert out["poll_week"].tolist() == list(range(2, 11))
    assert out["coef"].is_monotonic_decreasing
    assert (out["ci_lo"] < out["coef"]).all() and (out["coef"] < out["ci_hi"]).all()
    assert out["n_obs"].eq(300).all()
    assert half_life(out) == 5


def test_half_life_none_when_flat():
    df = _synthetic(lambda k: 0.8)
    out = preseason_decay(df, "y", "AP Top 25")
    assert half_life(out) is None


def test_half_life_none_when_reference_nonpositive():
    d = pd.DataFrame({"poll_week": [2, 3], "coef": [-0.1, -0.2]})
    assert half_life(d) is None


def test_small_weeks_and_nan_rows_skipped():
    df = _synthetic(lambda k: 0.5, weeks=[2, 3])
    df = df[~((df["poll_week"] == 3) & (df["team"].str[1:].astype(int) >= 5))]
    df.loc[df.index[0], "y"] = np.nan
    out = preseason_decay(df, "y", "AP Top 25")
    assert out["poll_week"].tolist() == [2]
    assert out["n_obs"].iloc[0] == 299


def test_empty_result_has_columns():
    out = preseason_decay(_synthetic(lambda k: 0.5, weeks=[2], n=5), "y", "p")
    assert out.empty and list(out.columns) == COLS
    assert half_life(out) is None
