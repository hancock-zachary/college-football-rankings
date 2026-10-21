import pandas as pd
import pytest

from cfb_bias.analysis.history import alltime_win_pct, ap_titles


def games_frame(rows, completed=True):
    return pd.DataFrame(
        [
            {
                "game_id": i,
                "season": s,
                "week": 1,
                "season_type": "regular",
                "completed": completed,
                "home_team": h,
                "away_team": a,
                "home_points": hp,
                "away_points": ap,
            }
            for i, (s, h, a, hp, ap) in enumerate(rows)
        ]
    )


def polls_frame(rows, poll="AP Top 25"):
    return pd.DataFrame(
        [
            {"season": s, "season_type": st, "week": w, "poll": poll, "team": t, "rank": r}
            for s, st, w, t, r in rows
        ]
    )


def test_alltime_win_pct_counts_ties_half_and_excludes_current_season():
    g = games_frame([(1999, "A", "B", 21, 21), (1999, "A", "C", 10, 3), (2000, "A", "B", 0, 7)])
    s = alltime_win_pct(g, season=2000, first_season=1936)
    assert s["A"] == pytest.approx(0.75)  # 1 win + ½ tie over 2 games; 2000 excluded
    assert s["B"] == pytest.approx(0.5)  # brief said 0.25; B only has the 1999 tie


def test_alltime_win_pct_counts_away_games_and_respects_first_season():
    g = games_frame([(1930, "A", "B", 0, 7), (1999, "B", "A", 3, 10)])
    s = alltime_win_pct(g, season=2000, first_season=1936)
    assert s["A"] == pytest.approx(1.0)  # away win; 1930 excluded
    assert s["B"] == pytest.approx(0.0)


def test_alltime_win_pct_ignores_incomplete_and_null_points():
    g = pd.concat(
        [games_frame([(1999, "A", "B", 10, 3)]), games_frame([(1999, "A", "C", 0, 7)], completed=False)],
        ignore_index=True,
    )
    g = pd.concat([g, games_frame([(1999, "D", "E", None, None)])], ignore_index=True)
    s = alltime_win_pct(g, season=2000)
    assert s["A"] == pytest.approx(1.0)
    assert "C" not in s.index and "D" not in s.index and "E" not in s.index


def test_alltime_win_pct_empty_when_no_games():
    s = alltime_win_pct(games_frame([(2005, "A", "B", 1, 0)]), season=2000)
    assert s.empty


def test_ap_titles_uses_final_poll_only():
    p = polls_frame([(1999, "regular", 15, "B", 1), (1999, "postseason", 1, "A", 1),
                     (2000, "postseason", 1, "A", 1)])
    t = ap_titles(p, season=2001)
    assert t["A"] == 2 and t.get("B", 0) == 0


def test_ap_titles_ignores_other_polls_and_out_of_window_seasons():
    ap = polls_frame([(1999, "postseason", 1, "A", 1), (2001, "postseason", 1, "A", 1),
                      (1930, "postseason", 1, "A", 1)])
    coaches = polls_frame([(1999, "postseason", 1, "Z", 1)], poll="Coaches Poll")
    t = ap_titles(pd.concat([ap, coaches], ignore_index=True), season=2001)
    assert t["A"] == 1 and "Z" not in t.index


def test_ap_titles_latest_regular_week_when_no_postseason():
    p = polls_frame([(1999, "regular", 14, "B", 1), (1999, "regular", 15, "A", 1)])
    t = ap_titles(p, season=2000)
    assert t["A"] == 1 and t.get("B", 0) == 0
