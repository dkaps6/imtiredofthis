import pandas as pd

from scripts.build.build_coverage_v2 import build_authoritative_team_map


def test_coverage_v2_allows_bye_weeks():
    teams = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    ]
    rows = []
    for i in range(0, len(teams), 2):
        away, home = teams[i], teams[i + 1]
        rows.append({
            "season": 2026,
            "week": 8,
            "home": home,
            "away": away,
            "game_id": f"2026_08_{away}_{home}",
            "kickoff_utc": pd.Timestamp("2026-10-25T17:00:00Z") + pd.Timedelta(minutes=i),
        })
    out = build_authoritative_team_map(2026, 8, schedule=pd.DataFrame(rows))
    assert len(out) == 30
    assert out.team.nunique() == 30
    assert not out.opponent.fillna("").eq("").any()
