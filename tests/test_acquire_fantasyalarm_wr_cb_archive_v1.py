from __future__ import annotations

from scripts.research.acquire_fantasyalarm_wr_cb_archive_v1 import parse_page


HTML = """
<html>
<head><meta property="article:published_time" content="2025-10-05T12:00:00-04:00"></head>
<body>
<h2>Left WR vs Right CB</h2>
<table>
  <thead><tr><th>Wide Receiver</th><th>Team</th><th>Cornerback</th><th>Opp</th><th>Matchup</th></tr></thead>
  <tbody>
    <tr><td>Michael Pittman Jr.</td><td>IND</td><td>Mike Hughes</td><td>ATL</td><td>Moderate</td></tr>
  </tbody>
</table>
<h2>Slot WR vs Slot CB</h2>
<table>
  <thead><tr><th>Wide Receiver</th><th>Team</th><th>Cornerback</th><th>Opp</th><th>Matchup</th></tr></thead>
  <tbody>
    <tr><td>Amon-Ra St. Brown</td><td>DET</td><td>Daxton Hill</td><td>CIN</td><td>Safe</td></tr>
  </tbody>
</table>
</body>
</html>
"""


def test_parse_explicit_pairings_and_alignment():
    rows, audit = parse_page(
        HTML,
        season=2025,
        week=5,
        source_url="https://www.fantasyalarm.com/example",
    )
    assert len(rows) == 2
    assert set(rows["alignment_bucket"]) == {"LWR_VS_RCB", "SWR_VS_SCB"}
    assert rows["identity_status"].eq("READY").all()
    assert rows["published_at_utc"].eq("2025-10-05T16:00:00Z").all()
    assert rows["editorial_matchup_model_eligible"].eq(False).all()
    assert audit["rows_emitted"] == 2


def test_editorial_grade_is_preserved_but_never_model_eligible():
    rows, _ = parse_page(
        HTML,
        season=2025,
        week=5,
        source_url="https://www.fantasyalarm.com/example",
    )
    row = rows.loc[rows["wr_raw"].eq("Michael Pittman Jr.")].iloc[0]
    assert row["editorial_matchup_raw"] == "Moderate"
    assert bool(row["editorial_matchup_model_eligible"]) is False


def test_tables_without_explicit_cb_are_skipped_not_inferred():
    html = """
    <html><body>
    <h2>Best WRs</h2>
    <table><tr><th>Wide Receiver</th><th>Team</th><th>Analysis</th></tr>
    <tr><td>Example WR</td><td>IND</td><td>Good matchup</td></tr></table>
    </body></html>
    """
    rows, audit = parse_page(
        html,
        season=2025,
        week=5,
        source_url="https://www.fantasyalarm.com/example",
    )
    assert rows.empty
    assert audit["rows_emitted"] == 0
    assert any(x["status"] == "SKIP_NO_EXPLICIT_WR_CB_COLUMNS" for x in audit["table_audit"])
