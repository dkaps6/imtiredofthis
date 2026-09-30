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



def test_parse_2024_combined_card_layout_without_html_table():
    html = """
    <html><head><meta property="article:published_time" content="2024-11-09T12:00:00-05:00"></head>
    <body>
      <h2>Left WR vs Right CB</h2>
      <div>Wide Receiver</div><div>Team</div><div>DK / FD $</div>
      <div>Cornerback</div><div>Opp</div><div>Matchup</div>
      <div>Michael Wilson</div><div>ARI</div><div>$4600 / $5500</div>
      <div>D.J. Reed</div><div>NYJ</div><div>Downgrade</div>
      <p>Long narrative that must never be parsed as a second matchup.</p>
    </body></html>
    """
    rows, audit = parse_page(
        html, season=2024, week=10,
        source_url="https://www.fantasyalarm.com/example-2024",
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["wr_raw"] == "Michael Wilson"
    assert row["cb_raw"] == "D.J. Reed"
    assert row["alignment_bucket"] == "LWR_VS_RCB"
    assert row["source_layout"] == "TEXT_COMBINED_SIX_FIELD"
    assert audit["rows_emitted"] == 1


def test_parse_2025_split_card_layout_without_html_table():
    html = """
    <html><head><meta property="article:published_time" content="2025-10-05T12:00:00-04:00"></head>
    <body>
      <h2>Left WR vs Right CB</h2>
      <div>Wide Receiver</div><div>Team</div><div>DK / FD PPG</div>
      <div>Marvin Harrison</div><div>ARI</div><div>12.2 / 10.2</div>
      <div>Cornerback</div><div>Opp</div><div>Matchup</div>
      <div>L'Jarius Sneed</div><div>TEN</div><div>Safe</div>
      <p>Narrative text.</p>
    </body></html>
    """
    rows, audit = parse_page(
        html, season=2025, week=5,
        source_url="https://www.fantasyalarm.com/example-2025",
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["wr_raw"] == "Marvin Harrison"
    assert row["cb_raw"] == "L'Jarius Sneed"
    assert row["alignment_bucket"] == "LWR_VS_RCB"
    assert row["source_layout"] == "TEXT_SPLIT_WR_CB_CARDS"
    assert bool(row["editorial_matchup_model_eligible"]) is False
    assert audit["rows_emitted"] == 1


def test_split_card_bye_row_is_missing_not_fake_assignment():
    html = """
    <html><body>
      <h2>Left WR vs Right CB</h2>
      <div>Wide Receiver</div><div>Team</div><div>DK / FD PPG</div>
      <div>Darnell Mooney</div><div>ATL</div>
      <div>Cornerback</div><div>Opp</div><div>Matchup</div>
      <div>N/A</div><div>N/A</div><div>N/A</div>
    </body></html>
    """
    rows, _ = parse_page(
        html, season=2025, week=5,
        source_url="https://www.fantasyalarm.com/example-bye",
    )
    assert rows.empty



def test_combined_stream_recovers_multiple_rows_after_one_header():
    html = """
    <html><body>
      <h2>Left WR vs Right CB</h2>
      <div>Wide Receiver</div><div>Team</div><div>DK / FD $</div>
      <div>Cornerback</div><div>Opp</div><div>Matchup</div>
      <div>Michael Wilson</div><div>ARI</div><div>$4600 / $5500</div>
      <div>D.J. Reed</div><div>NYJ</div><div>Downgrade</div>
      <p>Narrative text one.</p>
      <div>Darnell Mooney</div><div>ATL</div><div>$6500 / $7500</div>
      <div>Alontae Taylor</div><div>NO</div><div>Upgrade</div>
      <p>Narrative text two.</p>
      <div>Rashod Bateman</div><div>BAL</div><div>$4300 / $5400</div>
      <div>Cam Taylor-Britt</div><div>CIN</div><div>Upgrade</div>
    </body></html>
    """
    rows, _ = parse_page(
        html, season=2024, week=10,
        source_url="https://www.fantasyalarm.com/example-stream",
    )
    assert len(rows) == 3
    assert set(rows["wr_raw"]) == {"Michael Wilson", "Darnell Mooney", "Rashod Bateman"}
    assert set(rows["cb_raw"]) == {"D.J. Reed", "Alontae Taylor", "Cam Taylor-Britt"}
    assert rows["alignment_bucket"].eq("LWR_VS_RCB").all()



def test_parse_2021_embedded_team_table():
    html = """
    <html><body>
      <h2>Left WR vs Right CB</h2>
      <table>
        <thead><tr><th>Left WR</th><th>Right CB</th><th>Analysis</th></tr></thead>
        <tbody>
          <tr><td>DeAndre Hopkins ARI</td><td>Emmanuel Moseley SF</td><td>analysis</td></tr>
          <tr><td>Marquez Callaway NO</td><td>William Jackson WFT</td><td>analysis</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    rows, _ = parse_page(
        html, season=2021, week=5,
        source_url="https://www.fantasyalarm.com/example-2021",
    )
    assert len(rows) == 2
    assert rows["alignment_bucket"].eq("LWR_VS_RCB").all()
    assert set(rows["opponent"]) == {"SF", "WAS"}
    assert rows["source_layout"].eq("HTML_TABLE_2021_EMBEDDED_TEAM").all()


def test_parse_2026_inline_pair_layout():
    html = """
    <html><body>
      <h3>Left Wide Receiver (LWR) vs. Right Cornerback (RCB)</h3>
      <p>Each team's outside receiver on the left side.</p>
      <div>Marvin Harrison (ARI)</div>
      <div>vs. Deommodore Lenoir (SF) • Matchup: Risky</div>
      <p>Narrative.</p>
      <div>Drake London (ATL)</div>
      <div>vs. Brandon Cisse (GB) • Matchup: Moderate</div>
      <h3>Right Wide Receiver (RWR) vs. Left Cornerback (LCB)</h3>
      <div>Michael Wilson (ARI)</div>
      <div>vs. Renardo Green (SF) • Matchup: Risky</div>
    </body></html>
    """
    rows, _ = parse_page(
        html, season=2026, week=3,
        source_url="https://www.fantasyalarm.com/example-2026",
    )
    assert len(rows) == 3
    assert set(rows["alignment_bucket"]) == {"LWR_VS_RCB", "RWR_VS_LCB"}
    assert set(rows["wr_raw"]) == {"Marvin Harrison", "Drake London", "Michael Wilson"}
    assert rows["source_layout"].eq("TEXT_2026_INLINE_PAIR").all()
