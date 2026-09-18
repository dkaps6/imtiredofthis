import json
from pathlib import Path

import pandas as pd

from scripts.research.build_historical_base_manifest_v1 import build_manifest


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path,index=False)


def test_historical_base_manifest_fingerprints_and_seasons(tmp_path):
    player=tmp_path/"player_game_logs_history.csv"
    team=tmp_path/"team_weekly_history.csv"
    sched=tmp_path/"schedule_history.csv"
    out=tmp_path/"manifest.json"

    _write(player,[
        {"season":2024,"week":1,"team":"BUF","player_clean_key":"a"},
        {"season":2025,"week":1,"team":"BUF","player_clean_key":"a"},
    ])
    _write(team,[
        {"season":2024,"week":1,"team":"BUF","plays":60},
        {"season":2025,"week":1,"team":"BUF","plays":62},
    ])
    _write(sched,[
        {"season":2024,"week":1,"team":"BUF","opponent":"ARI"},
        {"season":2025,"week":1,"team":"BUF","opponent":"BAL"},
    ])

    result=build_manifest(
        player_logs=player,
        team_weekly=team,
        schedule=sched,
        output=out,
        builder_commit="f"*40,
    )

    assert result["manifest_version"]=="HISTORICAL_BASE_MANIFEST_V1"
    assert result["common_seasons"]==[2024,2025]
    assert result["full_row_data_committed_to_git"] is False
    assert result["science_reopened"] is False
    for table in result["tables"].values():
        assert len(table["sha256"])==64
        assert table["duplicate_key_rows"]==0

    saved=json.loads(out.read_text())
    assert saved["reuse_policy"].startswith("REUSE_EXACT_ARTIFACT")
