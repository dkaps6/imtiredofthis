"""Build a deterministic manifest for canonical historical NFL base tables.

This does not download or rebuild source data. It fingerprints already-materialized
historical base tables so future research can reuse exact historical assets when
available and only rehydrate when necessary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

MANIFEST_VERSION = "HISTORICAL_BASE_MANIFEST_V1"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table_manifest(path: Path, *, key_columns: Iterable[str] = ()) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)

    df=pd.read_csv(path)
    cols=[str(c) for c in df.columns]

    seasons=[]
    weeks=[]
    if "season" in df.columns:
        seasons=sorted(int(x) for x in pd.to_numeric(df["season"],errors="coerce").dropna().unique())
    if "week" in df.columns:
        weeks=sorted(int(x) for x in pd.to_numeric(df["week"],errors="coerce").dropna().unique())

    duplicate_key_rows=None
    keys=[k for k in key_columns if k in df.columns]
    if keys:
        duplicate_key_rows=int(df.duplicated(keys, keep=False).sum())

    return {
        "file_name":path.name,
        "path":str(path),
        "bytes":int(path.stat().st_size),
        "sha256":sha256_file(path),
        "rows":int(len(df)),
        "columns":cols,
        "column_count":len(cols),
        "seasons":seasons,
        "weeks":weeks,
        "key_columns_checked":keys,
        "duplicate_key_rows":duplicate_key_rows,
    }


def build_manifest(
    *,
    player_logs: Path,
    team_weekly: Path,
    schedule: Path,
    output: Path,
    builder_commit: str,
    source_lineage: str = "canonical repo historical builders",
) -> dict:
    tables={
        "player_game_logs_history":table_manifest(
            player_logs,
            key_columns=("season","week","team","player_clean_key"),
        ),
        "team_weekly_history":table_manifest(
            team_weekly,
            key_columns=("season","week","team"),
        ),
        "schedule_history":table_manifest(
            schedule,
            key_columns=("season","week","team"),
        ),
    }

    common_seasons=None
    for x in tables.values():
        s=set(x["seasons"])
        common_seasons=s if common_seasons is None else common_seasons & s
    common_seasons=sorted(common_seasons or [])

    payload={
        "manifest_version":MANIFEST_VERSION,
        "status":"CANONICAL_HISTORICAL_BASE_FINGERPRINT",
        "builder_commit":builder_commit,
        "source_lineage":source_lineage,
        "tables":tables,
        "common_seasons":common_seasons,
        "full_row_data_committed_to_git":False,
        "reuse_policy":"REUSE_EXACT_ARTIFACT_IF_AVAILABLE_ELSE_REHYDRATE_WITH_FROZEN_CANONICAL_BUILDER",
        "science_reopened":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--player-logs",type=Path,required=True)
    p.add_argument("--team-weekly",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--builder-commit",required=True)
    p.add_argument("--source-lineage",default="canonical repo historical builders")
    a=p.parse_args()
    result=build_manifest(
        player_logs=a.player_logs,
        team_weekly=a.team_weekly,
        schedule=a.schedule,
        output=a.out,
        builder_commit=a.builder_commit,
        source_lineage=a.source_lineage,
    )
    print(json.dumps({
        "manifest_version":result["manifest_version"],
        "common_seasons":result["common_seasons"],
        "tables":{k:{"rows":v["rows"],"sha256":v["sha256"]} for k,v in result["tables"].items()},
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
