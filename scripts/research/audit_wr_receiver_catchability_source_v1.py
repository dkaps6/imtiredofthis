#!/usr/bin/env python3
"""Source-only audit for a possible receiver-specific FTN catchability lane.

No WR outcome/projection artifact is loaded. This only verifies whether nflverse FTN
charting can be joined by exact game/play identity to nflverse PBP target-receiver
GSIS IDs for 2022-2024, with enough catchability population for honest strictly-prior
receiver histories.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SEASONS = (2022, 2023, 2024)
REG_WEEKS = set(range(1, 19))
MIN_EXACT_JOIN = 0.95
MIN_TARGET_CATCHABLE_COVERAGE = 0.80


def _read_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "wr-catchability-source-audit-v1"})
    with urlopen(req, timeout=180) as r:
        raw = r.read()
        final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> int:
    out_dir = Path("data/backtests/wr_receiver_catchability_source_audit_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for season in SEASONS:
        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        ftn, fmeta = _read_parquet(ftn_url)
        pbp, pmeta = _read_parquet(pbp_url)
        ftn.columns = [str(c).strip().lower() for c in ftn.columns]
        pbp.columns = [str(c).strip().lower() for c in pbp.columns]

        req_f = {"nflverse_game_id", "nflverse_play_id", "season", "week", "is_catchable_ball"}
        req_p = {"game_id", "play_id", "season", "week", "season_type", "pass_attempt", "receiver_player_id", "receiver_player_name", "posteam"}
        miss_f = sorted(req_f - set(ftn.columns))
        miss_p = sorted(req_p - set(pbp.columns))
        if miss_f or miss_p:
            raise RuntimeError(f"schema missing season={season}: ftn={miss_f}, pbp={miss_p}")

        ftn = ftn.loc[_num(ftn["week"]).between(1, 18)].copy()
        pbp = pbp.loc[
            pbp["season_type"].astype(str).str.upper().eq("REG")
            & _num(pbp["week"]).between(1, 18)
        ].copy()

        ftn["join_game"] = ftn["nflverse_game_id"].astype(str)
        ftn["join_play"] = _num(ftn["nflverse_play_id"])
        pbp["join_game"] = pbp["game_id"].astype(str)
        pbp["join_play"] = _num(pbp["play_id"])

        if ftn.duplicated(["join_game", "join_play"]).any():
            raise RuntimeError(f"duplicate FTN game/play key season={season}")
        if pbp.duplicated(["join_game", "join_play"]).any():
            raise RuntimeError(f"duplicate PBP game/play key season={season}")

        p = pbp[["join_game", "join_play", "pass_attempt", "receiver_player_id", "receiver_player_name", "posteam"]].copy()
        merged = ftn.merge(p, on=["join_game", "join_play"], how="left", validate="one_to_one", indicator=True)
        exact_join_rate = float(merged["_merge"].eq("both").mean()) if len(merged) else 0.0

        pass_attempt = _num(merged["pass_attempt"]).fillna(0).eq(1)
        receiver_id = merged["receiver_player_id"].astype("string").str.strip()
        has_receiver = receiver_id.notna() & receiver_id.ne("") & receiver_id.str.lower().ne("nan")
        target_rows = merged.loc[pass_attempt & has_receiver].copy()
        catchable_nonnull = target_rows["is_catchable_ball"].notna()
        catchable_coverage = float(catchable_nonnull.mean()) if len(target_rows) else 0.0

        ftn_weeks = sorted(_num(ftn["week"]).dropna().astype(int).unique().tolist())
        pbp_weeks = sorted(_num(pbp["week"]).dropna().astype(int).unique().tolist())
        rows.append({
            "season": season,
            "ftn_rows_reg": int(len(ftn)),
            "pbp_rows_reg": int(len(pbp)),
            "ftn_weeks_1_18_complete": REG_WEEKS.issubset(set(ftn_weeks)),
            "pbp_weeks_1_18_complete": REG_WEEKS.issubset(set(pbp_weeks)),
            "exact_ftn_pbp_join_rate": exact_join_rate,
            "receiver_target_rows": int(len(target_rows)),
            "receiver_target_rows_with_catchable": int(catchable_nonnull.sum()),
            "receiver_target_catchable_coverage": catchable_coverage,
            "unique_target_receiver_ids": int(target_rows.loc[catchable_nonnull, "receiver_player_id"].astype(str).nunique()),
            "ftn_sha256": fmeta["sha256"],
            "ftn_bytes": fmeta["bytes"],
            "pbp_sha256": pmeta["sha256"],
            "pbp_bytes": pmeta["bytes"],
        })

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "wr_receiver_catchability_source_audit.csv", index=False)

    source_pass = bool(
        report["ftn_weeks_1_18_complete"].all()
        and report["pbp_weeks_1_18_complete"].all()
        and report["exact_ftn_pbp_join_rate"].ge(MIN_EXACT_JOIN).all()
        and report["receiver_target_catchable_coverage"].ge(MIN_TARGET_CATCHABLE_COVERAGE).all()
        and report["receiver_target_rows"].gt(0).all()
    )
    summary = {
        "audit": "WR_RECEIVER_CATCHABILITY_SOURCE_AUDIT_V1",
        "seasons": list(SEASONS),
        "source_only": True,
        "wr_outcomes_loaded": False,
        "wr_projection_artifact_loaded": False,
        "sportsbook_inputs": 0,
        "minimum_exact_join_rate": MIN_EXACT_JOIN,
        "minimum_receiver_target_catchable_coverage": MIN_TARGET_CATCHABLE_COVERAGE,
        "source_contract_pass": source_pass,
        "receiver_identity_contract": "FTN nflverse_game_id+nflverse_play_id -> PBP game_id+play_id -> receiver_player_id (GSIS)",
    }
    (out_dir / "wr_receiver_catchability_source_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if not source_pass:
        raise SystemExit("receiver catchability source contract did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
