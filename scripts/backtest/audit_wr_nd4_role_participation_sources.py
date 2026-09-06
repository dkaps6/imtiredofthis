#!/usr/bin/env python3
"""WR-ND4 frozen historical role/participation source audit.

Source audit only. No model fitting and no production projection changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import _key

EXPECTED_M38_N = 4647
EXPECTED_M38_MAE = 17.099904733366
EXPECTED_WR_ROWS = 2130
WR_POS = {"WR", "LWR", "RWR", "SWR"}


def _to_pandas(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def _lower(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    return _lower(pd.read_csv(path))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canon_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(canon_team)


def _parent_check(pred: pd.DataFrame) -> dict:
    x = pred.loc[pred["market"].astype(str).eq("rec_yards")].copy()
    x["actual"] = _num(x["actual"])
    x["mc_proj"] = _num(x["mc_proj"])
    x = x.loc[x["actual"].notna() & x["mc_proj"].notna()].copy()
    err = x["mc_proj"] - x["actual"]
    out = {
        "n": int(len(x)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "correlation": float(x["mc_proj"].corr(x["actual"])),
    }
    if out["n"] != EXPECTED_M38_N or abs(out["mae"] - EXPECTED_M38_MAE) > 1e-9:
        raise RuntimeError(f"M38 parent drift: {out}")
    return out


def _casebook(pred: pd.DataFrame, logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pred.loc[
        pred["market"].astype(str).eq("rec_yards")
        & pred["position"].fillna("").astype(str).str.upper().isin(WR_POS)
    ].copy()
    p["team"] = _canon_series(p["team"])
    p["player_clean_key"] = p["player_clean_key"].astype(str)

    l = logs.copy()
    l["season"] = _num(l["season"])
    l["week"] = _num(l["week"])
    l["team"] = _canon_series(l["team"])
    l["player_clean_key"] = l.get("player_clean_key", l.get("player", "")).map(_key)
    keep = [c for c in ["season", "week", "team", "player_clean_key", "player_id", "targets", "rec_yards"] if c in l.columns]
    l = l[keep].drop_duplicates(["season", "week", "team", "player_clean_key"], keep="last")
    x = p.merge(l, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    if x["player_id"].isna().any():
        bad = x.loc[x["player_id"].isna(), ["season", "week", "team", "player", "player_clean_key"]]
        raise RuntimeError(f"WR-ND4 missing canonical player_id:\n{bad.head(20).to_string(index=False)}")
    x["targets"] = _num(x["targets"]).fillna(0.0)
    x["rec_yards"] = _num(x["rec_yards"]).fillna(0.0)
    anomaly = x.loc[x["targets"].le(0) & x["rec_yards"].abs().gt(1e-9)].copy()
    if not anomaly.empty:
        anomaly["exclusion_reason"] = "NONZERO_REC_YARDS_WITH_ZERO_RECORDED_TARGETS"
        x = x.drop(index=anomaly.index).copy()
    x = x.reset_index(drop=True)
    if len(x) != EXPECTED_WR_ROWS:
        raise RuntimeError(f"WR-ND4 casebook drift: expected {EXPECTED_WR_ROWS}, got {len(x)}")
    return x, anomaly


def _player_map(players: pd.DataFrame) -> pd.DataFrame:
    p = _lower(players)
    gsis = "gsis_id" if "gsis_id" in p.columns else "player_id"
    pfr = "pfr_id" if "pfr_id" in p.columns else "pfr_player_id"
    if gsis not in p.columns or pfr not in p.columns:
        raise RuntimeError(f"player map missing GSIS/PFR columns: {list(p.columns)}")
    out = p[[gsis, pfr]].rename(columns={gsis: "player_id", pfr: "pfr_player_id"}).copy()
    out["player_id"] = out["player_id"].astype("string").str.strip()
    out["pfr_player_id"] = out["pfr_player_id"].astype("string").str.strip()
    out = out.dropna().loc[lambda d: d["player_id"].ne("") & d["pfr_player_id"].ne("")]
    return out.drop_duplicates("player_id", keep="last")


def _normalize_snaps(snaps: pd.DataFrame, pmap: pd.DataFrame) -> pd.DataFrame:
    s = _lower(snaps)
    required = {"season", "week", "team", "pfr_player_id", "offense_pct", "offense_snaps"}
    if not required.issubset(s.columns):
        raise RuntimeError(f"snap counts missing columns: {sorted(required - set(s.columns))}")
    s["season"] = _num(s["season"])
    s["week"] = _num(s["week"])
    s["team"] = _canon_series(s["team"])
    s["pfr_player_id"] = s["pfr_player_id"].astype("string").str.strip()
    s["offense_pct"] = _num(s["offense_pct"])
    s["offense_snaps"] = _num(s["offense_snaps"])
    if "game_type" in s.columns:
        reg = s["game_type"].fillna("").astype(str).str.upper().eq("REG")
        if reg.any():
            s = s.loc[reg].copy()
    inv = pmap[["player_id", "pfr_player_id"]].drop_duplicates("pfr_player_id", keep="last")
    s = s.merge(inv, on="pfr_player_id", how="left", validate="many_to_one")
    return s


def _attach_snap_priors(casebook: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in casebook.iterrows():
        prior = snaps.loc[
            snaps["player_id"].astype(str).eq(str(r["player_id"]))
            & snaps["team"].eq(r["team"])
            & (
                snaps["season"].lt(float(r["season"]))
                | (snaps["season"].eq(float(r["season"])) & snaps["week"].lt(float(r["week"])))
            )
        ].sort_values(["season", "week"])
        pct = prior["offense_pct"].dropna()
        rows.append(
            {
                "prior_snap_games": int(len(pct)),
                "prior1_offense_pct": float(pct.iloc[-1]) if len(pct) >= 1 else np.nan,
                "prior2_mean_offense_pct": float(pct.tail(2).mean()) if len(pct) >= 2 else np.nan,
                "prior4_mean_offense_pct": float(pct.tail(4).mean()) if len(pct) >= 4 else np.nan,
            }
        )
    return pd.concat([casebook.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _coverage(frame: pd.DataFrame, column: str, mask: pd.Series) -> float:
    sub = frame.loc[mask, column]
    return float(sub.notna().mean()) if len(sub) else np.nan


def _snap_audit(casebook: pd.DataFrame, snaps: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x = _attach_snap_priors(casebook, snaps)
    phase = pd.DataFrame(
        [
            {
                "slice": "ALL",
                "n": int(len(x)),
                "prior1_coverage": _coverage(x, "prior1_offense_pct", pd.Series(True, index=x.index)),
                "prior2_coverage": _coverage(x, "prior2_mean_offense_pct", pd.Series(True, index=x.index)),
                "prior4_coverage": _coverage(x, "prior4_mean_offense_pct", pd.Series(True, index=x.index)),
            },
            {
                "slice": "W2_18",
                "n": int(x["week"].ge(2).sum()),
                "prior1_coverage": _coverage(x, "prior1_offense_pct", x["week"].ge(2)),
                "prior2_coverage": _coverage(x, "prior2_mean_offense_pct", x["week"].ge(2)),
                "prior4_coverage": _coverage(x, "prior4_mean_offense_pct", x["week"].ge(2)),
            },
            {
                "slice": "W13_18",
                "n": int(x["week"].ge(13).sum()),
                "prior1_coverage": _coverage(x, "prior1_offense_pct", x["week"].ge(13)),
                "prior2_coverage": _coverage(x, "prior2_mean_offense_pct", x["week"].ge(13)),
                "prior4_coverage": _coverage(x, "prior4_mean_offense_pct", x["week"].ge(13)),
            },
            {
                "slice": "WEEK1",
                "n": int(x["week"].eq(1).sum()),
                "prior1_coverage": _coverage(x, "prior1_offense_pct", x["week"].eq(1)),
                "prior2_coverage": _coverage(x, "prior2_mean_offense_pct", x["week"].eq(1)),
                "prior4_coverage": _coverage(x, "prior4_mean_offense_pct", x["week"].eq(1)),
            },
        ]
    )
    by_week = (
        x.groupby("week", as_index=False)
        .agg(
            n=("player_id", "size"),
            prior1_coverage=("prior1_offense_pct", lambda s: float(s.notna().mean())),
            prior2_coverage=("prior2_mean_offense_pct", lambda s: float(s.notna().mean())),
            prior4_coverage=("prior4_mean_offense_pct", lambda s: float(s.notna().mean())),
        )
    )
    mapped = snaps["player_id"].notna()
    mapped_nonnull_pct = float(snaps.loc[mapped, "offense_pct"].notna().mean()) if mapped.any() else 0.0
    w2 = phase.loc[phase["slice"].eq("W2_18")].iloc[0]
    late = phase.loc[phase["slice"].eq("W13_18")].iloc[0]
    passed = bool(
        w2["prior1_coverage"] >= 0.85
        and w2["prior2_coverage"] >= 0.80
        and late["prior1_coverage"] >= 0.90
        and mapped_nonnull_pct >= 0.98
    )
    summary = {
        "loaded_rows": int(len(snaps)),
        "mapped_rows": int(mapped.sum()),
        "mapped_row_rate": float(mapped.mean()) if len(snaps) else 0.0,
        "mapped_nonnull_offense_pct_rate": mapped_nonnull_pct,
        "w2_18_prior1_coverage": float(w2["prior1_coverage"]),
        "w2_18_prior2_coverage": float(w2["prior2_coverage"]),
        "w13_18_prior1_coverage": float(late["prior1_coverage"]),
        "gate_passed": passed,
    }
    return summary, phase, by_week


def _schedule_team_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    s["season"] = _num(s["season"])
    s["week"] = _num(s["week"])
    date_col = next((c for c in ["gameday", "game_date", "date"] if c in s.columns), None)
    if date_col is None:
        raise RuntimeError(f"schedule missing game-date column; columns={list(s.columns)}")
    s["game_date"] = pd.to_datetime(s[date_col], errors="coerce").dt.date
    away_col = "away_team" if "away_team" in s.columns else "away"
    home_col = "home_team" if "home_team" in s.columns else "home"
    if away_col not in s.columns or home_col not in s.columns:
        raise RuntimeError("schedule missing away/home team columns")
    rows = []
    for _, r in s.iterrows():
        for team in [r[away_col], r[home_col]]:
            rows.append({"season": int(r["season"]), "week": int(r["week"]), "team": canon_team(team), "game_date": r["game_date"]})
    return pd.DataFrame(rows).drop_duplicates(["season", "week", "team"])


def _normalize_depth(depth: pd.DataFrame) -> pd.DataFrame:
    d = _lower(depth)
    if "dt" not in d.columns:
        raise RuntimeError(f"2025 depth chart lacks timestamp `dt`; columns={list(d.columns)}")
    if "team" not in d.columns or "gsis_id" not in d.columns:
        raise RuntimeError("2025 depth chart missing team/gsis_id")
    d["team"] = _canon_series(d["team"])
    d["gsis_id"] = d["gsis_id"].astype("string").str.strip()
    d["dt_parsed"] = pd.to_datetime(d["dt"], errors="coerce", utc=True)
    d["snapshot_date"] = d["dt_parsed"].dt.date
    for c in ["pos_rank", "pos_slot"]:
        if c in d.columns:
            d[c] = _num(d[c])
        else:
            d[c] = np.nan
    return d.loc[d["dt_parsed"].notna() & d["team"].ne("")].copy()


def _attach_depth(casebook: pd.DataFrame, schedule: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    dates = _schedule_team_dates(schedule)
    x = casebook.merge(dates, on=["season", "week", "team"], how="left", validate="many_to_one")
    if x["game_date"].isna().any():
        raise RuntimeError("missing game_date on WR-ND4 casebook")
    out_rows = []
    for _, r in x.iterrows():
        eligible = depth.loc[depth["team"].eq(r["team"]) & depth["snapshot_date"].lt(r["game_date"])].copy()
        if eligible.empty:
            out_rows.append({"depth_snapshot_dt": pd.NaT, "depth_matched": 0, "depth_pos_rank": np.nan, "depth_pos_slot": np.nan, "depth_pos_abb": ""})
            continue
        latest_dt = eligible["dt_parsed"].max()
        snap = eligible.loc[eligible["dt_parsed"].eq(latest_dt)].copy()
        hit = snap.loc[snap["gsis_id"].astype(str).eq(str(r["player_id"]))].copy()
        if hit.empty:
            out_rows.append({"depth_snapshot_dt": latest_dt, "depth_matched": 0, "depth_pos_rank": np.nan, "depth_pos_slot": np.nan, "depth_pos_abb": ""})
            continue
        rank = hit["pos_rank"].dropna().min() if hit["pos_rank"].notna().any() else np.nan
        slot = hit["pos_slot"].dropna().min() if hit["pos_slot"].notna().any() else np.nan
        pos_abb = str(hit.get("pos_abb", pd.Series("", index=hit.index)).dropna().astype(str).iloc[0]) if len(hit) else ""
        out_rows.append({"depth_snapshot_dt": latest_dt, "depth_matched": 1, "depth_pos_rank": rank, "depth_pos_slot": slot, "depth_pos_abb": pos_abb})
    return pd.concat([x.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


def _depth_audit(casebook: pd.DataFrame, schedule: pd.DataFrame, depth: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x = _attach_depth(casebook, schedule, depth)
    x["depth_covered"] = x["depth_matched"].eq(1)
    x["depth_rank_usable"] = x["depth_covered"] & x["depth_pos_rank"].notna()
    snap_dates = pd.to_datetime(x["depth_snapshot_dt"], errors="coerce", utc=True).dt.date
    x["timestamp_violation"] = snap_dates.notna() & snap_dates.ge(x["game_date"])

    def row(name: str, mask: pd.Series) -> dict:
        sub = x.loc[mask]
        return {
            "slice": name,
            "n": int(len(sub)),
            "depth_coverage": float(sub["depth_covered"].mean()) if len(sub) else np.nan,
            "rank_usable_among_matched": float(sub.loc[sub["depth_covered"], "depth_rank_usable"].mean()) if sub["depth_covered"].any() else np.nan,
            "timestamp_violations": int(sub["timestamp_violation"].sum()),
        }

    phase = pd.DataFrame([
        row("ALL", pd.Series(True, index=x.index)),
        row("W2_18", x["week"].ge(2)),
        row("W13_18", x["week"].ge(13)),
        row("WEEK1", x["week"].eq(1)),
    ])
    by_week = x.groupby("week", as_index=False).agg(
        n=("player_id", "size"),
        depth_coverage=("depth_covered", "mean"),
        rank_usable_rate=("depth_rank_usable", "mean"),
        timestamp_violations=("timestamp_violation", "sum"),
    )
    allr = phase.loc[phase["slice"].eq("ALL")].iloc[0]
    w2 = phase.loc[phase["slice"].eq("W2_18")].iloc[0]
    late = phase.loc[phase["slice"].eq("W13_18")].iloc[0]
    rank_usable_matched = float(x.loc[x["depth_covered"], "depth_rank_usable"].mean()) if x["depth_covered"].any() else 0.0
    passed = bool(
        allr["depth_coverage"] >= 0.85
        and w2["depth_coverage"] >= 0.90
        and late["depth_coverage"] >= 0.90
        and rank_usable_matched >= 0.95
        and int(x["timestamp_violation"].sum()) == 0
    )
    summary = {
        "loaded_rows": int(len(depth)),
        "all_depth_coverage": float(allr["depth_coverage"]),
        "w2_18_depth_coverage": float(w2["depth_coverage"]),
        "w13_18_depth_coverage": float(late["depth_coverage"]),
        "rank_usable_among_matched": rank_usable_matched,
        "timestamp_violations": int(x["timestamp_violation"].sum()),
        "gate_passed": passed,
    }
    return summary, phase, by_week


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_nd4_role_participation_source_audit"))
    args = ap.parse_args()

    pred = _read(args.predictions, "component predictions")
    logs = _read(args.player_logs, "player logs")
    schedule = _read(args.schedule, "schedule")
    parent = _parent_check(pred)
    casebook, anomaly = _casebook(pred, logs)

    import nflreadpy as nfl

    players = _to_pandas(nfl.load_players())
    pmap = _player_map(players)
    snaps_raw = _to_pandas(nfl.load_snap_counts(seasons=[2024, 2025]))
    snaps = _normalize_snaps(snaps_raw, pmap)
    snap_summary, snap_phase, snap_week = _snap_audit(casebook, snaps)

    depth_raw = _to_pandas(nfl.load_depth_charts(seasons=[2025]))
    depth = _normalize_depth(depth_raw)
    depth_summary, depth_phase, depth_week = _depth_audit(casebook, schedule, depth)

    snap_pass = bool(snap_summary["gate_passed"])
    depth_pass = bool(depth_summary["gate_passed"])
    if snap_pass and depth_pass:
        disposition = "SNAP_AND_DEPTH_SOURCES_RECOVERED"
    elif snap_pass:
        disposition = "SNAP_SOURCE_RECOVERED"
    elif depth_pass:
        disposition = "DEPTH_SOURCE_RECOVERED"
    else:
        disposition = "ROLE_PARTICIPATION_SOURCE_BLOCKED"

    mapped_case = casebook.merge(pmap, on="player_id", how="left", validate="many_to_one")
    id_audit = {
        "casebook_rows": int(len(casebook)),
        "casebook_gsis_to_pfr_coverage": float(mapped_case["pfr_player_id"].notna().mean()),
        "casebook_unique_gsis": int(casebook["player_id"].nunique()),
        "casebook_unique_pfr_mapped": int(mapped_case["pfr_player_id"].dropna().nunique()),
    }
    schemas = pd.DataFrame([
        {"source": "snap_counts", "source_time_status": "ELIGIBLE_PRIOR_GAME", "columns": "|".join(map(str, snaps_raw.columns))},
        {"source": "depth_charts_2025", "source_time_status": "ELIGIBLE_STRICT_PRE_GAME_DATE", "columns": "|".join(map(str, depth_raw.columns))},
        {"source": "participation_2025", "source_time_status": "INELIGIBLE_POSTSEASON_RELEASE", "columns": "documented play-level offense_players; not loaded into canonical gate"},
    ])

    result = {
        "migration": "WR-ND4",
        "m38_parent_check": parent,
        "evaluation_rows": int(len(casebook)),
        "factorization_anomalies_excluded": int(len(anomaly)),
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "snap_source": snap_summary,
        "depth_source": depth_summary,
        "id_audit": id_audit,
        "participation_2025": {
            "canonical_gate_eligible": False,
            "reason": "2023+ nflverse participation is released after postseason; source-time unsafe for same-season 2025 walk-forward",
        },
        "frozen_gates": {
            "snap": {"w2_18_prior1": 0.85, "w2_18_prior2": 0.80, "w13_18_prior1": 0.90, "mapped_nonnull_pct": 0.98},
            "depth": {"all": 0.85, "w2_18": 0.90, "w13_18": 0.90, "rank_usable_matched": 0.95, "timestamp_violations": 0},
        },
        "disposition": disposition,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    schemas.to_csv(args.out_dir / "wr_nd4_source_schema_inventory.csv", index=False)
    snap_phase.to_csv(args.out_dir / "wr_nd4_snap_phase_coverage.csv", index=False)
    snap_week.to_csv(args.out_dir / "wr_nd4_snap_week_coverage.csv", index=False)
    depth_phase.to_csv(args.out_dir / "wr_nd4_depth_phase_coverage.csv", index=False)
    depth_week.to_csv(args.out_dir / "wr_nd4_depth_week_coverage.csv", index=False)
    anomaly.to_csv(args.out_dir / "wr_nd4_factorization_anomalies.csv", index=False)
    (args.out_dir / "wr_nd4_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[wr-nd4] snap summary")
    print(json.dumps(snap_summary, indent=2, sort_keys=True))
    print("[wr-nd4] depth summary")
    print(json.dumps(depth_summary, indent=2, sort_keys=True))
    print("[wr-nd4] result")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
