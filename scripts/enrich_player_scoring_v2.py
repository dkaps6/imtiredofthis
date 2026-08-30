#!/usr/bin/env python3
"""Enrich PlayerForm v2 with real historical red-zone and touchdown usage.

Player Identity v3 contract:
- stable nflverse/GSIS receiver/rusher IDs are authoritative when available;
- PBP names are only aliases used to bridge rows without a stable ID;
- prior/current scoring evidence joins PlayerForm on ``player_identity_key``;
- the PlayerForm prior/current split is preserved for Bayesian shrinkage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_slate_date, resolve_week
from scripts.utils.pbp import get_pbp
from scripts.utils.player_identity_v3 import build_identity_registry, resolve_slate_identities

DATA = Path("data")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _resolve_pbp_players(
    frame: pd.DataFrame,
    *,
    name_col: str,
    id_candidates: tuple[str, ...],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve PBP participant rows to the same identity grain as PlayerForm."""
    if frame.empty:
        return frame
    out = frame.copy()
    out["player"] = out[name_col].astype("string").fillna("").str.strip()
    id_col = next((c for c in id_candidates if c in out.columns), None)
    if id_col:
        out["provider_player_id"] = out[id_col].astype("string").fillna("").str.strip()
    else:
        out["provider_player_id"] = ""
    return resolve_slate_identities(
        out,
        registry,
        name_col="player",
        team_col="team",
        provider_id_col="provider_player_id",
        strict_ambiguous=True,
        allow_temporary=True,
    )


def _season_scoring(
    season: int,
    *,
    before_week: int | None = None,
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pbp = get_pbp(int(season), min_rows=1).copy()
    pbp.columns = [str(c).lower() for c in pbp.columns]
    if "season_type" in pbp.columns:
        reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")]
        if not reg.empty:
            pbp = reg.copy()
    if before_week is not None and "week" in pbp.columns:
        w = pd.to_numeric(pbp["week"], errors="coerce")
        pbp = pbp.loc[w.lt(int(before_week))].copy()
    if pbp.empty:
        return pd.DataFrame()

    registry = registry if registry is not None else pd.DataFrame()
    pbp["team"] = pbp.get("posteam", pd.Series("", index=pbp.index)).map(canon_team)
    yardline = pd.to_numeric(
        pbp.get("yardline_100", pd.Series(np.nan, index=pbp.index)), errors="coerce"
    )
    pass_attempt = _num(pbp, "pass_attempt").eq(1)
    rush_attempt = _num(pbp, "rush_attempt").eq(1)

    rec_col = "receiver_player_name" if "receiver_player_name" in pbp.columns else None
    rush_col = "rusher_player_name" if "rusher_player_name" in pbp.columns else None
    rows: list[pd.DataFrame] = []

    if rec_col:
        rec = pbp.loc[pass_attempt & pbp[rec_col].notna()].copy()
        rec = _resolve_pbp_players(
            rec,
            name_col=rec_col,
            id_candidates=("receiver_player_id", "receiver_id", "receiver_gsis_id"),
            registry=registry,
        )
        rec["rz_target"] = (
            pd.to_numeric(rec.get("yardline_100"), errors="coerce") <= 20
        ).astype(int)
        rec["rec_td"] = _num(rec, "pass_touchdown").eq(1).astype(int)
        agg = rec.groupby(["team", "player_identity_key"], dropna=False).agg(
            rz_targets=("rz_target", "sum"),
            rec_tds=("rec_td", "sum"),
        ).reset_index()
        rows.append(agg)

    if rush_col:
        rush = pbp.loc[rush_attempt & pbp[rush_col].notna()].copy()
        rush = _resolve_pbp_players(
            rush,
            name_col=rush_col,
            id_candidates=("rusher_player_id", "rusher_id", "rusher_gsis_id"),
            registry=registry,
        )
        rush["rz_rush"] = (
            pd.to_numeric(rush.get("yardline_100"), errors="coerce") <= 20
        ).astype(int)
        rush["rush_td"] = _num(rush, "rush_touchdown").eq(1).astype(int)
        agg = rush.groupby(["team", "player_identity_key"], dropna=False).agg(
            rz_rushes=("rz_rush", "sum"),
            rush_tds=("rush_td", "sum"),
        ).reset_index()
        rows.append(agg)

    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for part in rows[1:]:
        out = out.merge(part, on=["team", "player_identity_key"], how="outer")
    for c in ("rz_targets", "rec_tds", "rz_rushes", "rush_tds"):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    team_rz_targets = (
        pbp.loc[pass_attempt & yardline.le(20)].groupby("team").size().rename("team_rz_targets")
    )
    team_rz_rushes = (
        pbp.loc[rush_attempt & yardline.le(20)].groupby("team").size().rename("team_rz_rushes")
    )
    out = out.merge(team_rz_targets, on="team", how="left").merge(
        team_rz_rushes, on="team", how="left"
    )
    out["rz_tgt_share"] = np.where(
        out["team_rz_targets"] > 0,
        out["rz_targets"] / out["team_rz_targets"],
        np.nan,
    )
    out["rz_carry_share"] = np.where(
        out["team_rz_rushes"] > 0,
        out["rz_rushes"] / out["team_rz_rushes"],
        np.nan,
    )
    out["offensive_tds"] = out["rec_tds"] + out["rush_tds"]
    out["season"] = int(season)
    return out


def _blend_metric(prior, current, current_games):
    p = pd.to_numeric(prior, errors="coerce")
    c = pd.to_numeric(current, errors="coerce")
    games = pd.to_numeric(current_games, errors="coerce").fillna(0.0)
    weight = games / (games + 4.0)
    result = p.copy()
    only_current = c.notna() & p.isna()
    both = c.notna() & p.notna()
    result.loc[only_current] = c.loc[only_current]
    result.loc[both] = (1.0 - weight.loc[both]) * p.loc[both] + weight.loc[both] * c.loc[both]
    return result


def enrich(season: int, prior_season: int, week: int) -> pd.DataFrame:
    form_path = DATA / "player_form_consensus.csv"
    logs_path = DATA / "player_game_logs.csv"
    if not form_path.exists() or not logs_path.exists():
        raise RuntimeError("PlayerForm v2 outputs must exist before scoring enrichment")
    form = pd.read_csv(form_path)
    logs = pd.read_csv(logs_path)
    form.columns = [str(c).lower() for c in form.columns]
    logs.columns = [str(c).lower() for c in logs.columns]

    if "player_identity_key" not in form.columns or "player_identity_key" not in logs.columns:
        raise RuntimeError(
            "Player Identity v3 is required before scoring enrichment; missing player_identity_key"
        )

    # Build the alias/ID registry only from information legitimately available
    # before the target week.
    eligible_logs = logs.loc[
        pd.to_numeric(logs.get("season"), errors="coerce").eq(int(prior_season))
        | (
            pd.to_numeric(logs.get("season"), errors="coerce").eq(int(season))
            & pd.to_numeric(logs.get("week"), errors="coerce").lt(int(week))
        )
    ].copy()
    registry = build_identity_registry(eligible_logs)

    # PlayerForm creates these columns specifically for downstream shrinkage.
    # Do not silently erase them during scoring enrichment.
    player_evidence_cols = [
        f"{metric}_{era}"
        for metric in (
            "tgt_share", "rush_share", "route_rate", "yprr", "ypt", "ypc", "ypa",
            "receptions_per_target",
        )
        for era in ("prior", "current")
    ]
    evidence_present_before = {c for c in player_evidence_cols if c in form.columns}

    prior = _season_scoring(prior_season, registry=registry)
    try:
        current = _season_scoring(season, before_week=week, registry=registry)
    except Exception as exc:
        print(f"[player_scoring] current-season PBP unavailable; using prior only ({exc})")
        current = pd.DataFrame()

    if prior.empty:
        raise RuntimeError(f"Historical scoring PBP produced no rows for prior season {prior_season}")

    join_key = "player_identity_key"
    prior_keep = [join_key, "rz_tgt_share", "rz_carry_share", "offensive_tds"]
    p = prior[prior_keep].groupby(join_key, as_index=False).agg(
        rz_tgt_share=("rz_tgt_share", "mean"),
        rz_carry_share=("rz_carry_share", "mean"),
        offensive_tds=("offensive_tds", "sum"),
    ).rename(
        columns={c: f"{c}_prior" for c in ("rz_tgt_share", "rz_carry_share", "offensive_tds")}
    )

    if current.empty:
        c = pd.DataFrame(
            columns=[
                join_key,
                "rz_tgt_share_current",
                "rz_carry_share_current",
                "offensive_tds_current",
            ]
        )
    else:
        c = current[prior_keep].groupby(join_key, as_index=False).agg(
            rz_tgt_share=("rz_tgt_share", "mean"),
            rz_carry_share=("rz_carry_share", "mean"),
            offensive_tds=("offensive_tds", "sum"),
        ).rename(
            columns={x: f"{x}_current" for x in ("rz_tgt_share", "rz_carry_share", "offensive_tds")}
        )

    current_logs = logs.loc[
        pd.to_numeric(logs.get("season"), errors="coerce").eq(int(season))
        & pd.to_numeric(logs.get("week"), errors="coerce").lt(int(week))
    ].copy()
    games = (
        current_logs.groupby(join_key)["week"]
        .nunique()
        .rename("scoring_current_games")
        .reset_index()
        if not current_logs.empty
        else pd.DataFrame(columns=[join_key, "scoring_current_games"])
    )

    out = (
        form.merge(p, on=join_key, how="left", validate="many_to_one")
        .merge(c, on=join_key, how="left", validate="many_to_one")
        .merge(games, on=join_key, how="left", validate="many_to_one")
    )
    out["rz_tgt_share"] = _blend_metric(
        out["rz_tgt_share_prior"], out.get("rz_tgt_share_current"), out["scoring_current_games"]
    )
    out["rz_carry_share"] = _blend_metric(
        out["rz_carry_share_prior"], out.get("rz_carry_share_current"), out["scoring_current_games"]
    )
    out["rz_rush_share"] = out["rz_carry_share"]
    out["rz_share"] = out[["rz_tgt_share", "rz_carry_share"]].max(axis=1, skipna=True)

    prior_games = pd.to_numeric(out.get("prior_games"), errors="coerce").replace(0, np.nan)
    current_games = pd.to_numeric(out["scoring_current_games"], errors="coerce").replace(0, np.nan)
    prior_rate = pd.to_numeric(out["offensive_tds_prior"], errors="coerce") / prior_games
    current_rate = pd.to_numeric(out.get("offensive_tds_current"), errors="coerce") / current_games
    out["offensive_td_rate"] = _blend_metric(
        prior_rate, current_rate, out["scoring_current_games"]
    )

    # Remove only helper columns introduced above.  Preserve PlayerForm's raw
    # prior/current evidence and the stable identity columns.
    scoring_helper_cols = [
        "rz_tgt_share_prior", "rz_tgt_share_current",
        "rz_carry_share_prior", "rz_carry_share_current",
        "offensive_tds_prior", "offensive_tds_current",
        "scoring_current_games",
    ]
    out.drop(columns=scoring_helper_cols, inplace=True, errors="ignore")

    evidence_missing_after = sorted(evidence_present_before - set(out.columns))
    if evidence_missing_after:
        raise RuntimeError(f"Scoring enrichment erased PlayerForm evidence columns: {evidence_missing_after}")
    if out["player_identity_key"].isna().any():
        raise RuntimeError("Scoring enrichment lost stable player identity")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--prior-season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    prior = int(args.prior_season if args.prior_season is not None else resolve_prior_season())
    slate = (args.date if args.date is not None else resolve_slate_date()) or ""
    week = int(args.week if args.week is not None else resolve_week(season=season, slate_date=slate))
    out = enrich(season, prior, week)
    out.to_csv(DATA / "player_form_consensus.csv", index=False)
    out.to_csv(DATA / "player_form.csv", index=False)
    print(
        f"[player_scoring] enriched {len(out)} PlayerForm rows season={season} week={week}; "
        "preserved prior/current evidence and Player Identity v3"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
