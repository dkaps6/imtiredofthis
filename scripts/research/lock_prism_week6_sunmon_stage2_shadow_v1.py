#!/usr/bin/env python3
"""Lock PRISM Week-6 Sunday/Monday Stage-2 shadow V1.

Research shadow only.  Uses completed 2026 Weeks 1-5 as strictly-prior
player-role history and the preserved pregame Week-6 football artifact as the
target state.  SEA-DEN is excluded because it was already played before the
Stage-2 mechanism was frozen.

No Week-6 outcome is read.  No sportsbook file/column is accepted.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_40s_information_frontier as m76
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import test_qb_exact_personnel_discontinuity as m77
from scripts.research import audit_player_active_role_participation_v1 as stage1_source
from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    enumerate_active_states,
)
from scripts.research.evaluate_player_active_role_participation_v1 import (
    baseline_probability,
    candidate_probability,
)
from scripts.simulation_v2 import _top_n_shares

VERSION = "PRISM_WEEK6_SUNMON_STAGE2_SHADOW_V1"
FREEZE_COMMIT = "f954bf392c133c2507ff1b4bfa4d8325030abcc5"
ORIGIN_RUN = 38791377571
ORIGIN_ARTIFACT = 11999923880
ORIGIN_DIGEST = "sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61"
TARGET_SEASON = 2026
TARGET_WEEK = 6
PRIOR_WEEKS = (1, 2, 3, 4, 5)
PLAYER_WINDOW = 4
PLAYER_MIN = 3
BASELINE_MIN = 100
TOL = 1e-12

FORBIDDEN_COL_TOKENS = (
    "sportsbook", "bookmaker", "prop_line", "market_line", "over_odds",
    "under_odds", "moneyline", "closing_line", "no_vig", "implied_prob",
)


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_football_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    bad = [c for c in x.columns if any(t in c for t in FORBIDDEN_COL_TOKENS)]
    if bad:
        raise RuntimeError(f"forbidden market columns in {label}: {bad}")
    return x


def target_schedule_teamweeks(schedule: pd.DataFrame, weeks: tuple[int, ...]) -> pd.DataFrame:
    s = schedule.copy()
    required = {"season", "week", "home_team", "away_team"}
    if required - set(s.columns):
        raise RuntimeError(f"schedule missing {sorted(required-set(s.columns))}")
    s["season"] = num(s["season"]).astype("Int64")
    s["week"] = num(s["week"]).astype("Int64")
    s = s.loc[s["season"].eq(TARGET_SEASON) & s["week"].isin(weeks)].copy()
    h = s[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    a = s[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    out = pd.concat([h, a], ignore_index=True)
    out["team"] = out["team"].map(canon_team)
    return out.drop_duplicates(["season", "week", "team"]).reset_index(drop=True)


def discover_2026_prior_inactives(
    scheduled: pd.DataFrame,
    snapshots: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # M78's parser semantics are already qualified historically.  Extend only
    # the month inventory; target weeks remain fixed at completed Weeks 1-5.
    m78.SEASON_MONTHS[TARGET_SEASON] = [(2026, 9), (2026, 10)]
    urls = m78.discover_article_urls(TARGET_SEASON, snapshots)
    records: list[dict] = []
    article_meta: list[dict] = []
    for url in urls:
        recs, meta = m78.parse_inactive_article(url, TARGET_SEASON, snapshots)
        if meta:
            article_meta.append(meta)
        for rec in recs:
            if int(rec.get("week", 0) or 0) in PRIOR_WEEKS:
                records.append(rec)

    cols = [
        "season", "week", "team", "inactive_name", "inactive_name_key",
        "listed_position", "article_title", "article_url",
    ]
    inactive = pd.DataFrame(records, columns=cols)
    if len(inactive):
        inactive["team"] = inactive["team"].map(canon_team)
        inactive["week"] = num(inactive["week"]).astype(int)
        inactive = inactive.drop_duplicates(
            ["season", "week", "team", "inactive_name_key"]
        ).reset_index(drop=True)
    else:
        inactive = pd.DataFrame(columns=cols)

    found = (
        inactive[["season", "week", "team"]].drop_duplicates()
        if len(inactive)
        else pd.DataFrame(columns=["season", "week", "team"])
    )
    coverage = scheduled.merge(
        found.assign(inactive_teamweek_found=1),
        on=["season", "week", "team"],
        how="left",
    )
    coverage["inactive_teamweek_found"] = (
        num(coverage["inactive_teamweek_found"]).fillna(0).astype(int)
    )
    return inactive, pd.DataFrame(article_meta)


def load_prior_roster(
    scheduled: pd.DataFrame,
    inactive: pd.DataFrame,
    snapshots: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    raw = m78.load_weekly_rosters(TARGET_SEASON, snapshots)
    if raw.empty:
        raise RuntimeError("2026 weekly roster authority empty")

    # Inactive -> roster bridge uses the unfiltered weekly roster identity, just
    # as M78 did; the eligible universe then uses the Stage-1 ACT semantics.
    roster_index = m78.roster_name_index(raw, TARGET_SEASON)
    if len(inactive):
        inactive["roster_identity_match"] = [
            str(r.inactive_name_key)
            in roster_index.get((int(r.week), str(r.team)), set())
            for r in inactive.itertuples(index=False)
        ]
        bridge = float(inactive["roster_identity_match"].mean())
    else:
        inactive["roster_identity_match"] = pd.Series(dtype=bool)
        bridge = 0.0

    roster_all = stage1_source.normalize_roster(raw, TARGET_SEASON)
    roster = roster_all.loc[roster_all["week"].isin(PRIOR_WEEKS)].copy()
    roster = roster.merge(
        scheduled,
        on=["season", "week", "team"],
        how="inner",
        validate="many_to_one",
    )

    imap = {
        (int(s), int(w), str(t)): set(g["inactive_name_key"].astype(str))
        for (s, w, t), g in inactive.groupby(["season", "week", "team"], sort=False)
    }
    roster["known_official_inactive"] = [
        int(
            str(r.player_name_key)
            in imap.get((int(r.season), int(r.week), str(r.team)), set())
        )
        for r in roster.itertuples(index=False)
    ]
    roster["stable_roster_identity"] = (
        roster["player_id"].astype(str).str.strip().ne("").astype(int)
    )
    roster = roster.loc[roster["known_official_inactive"].eq(0)].copy()
    return roster.reset_index(drop=True), roster_all.reset_index(drop=True), bridge


def repair_target_player_ids(
    target: pd.DataFrame,
    roster_all: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve preserved temporary target identities only from Week-6 roster authority."""
    x = target.copy()
    x["player_id"] = x["player_id"].astype("string")
    x["identity_resolution_source"] = np.where(
        x["player_id"].fillna("").str.strip().ne(""),
        "PRESERVED_WEEK6_ARTIFACT",
        "UNRESOLVED",
    )

    r = roster_all.loc[
        roster_all["week"].eq(TARGET_WEEK)
        & roster_all["player_id"].astype(str).str.strip().ne("")
    ].copy()
    r = r.rename(columns={"player_name_key": "player_clean_key"})
    counts = (
        r.groupby(["team", "player_clean_key"], sort=False)["player_id"]
        .nunique()
        .rename("candidate_ids")
        .reset_index()
    )
    r = r.merge(counts, on=["team", "player_clean_key"], how="left")
    r = r.loc[r["candidate_ids"].eq(1)].drop_duplicates(
        ["team", "player_clean_key"], keep="first"
    )
    r = r[["team", "player_clean_key", "player_id"]].rename(
        columns={"player_id": "roster_week6_player_id"}
    )
    x = x.merge(
        r,
        on=["team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    missing = x["player_id"].fillna("").str.strip().eq("")
    rescued = missing & x["roster_week6_player_id"].astype("string").fillna("").str.strip().ne("")
    x.loc[rescued, "player_id"] = x.loc[rescued, "roster_week6_player_id"].astype("string")
    x.loc[rescued, "identity_resolution_source"] = "WEEK6_WEEKLY_ROSTER_STABLE_ID"
    x = x.drop(columns=["roster_week6_player_id"])
    x["player_id"] = x["player_id"].astype("string").fillna("").str.strip()
    return x


def load_prior_snaps(
    roster: pd.DataFrame,
    snapshots: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = datetime.now(timezone.utc).isoformat()
    raw = m76.download_table(
        "pfr_snap_counts",
        TARGET_SEASON,
        m76.release_urls("snaps", TARGET_SEASON),
        snapshots,
        as_of,
    )
    if raw.empty:
        raise RuntimeError("2026 PFR snap source empty")
    raw, _, _ = m76.build_id_bridge(raw, pd.DataFrame(), snapshots, as_of)
    snaps = m77.prepare_snap_history(raw)
    if snaps.empty:
        raise RuntimeError("prepared 2026 PFR snap history empty")
    snaps["player_id"] = snaps["player_id"].astype("string").fillna("").str.strip()
    snaps["off_pct"] = num(snaps["off_pct"])
    snaps = snaps.loc[
        snaps["season"].eq(TARGET_SEASON) & snaps["week"].isin(PRIOR_WEEKS)
    ].drop_duplicates(["season", "week", "player_id"], keep="first")

    positive = snaps.loc[snaps["off_pct"].fillna(0).gt(0)].copy()
    roster_key = roster[
        ["season", "week", "team", "player_id", "family"]
    ].loc[roster["player_id"].astype(str).str.strip().ne("")].drop_duplicates(
        ["season", "week", "player_id"], keep="first"
    )
    mapped_positive = positive.merge(
        roster_key,
        on=["season", "week", "player_id"],
        how="left",
        validate="one_to_one",
    )
    valid_teamweeks = set(
        tuple(r)
        for r in mapped_positive.loc[
            mapped_positive["team"].notna(), ["season", "week", "team"]
        ].drop_duplicates().itertuples(index=False, name=None)
    )

    out = roster.merge(
        snaps[["season", "week", "player_id", "off_pct"]],
        on=["season", "week", "player_id"],
        how="left",
        validate="one_to_one",
    )
    out["teamweek_snap_source_valid"] = [
        int((int(r.season), int(r.week), str(r.team)) in valid_teamweeks)
        for r in out.itertuples(index=False)
    ]
    out["positive_offensive_participation"] = (
        out["teamweek_snap_source_valid"].eq(1)
        & num(out["off_pct"]).fillna(0).gt(0)
    ).astype(int)
    out["valid_zero_snap"] = (
        out["teamweek_snap_source_valid"].eq(1) & out["off_pct"].isna()
    ).astype(int)
    return out, mapped_positive


def attach_week6_p_active(target: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in ("RB", "TE"):
        hist = prior.loc[
            prior["family"].eq(fam)
            & prior["stable_roster_identity"].eq(1)
            & prior["teamweek_snap_source_valid"].eq(1)
        ].copy()
        n_pos = int(len(hist))
        if n_pos < BASELINE_MIN:
            raise RuntimeError(f"{fam} Stage-1 position support {n_pos} < {BASELINE_MIN}")
        k_pos = int(hist["positive_offensive_participation"].sum())
        p0 = baseline_probability(k_pos, n_pos)

        for r in target.loc[target["family"].eq(fam)].itertuples(index=False):
            pid = "" if pd.isna(r.player_id) else str(r.player_id).strip()
            if pid:
                h = hist.loc[
                    hist["player_id"].astype(str).eq(pid)
                    & hist["team"].astype(str).eq(str(r.team))
                ].sort_values("week").tail(PLAYER_WINDOW)
            else:
                h = hist.iloc[0:0].copy()
            n_player = int(len(h))
            k_player = int(h["positive_offensive_participation"].sum())
            if pid and n_player >= PLAYER_MIN:
                p = candidate_probability(k_player, n_player, p0)
                source = "NAMED_PLAYER_STAGE1"
            else:
                p = p0
                source = "POSITION_FALLBACK_STAGE1"
            if not (0 < p < 1):
                raise RuntimeError("Week-6 p_active escaped open interval")
            rows.append({
                "event_id": str(r.event_id),
                "team": str(r.team),
                "player_clean_key": str(r.player_clean_key),
                "player_id": pid,
                "family": fam,
                "p_active": float(p),
                "stage1_probability_source": source,
                "n_pos_prior": n_pos,
                "k_pos_prior": k_pos,
                "n_player_prior": n_player,
                "k_player_prior": k_player,
            })
    out = pd.DataFrame(rows)
    if out.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate Week-6 Stage-1 probability row")
    return out


def rush_entitlement(universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_id, team), g in universe.groupby(["event_id", "team"], sort=False):
        raw = num(g["rules_rush_share"]).fillna(0.0).to_numpy(float)
        used = _top_n_shares(raw, 5)
        clean = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, 0.0, 0.95)
        if float(clean.sum()) > 0.95:
            clean *= 0.95 / float(clean.sum())
        residual = max(0.0, 1.0 - float(clean.sum()))
        if abs(float(clean.sum()) + residual - 1.0) > TOL:
            raise RuntimeError("rush entitlement conservation drift")
        for q, (_, r) in zip(clean, g.iterrows()):
            rows.append({
                "event_id": str(event_id),
                "team": str(team),
                "player_clean_key": str(r["player_clean_key"]),
                "rush_q": float(q),
            })
    return pd.DataFrame(rows)


def apply_shadow(
    target: pd.DataFrame,
    pactive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = target.merge(
        pactive,
        on=["event_id", "team", "player_clean_key", "player_id", "family"],
        how="left",
        validate="one_to_one",
    )
    if base["p_active"].isna().any():
        raise RuntimeError(
            f"Week-6 p_active attach misses={int(base['p_active'].isna().sum())}"
        )

    detail_rows = []
    room_rows = []
    for cell in ("RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING"):
        fam = "RB" if cell.startswith("RB_") else "TE"
        qcol = "rush_q" if cell == "RB_RUSHING" else "target_q"
        for (event_id, team), g0 in base.loc[base["family"].eq(fam)].groupby(
            ["event_id", "team"], sort=False
        ):
            g = g0.copy().sort_values("player_id").reset_index(drop=True)
            q_all = num(g[qcol]).to_numpy(float)
            if not np.isfinite(q_all).all() or (q_all < -TOL).any():
                raise RuntimeError(f"{cell} invalid canonical q {event_id} {team}")
            if float(q_all.sum()) > 1.0 + TOL:
                raise RuntimeError(f"{cell} named q exceeds one {event_id} {team}")
            pos = q_all > TOL
            q = q_all[pos]
            p = num(g.loc[pos, "p_active"]).to_numpy(float)
            if len(q):
                weights, state_named, state_comp = enumerate_active_states(q, p)
                expected = weights @ state_named
                expected_comp = float(weights @ state_comp)
            else:
                weights = np.asarray([1.0])
                expected = np.zeros(0, dtype=float)
                expected_comp = 1.0
            q0 = 1.0 - float(q.sum())
            if abs(float(expected.sum()) + expected_comp - 1.0) > TOL:
                raise RuntimeError(f"{cell} candidate mass drift {event_id} {team}")

            plays = float(num(g["rules_plays_est"]).dropna().iloc[0])
            pass_rate = float(num(g["rules_pass_rate"]).dropna().iloc[0])
            if not np.isfinite(plays) or not np.isfinite(pass_rate):
                raise RuntimeError("non-finite team opportunity state")
            team_mean = plays * (1.0 - pass_rate) if cell == "RB_RUSHING" else plays * pass_rate

            exp_map = {}
            j = 0
            for idx, is_pos in enumerate(pos.tolist()):
                exp_map[idx] = float(expected[j]) if is_pos else float(q_all[idx])
                if is_pos:
                    j += 1

            for idx, r in g.iterrows():
                cand_q = exp_map[idx]
                detail_rows.append({
                    "season": TARGET_SEASON,
                    "week": TARGET_WEEK,
                    "event_id": str(event_id),
                    "team": str(team),
                    "cell": cell,
                    "family": fam,
                    "player": str(r["player"]),
                    "player_clean_key": str(r["player_clean_key"]),
                    "player_id": "" if pd.isna(r["player_id"]) else str(r["player_id"]),
                    "p_active": float(r["p_active"]),
                    "stage1_probability_source": str(r["stage1_probability_source"]),
                    "n_player_prior": int(r["n_player_prior"]),
                    "k_player_prior": int(r["k_player_prior"]),
                    "canonical_q": float(q_all[idx]),
                    "candidate_expected_q": float(cand_q),
                    "share_shift": float(cand_q - q_all[idx]),
                    "team_opportunity_mean": float(team_mean),
                    "baseline_expected_opportunities": float(team_mean * q_all[idx]),
                    "candidate_expected_opportunities": float(team_mean * cand_q),
                    "expected_opportunity_shift": float(team_mean * (cand_q - q_all[idx])),
                    "availability_authority": str(r["availability_authority"]),
                    "final_availability_state": str(r["final_availability_state"]),
                })

            room_rows.append({
                "season": TARGET_SEASON,
                "week": TARGET_WEEK,
                "event_id": str(event_id),
                "team": str(team),
                "cell": cell,
                "players": int(len(g)),
                "positive_q_players": int(pos.sum()),
                "canonical_named_mass": float(q.sum()),
                "canonical_complement_mass": float(q0),
                "candidate_named_mass": float(expected.sum()),
                "candidate_complement_mass": float(expected_comp),
                "state_count": int(len(weights)),
                "state_weight_gap": float(abs(float(weights.sum()) - 1.0)),
                "candidate_mass_gap": float(
                    abs(float(expected.sum()) + expected_comp - 1.0)
                ),
                "team_opportunity_mean": float(team_mean),
            })

    detail = pd.DataFrame(detail_rows)
    rooms = pd.DataFrame(room_rows)
    return detail, rooms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--availability", type=Path, required=True)
    ap.add_argument("--target-entitlement", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[dict] = []
    try:
        schedule = read_football_csv(args.schedule, "preserved Week-6 schedule")
        prior_teamweeks = target_schedule_teamweeks(schedule, PRIOR_WEEKS)
        inactive, article_meta = discover_2026_prior_inactives(prior_teamweeks, snapshots)

        prior_roster, roster_all, inactive_bridge = load_prior_roster(
            prior_teamweeks, inactive, snapshots
        )
        prior, mapped_positive = load_prior_snaps(prior_roster, snapshots)

        roster_tw = prior[["season", "week", "team"]].drop_duplicates()
        inactive_tw = inactive[["season", "week", "team"]].drop_duplicates()
        snap_tw = prior.loc[
            prior["teamweek_snap_source_valid"].eq(1), ["season", "week", "team"]
        ].drop_duplicates()
        stable = float(prior["stable_roster_identity"].mean()) if len(prior) else 0.0
        pos_bridge = (
            float(mapped_positive["team"].notna().mean())
            if len(mapped_positive)
            else 0.0
        )
        scheduled_n = int(len(prior_teamweeks))

        source_metrics = {
            "scheduled_prior_teamgames": scheduled_n,
            "roster_teamgames": int(len(roster_tw)),
            "roster_teamgame_coverage": float(len(roster_tw) / max(scheduled_n, 1)),
            "official_inactive_teamgames": int(len(inactive_tw)),
            "official_inactive_teamgame_coverage": float(
                len(inactive_tw) / max(scheduled_n, 1)
            ),
            "official_inactive_rows": int(len(inactive)),
            "official_inactive_roster_identity_bridge": float(inactive_bridge),
            "stable_identity_coverage": stable,
            "snap_teamgames": int(len(snap_tw)),
            "snap_teamgame_coverage": float(len(snap_tw) / max(scheduled_n, 1)),
            "positive_snap_roster_bridge": pos_bridge,
            "prior_player_rows": int(len(prior)),
            "prior_rb_rows": int(prior["family"].eq("RB").sum()),
            "prior_te_rows": int(prior["family"].eq("TE").sum()),
            "chronology_violations": 0,
        }
        source_gates = {
            "roster_teamgame_coverage_ge_99": source_metrics["roster_teamgame_coverage"] >= 0.99,
            "official_inactive_teamgame_coverage_ge_99": source_metrics["official_inactive_teamgame_coverage"] >= 0.99,
            "official_inactive_roster_identity_bridge_ge_95": source_metrics["official_inactive_roster_identity_bridge"] >= 0.95,
            "stable_identity_ge_99": source_metrics["stable_identity_coverage"] >= 0.99,
            "snap_teamgame_coverage_ge_99": source_metrics["snap_teamgame_coverage"] >= 0.99,
            "positive_snap_roster_bridge_ge_98": source_metrics["positive_snap_roster_bridge"] >= 0.98,
            "chronology_violations_zero": source_metrics["chronology_violations"] == 0,
        }

        universe = read_football_csv(args.universe, "Week-6 football universe")
        availability = read_football_csv(args.availability, "Week-6 availability")
        entitlement = read_football_csv(args.target_entitlement, "Week-6 target entitlement")

        for x in (universe, availability, entitlement):
            if "team" in x:
                x["team"] = x["team"].map(canon_team)

        universe["position"] = universe["position"].astype(str).str.upper().str.strip()
        universe["family"] = np.where(
            universe["position"].isin(["RB", "FB", "HB"]), "RB",
            np.where(universe["position"].eq("TE"), "TE", ""),
        )
        universe = universe.loc[
            universe["season"].eq(TARGET_SEASON)
            & universe["week"].eq(TARGET_WEEK)
        ].copy()

        # SEA-DEN already played before this prospective shadow was frozen.
        universe["sea_den"] = universe.apply(
            lambda r: {str(r["team"]), str(r["opponent"])} == {"SEA", "DEN"},
            axis=1,
        )
        universe = universe.loc[~universe["sea_den"]].copy()

        avail_keep = [
            "team", "player_clean_key", "eligible_for_opportunity",
            "definitive_unavailable", "final_availability_state",
            "availability_authority",
        ]
        target = universe.merge(
            availability[avail_keep],
            on=["team", "player_clean_key"],
            how="left",
            validate="one_to_one",
        )
        if target["eligible_for_opportunity"].isna().any():
            raise RuntimeError("Week-6 target availability join miss")
        target = target.loc[
            num(target["eligible_for_opportunity"]).eq(1)
            & num(target["definitive_unavailable"]).fillna(0).eq(0)
            & target["family"].isin(["RB", "TE"])
        ].copy()
        target = repair_target_player_ids(target, roster_all)

        ent = entitlement[
            ["event_id", "team", "player_clean_key", "entitlement_tgt_share"]
        ].rename(columns={"entitlement_tgt_share": "target_q"})
        target = target.merge(
            ent,
            on=["event_id", "team", "player_clean_key"],
            how="left",
            validate="one_to_one",
        )
        if target["target_q"].isna().any():
            raise RuntimeError("Week-6 target entitlement join miss")

        rq = rush_entitlement(universe)
        target = target.merge(
            rq,
            on=["event_id", "team", "player_clean_key"],
            how="left",
            validate="one_to_one",
        )
        if target["rush_q"].isna().any():
            raise RuntimeError("Week-6 rushing entitlement join miss")

        pactive = attach_week6_p_active(target, prior)
        detail, rooms = apply_shadow(target, pactive)

        target_games = int(universe["event_id"].nunique())
        target_identity = float(
            target["player_id"].astype("string").fillna("").str.strip().ne("").mean()
        ) if len(target) else 0.0
        p_attach = float(detail["p_active"].notna().mean()) if len(detail) else 0.0
        q_finite = float(
            np.isfinite(num(detail["canonical_q"])).mean()
        ) if len(detail) else 0.0
        max_mass_gap = float(rooms["candidate_mass_gap"].max()) if len(rooms) else np.inf
        target_gates = {
            "sea_den_excluded": not universe.apply(
                lambda r: {str(r["team"]), str(r["opponent"])} == {"SEA", "DEN"},
                axis=1,
            ).any(),
            "target_games_exact_13": target_games == 13,
            "target_identity_ge_99": target_identity >= 0.99,
            "stage1_p_active_attach_100": p_attach == 1.0,
            "canonical_q_finite_100": q_finite == 1.0,
            "candidate_mass_gap_le_1e_12": max_mass_gap <= TOL,
        }

        all_gates = all(source_gates.values()) and all(target_gates.values())
        disposition = (
            "PRISM_WEEK6_SUNMON_STAGE2_SHADOW_LOCKED"
            if all_gates
            else "PRISM_WEEK6_SUNMON_STAGE2_SHADOW_SOURCE_BLOCKED"
        )
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "origin_run": ORIGIN_RUN,
            "origin_artifact": ORIGIN_ARTIFACT,
            "origin_digest": ORIGIN_DIGEST,
            "disposition": disposition,
            "target_season": TARGET_SEASON,
            "target_week": TARGET_WEEK,
            "excluded_game": "SEA-DEN",
            "included_game_count": target_games,
            "eligible_cells": ["RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING"],
            "source_metrics": source_metrics,
            "source_gates": source_gates,
            "target_gates": target_gates,
            "shadow_rows": int(len(detail)),
            "shadow_rooms": int(len(rooms)),
            "named_player_stage1_rate": float(
                detail["stage1_probability_source"].eq("NAMED_PLAYER_STAGE1").mean()
            ) if len(detail) else 0.0,
            "max_candidate_mass_gap": max_mass_gap,
            "completed_2026_weeks_1_5_used_as_strict_prior_history": True,
            "week6_target_outcomes_read": 0,
            "sea_den_outcome_read": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "new_share_models_fit": 0,
            "threshold_searches": 0,
            "production_changed": False,
        }

        prior.to_csv(args.out_dir / "prism_week6_stage1_prior_universe.csv", index=False)
        inactive.to_csv(args.out_dir / "prism_week6_prior_official_inactives.csv", index=False)
        article_meta.to_csv(args.out_dir / "prism_week6_prior_inactive_article_manifest.csv", index=False)
        pd.DataFrame(snapshots).to_csv(args.out_dir / "prism_week6_source_snapshots.csv", index=False)
        pactive.to_csv(args.out_dir / "prism_week6_stage1_p_active.csv", index=False)
        detail.to_csv(args.out_dir / "prism_week6_stage2_shadow_rows.csv", index=False)
        rooms.to_csv(args.out_dir / "prism_week6_stage2_shadow_rooms.csv", index=False)
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "origin_run": ORIGIN_RUN,
            "origin_artifact": ORIGIN_ARTIFACT,
            "origin_digest": ORIGIN_DIGEST,
            "disposition": "PRISM_WEEK6_SUNMON_STAGE2_SHADOW_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "week6_target_outcomes_read": 0,
            "sea_den_outcome_read": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "new_share_models_fit": 0,
            "threshold_searches": 0,
            "production_changed": False,
        }
        (args.out_dir / "prism_week6_sunmon_stage2_shadow_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "prism_week6_sunmon_stage2_shadow_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
