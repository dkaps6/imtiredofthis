#!/usr/bin/env python3
"""Diagnostic-only audit of WR anchor / current-role transmission.

The first audit launch proved that the frozen WR-R15 scored prediction artifact
contains only its scored outcome subset and therefore can omit the true M38 WR1
anchor. This bounded repair reconstructs the full fold-safe pregame WR state
using the same historical context, explicit M38 entitlement, frozen WR-R15 fold
coefficients, and exact strict-prior participation builder.

The canonical WR-R15 scored artifact remains the numerical parity authority and
the target-error authority. No new projection candidate is constructed or
scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.research.persist_historical_simulated_outcomes_v1 import _exact_week
from scripts.research.persist_wr_te_production_order_historical_v1 import (
    WR_FEATURES,
    _load_fold_params,
    _load_participation_snaps,
    _wr_strict_prior_features,
    apply_wr_fold,
)

VERSION = "WR_ANCHOR_ROLE_TRANSMISSION_AUDIT_V1"
BASELINE_VARIANT = "M38_EXPLICIT_BASELINE"
CANDIDATE_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
SEASONS = (2023, 2024)
WR_POS = {"WR", "LWR", "RWR", "SWR"}
AUTH_KEYS = ["season", "week", "team", "player_clean_key"]
FULL_KEYS = ["season", "week", "event_id", "team", "player_clean_key"]


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def is_wr(value: object) -> bool:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    return p in WR_POS or p.startswith("WR")


def prepare_authority(pred: pd.DataFrame) -> pd.DataFrame:
    need = {
        "variant", "team", "player_clean_key", "season", "week",
        "entitlement_tgt_share", "pred_targets", "actual_targets",
    }
    missing = need - set(pred.columns)
    if missing:
        raise RuntimeError(f"WR-R15 predictions missing {sorted(missing)}")
    x = pred.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str)
    x = x.loc[x["season"].isin(SEASONS)].copy()

    observed = set(x["variant"].astype(str).unique())
    expected = {BASELINE_VARIANT, CANDIDATE_VARIANT}
    if not expected.issubset(observed):
        raise RuntimeError(
            f"WR-R15 prediction variants missing expected={expected} observed={observed}"
        )

    b = x.loc[x["variant"].eq(BASELINE_VARIANT)].copy()
    c = x.loc[x["variant"].eq(CANDIDATE_VARIANT)].copy()
    for label, d in (("baseline", b), ("candidate", c)):
        if d.duplicated(AUTH_KEYS).any():
            bad = d.loc[d.duplicated(AUTH_KEYS, keep=False), AUTH_KEYS].head(10).to_dict("records")
            raise RuntimeError(f"{label} duplicate prediction identities: {bad}")

    b = b[AUTH_KEYS + ["entitlement_tgt_share", "pred_targets", "actual_targets"]].rename(
        columns={
            "entitlement_tgt_share": "authority_baseline_entitlement",
            "pred_targets": "authority_baseline_pred_targets",
            "actual_targets": "authority_actual_targets",
        }
    )
    c = c[AUTH_KEYS + ["entitlement_tgt_share", "pred_targets", "actual_targets"]].rename(
        columns={
            "entitlement_tgt_share": "authority_candidate_entitlement",
            "pred_targets": "authority_candidate_pred_targets",
            "actual_targets": "authority_candidate_actual_targets",
        }
    )
    out = b.merge(c, on=AUTH_KEYS, how="inner", validate="one_to_one")
    if len(out) != len(b) or len(out) != len(c):
        raise RuntimeError(
            f"authority baseline/candidate identity mismatch b={len(b)} c={len(c)} joined={len(out)}"
        )
    label_gap = (
        num(out["authority_actual_targets"]) - num(out["authority_candidate_actual_targets"])
    ).abs()
    if len(label_gap) and float(label_gap.max()) > 1e-12:
        raise RuntimeError("authority baseline/candidate actual labels differ")
    for c in [
        "authority_baseline_entitlement", "authority_candidate_entitlement",
        "authority_baseline_pred_targets", "authority_candidate_pred_targets",
        "authority_actual_targets",
    ]:
        out[c] = num(out[c])
        if out[c].isna().any():
            raise RuntimeError(f"authority non-numeric values in {c}")
    return out


def actual_target_labels(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    need = {"season", "week", "team", "player_clean_key", "targets"}
    missing = need - set(x.columns)
    if missing:
        raise RuntimeError(f"historical player logs missing {sorted(missing)}")
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["targets"] = num(x["targets"]).fillna(0.0)
    z = x.loc[
        x["season"].eq(int(season)) & x["week"].eq(int(week))
    ].copy()
    return (
        z.groupby(["team", "player_clean_key"], as_index=False)
        .agg(actual_targets=("targets", "sum"))
    )


def reconstruct_full_state(
    *,
    player_logs: pd.DataFrame,
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    universe_dirs: dict[int, Path],
    injuries_history: pd.DataFrame,
    weather_history: pd.DataFrame,
    coefficients: Path,
) -> tuple[pd.DataFrame, dict]:
    snaps, dup_rate, source_seasons = _load_participation_snaps()
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    total_future = 0

    for season in SEASONS:
        prior = season - 1
        params = _load_fold_params(
            coefficients,
            test_season=int(season),
            features=WR_FEATURES,
            label="WR-R15",
        )
        for week in range(1, 19):
            u_path = universe_dirs[int(season)] / f"{season}_week_{week:02d}.csv"
            if not u_path.exists():
                raise RuntimeError(f"missing pregame universe: {u_path}")
            universe = read(u_path, f"{season} W{week} pregame universe")
            bundle = build_historical_context_bundle(
                player_logs=player_logs,
                team_weekly=team_weekly,
                pregame_universe=universe,
                schedule=schedule,
                season=int(season),
                week=int(week),
                prior_season=int(prior),
                injuries=_exact_week(injuries_history, season, week),
                weather=_exact_week(weather_history, season, week),
            )
            metrics = build_mc_predictions(bundle, iterations=20, seed=42 + int(week))
            players = (
                metrics.sort_values(["event_id", "team", "player_clean_key"])
                .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
                .copy()
            )
            baseline, _ = materialize_target_entitlement(players)
            baseline["team"] = baseline["team"].map(canon_team)
            baseline["season"] = int(season)
            baseline["week"] = int(week)

            pos = baseline["position"].fillna("").astype(str).str.upper().str.strip()
            wr_base = baseline.loc[pos.map(is_wr)].copy()
            if wr_base.empty:
                raise RuntimeError(f"{season} W{week} reconstructed zero WR rows")

            # Attach the exact WR-R15 strict-prior participation state to every
            # WR, including the otherwise-immutable M38 anchor.
            all_feat, future = _wr_strict_prior_features(wr_base.copy(), snaps)
            total_future += int(future)
            if int(future) != 0:
                raise RuntimeError(
                    f"{season} W{week} full-WR participation used same/future rows: {future}"
                )

            final, _, wr_audit = apply_wr_fold(
                baseline.copy(), snaps=snaps, params=params
            )
            final["team"] = final["team"].map(canon_team)
            final_pos = final["position"].fillna("").astype(str).str.upper().str.strip()
            wr_final = final.loc[final_pos.map(is_wr)].copy()

            key = ["event_id", "team", "player_clean_key"]
            base_cols = key + [
                "player", "position", "entitlement_tgt_share",
            ]
            feat_cols = key + [
                "prior1_same_team", "prior1_same_team_offense_pct",
                "prior1_same_team_offense_snaps", "prior_count_same_team",
            ]
            final_cols = key + ["entitlement_tgt_share"]
            z = (
                wr_base[base_cols]
                .rename(columns={"entitlement_tgt_share": "baseline_entitlement_tgt_share"})
                .merge(all_feat[feat_cols], on=key, how="left", validate="one_to_one")
                .merge(
                    wr_final[final_cols].rename(
                        columns={"entitlement_tgt_share": "candidate_entitlement_tgt_share"}
                    ),
                    on=key,
                    how="left",
                    validate="one_to_one",
                )
            )
            z["season"] = int(season)
            z["week"] = int(week)
            labels = actual_target_labels(player_logs, season, week)
            z = z.merge(
                labels, on=["team", "player_clean_key"], how="left", validate="one_to_one"
            )
            z["actual_targets"] = num(z["actual_targets"]).fillna(0.0)
            rows.append(z)

            audit_rows.append({
                "season": int(season),
                "week": int(week),
                "wr_rows": int(len(z)),
                "full_wr_future_violations": int(future),
                "wr_fold_same_future_participation": int(
                    wr_audit.get("same_future_participation", -1)
                ),
                "wr1_anchor_max_abs_gap": float(
                    wr_audit.get("m38_wr1_anchor_max_abs_gap", np.nan)
                ),
                "wr_room_mass_max_abs_gap": float(
                    wr_audit.get("wr_room_mass_max_abs_gap", np.nan)
                ),
            })

    detail = pd.concat(rows, ignore_index=True)
    if detail.duplicated(FULL_KEYS).any():
        bad = detail.loc[
            detail.duplicated(FULL_KEYS, keep=False), FULL_KEYS
        ].head(10).to_dict("records")
        raise RuntimeError(f"reconstructed duplicate full WR identities: {bad}")

    audit = {
        "raw_snap_duplicate_rate": float(dup_rate),
        "snap_source_seasons": [int(x) for x in source_seasons],
        "strict_prior_future_violations": int(total_future),
        "reconstructed_wr_rows": int(len(detail)),
        "reconstructed_team_games": int(
            detail[["season", "week", "event_id", "team"]].drop_duplicates().shape[0]
        ),
        "max_wr1_anchor_gap": float(
            pd.DataFrame(audit_rows)["wr1_anchor_max_abs_gap"].max()
        ),
        "max_wr_room_mass_gap": float(
            pd.DataFrame(audit_rows)["wr_room_mass_max_abs_gap"].max()
        ),
    }
    return detail.sort_values(FULL_KEYS).reset_index(drop=True), audit


def authority_parity(
    full: pd.DataFrame, authority: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    recon = full[
        AUTH_KEYS + [
            "baseline_entitlement_tgt_share", "candidate_entitlement_tgt_share",
            "actual_targets",
        ]
    ].copy()
    if recon.duplicated(AUTH_KEYS).any():
        raise RuntimeError("reconstructed authority join keys are not unique")
    joined = authority.merge(recon, on=AUTH_KEYS, how="left", validate="one_to_one")
    missing = int(joined["baseline_entitlement_tgt_share"].isna().sum())
    if missing:
        sample = joined.loc[
            joined["baseline_entitlement_tgt_share"].isna(), AUTH_KEYS
        ].head(10).to_dict("records")
        raise RuntimeError(
            f"reconstruction missing {missing} scored authority rows sample={sample}"
        )
    base_gap = (
        joined["authority_baseline_entitlement"]
        - joined["baseline_entitlement_tgt_share"]
    ).abs()
    cand_gap = (
        joined["authority_candidate_entitlement"]
        - joined["candidate_entitlement_tgt_share"]
    ).abs()
    actual_gap = (
        joined["authority_actual_targets"] - joined["actual_targets"]
    ).abs()
    audit = {
        "authority_rows": int(len(joined)),
        "authority_reconstruction_coverage": 1.0,
        "max_baseline_entitlement_gap": float(base_gap.max()) if len(base_gap) else 0.0,
        "max_candidate_entitlement_gap": float(cand_gap.max()) if len(cand_gap) else 0.0,
        "max_actual_target_label_gap": float(actual_gap.max()) if len(actual_gap) else 0.0,
    }
    # This is a mechanical authority reconstruction. Fail closed if the current
    # replay cannot reproduce the frozen OOS entitlement values.
    if audit["max_baseline_entitlement_gap"] > 1e-8:
        raise RuntimeError(f"baseline authority parity failed: {audit}")
    if audit["max_candidate_entitlement_gap"] > 1e-8:
        raise RuntimeError(f"candidate authority parity failed: {audit}")
    if audit["max_actual_target_label_gap"] > 1e-12:
        raise RuntimeError(f"actual-label authority parity failed: {audit}")
    return joined, audit


def deterministic_leader(
    g: pd.DataFrame, value_col: str, *, eligibility: pd.Series | None = None
) -> str:
    z = g.copy()
    if eligibility is not None:
        z = z.loc[eligibility.loc[z.index]].copy()
    z[value_col] = num(z[value_col])
    z = z.loc[z[value_col].notna()].copy()
    if z.empty:
        return ""
    z = z.sort_values(
        [value_col, "player_clean_key"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(z.iloc[0]["player_clean_key"])


def player_value(g: pd.DataFrame, player_key: str, col: str) -> float:
    z = g.loc[g["player_clean_key"].eq(str(player_key))]
    if len(z) != 1:
        return np.nan
    v = pd.to_numeric(z.iloc[0][col], errors="coerce")
    return float(v) if pd.notna(v) else np.nan


def rank_for_player(g: pd.DataFrame, player_key: str, col: str) -> float:
    z = g[["player_clean_key", col]].copy()
    z[col] = num(z[col])
    z = z.sort_values(
        [col, "player_clean_key"], ascending=[False, True], kind="mergesort"
    )
    z["rank"] = np.arange(1, len(z) + 1, dtype=float)
    hit = z.loc[z["player_clean_key"].eq(str(player_key)), "rank"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def build_team_games(
    detail: pd.DataFrame, authority: pd.DataFrame
) -> pd.DataFrame:
    auth = authority.copy()
    auth["authority_baseline_abs_error"] = (
        auth["authority_baseline_pred_targets"] - auth["authority_actual_targets"]
    ).abs()
    auth["authority_candidate_abs_error"] = (
        auth["authority_candidate_pred_targets"] - auth["authority_actual_targets"]
    ).abs()
    err = (
        auth.groupby(["season", "week", "team"], as_index=False)
        .agg(
            authority_scored_wr_rows=("player_clean_key", "size"),
            baseline_target_abs_error_sum=("authority_baseline_abs_error", "sum"),
            candidate_target_abs_error_sum=("authority_candidate_abs_error", "sum"),
        )
    )

    rows = []
    group_cols = ["season", "week", "event_id", "team"]
    for keys, g in detail.groupby(group_cols, sort=True):
        season, week, event_id, team = keys
        anchor = deterministic_leader(g, "baseline_entitlement_tgt_share")
        if not anchor:
            raise RuntimeError(f"missing reconstructed M38 anchor {keys}")

        eligible = (
            g["prior1_same_team"].fillna(False).astype(bool)
            & num(g["prior1_same_team_offense_pct"]).notna()
        )
        participation = deterministic_leader(
            g, "prior1_same_team_offense_pct", eligibility=eligible
        )
        if not participation:
            # No imputation: this team-game is outside the frozen participation-
            # transmission cohort and is accounted for in coverage.
            continue

        candidate_leader = deterministic_leader(
            g, "candidate_entitlement_tgt_share"
        )
        actual_max = float(num(g["actual_targets"]).max())
        actual_top = set(
            g.loc[
                num(g["actual_targets"]).eq(actual_max), "player_clean_key"
            ].astype(str)
        )
        rec = {
            "season": int(season),
            "week": int(week),
            "event_id": str(event_id),
            "team": str(team),
            "n_wr_full_pregame": int(len(g)),
            "m38_anchor_key": anchor,
            "participation_leader_key": participation,
            "candidate_entitlement_leader_key": candidate_leader,
            "anchor_participation_mismatch": int(anchor != participation),
            "candidate_followed_participation_leader": int(
                candidate_leader == participation
            ),
            "candidate_leader_changed_from_anchor": int(candidate_leader != anchor),
            "anchor_actual_top_hit": int(anchor in actual_top),
            "participation_leader_actual_top_hit": int(participation in actual_top),
            "candidate_leader_actual_top_hit": int(candidate_leader in actual_top),
            "anchor_actual_targets": player_value(g, anchor, "actual_targets"),
            "participation_leader_actual_targets": player_value(
                g, participation, "actual_targets"
            ),
            "anchor_baseline_entitlement": player_value(
                g, anchor, "baseline_entitlement_tgt_share"
            ),
            "anchor_candidate_entitlement": player_value(
                g, anchor, "candidate_entitlement_tgt_share"
            ),
            "participation_leader_baseline_entitlement": player_value(
                g, participation, "baseline_entitlement_tgt_share"
            ),
            "participation_leader_candidate_entitlement": player_value(
                g, participation, "candidate_entitlement_tgt_share"
            ),
            "participation_leader_baseline_rank": rank_for_player(
                g, participation, "baseline_entitlement_tgt_share"
            ),
            "participation_leader_candidate_rank": rank_for_player(
                g, participation, "candidate_entitlement_tgt_share"
            ),
            "max_actual_targets": actual_max,
        }
        rec["participation_minus_anchor_actual_targets"] = (
            rec["participation_leader_actual_targets"]
            - rec["anchor_actual_targets"]
        )
        rec["anchor_entitlement_immutability_gap"] = abs(
            rec["anchor_candidate_entitlement"]
            - rec["anchor_baseline_entitlement"]
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("anchor transmission audit produced zero eligible team-games")
    out = out.merge(
        err, on=["season", "week", "team"], how="left", validate="one_to_one"
    )
    out = out.loc[out["authority_scored_wr_rows"].notna()].copy()
    if out.empty:
        raise RuntimeError("zero reconstructed team-games overlap scored WR-R15 authority")
    out["baseline_target_abs_error_per_scored_wr"] = (
        out["baseline_target_abs_error_sum"] / out["authority_scored_wr_rows"]
    )
    out["candidate_target_abs_error_per_scored_wr"] = (
        out["candidate_target_abs_error_sum"] / out["authority_scored_wr_rows"]
    )
    return out.sort_values(group_cols).reset_index(drop=True)


def cohort_row(d: pd.DataFrame, scope: str, cohort: str) -> dict:
    scored = int(d["authority_scored_wr_rows"].sum()) if len(d) else 0
    return {
        "scope": scope,
        "cohort": cohort,
        "team_games": int(len(d)),
        "authority_scored_wr_rows": scored,
        "mismatch_rate": (
            float(d["anchor_participation_mismatch"].mean()) if len(d) else np.nan
        ),
        "candidate_follow_participation_rate": (
            float(d["candidate_followed_participation_leader"].mean())
            if len(d) else np.nan
        ),
        "anchor_actual_top_hit_rate": (
            float(d["anchor_actual_top_hit"].mean()) if len(d) else np.nan
        ),
        "participation_actual_top_hit_rate": (
            float(d["participation_leader_actual_top_hit"].mean())
            if len(d) else np.nan
        ),
        "candidate_actual_top_hit_rate": (
            float(d["candidate_leader_actual_top_hit"].mean()) if len(d) else np.nan
        ),
        "participation_minus_anchor_actual_top_hit_rate": (
            float(
                d["participation_leader_actual_top_hit"].mean()
                - d["anchor_actual_top_hit"].mean()
            )
            if len(d) else np.nan
        ),
        "mean_participation_minus_anchor_actual_targets": (
            float(d["participation_minus_anchor_actual_targets"].mean())
            if len(d) else np.nan
        ),
        "baseline_abs_error_per_scored_wr": (
            float(d["baseline_target_abs_error_sum"].sum() / max(1, scored))
            if len(d) else np.nan
        ),
        "candidate_abs_error_per_scored_wr": (
            float(d["candidate_target_abs_error_sum"].sum() / max(1, scored))
            if len(d) else np.nan
        ),
    }


def summaries(
    team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows = []
    season_rows = []
    for scope, d in [("POOLED", team_games)] + [
        (str(s), team_games.loc[team_games["season"].eq(s)]) for s in SEASONS
    ]:
        mismatch = d.loc[d["anchor_participation_mismatch"].eq(1)]
        match = d.loc[d["anchor_participation_mismatch"].eq(0)]
        cohort_rows += [
            cohort_row(d, scope, "ALL"),
            cohort_row(mismatch, scope, "ANCHOR_PARTICIPATION_MISMATCH"),
            cohort_row(match, scope, "ANCHOR_PARTICIPATION_MATCH"),
        ]
        mrows = int(mismatch["authority_scored_wr_rows"].sum())
        arows = int(match["authority_scored_wr_rows"].sum())
        season_rows.append({
            "scope": scope,
            "eligible_team_games": int(len(d)),
            "mismatch_team_games": int(len(mismatch)),
            "mismatch_rate": float(len(mismatch) / len(d)) if len(d) else np.nan,
            "mismatch_participation_hit_minus_anchor": (
                float(
                    mismatch["participation_leader_actual_top_hit"].mean()
                    - mismatch["anchor_actual_top_hit"].mean()
                )
                if len(mismatch) else np.nan
            ),
            "mismatch_mean_participation_minus_anchor_actual_targets": (
                float(mismatch["participation_minus_anchor_actual_targets"].mean())
                if len(mismatch) else np.nan
            ),
            "mismatch_candidate_follow_participation_rate": (
                float(mismatch["candidate_followed_participation_leader"].mean())
                if len(mismatch) else np.nan
            ),
            "mismatch_candidate_abs_error_per_scored_wr": (
                float(mismatch["candidate_target_abs_error_sum"].sum() / max(1, mrows))
                if len(mismatch) else np.nan
            ),
            "match_candidate_abs_error_per_scored_wr": (
                float(match["candidate_target_abs_error_sum"].sum() / max(1, arows))
                if len(match) else np.nan
            ),
        })
    return pd.DataFrame(cohort_rows), pd.DataFrame(season_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-predictions", type=Path, required=True)
    ap.add_argument("--wr-coefficients", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-2023", type=Path, required=True)
    ap.add_argument("--universe-2024", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, required=True)
    ap.add_argument("--weather", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = prepare_authority(
        read(args.authority_predictions, "WR-R15 OOS predictions")
    )
    player_logs = read(args.player_logs, "historical player logs")
    team_weekly = read(args.team_weekly, "historical team-week")
    schedule = read(args.schedule, "historical schedule")
    injuries = optional(args.injuries)
    weather = optional(args.weather)

    full, reconstruction_audit = reconstruct_full_state(
        player_logs=player_logs,
        team_weekly=team_weekly,
        schedule=schedule,
        universe_dirs={2023: args.universe_2023, 2024: args.universe_2024},
        injuries_history=injuries,
        weather_history=weather,
        coefficients=args.wr_coefficients,
    )
    parity_rows, parity_audit = authority_parity(full, authority)
    team_games = build_team_games(full, parity_rows)
    cohort, season = summaries(team_games)

    pooled = season.loc[season["scope"].eq("POOLED")].iloc[0]
    s23 = season.loc[season["scope"].eq("2023")].iloc[0]
    s24 = season.loc[season["scope"].eq("2024")].iloc[0]
    mismatch = team_games.loc[team_games["anchor_participation_mismatch"].eq(1)]
    match = team_games.loc[team_games["anchor_participation_mismatch"].eq(0)]
    mismatch_rows = int(mismatch["authority_scored_wr_rows"].sum())
    match_rows = int(match["authority_scored_wr_rows"].sum())
    mismatch_err = float(
        mismatch["candidate_target_abs_error_sum"].sum() / max(1, mismatch_rows)
    )
    match_err = float(
        match["candidate_target_abs_error_sum"].sum() / max(1, match_rows)
    )

    criteria = {
        "mismatch_team_games_ge150": int(len(mismatch)) >= 150,
        "pooled_participation_top_hit_advantage_ge5pp":
            float(pooled["mismatch_participation_hit_minus_anchor"]) >= 0.05,
        "participation_top_hit_advantage_nonnegative_both_seasons":
            float(s23["mismatch_participation_hit_minus_anchor"]) >= 0.0
            and float(s24["mismatch_participation_hit_minus_anchor"]) >= 0.0,
        "participation_actual_target_advantage_positive_pooled_nonnegative_both":
            float(pooled["mismatch_mean_participation_minus_anchor_actual_targets"]) > 0.0
            and float(s23["mismatch_mean_participation_minus_anchor_actual_targets"]) >= 0.0
            and float(s24["mismatch_mean_participation_minus_anchor_actual_targets"]) >= 0.0,
        "wr_r15_follows_participation_lt50pct":
            float(pooled["mismatch_candidate_follow_participation_rate"]) < 0.50,
        "mismatch_final_target_error_ge5pct_worse_than_match":
            float(mismatch_err) >= 1.05 * float(match_err),
        "strict_prior_future_violations_zero":
            int(reconstruction_audit["strict_prior_future_violations"]) == 0,
        "sportsbook_inputs_zero": True,
        "candidate_variants_scored_zero": True,
    }
    warranted = all(criteria.values())
    disposition = (
        "WR_ANCHOR_ROLE_TRANSMISSION_GAP_WARRANTED"
        if warranted else "WR_ANCHOR_ROLE_TRANSMISSION_NO_GAP_CLOSED"
    )

    source_audit = {
        "version": VERSION,
        "wr_r15_authority_run": 34238301577,
        "wr_r15_authority_artifact": 10061328722,
        "authority_seasons": list(SEASONS),
        **reconstruction_audit,
        **parity_audit,
        "eligible_team_games": int(len(team_games)),
        "max_anchor_entitlement_immutability_gap": float(
            team_games["anchor_entitlement_immutability_gap"].max()
        ),
        "sportsbook_inputs_used": 0,
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
    }
    payload = {
        "version": VERSION,
        "disposition": disposition,
        "structural_hypothesis_warranted": bool(warranted),
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "production_mutations": 0,
        "mechanical_repair": (
            "reconstruct_full_fold_safe_WR_universe_to_restore_true_M38_anchor;"
            "frozen_criteria_unchanged"
        ),
        "criteria": criteria,
        "pooled": {
            "eligible_team_games": int(pooled["eligible_team_games"]),
            "mismatch_team_games": int(pooled["mismatch_team_games"]),
            "mismatch_rate": float(pooled["mismatch_rate"]),
            "participation_top_hit_minus_anchor": float(
                pooled["mismatch_participation_hit_minus_anchor"]
            ),
            "participation_minus_anchor_actual_targets": float(
                pooled["mismatch_mean_participation_minus_anchor_actual_targets"]
            ),
            "candidate_follow_participation_rate": float(
                pooled["mismatch_candidate_follow_participation_rate"]
            ),
            "mismatch_candidate_abs_error_per_scored_wr": mismatch_err,
            "match_candidate_abs_error_per_scored_wr": match_err,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out_dir / "player_detail_anchor_transmission.csv", index=False)
    team_games.to_csv(args.out_dir / "team_game_anchor_transmission.csv", index=False)
    cohort.to_csv(args.out_dir / "cohort_summary.csv", index=False)
    season.to_csv(args.out_dir / "season_summary.csv", index=False)
    parity_rows.to_csv(args.out_dir / "authority_parity_rows.csv", index=False)
    (args.out_dir / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# WR Anchor / Role-Transmission Audit V1",
        "",
        f"Disposition: **{disposition}**",
        "",
        "- candidate variants scored: **0**",
        "- parameters fit: **0**",
        "- sportsbook inputs: **0**",
        "- first-run stop: **MECHANICAL / INCOMPLETE SCORED ARTIFACT OMITTED TRUE ANCHORS**",
        "- bounded repair: **FULL FOLD-SAFE AUTHORITY RECONSTRUCTION; SCIENCE UNCHANGED**",
        "",
        "## Authority parity",
        "",
        f"- scored authority rows reproduced: {parity_audit['authority_rows']}",
        f"- max baseline entitlement gap: {parity_audit['max_baseline_entitlement_gap']:.3g}",
        f"- max candidate entitlement gap: {parity_audit['max_candidate_entitlement_gap']:.3g}",
        f"- max actual-label gap: {parity_audit['max_actual_target_label_gap']:.3g}",
        "",
        "## Pooled",
        "",
        f"- eligible team-games: {int(pooled['eligible_team_games'])}",
        f"- anchor/participation mismatch team-games: {int(pooled['mismatch_team_games'])} ({float(pooled['mismatch_rate']):.2%})",
        f"- mismatch participation-leader actual-top hit advantage vs M38 anchor: {float(pooled['mismatch_participation_hit_minus_anchor']):+.2%}",
        f"- mismatch mean actual targets, participation leader minus anchor: {float(pooled['mismatch_mean_participation_minus_anchor_actual_targets']):+.4f}",
        f"- mismatch WR-R15 final leader follows participation leader: {float(pooled['mismatch_candidate_follow_participation_rate']):.2%}",
        f"- mismatch candidate target AE / scored WR: {mismatch_err:.6f}",
        f"- match candidate target AE / scored WR: {match_err:.6f}",
        "",
        "## Frozen criteria",
        "",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in criteria.items()]
    lines += [
        "",
        "This audit does not authorize a WR1 or hierarchy change. "
        "A separate candidate may be frozen only if every diagnostic criterion passes.",
    ]
    (args.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
