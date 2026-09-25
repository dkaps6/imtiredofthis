#!/usr/bin/env python3
"""Diagnostic-only audit of WR anchor / current-role transmission.

Bounded mechanical repair history
---------------------------------
Run 1 proved the frozen WR-R15 scored artifact is a scored subset and can omit
true M38 anchors. Run 2 rebuilt the full state through today's production
helpers, but the frozen-authority parity gate correctly rejected that replay.

This version intentionally imports the exact authority-era WR-R14/R15 builders
from the WR-R15 run head (02c3dd1...) supplied first on PYTHONPATH by the
workflow. It consumes the frozen WR-R15 fold coefficients rather than refitting.

No scientific criterion, cohort, season, target label, or candidate is changed.
Candidate variants scored remains zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_wr_r14_participation_entitlement_v1 import (
    WR_POS,
    _build_bundle_frame,
    _strict_prior_snap_features,
    _target_actuals,
)
from scripts.backtest.evaluate_wr_r15_wr1_anchor_participation_v1 import (
    FEATURES as WR_FEATURES,
    _apply_model as authority_apply_model,
)
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps
from scripts.utils.canonical_names import canon_team

VERSION = "WR_ANCHOR_ROLE_TRANSMISSION_AUDIT_V1"
AUTHORITY_SOURCE_COMMIT = "02c3dd1a681d4ab2953683039e39830554f9ec9f"
BASELINE_VARIANT = "M38_EXPLICIT_BASELINE"
CANDIDATE_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
SEASONS = (2023, 2024)
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


def num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


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
            bad = d.loc[
                d.duplicated(AUTH_KEYS, keep=False), AUTH_KEYS
            ].head(10).to_dict("records")
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
        raise RuntimeError("authority baseline/candidate identity universe mismatch")
    if (
        num(out["authority_actual_targets"])
        - num(out["authority_candidate_actual_targets"])
    ).abs().max() > 1e-12:
        raise RuntimeError("authority actual labels differ by variant")
    for col in [
        "authority_baseline_entitlement", "authority_candidate_entitlement",
        "authority_baseline_pred_targets", "authority_candidate_pred_targets",
        "authority_actual_targets",
    ]:
        out[col] = num(out[col])
        if out[col].isna().any():
            raise RuntimeError(f"authority non-numeric values in {col}")
    return out


class FrozenFoldModel:
    """Exact predict() contract reconstructed from frozen scaler/ridge rows."""

    def __init__(self, coef_path: Path, test_season: int):
        x = read(coef_path, "WR-R15 fold coefficients")
        x["test_season"] = num(x["test_season"]).astype("Int64")
        g = x.loc[x["test_season"].eq(int(test_season))].copy()
        if set(g["feature"].astype(str)) != set(WR_FEATURES):
            raise RuntimeError(
                f"WR-R15 feature contract mismatch test={test_season}"
            )
        g = g.set_index("feature").loc[WR_FEATURES]
        self.mean = num(g["scaler_mean"]).to_numpy(float)
        self.scale = num(g["scaler_scale"]).to_numpy(float)
        self.coef = num(g["standardized_coefficient"]).to_numpy(float)
        ints = num(g["ridge_intercept"]).dropna().unique()
        if len(ints) != 1:
            raise RuntimeError("WR-R15 frozen fold intercept is not unique")
        self.intercept = float(ints[0])
        if not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise RuntimeError("WR-R15 frozen scaler invalid")

    def predict(self, x) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            a = x[WR_FEATURES].to_numpy(float)
        else:
            a = np.asarray(x, dtype=float)
        return ((a - self.mean) / self.scale) @ self.coef + self.intercept


def is_wr(value: object) -> bool:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    return p in WR_POS or p.startswith("WR")


def reconstruct_authority_state(
    *,
    data_dirs: dict[int, Path],
    logs_by_season: dict[int, pd.DataFrame],
    coefficients: Path,
) -> tuple[pd.DataFrame, dict]:
    snaps, dup_rate, source_seasons = _load_snaps()
    rows: list[pd.DataFrame] = []
    future_total = 0
    fold_future_total = 0
    max_anchor_gap = 0.0
    max_room_gap = 0.0

    for season in SEASONS:
        model = FrozenFoldModel(coefficients, season)
        logs = logs_by_season[int(season)]
        for week in range(1, 19):
            baseline = _build_bundle_frame(
                season=int(season),
                week=int(week),
                prior_season=int(season - 1),
                data_dir=data_dirs[int(season)],
                logs=logs,
            )
            baseline["team"] = baseline["team"].map(canon_team)
            pos = baseline["position"].fillna("").astype(str).str.upper().str.strip()
            wr_base = baseline.loc[pos.map(is_wr)].copy()
            if wr_base.empty:
                raise RuntimeError(f"{season} W{week} authority replay found zero WR rows")

            # Full-WR strict-prior participation state, including the immutable
            # M38 anchor. Same source and cutoff function as the original R15.
            all_feat, future = _strict_prior_snap_features(wr_base.copy(), snaps)
            future_total += int(future)
            if int(future) != 0:
                raise RuntimeError(
                    f"{season} W{week} full-WR participation leakage={future}"
                )

            candidate, _, audits, fold_future = authority_apply_model(
                baseline.copy(), snaps, model
            )
            fold_future_total += int(fold_future)
            if int(fold_future) != 0:
                raise RuntimeError(
                    f"{season} W{week} R15 apply participation leakage={fold_future}"
                )
            if audits:
                max_anchor_gap = max(
                    max_anchor_gap,
                    max(abs(float(a["anchor_entitlement_delta"])) for a in audits),
                )
                max_room_gap = max(
                    max_room_gap,
                    max(abs(float(a["wr_room_mass_gap"])) for a in audits),
                )

            candidate["team"] = candidate["team"].map(canon_team)
            cpos = candidate["position"].fillna("").astype(str).str.upper().str.strip()
            wr_final = candidate.loc[cpos.map(is_wr)].copy()

            key = ["event_id", "team", "player_clean_key"]
            z = (
                wr_base[key + ["player", "position", "entitlement_tgt_share"]]
                .rename(
                    columns={
                        "entitlement_tgt_share": "baseline_entitlement_tgt_share"
                    }
                )
                .merge(
                    all_feat[key + [
                        "prior1_same_team",
                        "prior1_same_team_offense_pct",
                        "prior1_same_team_offense_snaps",
                        "prior_count_same_team",
                    ]],
                    on=key,
                    how="left",
                    validate="one_to_one",
                )
                .merge(
                    wr_final[key + ["entitlement_tgt_share"]].rename(
                        columns={
                            "entitlement_tgt_share":
                                "candidate_entitlement_tgt_share"
                        }
                    ),
                    on=key,
                    how="left",
                    validate="one_to_one",
                )
            )
            z["season"] = int(season)
            z["week"] = int(week)

            actual = _target_actuals(logs, int(season), int(week))
            actual["team"] = actual["team"].map(canon_team)
            z = z.merge(
                actual,
                on=["team", "player_clean_key"],
                how="left",
                validate="one_to_one",
            )
            z["actual_targets"] = num(z["actual_targets"]).fillna(0.0)
            rows.append(z)

    full = pd.concat(rows, ignore_index=True)
    if full.duplicated(FULL_KEYS).any():
        raise RuntimeError("authority replay produced duplicate full WR identities")
    return full.sort_values(FULL_KEYS).reset_index(drop=True), {
        "authority_source_commit": AUTHORITY_SOURCE_COMMIT,
        "raw_snap_duplicate_rate": float(dup_rate),
        "snap_source_seasons": [int(x) for x in source_seasons],
        "strict_prior_future_violations": int(future_total),
        "r15_apply_future_violations": int(fold_future_total),
        "reconstructed_wr_rows": int(len(full)),
        "reconstructed_team_games": int(
            full[["season", "week", "event_id", "team"]]
            .drop_duplicates().shape[0]
        ),
        "max_anchor_entitlement_gap": float(max_anchor_gap),
        "max_wr_room_mass_gap": float(max_room_gap),
    }


def authority_parity(
    full: pd.DataFrame, authority: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    recon = full[
        AUTH_KEYS + [
            "baseline_entitlement_tgt_share",
            "candidate_entitlement_tgt_share",
            "actual_targets",
        ]
    ].copy()
    if recon.duplicated(AUTH_KEYS).any():
        raise RuntimeError("authority replay parity keys are not unique")
    joined = authority.merge(recon, on=AUTH_KEYS, how="left", validate="one_to_one")
    missing = int(joined["baseline_entitlement_tgt_share"].isna().sum())
    if missing:
        sample = joined.loc[
            joined["baseline_entitlement_tgt_share"].isna(), AUTH_KEYS
        ].head(10).to_dict("records")
        raise RuntimeError(
            f"authority replay missing scored rows n={missing} sample={sample}"
        )
    base_gap = (
        joined["authority_baseline_entitlement"]
        - joined["baseline_entitlement_tgt_share"]
    ).abs()
    cand_gap = (
        joined["authority_candidate_entitlement"]
        - joined["candidate_entitlement_tgt_share"]
    ).abs()
    label_gap = (
        joined["authority_actual_targets"] - joined["actual_targets"]
    ).abs()
    audit = {
        "authority_rows": int(len(joined)),
        "authority_reconstruction_coverage": 1.0,
        "max_baseline_entitlement_gap": float(base_gap.max()),
        "max_candidate_entitlement_gap": float(cand_gap.max()),
        "max_actual_target_label_gap": float(label_gap.max()),
    }
    if audit["max_baseline_entitlement_gap"] > 1e-10:
        raise RuntimeError(f"baseline authority parity failed: {audit}")
    if audit["max_candidate_entitlement_gap"] > 1e-10:
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
            raise RuntimeError(f"missing M38 anchor {keys}")
        eligible = (
            g["prior1_same_team"].fillna(False).astype(bool)
            & num(g["prior1_same_team_offense_pct"]).notna()
        )
        participation = deterministic_leader(
            g, "prior1_same_team_offense_pct", eligibility=eligible
        )
        if not participation:
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
        raise RuntimeError("zero full-state team-games overlap scored WR-R15 authority")
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


def summaries(team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    ap.add_argument("--data-2023", type=Path, required=True)
    ap.add_argument("--logs-2023", type=Path, required=True)
    ap.add_argument("--data-2024", type=Path, required=True)
    ap.add_argument("--logs-2024", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = prepare_authority(
        read(args.authority_predictions, "WR-R15 OOS predictions")
    )
    full, reconstruction_audit = reconstruct_authority_state(
        data_dirs={2023: args.data_2023, 2024: args.data_2024},
        logs_by_season={
            2023: read(args.logs_2023, "2023 fold player logs"),
            2024: read(args.logs_2024, "2024 fold player logs"),
        },
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
            int(reconstruction_audit["strict_prior_future_violations"]) == 0
            and int(reconstruction_audit["r15_apply_future_violations"]) == 0,
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
            "exact_authority-era_R14_R15_source_replay;"
            "frozen_science_unchanged"
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
        f"- authority source commit: **{AUTHORITY_SOURCE_COMMIT}**",
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
