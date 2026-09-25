#!/usr/bin/env python3
"""Diagnostic-only audit of WR anchor / current-role transmission.

Consumes the frozen WR-R15 OOS confirmation predictions and the exact
participation source / strict-prior feature builder used by WR-R15. It does not
construct or score a new projection candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.persist_wr_te_production_order_historical_v1 import (
    _load_participation_snaps,
    _wr_strict_prior_features,
)
from scripts.utils.canonical_names import canon_team

VERSION = "WR_ANCHOR_ROLE_TRANSMISSION_AUDIT_V1"
BASELINE_VARIANT = "M38_EXPLICIT_BASELINE"
CANDIDATE_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
SEASONS = (2023, 2024)
KEYS = ["season", "week", "event_id", "team", "player_clean_key"]


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


def prepare_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    need = {
        "variant", "event_id", "team", "player_clean_key", "player",
        "wr_rank", "entitlement_tgt_share", "pred_targets", "season",
        "week", "actual_targets",
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
        if d.duplicated(KEYS).any():
            bad = d.loc[d.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
            raise RuntimeError(f"{label} duplicate prediction identities: {bad}")

    b = b[KEYS + [
        "player", "wr_rank", "entitlement_tgt_share", "pred_targets", "actual_targets"
    ]].rename(columns={
        "wr_rank": "baseline_wr_rank",
        "entitlement_tgt_share": "baseline_entitlement_tgt_share",
        "pred_targets": "baseline_pred_targets",
        "actual_targets": "baseline_actual_targets",
    })
    c = c[KEYS + [
        "entitlement_tgt_share", "pred_targets", "actual_targets"
    ]].rename(columns={
        "entitlement_tgt_share": "candidate_entitlement_tgt_share",
        "pred_targets": "candidate_pred_targets",
        "actual_targets": "candidate_actual_targets",
    })
    out = b.merge(c, on=KEYS, how="inner", validate="one_to_one")
    if len(out) != len(b) or len(out) != len(c):
        raise RuntimeError(
            f"baseline/candidate identity universe mismatch b={len(b)} c={len(c)} joined={len(out)}"
        )
    actual_gap = (
        num(out["baseline_actual_targets"]) - num(out["candidate_actual_targets"])
    ).abs()
    if actual_gap.max() > 1e-12:
        raise RuntimeError("baseline/candidate actual target labels differ")
    out["actual_targets"] = num(out["baseline_actual_targets"])
    out["baseline_wr_rank"] = num(out["baseline_wr_rank"]).astype("Int64")
    for col in (
        "baseline_entitlement_tgt_share", "candidate_entitlement_tgt_share",
        "baseline_pred_targets", "candidate_pred_targets",
    ):
        out[col] = num(out[col])
    if out[[
        "actual_targets", "baseline_entitlement_tgt_share",
        "candidate_entitlement_tgt_share", "baseline_pred_targets",
        "candidate_pred_targets",
    ]].isna().any().any():
        raise RuntimeError("non-numeric authority prediction values")
    return out.sort_values(KEYS).reset_index(drop=True)


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
    z = z.sort_values([col, "player_clean_key"], ascending=[False, True], kind="mergesort")
    z["rank"] = np.arange(1, len(z) + 1, dtype=float)
    hit = z.loc[z["player_clean_key"].eq(str(player_key)), "rank"]
    return float(hit.iloc[0]) if len(hit) else np.nan


def build_team_games(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["season", "week", "event_id", "team"]
    for keys, g in detail.groupby(group_cols, sort=True):
        season, week, event_id, team = keys
        anchors = g.loc[g["baseline_wr_rank"].eq(1)]
        if len(anchors) != 1:
            raise RuntimeError(
                f"expected one M38 anchor {season} W{week} {team}, found {len(anchors)}"
            )
        anchor = str(anchors.iloc[0]["player_clean_key"])

        eligible = (
            g["prior1_same_team"].fillna(False).astype(bool)
            & num(g["prior1_same_team_offense_pct"]).notna()
        )
        snap_leader = deterministic_leader(
            g, "prior1_same_team_offense_pct", eligibility=eligible
        )
        if not snap_leader:
            # Source coverage is measured explicitly. No imputation.
            continue

        candidate_leader = deterministic_leader(g, "candidate_entitlement_tgt_share")
        actual_max = float(num(g["actual_targets"]).max())
        actual_top = set(
            g.loc[num(g["actual_targets"]).eq(actual_max), "player_clean_key"].astype(str)
        )

        baseline_abs = (num(g["baseline_pred_targets"]) - num(g["actual_targets"])).abs()
        candidate_abs = (num(g["candidate_pred_targets"]) - num(g["actual_targets"])).abs()

        rec = {
            "season": int(season),
            "week": int(week),
            "event_id": str(event_id),
            "team": str(team),
            "n_wr": int(len(g)),
            "m38_anchor_key": anchor,
            "participation_leader_key": snap_leader,
            "candidate_entitlement_leader_key": candidate_leader,
            "anchor_participation_mismatch": int(anchor != snap_leader),
            "candidate_followed_participation_leader": int(candidate_leader == snap_leader),
            "candidate_leader_changed_from_anchor": int(candidate_leader != anchor),
            "anchor_actual_top_hit": int(anchor in actual_top),
            "participation_leader_actual_top_hit": int(snap_leader in actual_top),
            "candidate_leader_actual_top_hit": int(candidate_leader in actual_top),
            "actual_top_target_count": int(len(actual_top)),
            "anchor_actual_targets": player_value(g, anchor, "actual_targets"),
            "participation_leader_actual_targets": player_value(g, snap_leader, "actual_targets"),
            "candidate_leader_actual_targets": player_value(g, candidate_leader, "actual_targets"),
            "anchor_baseline_entitlement": player_value(
                g, anchor, "baseline_entitlement_tgt_share"
            ),
            "anchor_candidate_entitlement": player_value(
                g, anchor, "candidate_entitlement_tgt_share"
            ),
            "participation_leader_baseline_entitlement": player_value(
                g, snap_leader, "baseline_entitlement_tgt_share"
            ),
            "participation_leader_candidate_entitlement": player_value(
                g, snap_leader, "candidate_entitlement_tgt_share"
            ),
            "participation_leader_baseline_rank": rank_for_player(
                g, snap_leader, "baseline_entitlement_tgt_share"
            ),
            "participation_leader_candidate_rank": rank_for_player(
                g, snap_leader, "candidate_entitlement_tgt_share"
            ),
            "baseline_target_abs_error_sum": float(baseline_abs.sum()),
            "candidate_target_abs_error_sum": float(candidate_abs.sum()),
            "baseline_target_abs_error_per_wr": float(baseline_abs.mean()),
            "candidate_target_abs_error_per_wr": float(candidate_abs.mean()),
            "candidate_minus_baseline_abs_error_per_wr": float(
                candidate_abs.mean() - baseline_abs.mean()
            ),
            "max_actual_targets": actual_max,
        }
        rec["participation_minus_anchor_actual_targets"] = (
            rec["participation_leader_actual_targets"] - rec["anchor_actual_targets"]
        )
        rec["anchor_entitlement_immutability_gap"] = abs(
            rec["anchor_candidate_entitlement"] - rec["anchor_baseline_entitlement"]
        )
        rows.append(rec)

    if not rows:
        raise RuntimeError("anchor transmission audit produced zero eligible team-games")
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def cohort_row(d: pd.DataFrame, scope: str, cohort: str) -> dict:
    return {
        "scope": scope,
        "cohort": cohort,
        "team_games": int(len(d)),
        "wr_rows": int(d["n_wr"].sum()) if len(d) else 0,
        "mismatch_rate": float(d["anchor_participation_mismatch"].mean()) if len(d) else np.nan,
        "candidate_follow_participation_rate": (
            float(d["candidate_followed_participation_leader"].mean()) if len(d) else np.nan
        ),
        "candidate_leader_changed_from_anchor_rate": (
            float(d["candidate_leader_changed_from_anchor"].mean()) if len(d) else np.nan
        ),
        "anchor_actual_top_hit_rate": (
            float(d["anchor_actual_top_hit"].mean()) if len(d) else np.nan
        ),
        "participation_actual_top_hit_rate": (
            float(d["participation_leader_actual_top_hit"].mean()) if len(d) else np.nan
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
            float(d["participation_minus_anchor_actual_targets"].mean()) if len(d) else np.nan
        ),
        "baseline_abs_error_per_wr": (
            float(
                d["baseline_target_abs_error_sum"].sum() / max(1, int(d["n_wr"].sum()))
            )
            if len(d) else np.nan
        ),
        "candidate_abs_error_per_wr": (
            float(
                d["candidate_target_abs_error_sum"].sum() / max(1, int(d["n_wr"].sum()))
            )
            if len(d) else np.nan
        ),
    }


def summaries(team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows = []
    season_rows = []
    for scope, d in [("POOLED", team_games)] + [
        (str(s), team_games.loc[team_games["season"].eq(s)]) for s in SEASONS
    ]:
        cohort_rows.append(cohort_row(d, scope, "ALL"))
        cohort_rows.append(
            cohort_row(
                d.loc[d["anchor_participation_mismatch"].eq(1)],
                scope,
                "ANCHOR_PARTICIPATION_MISMATCH",
            )
        )
        cohort_rows.append(
            cohort_row(
                d.loc[d["anchor_participation_mismatch"].eq(0)],
                scope,
                "ANCHOR_PARTICIPATION_MATCH",
            )
        )
        mismatch = d.loc[d["anchor_participation_mismatch"].eq(1)]
        match = d.loc[d["anchor_participation_mismatch"].eq(0)]
        season_rows.append({
            "scope": scope,
            "eligible_team_games": int(len(d)),
            "mismatch_team_games": int(len(mismatch)),
            "mismatch_rate": float(mismatch.shape[0] / len(d)) if len(d) else np.nan,
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
            "mismatch_candidate_abs_error_per_wr": (
                float(
                    mismatch["candidate_target_abs_error_sum"].sum()
                    / max(1, int(mismatch["n_wr"].sum()))
                )
                if len(mismatch) else np.nan
            ),
            "match_candidate_abs_error_per_wr": (
                float(
                    match["candidate_target_abs_error_sum"].sum()
                    / max(1, int(match["n_wr"].sum()))
                )
                if len(match) else np.nan
            ),
        })
    return pd.DataFrame(cohort_rows), pd.DataFrame(season_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = prepare_predictions(read(args.predictions, "WR-R15 OOS predictions"))

    snaps, dup_rate, source_seasons = _load_participation_snaps()
    feat, future_violations = _wr_strict_prior_features(authority.copy(), snaps)
    if len(feat) != len(authority):
        raise RuntimeError("strict-prior feature attachment changed authority row count")

    detail = feat.copy()
    team_games = build_team_games(detail)
    cohort, season = summaries(team_games)

    pooled = season.loc[season["scope"].eq("POOLED")].iloc[0]
    s23 = season.loc[season["scope"].eq("2023")].iloc[0]
    s24 = season.loc[season["scope"].eq("2024")].iloc[0]

    mismatch = team_games.loc[team_games["anchor_participation_mismatch"].eq(1)]
    match = team_games.loc[team_games["anchor_participation_mismatch"].eq(0)]
    mismatch_err = (
        mismatch["candidate_target_abs_error_sum"].sum()
        / max(1, int(mismatch["n_wr"].sum()))
    )
    match_err = (
        match["candidate_target_abs_error_sum"].sum()
        / max(1, int(match["n_wr"].sum()))
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
        "strict_prior_future_violations_zero": int(future_violations) == 0,
        "sportsbook_inputs_zero": True,
        "candidate_variants_scored_zero": True,
    }
    warranted = all(criteria.values())
    disposition = (
        "WR_ANCHOR_ROLE_TRANSMISSION_GAP_WARRANTED"
        if warranted
        else "WR_ANCHOR_ROLE_TRANSMISSION_NO_GAP_CLOSED"
    )

    source_audit = {
        "version": VERSION,
        "wr_r15_authority_run": 34238301577,
        "wr_r15_authority_artifact": 10061328722,
        "authority_seasons": list(SEASONS),
        "raw_snap_duplicate_rate": float(dup_rate),
        "snap_source_seasons": [int(x) for x in source_seasons],
        "strict_prior_future_violations": int(future_violations),
        "eligible_team_games": int(len(team_games)),
        "eligible_player_rows": int(len(detail)),
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
            "mismatch_candidate_abs_error_per_wr": float(mismatch_err),
            "match_candidate_abs_error_per_wr": float(match_err),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "player_detail_anchor_transmission.csv", index=False)
    team_games.to_csv(args.out_dir / "team_game_anchor_transmission.csv", index=False)
    cohort.to_csv(args.out_dir / "cohort_summary.csv", index=False)
    season.to_csv(args.out_dir / "season_summary.csv", index=False)
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
        "",
        "## Pooled",
        "",
        f"- eligible team-games: {int(pooled['eligible_team_games'])}",
        f"- anchor/participation mismatch team-games: {int(pooled['mismatch_team_games'])} ({float(pooled['mismatch_rate']):.2%})",
        f"- mismatch participation-leader actual-top hit advantage vs M38 anchor: {float(pooled['mismatch_participation_hit_minus_anchor']):+.2%}",
        f"- mismatch mean actual targets, participation leader minus anchor: {float(pooled['mismatch_mean_participation_minus_anchor_actual_targets']):+.4f}",
        f"- mismatch WR-R15 final leader follows participation leader: {float(pooled['mismatch_candidate_follow_participation_rate']):.2%}",
        f"- mismatch candidate target AE / WR: {float(mismatch_err):.6f}",
        f"- match candidate target AE / WR: {float(match_err):.6f}",
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
