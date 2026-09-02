#!/usr/bin/env python3
"""RB STACK6E: full-roster competition parity + exact inactive competitor state.

Frozen protocol:
  docs/migrations/RB_STACK6E_INACTIVE_COMPETITOR_PARITY_PLAN.md

Research only. 2024 fit, 2025 evaluation. Sportsbook is loaded only after the
football disposition is frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_stack2_enriched_allocation as s2

START_WEEK = 6
AVAIL_FEATURES = [
    "inactive_comp_count",
    "inactive_comp_prior3_share",
    "inactive_comp_prior3_snap",
    "inactive_above_count",
    "effective_active_depth_rank",
]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    z = pd.read_csv(hits[0], low_memory=False)
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def metric(y, p):
    q = pd.DataFrame({"y": num(y), "p": num(p)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = q.p - q.y
    corr = q.y.corr(q.p) if len(q) >= 3 and q.y.nunique() > 1 and q.p.nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.square(e).mean())),
        "bias": float(e.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
    }


def prepare_parent(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    required = [
        "season", "week", "team", "player", "depth_rank", "m94c_share", "m94c_att",
        "stack_att", "enriched_att", "p3_parent", "actual_rush_att", "actual_rush_yards",
    ]
    miss = [c for c in required if c not in z.columns]
    if miss:
        raise RuntimeError(f"STACK6E parent missing {miss}")
    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(s2.tm)
    z["name_key"] = z.get("name_key", z.get("join_key", z.player)).astype(str).map(s2.nk)
    z["depth_rank"] = num(z.depth_rank)
    z["m94c_share"] = num(z.m94c_share).fillna(0.0)
    z["m94c_att"] = num(z.m94c_att)
    z["actual_rush_att"] = num(z.actual_rush_att)
    z["actual_rush_yards"] = num(z.actual_rush_yards)
    z["parent_att"] = np.where(z.week.eq(1), num(z.stack_att), num(z.enriched_att))
    z["parent_yards"] = num(z.p3_parent)
    z["parent_ypc"] = np.where(z.parent_att.abs().gt(0.20), z.parent_yards / z.parent_att, np.nan)
    risk_col = "stack6_risk" if "stack6_risk" in z.columns else "state_m95f_risk_stack4"
    if risk_col not in z.columns:
        raise RuntimeError("STACK6E parent missing frozen M95F risk state")
    z["stack6e_risk"] = z[risk_col].astype(str).str.lower().isin(["true", "1", "yes"])
    z["stack6e_domain"] = z.week.ge(START_WEEK) & (~z.stack6e_risk) & z.depth_rank.ge(2)
    return z


def load_stack2_sources():
    import nflreadpy as nfl

    seasons = [2023, 2024, 2025]
    logs = s2.load_weekly_logs(seasons)
    rosters = s2.load_rosters([2024, 2025])
    sched = s2.lower(s2.pdx(nfl.load_schedules(seasons=[2024, 2025])))
    depth = s2.depth_tables([2024, 2025], sched)
    snaps = s2.load_snaps(seasons)
    injuries = s2.load_injuries([2024, 2025])
    return logs, rosters, depth, snaps, injuries


def full_roster_eval(rosters, logs, depth, snaps, injuries) -> pd.DataFrame:
    q = rosters.loc[rosters.season.eq(2025)].copy()
    q = q.merge(depth, on=["season", "week", "team", "name_key"], how="left")
    q = q.merge(injuries, on=["season", "week", "team", "name_key"], how="left")
    q = s2.enrich_history(q, logs, snaps)
    q = s2.add_team_competition(q)
    q = s2.finalize_features(q)
    return q


def add_inactive_competitor_features(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    for c in ["roster_inactive", "prior3_rb_share", "prior3_snap_pct", "depth_rank", "depth_rank_missing"]:
        if c not in z.columns:
            z[c] = np.nan
    z["roster_inactive"] = num(z.roster_inactive).fillna(0.0)
    z["prior3_rb_share"] = num(z.prior3_rb_share).fillna(0.0)
    z["prior3_snap_pct"] = num(z.prior3_snap_pct).fillna(0.0)
    z["depth_rank"] = num(z.depth_rank)
    z["depth_rank_missing"] = num(z.depth_rank_missing).fillna(1.0)

    rows = []
    for _, g in z.groupby(["season", "week", "team"], sort=False):
        for idx, r in g.iterrows():
            oth = g.loc[g.index != idx].copy()
            ina = num(oth.roster_inactive).fillna(0.0).eq(1.0)
            own_rank = float(r.depth_rank) if pd.notna(r.depth_rank) else np.nan
            known = num(oth.depth_rank_missing).fillna(1.0).eq(0.0) & num(oth.depth_rank).notna()
            inactive_above = int((ina & known & num(oth.depth_rank).lt(own_rank)).sum()) if np.isfinite(own_rank) else 0
            if float(r.depth_rank_missing) >= 0.5 or not np.isfinite(own_rank):
                effective_rank = own_rank if np.isfinite(own_rank) else 4.0
            else:
                effective_rank = max(1.0, own_rank - inactive_above)
            rows.append(
                {
                    "_idx": idx,
                    "inactive_comp_count": float(ina.sum()),
                    "inactive_comp_prior3_share": float((num(oth.prior3_rb_share).fillna(0.0) * ina.astype(float)).sum()),
                    "inactive_comp_prior3_snap": float((num(oth.prior3_snap_pct).fillna(0.0) * ina.astype(float)).sum()),
                    "inactive_above_count": float(inactive_above),
                    "effective_active_depth_rank": float(effective_rank),
                }
            )
    f = pd.DataFrame(rows).set_index("_idx")
    for c in AVAIL_FEATURES:
        z.loc[f.index, c] = f[c]
    return z


def target_context(parent: pd.DataFrame, full: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    keys = ["season", "week", "team", "name_key"]
    keep = keys + features
    if full.duplicated(keys).any():
        dup = full.loc[full.duplicated(keys, keep=False), keys].head(10)
        raise RuntimeError(f"STACK6E full-roster context duplicate keys:\n{dup}")
    ctx = full[keep].copy()
    z = parent.merge(ctx, on=keys, how="left", validate="one_to_one", suffixes=("", "_ctx"))
    return z


def normalize_scores(z: pd.DataFrame, raw_col: str, share_col: str) -> None:
    out = pd.Series(index=z.index, dtype=float)
    for _, g in z.groupby(["season", "week", "team"], sort=False):
        s = num(g[raw_col]).fillna(0.0).clip(lower=0.0)
        total = float(s.sum())
        if total <= 0:
            base = num(g.m94c_share).fillna(0.0).clip(lower=0.0)
            btot = float(base.sum())
            s = base / btot if btot > 0 else pd.Series(np.ones(len(g)) / max(len(g), 1), index=g.index)
        else:
            s = s / total
        out.loc[g.index] = s
    z[share_col] = out


def apply_arm(parent: pd.DataFrame, raw_score: np.ndarray, arm: str) -> pd.DataFrame:
    z = parent.copy()
    z[f"alloc_score_{arm}"] = raw_score
    normalize_scores(z, f"alloc_score_{arm}", f"alloc_share_{arm}")
    pool = z.groupby(["season", "week", "team"])["m94c_att"].transform("sum")
    z[f"candidate_share_{arm}"] = 0.50 * num(z.m94c_share).fillna(0.0) + 0.50 * num(z[f"alloc_share_{arm}"]).fillna(0.0)
    z[f"candidate_att_raw_{arm}"] = z[f"candidate_share_{arm}"] * pool

    eligible = z.stack6e_domain
    z[f"pred_att_{arm}"] = num(z.parent_att)
    z.loc[eligible, f"pred_att_{arm}"] = num(z.loc[eligible, f"candidate_att_raw_{arm}"])
    z[f"pred_yards_{arm}"] = num(z.parent_yards)
    yp = num(z.parent_ypc)
    repl = num(z[f"pred_att_{arm}"]) * yp
    usable = eligible & yp.notna()
    z.loc[usable, f"pred_yards_{arm}"] = repl.loc[usable]
    z[f"delta_att_{arm}"] = num(z[f"pred_att_{arm}"]) - num(z.parent_att)
    return z


def score_table(z: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    masks = {
        "all_rb_w6_18": z.week.ge(START_WEEK),
        "eligible_w6_18": z.stack6e_domain,
        "eligible_w13_18": z.stack6e_domain & z.week.ge(13),
        "m95f_risk_w6_18": z.stack6e_risk & z.week.ge(START_WEEK),
        "depth1_w6_18": z.depth_rank.eq(1) & z.week.ge(START_WEEK),
        "depth2_w6_18": z.depth_rank.eq(2) & z.week.ge(START_WEEK),
        "depth3plus_w6_18": z.depth_rank.ge(3) & z.week.ge(START_WEEK),
    }
    cols = {"P3_PARENT": ("parent_att", "parent_yards")}
    cols.update({a: (f"pred_att_{a}", f"pred_yards_{a}") for a in arms})
    rows = []
    for scope, mask in masks.items():
        g = z.loc[mask]
        for arm, (ac, yc) in cols.items():
            cm = metric(g.actual_rush_att, g[ac])
            ym = metric(g.actual_rush_yards, g[yc])
            rows.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "n": ym["n"],
                    "carry_mae": cm["mae"],
                    "carry_rmse": cm["rmse"],
                    "carry_bias": cm["bias"],
                    "carry_corr": cm["corr"],
                    "yard_mae": ym["mae"],
                    "yard_rmse": ym["rmse"],
                    "yard_bias": ym["bias"],
                    "yard_corr": ym["corr"],
                }
            )
    return pd.DataFrame(rows)


def retention(z: pd.DataFrame, scores: pd.DataFrame, arms: list[str]) -> tuple[pd.DataFrame, str]:
    def r(scope, arm):
        q = scores.loc[scores.scope.eq(scope) & scores.arm.eq(arm)]
        if q.empty:
            raise RuntimeError(f"missing score {scope}/{arm}")
        return q.iloc[0]

    base = r("eligible_w6_18", "P3_PARENT")
    allbase = r("all_rb_w6_18", "P3_PARENT")
    latebase = r("eligible_w13_18", "P3_PARENT")
    rows = []
    for arm in arms:
        e = r("eligible_w6_18", arm)
        a = r("all_rb_w6_18", arm)
        l = r("eligible_w13_18", arm)
        carry_gain = float(base.carry_mae - e.carry_mae)
        yard_gain = float(base.yard_mae - e.yard_mae)
        late_gain = float(latebase.yard_mae - l.yard_mae)
        all_reg = float(a.yard_mae - allbase.yard_mae)
        bias_worsen = abs(float(e.carry_bias)) - abs(float(base.carry_bias))
        risk = z.stack6e_risk & z.week.ge(START_WEEK)
        d1 = z.depth_rank.eq(1) & z.week.ge(START_WEEK)
        risk_change = float((num(z.loc[risk, f"pred_yards_{arm}"]) - num(z.loc[risk, "parent_yards"])).abs().max()) if risk.any() else 0.0
        d1_change = float((num(z.loc[d1, f"pred_yards_{arm}"]) - num(z.loc[d1, "parent_yards"])).abs().max()) if d1.any() else 0.0
        passed = int(
            carry_gain >= 0.20
            and yard_gain >= 0.15
            and late_gain > 0
            and all_reg <= 0.05
            and bias_worsen <= 0.25
            and risk_change <= 1e-9
            and d1_change <= 1e-9
        )
        rows.append(
            {
                "arm": arm,
                "feature_count": len(s2.FULL) + (len(AVAIL_FEATURES) if arm == "INACTIVE_COMPETITOR_STATE" else 0),
                "carry_mae_gain": carry_gain,
                "yard_mae_gain": yard_gain,
                "late_yard_mae_gain": late_gain,
                "all_rb_yard_mae_regression": all_reg,
                "carry_abs_bias_worsening": bias_worsen,
                "max_risk_yard_change": risk_change,
                "max_depth1_yard_change": d1_change,
                "gate_pass": passed,
            }
        )
    gates = pd.DataFrame(rows)
    passing = gates.loc[gates.gate_pass.eq(1)].copy()
    selected = "NONE"
    if len(passing):
        best = float(passing.yard_mae_gain.max())
        parity = passing.loc[passing.arm.eq("FULL_ROSTER_PARITY")]
        if len(parity) and float(parity.iloc[0].yard_mae_gain) >= best - 0.05:
            selected = "FULL_ROSTER_PARITY"
        else:
            selected = str(passing.sort_values(["yard_mae_gain", "arm"], ascending=[False, True]).iloc[0].arm)
    gates["selected_arm"] = selected
    return gates, selected


def market_audit(z: pd.DataFrame, market: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    c = market.copy()
    c["season"] = num(c.get("season", 2025)).fillna(2025).astype(int)
    c["week"] = num(c.week).astype(int)
    c["team"] = c.team.map(s2.tm)
    c["name_key"] = c.player.astype(str).map(s2.nk)
    keys = ["season", "week", "team", "name_key"]
    q = z.merge(c[keys + ["consensus_line"]].drop_duplicates(keys), on=keys, how="inner", validate="one_to_one")
    scopes = {
        "all": pd.Series(True, index=q.index),
        "eligible": q.stack6e_domain,
        "m95f_nonrisk": ~q.stack6e_risk,
        "depth2": q.depth_rank.eq(2),
        "depth3plus": q.depth_rank.ge(3),
    }
    pred = {"P3_PARENT": "parent_yards", **{a: f"pred_yards_{a}" for a in arms}, "VEGAS_CONSENSUS": "consensus_line"}
    rows = []
    for arm, col in pred.items():
        for scope, mask in scopes.items():
            rows.append({"scope": scope, "arm": arm, **metric(q.loc[mask, "actual_rush_yards"], q.loc[mask, col])})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6-root", type=Path, required=True)
    ap.add_argument("--market-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    parent = prepare_parent(one(a.stack6_root, "stack6_2025_casebook.csv"))
    # Hard audit the discovered mismatch before rebuilding it.
    original_inactive_target_rows = int(num(parent.get("roster_inactive", 0)).fillna(0).eq(1).sum())
    if original_inactive_target_rows != 0:
        raise RuntimeError(f"STACK6E expected frozen P3 target roster_inactive to be degenerate zero; found {original_inactive_target_rows}")

    logs, rosters, depth, snaps, injuries = load_stack2_sources()
    train = s2.finalize_features(s2.build_training(rosters, logs, depth, snaps, injuries))
    full25 = full_roster_eval(rosters, logs, depth, snaps, injuries)
    train = add_inactive_competitor_features(train)
    full25 = add_inactive_competitor_features(full25)

    # Source/state audit is independent of outcomes.
    ina25 = full25.loc[num(full25.roster_inactive).eq(1)]
    source_audit = pd.DataFrame([
        {
            "p3_target_rows": int(len(parent)),
            "p3_target_inactive_rows": original_inactive_target_rows,
            "full_2025_roster_rows": int(len(full25)),
            "full_2025_inactive_rbfb_rows": int(len(ina25)),
            "target_team_weeks": int(parent[["season", "week", "team"]].drop_duplicates().shape[0]),
            "full_roster_team_weeks": int(full25[["season", "week", "team"]].drop_duplicates().shape[0]),
            "training_rows_2024": int(len(train)),
            "full_feature_count": int(len(s2.FULL)),
            "availability_feature_count": int(len(AVAIL_FEATURES)),
            "sportsbook_used_for_source_audit": 0,
            "outcome_used_for_feature_selection": 0,
        }
    ])

    parity = target_context(parent, full25, list(s2.FULL))
    avail = target_context(parent, full25, list(s2.FULL) + AVAIL_FEATURES)
    parity_match = float(parity[s2.FULL].notna().any(axis=1).mean())
    avail_match = float(avail[AVAIL_FEATURES].notna().all(axis=1).mean())
    if parity_match < 0.98 or avail_match < 0.98:
        raise RuntimeError(f"STACK6E target context coverage too low: parity={parity_match:.4f} availability={avail_match:.4f}")

    # Frozen 2024-only fit.
    pred_parity = s2.fit_predict(train, parity, list(s2.FULL), seed=17)
    pred_avail = s2.fit_predict(train, avail, list(s2.FULL) + AVAIL_FEATURES, seed=17)

    out = apply_arm(parity, pred_parity, "FULL_ROSTER_PARITY")
    # Keep one canonical frame; availability context contains the same parent rows.
    tmp = apply_arm(avail, pred_avail, "INACTIVE_COMPETITOR_STATE")
    for c in [x for x in tmp.columns if x.startswith(("alloc_score_INACTIVE", "alloc_share_INACTIVE", "candidate_share_INACTIVE", "candidate_att_raw_INACTIVE", "pred_att_INACTIVE", "pred_yards_INACTIVE", "delta_att_INACTIVE"))]:
        out[c] = tmp[c].to_numpy()

    arms = ["FULL_ROSTER_PARITY", "INACTIVE_COMPETITOR_STATE"]
    scores = score_table(out, arms)
    gates, selected = retention(out, scores, arms)

    if selected == "FULL_ROSTER_PARITY":
        disposition = "STACK6E_RETAIN_FULL_ROSTER_PARITY"
    elif selected == "INACTIVE_COMPETITOR_STATE":
        disposition = "STACK6E_RETAIN_INACTIVE_COMPETITOR_STATE"
    else:
        disposition = "STACK6E_NO_RETAINABLE_INACTIVE_COMPETITOR_REPAIR"

    disposition_df = pd.DataFrame([
        {
            "selected_arm": selected,
            "passing_arm_count": int(gates.gate_pass.sum()),
            "disposition": disposition,
            "fit_season": 2024,
            "evaluation_season": 2025,
            "stack2_model_reused": 1,
            "hyperparameter_search": 0,
            "feature_search": 0,
            "threshold_search": 0,
            "weight_search": 0,
            "population_search": 0,
            "sportsbook_upstream": 0,
            "production_change": 0,
            "live_2026_official_inactive_feed_required_before_promotion": 1,
        }
    ])

    # Freeze football disposition on disk before market is read.
    source_audit.to_csv(a.out_dir / "stack6e_source_state_audit.csv", index=False)
    scores.to_csv(a.out_dir / "stack6e_score_table.csv", index=False)
    gates.to_csv(a.out_dir / "stack6e_retention_gates.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6e_disposition.csv", index=False)

    market = one(a.market_root, "rb_market_casebook.csv")
    market_metrics = market_audit(out, market, arms)
    market_metrics.to_csv(a.out_dir / "stack6e_market_metrics.csv", index=False)

    keep_case = [
        "season", "week", "team", "player", "name_key", "depth_rank", "stack6e_risk", "stack6e_domain",
        "actual_rush_att", "actual_rush_yards", "parent_att", "parent_yards", "parent_ypc", "m94c_share", "m94c_att",
    ]
    keep_case += [c for c in AVAIL_FEATURES if c in out.columns]
    for arm in arms:
        keep_case += [
            f"alloc_score_{arm}", f"alloc_share_{arm}", f"candidate_att_raw_{arm}",
            f"pred_att_{arm}", f"pred_yards_{arm}", f"delta_att_{arm}",
        ]
    out[[c for c in keep_case if c in out.columns]].to_csv(a.out_dir / "stack6e_2025_casebook.csv", index=False)

    manifest = pd.DataFrame([
        {"item": "parent", "value": "P3 frozen STACK6 casebook"},
        {"item": "fit", "value": "2024 full weekly RB/FB roster"},
        {"item": "evaluation", "value": "2025 frozen P3/M94C target universe"},
        {"item": "arm_1", "value": "FULL_ROSTER_PARITY original STACK2 FULL features"},
        {"item": "arm_2", "value": "INACTIVE_COMPETITOR_STATE FULL + exactly five frozen availability features"},
        {"item": "market", "value": "downstream only after football disposition"},
    ])
    manifest.to_csv(a.out_dir / "stack6e_manifest.csv", index=False)

    print("=== STACK6E source/state audit ===")
    print(source_audit.to_string(index=False))
    print("=== STACK6E football scores ===")
    print(scores.to_string(index=False))
    print("=== STACK6E frozen gates ===")
    print(gates.to_string(index=False))
    print("=== STACK6E disposition ===")
    print(disposition_df.to_string(index=False))
    print("=== STACK6E downstream market ===")
    print(market_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
