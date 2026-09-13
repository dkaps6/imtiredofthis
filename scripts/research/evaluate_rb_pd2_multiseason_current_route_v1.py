#!/usr/bin/env python3
"""RB-PD2 multi-season replication on the current production-equivalent mean route.

Frozen plan:
  docs/research/RB_PD2_MULTISEASON_P3_EQUIVALENT_REPLICATION_V1_PLAN.md

This is research only.  It uses the parity-checked M95Q M91 temporal component
artifacts, fits canonical ensemble weights on S-1 only, applies them to S, and
then reruns the original PD2 last8/min4 player-error-persistence diagnostics on
2021-2024.  2025 is forbidden.

No sportsbook inputs. No production change. No uncertainty-width candidate is
implemented here; a replicated difficulty diagnostic only authorizes a separate
prospectively frozen width study.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble, fit_market_weights

TARGET_SEASONS = (2021, 2022, 2023, 2024)
SOURCE_SEASONS = (2020, 2021, 2022, 2023, 2024)
RB_POS = {"RB", "HB", "FB"}
MARKETS = {"rush_att", "rush_yards"}
HIST = 8
MIN_PRIOR = 4
MIN_ROWS = 700
TEAM_ALIAS = {
    "JAC": "JAX", "JAX": "JAX", "LA": "LAR", "LAR": "LAR",
    "STL": "LAR", "OAK": "LV", "SD": "LAC", "ARZ": "ARI",
}


def key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def team(v) -> str:
    s = str(v or "").strip().upper()
    return TEAM_ALIAS.get(s, s)


def num(v):
    return pd.to_numeric(v, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing/empty input: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def verify_m95q_parity(parity_root: Path) -> dict:
    disposition = read(parity_root / "m95q_disposition.csv")
    m91 = read(parity_root / "m95q_2024_m91_universe_parity.csv")
    downstream = read(parity_root / "m95q_2024_parity_audit.csv")

    if len(disposition) != 1 or str(disposition.iloc[0].get("disposition", "")) != "M95Q_EXPANDED_PANEL_READY":
        raise RuntimeError("M95Q source disposition is not M95Q_EXPANDED_PANEL_READY")
    if int(num(pd.Series([m91.iloc[0].get("m91_universe_parity_pass", 0)])).fillna(0).iloc[0]) != 1:
        raise RuntimeError("M95Q 2024 M91 universe parity did not pass")
    if int(num(pd.Series([downstream.iloc[0].get("parity_pass", 0)])).fillna(0).iloc[0]) != 1:
        raise RuntimeError("M95Q downstream 2024 parity did not pass")

    return {
        "m95q_disposition": "M95Q_EXPANDED_PANEL_READY",
        "m91_universe_2024_pass": True,
        "downstream_2024_parity_pass": True,
    }


def prep_components(x: pd.DataFrame, expected_season: int) -> pd.DataFrame:
    required = {
        "season", "week", "team", "player", "position", "market", "actual",
        "mc_proj", "ml_proj", "state_proj",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"season {expected_season} component predictions missing {missing}")

    out = x.copy()
    out["season"] = num(out["season"])
    out["week"] = num(out["week"])
    if not out["season"].dropna().eq(expected_season).all():
        vals = sorted(out["season"].dropna().astype(int).unique().tolist())
        raise RuntimeError(f"source-season drift expected={expected_season} observed={vals}")
    out["position"] = out["position"].fillna("").astype(str).str.upper().str.strip()
    out["market"] = out["market"].fillna("").astype(str).str.lower().str.strip()
    return out


def build_parent_panel(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = {
        s: prep_components(read(root / str(s) / "component_predictions.csv"), s)
        for s in SOURCE_SEASONS
    }

    panels: list[pd.DataFrame] = []
    weights_out: list[pd.DataFrame] = []

    for season in TARGET_SEASONS:
        # Canonical STACK1 temporal contract: fit only on immediately prior OOS
        # component predictions, freeze, then apply to the target season.
        weights = fit_market_weights(src[season - 1])
        if weights.empty:
            raise RuntimeError(f"no ensemble weights fitted for target season {season}")
        for market in MARKETS:
            if not weights["market"].astype(str).str.lower().eq(market).any():
                raise RuntimeError(f"missing frozen {market} weights for target season {season}")
        wsave = weights.copy()
        wsave["fit_season"] = season - 1
        wsave["target_season"] = season
        weights_out.append(wsave)

        scored = apply_ensemble(src[season], weights=weights)
        q = scored.loc[
            scored["position"].isin(RB_POS)
            & scored["market"].isin(MARKETS)
            & num(scored["week"]).between(1, 18)
        ].copy()
        q["season"] = num(q["season"]).astype(int)
        q["week"] = num(q["week"]).astype(int)
        q["team"] = q["team"].map(team)
        q["player_key"] = q.get("player_clean_key", q["player"]).map(key)
        if q["player_key"].eq("").any():
            raise RuntimeError(f"empty player key in target season {season}")

        keys = ["season", "week", "team", "player_key"]
        if q.duplicated(keys + ["market"]).any():
            bad = q.loc[q.duplicated(keys + ["market"], keep=False), keys + ["market", "player"]].head(20)
            raise RuntimeError(f"duplicate RB market rows season={season}: {bad.to_dict('records')}")

        market_sets = q.groupby(keys, sort=False)["market"].agg(lambda z: tuple(sorted(z.tolist())))
        bad_groups = market_sets.loc[market_sets.ne(("rush_att", "rush_yards"))]
        if len(bad_groups):
            raise RuntimeError(
                f"missing RB market pair season={season} bad_groups={len(bad_groups)} "
                f"sample={bad_groups.head(10).to_dict()}"
            )

        rows = []
        for ident, g in q.groupby(keys, sort=False):
            rec = dict(zip(keys, ident))
            rec["player"] = g.iloc[0].get("player", "")
            for market, suffix in (("rush_att", "carry"), ("rush_yards", "yard")):
                r = g.loc[g["market"].eq(market)].iloc[0]
                pred = num(pd.Series([r.get("ensemble_proj")])).iloc[0]
                actual = num(pd.Series([r.get("actual")])).iloc[0]
                if not np.isfinite(pred) or not np.isfinite(actual):
                    raise RuntimeError(f"non-finite {market} parent row target={season} identity={ident}")
                rec[f"pred_{suffix}"] = float(pred)
                rec[f"actual_{suffix}"] = float(actual)
            rows.append(rec)
        panels.append(pd.DataFrame(rows))

    panel = pd.concat(panels, ignore_index=True).sort_values(
        ["season", "week", "player_key"], kind="stable"
    )
    if set(panel["season"].unique().tolist()) != set(TARGET_SEASONS):
        raise RuntimeError(f"target season set drift: {sorted(panel['season'].unique().tolist())}")
    if panel["season"].eq(2025).any():
        raise RuntimeError("forbidden 2025 row entered multi-season replication")
    if panel.duplicated(["season", "week", "team", "player_key"]).any():
        raise RuntimeError("final RB parent identity is not unique")

    return panel, pd.concat(weights_out, ignore_index=True)


def build_walkforward(panel: pd.DataFrame) -> pd.DataFrame:
    hist: dict[str, list[dict]] = {}
    rows = []
    for r in panel.sort_values(["season", "week", "player_key"], kind="stable").itertuples(index=False):
        carry_error = float(r.pred_carry - r.actual_carry)
        yard_error = float(r.pred_yard - r.actual_yard)
        h = hist.get(r.player_key, [])[-HIST:]
        rec = {
            "season": int(r.season), "week": int(r.week), "team": r.team,
            "player": r.player, "player_key": r.player_key,
            "target_carry_error": carry_error,
            "target_carry_abs_error": abs(carry_error),
            "target_yard_error": yard_error,
            "target_yard_abs_error": abs(yard_error),
            "prior_games": len(h),
        }
        if h:
            d = pd.DataFrame(h)
            rec.update(
                prior8_carry_bias=float(d["carry_error"].mean()),
                prior8_carry_mae=float(d["carry_abs"].mean()),
                prior8_yard_bias=float(d["yard_error"].mean()),
                prior8_yard_mae=float(d["yard_abs"].mean()),
                last_prior_ord=int(d.iloc[-1]["ord"]),
            )
        else:
            rec.update(
                prior8_carry_bias=np.nan, prior8_carry_mae=np.nan,
                prior8_yard_bias=np.nan, prior8_yard_mae=np.nan,
                last_prior_ord=np.nan,
            )
        rows.append(rec)
        hist.setdefault(r.player_key, []).append({
            "ord": int(r.season) * 100 + int(r.week),
            "carry_error": carry_error, "carry_abs": abs(carry_error),
            "yard_error": yard_error, "yard_abs": abs(yard_error),
        })

    out = pd.DataFrame(rows)
    current_ord = out["season"] * 100 + out["week"]
    bad = out["last_prior_ord"].notna() & num(out["last_prior_ord"]).ge(current_ord)
    if bad.any():
        raise RuntimeError(f"walk-forward leakage detected rows={int(bad.sum())}")
    return out


def quartile_gap(g: pd.DataFrame, feat: str, outcome: str) -> float:
    f = num(g[feat])
    y = num(g[outcome])
    q25 = float(f.quantile(0.25))
    q75 = float(f.quantile(0.75))
    return float(y.loc[f.ge(q75)].mean() - y.loc[f.le(q25)].mean())


def sliced_gap(g: pd.DataFrame, feat: str, outcome: str, lo: int, hi: int) -> float:
    q = g.loc[g["week"].between(lo, hi)].copy()
    if len(q) < 100 or num(q[feat]).nunique() < 4:
        return np.nan
    return quartile_gap(q, feat, outcome)


def diagnostic_values(g: pd.DataFrame, feat: str, outcome: str, sign_min: float | None):
    spearman = float(num(g[feat]).corr(num(g[outcome]), method="spearman"))
    gap = quartile_gap(g, feat, outcome)
    sign = np.nan
    if sign_min is not None:
        q = g.loc[num(g[feat]).abs().ge(sign_min)].copy()
        sign = float((np.sign(num(q[feat])) == np.sign(num(q[outcome]))).mean()) if len(q) else np.nan
    return spearman, gap, sign, sliced_gap(g, feat, outcome, 5, 12), sliced_gap(g, feat, outcome, 13, 18)


def score(walkforward: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    g = walkforward.loc[walkforward["prior_games"].ge(MIN_PRIOR)].copy()
    specs = [
        ("CARRY_DIRECTIONAL_PERSISTENCE", "prior8_carry_bias", "target_carry_error", 1.0, 0.5),
        ("CARRY_DIFFICULTY_PERSISTENCE", "prior8_carry_mae", "target_carry_abs_error", 0.75, None),
        ("YARD_DIRECTIONAL_PERSISTENCE", "prior8_yard_bias", "target_yard_error", 6.0, 3.0),
        ("YARD_DIFFICULTY_PERSISTENCE", "prior8_yard_mae", "target_yard_abs_error", 5.0, None),
    ]

    metric_rows = []
    season_rows = []
    for name, feat, outcome, min_gap, sign_min in specs:
        sp, gap, sign, early, late = diagnostic_values(g, feat, outcome, sign_min)
        sign_ok = sign_min is None or (np.isfinite(sign) and sign >= 0.55)
        pooled_pass = bool(
            len(g) >= MIN_ROWS
            and sp >= 0.08
            and gap >= min_gap
            and sign_ok
            and np.isfinite(early) and early > 0
            and np.isfinite(late) and late > 0
        )

        positive_seasons = 0
        recent_joint_negative = False
        for season in TARGET_SEASONS:
            q = g.loc[g["season"].eq(season)]
            ssp = float(num(q[feat]).corr(num(q[outcome]), method="spearman")) if len(q) > 2 else np.nan
            sgap = quartile_gap(q, feat, outcome) if len(q) > 10 else np.nan
            if np.isfinite(ssp) and np.isfinite(sgap) and ssp > 0 and sgap > 0:
                positive_seasons += 1
            if season in (2023, 2024) and np.isfinite(ssp) and np.isfinite(sgap) and ssp <= 0 and sgap <= 0:
                recent_joint_negative = True
            season_rows.append({
                "diagnostic": name, "season": season, "rows": int(len(q)),
                "spearman": ssp, "quartile_gap": sgap,
            })

        replicated = bool(pooled_pass and positive_seasons >= 3 and not recent_joint_negative)
        metric_rows.append({
            "diagnostic": name, "rows": int(len(g)), "spearman": sp,
            "quartile_gap": gap, "sign_agreement": sign,
            "gap_weeks5_12": early, "gap_weeks13_18": late,
            "original_pooled_gate_pass": pooled_pass,
            "positive_seasons": positive_seasons,
            "recent_joint_negative": recent_joint_negative,
            "replicated": replicated,
        })

    metrics = pd.DataFrame(metric_rows)
    by_season = pd.DataFrame(season_rows)
    winners = metrics.loc[metrics["replicated"], "diagnostic"].tolist()
    result = {
        "migration": "RB_PD2_MULTISEASON_CURRENT_PRODUCTION_ROUTE_REPLICATION_V1",
        "source_rows": int(len(walkforward)),
        "scoreable_rows": int(len(g)),
        "players": int(walkforward["player_key"].nunique()),
        "target_seasons": list(TARGET_SEASONS),
        "history_window": HIST,
        "minimum_prior_games": MIN_PRIOR,
        "walk_forward_leakage_violations": 0,
        "sportsbook_inputs_used": False,
        "production_changed": False,
        "replicated_diagnostics": winners,
        "carry_width_unlocked": "CARRY_DIFFICULTY_PERSISTENCE" in winners,
        "yard_width_unlocked": "YARD_DIFFICULTY_PERSISTENCE" in winners,
        "literal_pd6_p3_equivalent_blocker_resolved": False,
        "disposition": (
            "MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE_REPRODUCED"
            if winners else "NO_MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE"
        ),
    }
    return metrics, by_season, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="M95Q per-season M91 artifact root")
    ap.add_argument("--parity-root", type=Path, required=True, help="final M95Q result artifact root")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    parity = verify_m95q_parity(a.parity_root)
    panel, weights = build_parent_panel(a.root)
    walkforward = build_walkforward(panel)
    metrics, by_season, result = score(walkforward)
    result["source_parity"] = parity

    panel.to_csv(a.out_dir / "rb_pd2_multiseason_parent_panel.csv", index=False)
    weights.to_csv(a.out_dir / "rb_pd2_multiseason_weights.csv", index=False)
    walkforward.to_csv(a.out_dir / "rb_pd2_multiseason_walkforward_casebook.csv", index=False)
    metrics.to_csv(a.out_dir / "rb_pd2_multiseason_metrics.csv", index=False)
    by_season.to_csv(a.out_dir / "rb_pd2_multiseason_by_season.csv", index=False)
    (a.out_dir / "rb_pd2_multiseason_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(metrics.to_string(index=False))
    print("--- by season ---")
    print(by_season.to_string(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
