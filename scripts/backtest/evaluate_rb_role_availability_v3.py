"""M95G mechanical corrections v3.

Corrects two input-contract issues only:
1. TEAM_KEYS are ordered [season, week, team] when reconstructing prior-game RB leaders.
2. The frozen M95F holdout/validation traces name the target-specific raw tail scores
   raw_prob_20/raw_prob_25, while the temporal OOF artifact calls the same field raw_score.
   This wrapper maps the target-specific frozen column to raw_score before prediction.

No features, model families, coefficients, thresholds, selection criteria, or
validation gates are changed.
"""
import pandas as pd

import scripts.backtest.evaluate_rb_role_availability as g


def _previous_team_leaders_fixed(trace: pd.DataFrame) -> pd.DataFrame:
    z = trace.copy()
    if "actual_carries" not in z.columns:
        if "actual_rush_att" in z.columns:
            z["actual_carries"] = g.num(z["actual_rush_att"])
        else:
            raise RuntimeError("M95B trace missing actual carry truth for prior-game leader construction")
    z = z.loc[z["season"].isin([2024, 2025])].copy()
    rows = []
    for (season, week, team), frame in z.groupby(g.TEAM_KEYS):
        q = frame.loc[g.num(frame["actual_carries"]).notna()].copy()
        if q.empty:
            continue
        q["actual_carries"] = g.num(q["actual_carries"])
        q = q.sort_values(["actual_carries", "player_clean_key"], ascending=[False, True])
        rows.append({
            "season": int(season), "team": g.canon(team), "week": int(week),
            "game_top1_key": str(q.iloc[0]["player_clean_key"]),
            "game_top1_carries": float(q.iloc[0]["actual_carries"]),
            "game_top2_key": str(q.iloc[1]["player_clean_key"]) if len(q) > 1 else "",
            "game_top2_carries": float(q.iloc[1]["actual_carries"]) if len(q) > 1 else 0.0,
        })
    game = pd.DataFrame(rows).sort_values(["season", "team", "week"])
    grp = game.groupby(["season", "team"], sort=False)
    game["prior_top1_key"] = grp["game_top1_key"].shift(1)
    game["prior_top1_carries"] = grp["game_top1_carries"].shift(1)
    game["prior_top2_key"] = grp["game_top2_key"].shift(1)
    game["prior_top2_carries"] = grp["game_top2_carries"].shift(1)
    return game[g.TEAM_KEYS + ["prior_top1_key", "prior_top1_carries", "prior_top2_key", "prior_top2_carries"]]


_original_fit_apply = g.fit_apply


def _fit_apply_score_contract(train, test, spec, c, target):
    tr = train.copy()
    te = test.copy()
    suffix = "20" if target == "actual_20plus" else "25"
    if "raw_score" not in tr.columns and f"raw_prob_{suffix}" in tr.columns:
        tr["raw_score"] = tr[f"raw_prob_{suffix}"]
    if "raw_score" not in te.columns and f"raw_prob_{suffix}" in te.columns:
        te["raw_score"] = te[f"raw_prob_{suffix}"]
    return _original_fit_apply(tr, te, spec, c, target)


g.previous_team_leaders = _previous_team_leaders_fixed
g.fit_apply = _fit_apply_score_contract

if __name__ == "__main__":
    raise SystemExit(g.main())
