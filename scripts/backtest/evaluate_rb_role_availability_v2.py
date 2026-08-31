"""M95G mechanical correction: TEAM_KEYS iteration order.

The original M95G evaluator grouped by TEAM_KEYS = [season, week, team] but
unpacked the tuple as (season, team, week). This wrapper corrects only that
identity/order bug. No features, model families, thresholds, selection rules,
or validation gates are changed.
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


g.previous_team_leaders = _previous_team_leaders_fixed

if __name__ == "__main__":
    raise SystemExit(g.main())
