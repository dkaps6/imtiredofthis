"""M95B evaluator bug-fix shim.

This does not alter the pre-specified M95B model design. It only fixes two
implementation issues discovered before a successful scientific run:
1) nflverse PFR weekly rushing uses pfr_player_name and explicit
   rushing_yards_before/after_contact column names;
2) the incremental AUC reporter accidentally resolved Series.corr as the pandas
   method instead of the stored `corr` result column.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import scripts.backtest.evaluate_rb_offense_defense_matchup as m


def read_pfr_fixed(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("advstats_week_rush_*.csv")):
        frames.append(m.lower(pd.read_csv(p, low_memory=False)))
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    sc = m.alias(x, ["season"])
    wc = m.alias(x, ["week"])
    pc = m.alias(x, ["pfr_player_name", "player", "player_name"])
    tc = m.alias(x, ["team", "tm"])
    ac = m.alias(x, ["carries", "att", "rush_att"])
    if not all([sc, wc, pc, tc, ac]):
        raise RuntimeError(f"PFR rushing schema missing required identity/attempt columns: {list(x.columns)}")

    out = pd.DataFrame({
        "season": m.num(x[sc]),
        "week": m.num(x[wc]),
        "team": x[tc].map(m.team),
        "player_short_key": x[pc].map(m.short_name),
    })
    att = m.num(x[ac])
    out["pfr_att"] = att

    # Prefer already-normalized PFR averages when present. Otherwise calculate
    # from the corresponding total and carries.
    ybc_avg = m.alias(x, ["rushing_yards_before_contact_avg", "ybc_att", "ybc_per_att"])
    ybc_tot = m.alias(x, ["rushing_yards_before_contact", "ybc", "yards_before_contact"])
    yac_avg = m.alias(x, ["rushing_yards_after_contact_avg", "yac_att", "yac_per_att"])
    yac_tot = m.alias(x, ["rushing_yards_after_contact", "yac", "yards_after_contact"])
    br = m.alias(x, ["rushing_broken_tackles", "brktkl", "brk_tkl", "broken_tackles"])

    if ybc_avg:
        out["pfr_ybc_per_att"] = m.num(x[ybc_avg])
    elif ybc_tot:
        out["pfr_ybc_per_att"] = m.num(x[ybc_tot]) / att.replace(0, np.nan)
    else:
        out["pfr_ybc_per_att"] = np.nan

    if yac_avg:
        out["pfr_yac_per_att"] = m.num(x[yac_avg])
    elif yac_tot:
        out["pfr_yac_per_att"] = m.num(x[yac_tot]) / att.replace(0, np.nan)
    else:
        out["pfr_yac_per_att"] = np.nan

    out["pfr_brk_tkl_per_att"] = (
        m.num(x[br]) / att.replace(0, np.nan) if br else np.nan
    )
    out = out.loc[out["week"].between(1, 22) & out["player_short_key"].ne("")].copy()
    return out.drop_duplicates(["season", "week", "team", "player_short_key"], keep="last")


def incremental_fixed(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = [
        "role_baseline",
        "role_plus_offense",
        "role_offense_defense",
        "full_matchup_interactions",
    ]
    for (split, target), g in results.groupby(["split", "target"]):
        for a, b in zip(order[:-1], order[1:]):
            ga = g.loc[g["family"].eq(a)]
            gb = g.loc[g["family"].eq(b)]
            if ga.empty or gb.empty:
                continue
            if target.endswith("_auc"):
                gain = float(gb.iloc[0]["corr"] - ga.iloc[0]["corr"])
                metric = "auc_gain"
            else:
                gain = float(ga.iloc[0]["mae"] - gb.iloc[0]["mae"])
                metric = "mae_gain"
            rows.append({
                "split": split,
                "target": target,
                "from_family": a,
                "to_family": b,
                "metric": metric,
                "incremental_gain": gain,
            })
    return pd.DataFrame(rows)


m.read_pfr = read_pfr_fixed
m.incremental = incremental_fixed

if __name__ == "__main__":
    raise SystemExit(m.main())
