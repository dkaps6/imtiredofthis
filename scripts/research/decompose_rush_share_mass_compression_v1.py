"""Quantify the _allocate_counts 0.95 team-mass compression on the EXACT production
simulation universe, per GPT audit 6477974816 AUDIT TARGET 2.

Population: data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv
-- 396 rows, 28 teams, 0 duplicate (team, player_clean_key). This is the frame
_full_roster_simulate hands to canonical_simulate, so there is NO population mixing.

Diagnostic only. Changes no coefficient, no cap, nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.simulation_v2 import _top_n_shares  # real production truncation

UNIVERSE = Path("data/research/week7_opportunity_trace/"
                "week7_trace__data__football_simulation_universe.csv")
RB_POSITIONS = {"RB", "HB", "TB", "FB"}
CAP = 0.95          # scripts/simulation_v2._allocate_counts
TARGET_WEEK, TARGET_SEASON = 7, 2026


def _is_rb(p: object) -> bool:
    s = str(p).strip().upper()
    return s in RB_POSITIONS or s.startswith("RB")


def actual_rb_carries() -> pd.DataFrame:
    import nflreadpy as nfl
    w = nfl.load_player_stats(seasons=[TARGET_SEASON]).to_pandas()
    w = w[(w["week"] == TARGET_WEEK) & (w["position"].map(_is_rb))]
    w["carries"] = pd.to_numeric(w["carries"], errors="coerce").fillna(0.0)
    w["rushing_yards"] = pd.to_numeric(w["rushing_yards"], errors="coerce").fillna(0.0)
    return (w.groupby("team", as_index=False)[["carries", "rushing_yards"]].sum()
              .rename(columns={"carries": "actual_rb_carries",
                               "rushing_yards": "actual_rb_yards"}))


def main() -> int:
    d = pd.read_csv(UNIVERSE, low_memory=False)
    rows, teamrows = [], []
    for team, g in d.groupby("team", dropna=False):
        raw = pd.to_numeric(g["rules_rush_share"], errors="coerce").fillna(0.0).to_numpy()
        shares = _top_n_shares(raw, 5)                      # production truncation
        # team rush attempts exactly as simulate() derives the mean
        plays = pd.to_numeric(g["rules_plays_est"], errors="coerce").dropna()
        prate = pd.to_numeric(g["rules_pass_rate"], errors="coerce").dropna()
        if plays.empty or prate.empty:
            continue
        team_rush_att = float(plays.iloc[0]) * (1.0 - float(prate.iloc[0]))
        raw_sum = float(shares.sum())
        mult = min(1.0, CAP / raw_sum) if raw_sum > CAP else 1.0
        ypc = pd.to_numeric(g["rules_ypc"], errors="coerce")
        rb_mask = g["position"].fillna("").astype(str).map(_is_rb).to_numpy()
        pre = team_rush_att * shares
        post = pre * mult
        teamrows.append({
            "team": str(team), "team_rush_att": team_rush_att,
            "raw_share_sum": raw_sum, "cap_multiplier": mult,
            "compressed": bool(mult < 1.0),
            "rb_share_sum": float(shares[rb_mask].sum()),
            "rb_carries_pre_cap": float(pre[rb_mask].sum()),
            "rb_carries_post_cap": float(post[rb_mask].sum()),
            "rb_carries_lost": float((pre - post)[rb_mask].sum()),
            "rb_yards_lost": float(((pre - post)[rb_mask] * ypc.fillna(ypc.mean()).to_numpy()[rb_mask]).sum()),
            "n_modeled_backs": int(rb_mask.sum()),
        })
        order = np.argsort(-shares[rb_mask])
        for rank, idx in enumerate(order, start=1):
            rows.append({"team": str(team), "rank": rank,
                         "pre_cap_carries": float(pre[rb_mask][idx]),
                         "post_cap_carries": float(post[rb_mask][idx]),
                         "carries_lost": float((pre - post)[rb_mask][idx]),
                         "ypc": float(ypc.fillna(ypc.mean()).to_numpy()[rb_mask][idx])})
    t = pd.DataFrame(teamrows); r = pd.DataFrame(rows)
    if t.empty:
        print("FAIL CLOSED: no usable teams", file=sys.stderr); return 2

    print(f"=== 0.95 TEAM-MASS COMPRESSION, production universe, n={len(t)} teams ===")
    print(f"  teams whose raw rush-share sum exceeds {CAP}: "
          f"**{int(t['compressed'].sum())}/{len(t)}**")
    print(f"  raw share sum: mean {t['raw_share_sum'].mean():.4f}  "
          f"min {t['raw_share_sum'].min():.4f}  max {t['raw_share_sum'].max():.4f}")
    print(f"  cap multiplier: mean {t['cap_multiplier'].mean():.4f}  "
          f"min {t['cap_multiplier'].min():.4f}")
    print(f"  RB carries  pre-cap {t['rb_carries_pre_cap'].mean():6.2f}"
          f"  post-cap {t['rb_carries_post_cap'].mean():6.2f}"
          f"  LOST {t['rb_carries_lost'].mean():5.2f}/team")
    print(f"  RB yards lost to compression: {t['rb_yards_lost'].mean():.2f}/team")

    print("\n=== does compression cause a TEAM-LEVEL RB deficit? (same population, wk7) ===")
    act = actual_rb_carries()
    m = t.merge(act, on="team", how="inner")
    print(f"  matched teams {len(m)}")
    print(f"  model post-cap RB carries {m['rb_carries_post_cap'].mean():6.2f}"
          f"   actual {m['actual_rb_carries'].mean():6.2f}"
          f"   bias {(m['rb_carries_post_cap']-m['actual_rb_carries']).mean():+6.2f}")
    print(f"  model PRE-cap  RB carries {m['rb_carries_pre_cap'].mean():6.2f}"
          f"   actual {m['actual_rb_carries'].mean():6.2f}"
          f"   bias {(m['rb_carries_pre_cap']-m['actual_rb_carries']).mean():+6.2f}")
    b = m["rb_carries_post_cap"] - m["actual_rb_carries"]
    ci = 1.96 * b.std(ddof=1) / np.sqrt(len(b))
    print(f"  post-cap team RB carry bias 95% CI [{b.mean()-ci:+.2f}, {b.mean()+ci:+.2f}]")

    print("\n=== per-rank carries lost to compression ===")
    for k, g in r.groupby(r["rank"].clip(upper=3)):
        lab = "3+" if k == 3 else str(int(k))
        print(f"  rank {lab:2s} n={len(g):3d}  pre {g['pre_cap_carries'].mean():5.2f}"
              f"  post {g['post_cap_carries'].mean():5.2f}"
              f"  lost {g['carries_lost'].mean():5.2f} carries"
              f"  = {(g['carries_lost']*g['ypc']).mean():5.2f} yards")

    out = Path("data/research/rb_carry_concentration")
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "version": "RUSH_SHARE_MASS_COMPRESSION_DECOMPOSITION_V1",
        "requested_by": "GPT audit issue #535 comment 6477974816 (AUDIT TARGET 2)",
        "population": str(UNIVERSE), "teams": int(len(t)), "cap": CAP,
        "teams_compressed": int(t["compressed"].sum()),
        "raw_share_sum_mean": round(float(t["raw_share_sum"].mean()), 4),
        "cap_multiplier_mean": round(float(t["cap_multiplier"].mean()), 4),
        "rb_carries_lost_per_team": round(float(t["rb_carries_lost"].mean()), 3),
        "rb_yards_lost_per_team": round(float(t["rb_yards_lost"].mean()), 2),
        "team_rb_carry_bias_post_cap": round(float(b.mean()), 3),
        "team_rb_carry_bias_post_cap_ci95": [round(float(b.mean()-ci), 3), round(float(b.mean()+ci), 3)],
        "team_rb_carry_bias_pre_cap": round(float((m["rb_carries_pre_cap"]-m["actual_rb_carries"]).mean()), 3),
        "per_rank_yards_lost": {("3+" if int(k) == 3 else str(int(k))):
                                round(float((g["carries_lost"]*g["ypc"]).mean()), 2)
                                for k, g in r.groupby(r["rank"].clip(upper=3))},
        "changed_any_coefficient_or_cap": False,
        "sportsbook_inputs_used": False,
    }
    (out / "rush_share_mass_compression_v1.json").write_text(json.dumps(res, indent=2) + "\n")
    t.to_csv(out / "rush_share_mass_compression_teams_v1.csv", index=False)
    print(f"\nwrote {out}/rush_share_mass_compression_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
