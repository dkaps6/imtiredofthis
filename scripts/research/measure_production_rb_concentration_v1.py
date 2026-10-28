"""Measure PRODUCTION's RB rush-share concentration vs the strictly-prior football
target, controlling for the number of backs.

CORRECTED V1b. An earlier draft of this script read week7_trace__data__metrics_ready.csv
and called simulate() on it directly. That is NOT the production path: production calls
_full_roster_simulate -> _build_full_universe, which builds the universe from
player_form_consensus.csv and REFUSES duplicate (team, player_clean_key) rows. Because
metrics_ready carries one row per (player, market), the earlier draft measured a top-1
share of 0.2226 that was purely an artifact of duplicate rows consuming the top-5 pool.
That number is retracted; it never described production.

This version dedupes per (team, player) and reads rules_rush_share -- the column
simulate() actually resolves first on the production universe.

Diagnostic only: fits nothing, promotes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.simulation_v2 import _top_n_shares  # real production truncation

TRACE = Path("data/research/week7_opportunity_trace")
RULES = TRACE / "week7_trace__data__model_rule_simulation_inputs.csv"
RB_POSITIONS = {"RB", "HB", "TB", "FB"}
CALIBRATION_SEASONS = tuple(range(2013, 2026))
MIN_TEAM_PRIOR_CARRIES = 20
SHARE_COLS = ("rules_rush_share", "bayes_rush_share", "rush_share")


def _is_rb(pos: object) -> bool:
    p = str(pos).strip().upper()
    return p in RB_POSITIONS or p.startswith("RB")


def production_concentration() -> tuple[pd.DataFrame, str]:
    if not RULES.exists():
        raise SystemExit(f"FAIL CLOSED: missing {RULES}")
    d = pd.read_csv(RULES, low_memory=False)
    col = next((c for c in SHARE_COLS if c in d.columns), None)
    if col is None:
        raise SystemExit(f"FAIL CLOSED: none of {SHARE_COLS} present")
    # DEDUPE: production simulates one row per player, not one per (player, market)
    d = d.drop_duplicates(["team", "player"], keep="first").copy()
    d["_share"] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    rows = []
    for team, team_df in d.groupby("team", dropna=False):
        shares = _top_n_shares(team_df["_share"].to_numpy(), 5)  # production truncation
        rb_mask = team_df["position"].fillna("").astype(str).map(_is_rb).to_numpy()
        rb = shares[rb_mask]
        rb = rb[rb > 0]
        if len(rb) < 2:
            continue
        rb = rb / rb.sum()  # relative share WITHIN the RB subset
        rows.append({"team": str(team), "n_backs": int(len(rb)), "top1": float(rb.max())})
    return pd.DataFrame(rows), col


def football_concentration() -> pd.DataFrame:
    import nflreadpy as nfl
    w = nfl.load_player_stats(seasons=list(CALIBRATION_SEASONS)).to_pandas()
    w = w[["season", "week", "team", "player_id", "position", "carries"]].copy()
    w = w[w["position"].map(_is_rb)]
    w["carries"] = pd.to_numeric(w["carries"], errors="coerce").fillna(0.0)
    w["week"] = pd.to_numeric(w["week"], errors="coerce")
    w = w[w["week"].notna()]
    w["week"] = w["week"].astype(int)
    w = w.sort_values(["season", "team", "player_id", "week"])
    w["prior_player"] = w.groupby(["season", "team", "player_id"])["carries"].cumsum() - w["carries"]
    tw = (w.groupby(["season", "team", "week"], as_index=False)["carries"].sum()
            .rename(columns={"carries": "team_carries"}).sort_values(["season", "team", "week"]))
    tw["team_prior"] = tw.groupby(["season", "team"])["team_carries"].cumsum() - tw["team_carries"]
    w = w.merge(tw, on=["season", "team", "week"], how="left")
    w = w[(w["team_prior"] >= MIN_TEAM_PRIOR_CARRIES) & (w["team_carries"] > 0)]
    w["proxy"] = w["prior_player"] / w["team_prior"]
    w = w[w["proxy"] > 0].copy()
    w["actual_share"] = w["carries"] / w["team_carries"]
    grp = ["season", "week", "team"]
    w["actual_share"] = w["actual_share"] / w.groupby(grp)["actual_share"].transform("sum")
    g = w.groupby(grp)
    out = pd.DataFrame({
        "n_backs": g["player_id"].size(),
        "top1": g.apply(lambda x: x.loc[x["proxy"].idxmax(), "actual_share"], include_groups=False),
    }).reset_index()
    return out[out["n_backs"] >= 2]


def main() -> int:
    prod, col = production_concentration()
    if prod.empty:
        print("FAIL CLOSED: no production team with >=2 modeled backs", file=sys.stderr)
        return 2
    fb = football_concentration()

    print(f"=== PRODUCTION (W7, deduped per player, column '{col}') ===")
    print(f"  teams {len(prod)}   mean top-1 within-RB share {prod['top1'].mean():.4f}"
          f"   mean backs {prod['n_backs'].mean():.2f}")
    print(f"  n_backs: {prod['n_backs'].value_counts().sort_index().to_dict()}")
    print("\n=== FOOTBALL (2013-2025, actual share of top-back-by-PREGAME-proxy) ===")
    print(f"  team-weeks {len(fb)}   mean top-1 share {fb['top1'].mean():.4f}"
          f"   mean backs {fb['n_backs'].mean():.2f}")

    print("\n=== MATCHED BY NUMBER OF BACKS ===")
    print(f"  {'n_backs':>7} {'prod n':>7} {'prod top1':>10} {'fb n':>6} {'fb top1':>9} {'gap':>8}")
    matched = []
    for n in sorted(prod["n_backs"].unique()):
        p = prod[prod["n_backs"] == n]; f = fb[fb["n_backs"] == n]
        if len(f) < 30:
            print(f"  {n:>7} {len(p):>7} {p['top1'].mean():>10.4f} {len(f):>6} {'(thin)':>9}")
            continue
        gap = p["top1"].mean() - f["top1"].mean()
        matched.append({"n_backs": int(n), "prod_n": int(len(p)),
                        "prod_top1": round(float(p["top1"].mean()), 4), "fb_n": int(len(f)),
                        "fb_top1": round(float(f["top1"].mean()), 4), "gap": round(float(gap), 4)})
        print(f"  {n:>7} {len(p):>7} {p['top1'].mean():>10.4f} {len(f):>6}"
              f" {f['top1'].mean():>9.4f} {gap:>+8.4f}")

    sized = {}
    if matched:
        wsum = sum(m["prod_n"] for m in matched)
        pr = sum(m["prod_top1"] * m["prod_n"] for m in matched) / wsum
        fbs = sum(m["fb_top1"] * m["prod_n"] for m in matched) / wsum
        # paired-by-cell bootstrap CI on the size-standardized gap
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(4000):
            tot = 0.0
            for m in matched:
                pv = prod[prod["n_backs"] == m["n_backs"]]["top1"].to_numpy()
                fv = fb[fb["n_backs"] == m["n_backs"]]["top1"].to_numpy()
                tot += m["prod_n"] * (rng.choice(pv, len(pv)).mean() - rng.choice(fv, len(fv)).mean())
            boots.append(tot / wsum)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sized = {"production": round(pr, 4), "football": round(fbs, 4),
                 "gap": round(pr - fbs, 4), "gap_ci95": [round(float(lo), 4), round(float(hi), 4)]}
        print(f"\n  size-standardized: production {pr:.4f}  football {fbs:.4f}"
              f"  gap {pr - fbs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    out_dir = Path("data/research/rb_carry_concentration")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "version": "PRODUCTION_RB_CONCENTRATION_MEASUREMENT_V1B",
        "retracted_predecessor": {
            "value": 0.2226,
            "why_wrong": "read metrics_ready (one row per player-market) and called simulate() "
                         "directly; production builds a DEDUPED universe via _build_full_universe, "
                         "which raises on duplicate (team, player_clean_key). Off the production path.",
        },
        "source": str(RULES), "share_column": col,
        "production_teams": int(len(prod)),
        "production_mean_top1": round(float(prod["top1"].mean()), 4),
        "production_mean_backs": round(float(prod["n_backs"].mean()), 3),
        "football_team_weeks": int(len(fb)),
        "football_mean_top1": round(float(fb["top1"].mean()), 4),
        "matched_by_n_backs": matched, "size_standardized": sized,
        "population_limitation":
            "deduped PRICED universe (28 teams) rather than the full 396-row football universe; "
            "unpriced roster backs also carry rules_rush_share in production.",
        "sportsbook_inputs_used": False, "fits_anything": False,
    }
    p = out_dir / "production_rb_concentration_v1.json"
    p.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
