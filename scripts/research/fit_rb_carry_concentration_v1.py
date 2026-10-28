"""Fit the RB carry-concentration exponent gamma on strictly-prior seasons.

Preregistration: docs/research/PRISM_RB_CARRY_CONCENTRATION_PREREGISTRATION_V1.md

Ranks backs by a PREGAME trailing-share proxy, never by realized carries.
Ranking on the outcome manufactures concentration through max-selection; that
artifact produced three retracted claims in the parent diagnosis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CALIBRATION_SEASONS = tuple(range(2013, 2026))  # 2013-2025 inclusive; 2026 excluded
RB_POSITIONS = {"RB", "HB", "TB", "FB"}
MIN_TEAM_PRIOR_CARRIES = 20
MIN_BACKS = 2
GAMMA_GRID = np.round(np.arange(0.50, 3.001, 0.01), 2)


def _rb_family(pos: object) -> bool:
    p = str(pos).strip().upper()
    return p in RB_POSITIONS or p.startswith("RB")


def load_weekly(seasons) -> pd.DataFrame:
    import nflreadpy as nfl

    w = nfl.load_player_stats(seasons=list(seasons)).to_pandas()
    keep = ["season", "week", "team", "player_id", "player_display_name", "position", "carries"]
    w = w[[c for c in keep if c in w.columns]].copy()
    w = w[w["position"].map(_rb_family)]
    w["carries"] = pd.to_numeric(w["carries"], errors="coerce").fillna(0.0)
    w = w[pd.to_numeric(w["week"], errors="coerce").notna()]
    w["week"] = w["week"].astype(int)
    w["season"] = w["season"].astype(int)
    return w


def build_pairs(w: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): pregame trailing share proxy + actual share."""
    w = w.sort_values(["season", "team", "player_id", "week"])
    # prior-weeks-only cumulative carries, within season (shift before cumsum)
    w["prior_player_carries"] = (
        w.groupby(["season", "team", "player_id"])["carries"].cumsum() - w["carries"]
    )
    team_week = w.groupby(["season", "team", "week"], as_index=False)["carries"].sum()
    team_week = team_week.rename(columns={"carries": "team_carries"})
    team_week = team_week.sort_values(["season", "team", "week"])
    team_week["team_prior_carries"] = (
        team_week.groupby(["season", "team"])["team_carries"].cumsum() - team_week["team_carries"]
    )
    w = w.merge(team_week, on=["season", "team", "week"], how="left")

    w = w[w["team_prior_carries"] >= MIN_TEAM_PRIOR_CARRIES]
    w = w[w["team_carries"] > 0]
    w["proxy"] = w["prior_player_carries"] / w["team_prior_carries"]
    w["actual_share"] = w["carries"] / w["team_carries"]
    # the transform only acts on positive shares; renormalize both sides on that subset
    w = w[w["proxy"] > 0].copy()
    grp = ["season", "week", "team"]
    w["proxy"] = w["proxy"] / w.groupby(grp)["proxy"].transform("sum")
    asum = w.groupby(grp)["actual_share"].transform("sum")
    w = w[asum > 0].copy()
    w["actual_share"] = w["actual_share"] / w.groupby(grp)["actual_share"].transform("sum")
    n_backs = w.groupby(grp)["player_id"].transform("size")
    return w[n_backs >= MIN_BACKS].copy()


def sse_for_gamma(pairs: pd.DataFrame, gamma: float) -> float:
    grp = ["season", "week", "team"]
    p = np.power(pairs["proxy"].to_numpy(), gamma)
    s = pd.Series(p, index=pairs.index).groupby([pairs[c] for c in grp]).transform("sum")
    fitted = p / s.to_numpy()
    return float(np.sum((fitted - pairs["actual_share"].to_numpy()) ** 2))


def fit_gamma(pairs: pd.DataFrame) -> tuple[float, float]:
    sses = np.array([sse_for_gamma(pairs, g) for g in GAMMA_GRID])
    i = int(np.argmin(sses))
    return float(GAMMA_GRID[i]), float(sses[i])


def concentration_report(pairs: pd.DataFrame, gamma: float) -> dict:
    """Top-1-by-proxy share: proxy vs sharpened vs actual. Rank is pregame."""
    grp = ["season", "week", "team"]
    out = {}
    p = pairs["proxy"].to_numpy()
    sh = np.power(p, gamma)
    pairs = pairs.assign(_sh=sh)
    pairs["_sh"] = pairs["_sh"] / pairs.groupby(grp)["_sh"].transform("sum")
    top = pairs.loc[pairs.groupby(grp)["proxy"].idxmax()]
    out["n_team_weeks"] = int(len(top))
    out["top1_proxy_share"] = round(float(top["proxy"].mean()), 4)
    out["top1_sharpened_share"] = round(float(top["_sh"].mean()), 4)
    out["top1_actual_share"] = round(float(top["actual_share"].mean()), 4)
    return out


def main() -> int:
    out_dir = Path("data/research/rb_carry_concentration")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading nflverse weekly stats {CALIBRATION_SEASONS[0]}-{CALIBRATION_SEASONS[-1]} ...")
    w = load_weekly(CALIBRATION_SEASONS)
    if w.empty:
        print("FAIL CLOSED: no RB weekly rows loaded", file=sys.stderr)
        return 2
    pairs = build_pairs(w)
    if pairs.empty:
        print("FAIL CLOSED: no usable (proxy, actual) pairs", file=sys.stderr)
        return 2
    seasons = sorted(pairs["season"].unique().tolist())
    print(f"usable rows {len(pairs)}  team-weeks {pairs.groupby(['season','week','team']).ngroups}"
          f"  seasons {seasons[0]}-{seasons[-1]}")

    gamma, sse = fit_gamma(pairs)
    base_sse = sse_for_gamma(pairs, 1.0)
    rep = concentration_report(pairs, gamma)
    print(f"\n=== POOLED FIT (2013-2025) ===")
    print(f"  gamma = {gamma:.2f}   SSE {sse:.2f}   vs gamma=1 SSE {base_sse:.2f}"
          f"   improvement {100*(base_sse-sse)/base_sse:.2f}%")
    print(f"  top-1-by-proxy share: proxy {rep['top1_proxy_share']:.4f}"
          f" -> sharpened {rep['top1_sharpened_share']:.4f}"
          f"   actual {rep['top1_actual_share']:.4f}   (n={rep['n_team_weeks']} team-weeks)")

    print(f"\n=== WALK-FORWARD (gamma for S fitted on [2013, S-1]) ===")
    wf = []
    for s in seasons[1:]:
        tr = pairs[pairs["season"] < s]
        if tr.groupby(["season", "week", "team"]).ngroups < 50:
            continue
        g, _ = fit_gamma(tr)
        wf.append({"season": int(s), "gamma": g, "train_team_weeks":
                   int(tr.groupby(["season", "week", "team"]).ngroups)})
        print(f"  {s}: gamma {g:.2f}  (train team-weeks {wf[-1]['train_team_weeks']})")
    gs = [r["gamma"] for r in wf]
    stability = {"min": min(gs), "max": max(gs), "spread": round(max(gs) - min(gs), 3),
                 "mean": round(float(np.mean(gs)), 3),
                 "sd": round(float(np.std(gs, ddof=1)), 3)} if len(gs) > 1 else {}
    if stability:
        print(f"  stability: min {stability['min']:.2f} max {stability['max']:.2f}"
              f" spread {stability['spread']:.2f} mean {stability['mean']:.2f}"
              f" sd {stability['sd']:.3f}")

    result = {
        "version": "RB_CARRY_CONCENTRATION_FIT_V1",
        "preregistration": "docs/research/PRISM_RB_CARRY_CONCENTRATION_PREREGISTRATION_V1.md",
        "calibration_seasons": [int(s) for s in seasons],
        "excluded_seasons": [2026],
        "rows": int(len(pairs)),
        "team_weeks": int(pairs.groupby(["season", "week", "team"]).ngroups),
        "gamma_pooled": gamma,
        "sse": round(sse, 4),
        "sse_at_gamma_1": round(base_sse, 4),
        "concentration": rep,
        "walk_forward": wf,
        "walk_forward_stability": stability,
        "proxy": "within-season prior-weeks share of team RB carries; ranking is pregame only",
        "sportsbook_inputs_used": False,
    }
    p = out_dir / "rb_carry_concentration_fit_v1.json"
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {p}")

    if gamma <= 1.0:
        print("\nK1 TRIGGERED: gamma <= 1.0 -- shares are not flat; H1 is false.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
