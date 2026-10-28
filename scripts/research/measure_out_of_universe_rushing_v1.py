"""AUDIT TARGET 4 — what fraction of real team rush attempts belong to players outside
the pregame modeled universe, and does the share denominator match production's rush pool?

Specified by GPT audit issue #535 comment 6478678129. The point is to derive the team
rush-share BUDGET empirically instead of hard-coding 0.95 or 1.00:

    modeled team rush budget = 1 - expected share of real rush attempts belonging to
                               players legitimately outside the modeled pregame universe

Identity discipline (explicitly required): unresolved identity is reported SEPARATELY and
is never silently counted as "unmodeled". A join failure is not an unmodeled rusher.

Leakage note: the modeled universe is the PREGAME 396-row frame. Completeness is judged
against it, never defined from who happened to carry the ball afterwards.

Diagnostic only. Changes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.simulation_v2 import _top_n_shares
from scripts.utils.canonical_names import canonicalize_player_name_safe

UNIVERSE = Path("data/research/week7_opportunity_trace/"
                "week7_trace__data__football_simulation_universe.csv")
SEASON, WEEK = 2026, 7
# positions that can legitimately carry the ball without being in a skill-position universe
PLAUSIBLY_UNMODELED = {"K", "P", "LS", "DT", "DE", "NT", "DL", "OL", "T", "G", "C",
                       "CB", "S", "FS", "SS", "DB", "LB", "ILB", "OLB", "MLB"}
MODELED_FAMILIES = {"QB", "RB", "FB", "HB", "TB", "WR", "TE"}


def _key(name: object) -> str:
    """Canonical clean key.

    canonicalize_player_name_safe returns a TUPLE (display_name, clean_key). An earlier
    draft did str() on the whole tuple, so the join matched nothing and real universe
    members (Dak Prescott, C.J. Stroud, J.K. Dobbins) were misfiled as out-of-universe.
    Take element [1], and fall back to the same alnum-lower rule the repo uses elsewhere.
    """
    try:
        k = canonicalize_player_name_safe(str(name))
    except Exception:
        k = None
    if isinstance(k, (tuple, list)) and len(k) > 1 and str(k[1]).strip():
        return str(k[1]).strip()
    if isinstance(k, str) and k.strip():
        return k.strip()
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def main() -> int:
    import nflreadpy as nfl

    u = pd.read_csv(UNIVERSE, low_memory=False)
    u["_k"] = u["player"].map(_key)
    teams = sorted(u["team"].astype(str).unique())
    modeled = {(str(t), str(k)) for t, k in zip(u["team"], u["_k"])}

    w = nfl.load_player_stats(seasons=[SEASON]).to_pandas()
    w = w[(w["week"] == WEEK) & (w["team"].astype(str).isin(teams))].copy()
    w["carries"] = pd.to_numeric(w["carries"], errors="coerce").fillna(0.0)
    car = w[w["carries"] > 0].copy()
    car["_k"] = car["player_display_name"].map(_key)
    car["_in"] = [(str(t), str(k)) in modeled for t, k in zip(car["team"], car["_k"])]
    car["_pos"] = car["position"].fillna("").astype(str).str.upper()

    # classify the outside carriers
    out = car[~car["_in"]].copy()
    out["_class"] = np.where(out["_pos"].isin(PLAUSIBLY_UNMODELED), "genuinely_unmodeled",
                     np.where(out["_pos"].isin(MODELED_FAMILIES), "ABSENT_FROM_PREGAME_UNIVERSE",
                              "other_position"))

    rows = []
    for t in teams:
        c = car[car["team"].astype(str) == t]
        tot = float(c["carries"].sum())
        if tot <= 0:
            continue
        inside = float(c.loc[c["_in"], "carries"].sum())
        o = out[out["team"].astype(str) == t]
        rows.append({
            "team": t, "actual_rush_att": tot, "modeled_att": inside,
            "outside_att": float(o["carries"].sum()),
            "outside_genuine": float(o.loc[o["_class"] == "genuinely_unmodeled", "carries"].sum()),
            "outside_unresolved": float(o.loc[o["_class"] == "ABSENT_FROM_PREGAME_UNIVERSE", "carries"].sum()),
            "outside_frac": float(o["carries"].sum()) / tot,
            "outside_frac_genuine": float(o.loc[o["_class"] == "genuinely_unmodeled", "carries"].sum()) / tot,
        })
    t = pd.DataFrame(rows)
    if t.empty:
        print("FAIL CLOSED: no teams", file=sys.stderr)
        return 2

    print(f"=== OUT-OF-UNIVERSE RUSHING, {SEASON} wk{WEEK}, n={len(t)} teams ===")
    print(f"  actual team rush attempts: mean {t['actual_rush_att'].mean():.2f}")
    print(f"  attempts by MODELED players: mean {t['modeled_att'].mean():.2f}")
    print(f"  attempts OUTSIDE the universe: mean {t['outside_att'].mean():.2f}")
    for lab, col in (("ALL outside", "outside_frac"), ("GENUINELY unmodeled only", "outside_frac_genuine")):
        v = t[col]
        print(f"\n  {lab} fraction of team rush attempts:")
        print(f"    mean {v.mean():.4f}  median {v.median():.4f}"
              f"  p90 {v.quantile(0.90):.4f}  p95 {v.quantile(0.95):.4f}  max {v.max():.4f}")
        print(f"    teams with ZERO such carries: {int((v == 0).sum())}/{len(t)}")
    print(f"\n  => empirical budget 1 - mean(genuine outside) = "
          f"**{1 - t['outside_frac_genuine'].mean():.4f}**"
          f"   (vs production's hard-coded 0.95)")

    print("\n=== outside carriers, classified (identity failures kept SEPARATE) ===")
    agg = (out.groupby("_class")
              .agg(players=("player_display_name", "nunique"), carries=("carries", "sum"))
              .reset_index())
    print(agg.to_string(index=False))
    if (out["_class"] == "ABSENT_FROM_PREGAME_UNIVERSE").any():
        print("\n  UNRESOLVED (skill-position carriers absent from the pregame universe) —")
        print("  these are NOT evidence of unmodeled rushing; they are join/roster gaps:")
        bad = (out[out["_class"] == "ABSENT_FROM_PREGAME_UNIVERSE"]
               .groupby(["team", "player_display_name", "_pos"], as_index=False)["carries"].sum()
               .sort_values("carries", ascending=False))
        print(bad.head(20).to_string(index=False))
    if (out["_class"] == "genuinely_unmodeled").any():
        print("\n  GENUINELY unmodeled carriers:")
        print(out[out["_class"] == "genuinely_unmodeled"]
              .groupby(["team", "player_display_name", "_pos"], as_index=False)["carries"].sum()
              .sort_values("carries", ascending=False).head(15).to_string(index=False))

    print("\n=== DENOMINATOR SEMANTICS: does the share pool match production's rush_att? ===")
    dr = []
    for team, g in u.groupby("team"):
        plays = pd.to_numeric(g["rules_plays_est"], errors="coerce").dropna()
        prate = pd.to_numeric(g["rules_pass_rate"], errors="coerce").dropna()
        if plays.empty or prate.empty:
            continue
        dr.append({"team": str(team),
                   "model_rush_att": float(plays.iloc[0]) * (1.0 - float(prate.iloc[0]))})
    dmod = pd.DataFrame(dr).merge(t[["team", "actual_rush_att"]], on="team", how="inner")
    diff = dmod["model_rush_att"] - dmod["actual_rush_att"]
    ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"  model rush_att (plays x (1-pass_rate)) mean {dmod['model_rush_att'].mean():.2f}")
    print(f"  actual rush attempts (sum of carries)  mean {dmod['actual_rush_att'].mean():.2f}")
    print(f"  difference {diff.mean():+.2f}  95% CI [{diff.mean()-ci:+.2f}, {diff.mean()+ci:+.2f}]")
    print("  (a positive difference means production's pool is LARGER than official carries —")
    print("   e.g. sacks/kneels/aborted plays counted as rushes — which would belong in the")
    print("   budget contract explicitly rather than inside a 5% fudge factor.)")

    outdir = Path("data/research/rb_carry_concentration")
    outdir.mkdir(parents=True, exist_ok=True)
    res = {
        "version": "OUT_OF_UNIVERSE_RUSHING_V1",
        "requested_by": "GPT audit #535 comment 6478678129 (AUDIT TARGET 4)",
        "season": SEASON, "week": WEEK, "teams": int(len(t)),
        "identity_join": "repo canonicalize_player_name_safe on (team, player); universe has no gsis_id",
        "leakage_note": "completeness judged against the PREGAME 396-row universe only",
        "actual_rush_att_mean": round(float(t["actual_rush_att"].mean()), 3),
        "outside_frac_all": {k: round(float(v), 5) for k, v in
                             {"mean": t["outside_frac"].mean(), "median": t["outside_frac"].median(),
                              "p90": t["outside_frac"].quantile(.9), "p95": t["outside_frac"].quantile(.95),
                              "max": t["outside_frac"].max()}.items()},
        "outside_frac_genuine": {k: round(float(v), 5) for k, v in
                                 {"mean": t["outside_frac_genuine"].mean(),
                                  "median": t["outside_frac_genuine"].median(),
                                  "p90": t["outside_frac_genuine"].quantile(.9),
                                  "p95": t["outside_frac_genuine"].quantile(.95),
                                  "max": t["outside_frac_genuine"].max()}.items()},
        "teams_zero_outside": int((t["outside_frac"] == 0).sum()),
        "teams_zero_genuine_outside": int((t["outside_frac_genuine"] == 0).sum()),
        "empirical_budget_from_genuine": round(float(1 - t["outside_frac_genuine"].mean()), 5),
        "production_hardcoded_cap": 0.95,
        "outside_classified": {r["_class"]: {"players": int(r["players"]), "carries": float(r["carries"])}
                               for _, r in agg.iterrows()},
        "denominator": {
            "model_rush_att_mean": round(float(dmod["model_rush_att"].mean()), 3),
            "actual_rush_att_mean": round(float(dmod["actual_rush_att"].mean()), 3),
            "difference_mean": round(float(diff.mean()), 3),
            "difference_ci95": [round(float(diff.mean()-ci), 3), round(float(diff.mean()+ci), 3)],
        },
        "weeks_covered": 1,
        "weeks_limitation": "only W7 has a preserved leakage-safe pregame universe",
        "changed_anything": False, "sportsbook_inputs_used": False,
    }
    (outdir / "out_of_universe_rushing_v1.json").write_text(json.dumps(res, indent=2) + "\n")
    t.to_csv(outdir / "out_of_universe_rushing_teams_v1.csv", index=False)
    print(f"\nwrote {outdir}/out_of_universe_rushing_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
