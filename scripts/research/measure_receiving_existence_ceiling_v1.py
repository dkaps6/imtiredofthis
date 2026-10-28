"""Is the roster EXISTENCE ceiling an RB problem or a system-wide one?

GPT audit #535 comment 6479393261 item C approved this as diagnostic-only. Mirrors
scripts/research/measure_out_of_universe_rushing_v1.py on the receiving side.

Matching is SUFFIX-TOLERANT. canonicalize_player_name_safe does not strip Jr./Sr./II/III,
so nflverse "Mike Washington Jr." (mikewashingtonjr) never joins the universe's
"Mike Washington" (mikewashington). That gap produced a false out-of-universe
classification in the rushing pass; here suffix-only misses are reported as their own
class and NOT counted as absent.

State labels come from nflverse load_rosters_weekly. TIMING CAVEAT, per the same audit:
the current-season roster asset is MUTABLE, so today's file is strong evidence of what was
knowable but is NOT a contemporaneous snapshot of the Thursday run. Labels are evidence,
not proof.

Diagnostic only. Retunes nothing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.canonical_names import canonicalize_player_name_safe

UNIVERSE = Path("data/research/week7_opportunity_trace/"
                "week7_trace__data__football_simulation_universe.csv")
SEASON, WEEK = 2026, 7
SUFFIX = re.compile(r"(jr|sr|ii|iii|iv|v)$")
RECEIVING_POSITIONS = {"WR", "TE", "RB", "FB", "HB", "TB"}


def ckey(name: object) -> str:
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
    u["_k"] = u["player"].map(ckey)
    teams = sorted(u["team"].astype(str).unique())
    exact = {(str(t), k) for t, k in zip(u["team"], u["_k"])}
    stripped = {(str(t), SUFFIX.sub("", k)) for t, k in zip(u["team"], u["_k"])}

    w = nfl.load_player_stats(seasons=[SEASON]).to_pandas()
    w = w[(w["week"] == WEEK) & (w["team"].astype(str).isin(teams))].copy()
    for c in ("targets", "receptions", "receiving_yards"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)
    rec = w[w["targets"] > 0].copy()
    rec["_k"] = rec["player_display_name"].map(ckey)
    rec["_pos"] = rec["position"].fillna("").astype(str).str.upper()

    def classify_match(t, k):
        if (str(t), k) in exact:
            return "modeled"
        if (str(t), SUFFIX.sub("", k)) in stripped:
            return "SUFFIX_ONLY_JOIN_GAP"
        return "absent"

    rec["_match"] = [classify_match(t, k) for t, k in zip(rec["team"], rec["_k"])]

    # state-label the absent ones from weekly roster status
    rw = nfl.load_rosters_weekly(seasons=[SEASON]).to_pandas()
    ncol = "full_name" if "full_name" in rw.columns else "player_name"
    rw["_k"] = rw[ncol].map(ckey)
    rw["week"] = pd.to_numeric(rw["week"], errors="coerce")

    def state(team, k):
        s = rw[(rw["_k"] == k) & (rw["team"].astype(str) == str(team))]
        if s.empty:
            return "NO_ROSTER_ROW", ""
        prior = s[s["week"] < WEEK]["status"].astype(str).tolist()
        trail = "/".join(f"w{int(r.week)}:{r.status}" for r in s.sort_values("week").itertuples())
        if any(x == "ACT" for x in prior):
            return "PROVIDER_OMISSION_was_active_earlier", trail
        if any(x == "RES" for x in prior):
            return "RETURN_FROM_RESERVE", trail
        if any(x == "DEV" for x in prior):
            return "PRACTICE_SQUAD_ELEVATION", trail
        if not prior:
            return "LATE_ADDITION_no_prior_week", trail
        return "UNRESOLVED_CONTEXT", trail

    absent = rec[rec["_match"] == "absent"].copy()
    if not absent.empty:
        lab = [state(t, k) for t, k in zip(absent["team"], absent["_k"])]
        absent["_state"] = [a for a, _ in lab]
        absent["_trail"] = [b for _, b in lab]

    print(f"=== RECEIVING EXISTENCE CEILING, {SEASON} wk{WEEK}, {len(teams)} teams ===")
    print(f"  players with >0 targets: {len(rec)}")
    for m, g in rec.groupby("_match"):
        print(f"    {m:24s} players {len(g):4d}  targets {g['targets'].sum():6.0f}"
              f"  rec {g['receptions'].sum():5.0f}  yards {g['receiving_yards'].sum():7.0f}")

    tot_t = rec["targets"].sum()
    print(f"\n  share of all targets OUTSIDE the universe: "
          f"**{absent['targets'].sum()/tot_t:.4f}**" if not absent.empty else "\n  none absent")
    sfx = rec[rec["_match"] == "SUFFIX_ONLY_JOIN_GAP"]
    if not sfx.empty:
        print(f"  share lost to the SUFFIX JOIN GAP alone: {sfx['targets'].sum()/tot_t:.4f}"
              f"  ({len(sfx)} players, {sfx['targets'].sum():.0f} targets) — a repo identity"
              f" defect, NOT a roster gap")
        print(sfx[["team", "player_display_name", "_pos", "targets", "receptions"]]
              .sort_values("targets", ascending=False).head(8).to_string(index=False))

    by_pos, by_state = {}, {}
    if not absent.empty:
        print("\n=== ABSENT by position ===")
        p = (absent.groupby("_pos")
                   .agg(players=("player_display_name", "nunique"), targets=("targets", "sum"),
                        receptions=("receptions", "sum"), yards=("receiving_yards", "sum"))
                   .sort_values("targets", ascending=False))
        print(p.to_string())
        by_pos = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in p.to_dict("index").items()}

        print("\n=== ABSENT by state label (evidence, not proof — mutable roster asset) ===")
        st = (absent.groupby("_state")
                    .agg(players=("player_display_name", "nunique"), targets=("targets", "sum"))
                    .sort_values("targets", ascending=False))
        print(st.to_string())
        by_state = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in st.to_dict("index").items()}

        print("\n=== top absent receivers ===")
        print(absent.sort_values("targets", ascending=False)
              [["team", "player_display_name", "_pos", "targets", "receptions",
                "receiving_yards", "_state"]].head(14).to_string(index=False))

    print("\n=== team-level outside-universe TARGET-SHARE fraction ===")
    tr = []
    for t in teams:
        c = rec[rec["team"].astype(str) == t]
        if c["targets"].sum() <= 0:
            continue
        a = c[c["_match"] == "absent"]["targets"].sum()
        tr.append({"team": t, "team_targets": float(c["targets"].sum()),
                   "outside_targets": float(a),
                   "outside_frac": float(a) / float(c["targets"].sum())})
    tt = pd.DataFrame(tr)
    v = tt["outside_frac"]
    print(f"  mean {v.mean():.4f}  median {v.median():.4f}  p90 {v.quantile(.9):.4f}"
          f"  p95 {v.quantile(.95):.4f}  max {v.max():.4f}")
    print(f"  teams with ZERO outside-universe targets: {int((v == 0).sum())}/{len(tt)}")

    outdir = Path("data/research/rb_carry_concentration")
    outdir.mkdir(parents=True, exist_ok=True)
    res = {
        "version": "RECEIVING_EXISTENCE_CEILING_V1",
        "requested_by": "GPT audit #535 comment 6479393261 item C",
        "season": SEASON, "week": WEEK, "teams": len(teams),
        "players_with_targets": int(len(rec)),
        "match_breakdown": {m: {"players": int(len(g)), "targets": float(g["targets"].sum()),
                                "receptions": float(g["receptions"].sum()),
                                "yards": float(g["receiving_yards"].sum())}
                            for m, g in rec.groupby("_match")},
        "absent_target_share": float(absent["targets"].sum() / tot_t) if not absent.empty else 0.0,
        "suffix_gap_target_share": float(sfx["targets"].sum() / tot_t) if not sfx.empty else 0.0,
        "absent_by_position": by_pos, "absent_by_state": by_state,
        "team_outside_target_frac": {
            "mean": round(float(v.mean()), 5), "median": round(float(v.median()), 5),
            "p90": round(float(v.quantile(.9)), 5), "p95": round(float(v.quantile(.95)), 5),
            "max": round(float(v.max()), 5),
            "teams_zero": int((v == 0).sum()), "teams": int(len(tt))},
        "timing_caveat": "weekly roster asset is mutable; labels are evidence of what was "
                         "knowable, not a contemporaneous Thursday snapshot",
        "retuned_anything": False, "sportsbook_inputs_used": False,
    }
    (outdir / "receiving_existence_ceiling_v1.json").write_text(json.dumps(res, indent=2) + "\n")
    tt.to_csv(outdir / "receiving_existence_ceiling_teams_v1.csv", index=False)
    print(f"\nwrote {outdir}/receiving_existence_ceiling_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
