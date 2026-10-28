"""Three nulls on the opponent side, so no future session re-runs them.

Each answers a specific question that was raised and is worth settling once:

1. **Are tough matchups worse for some players than others?** Tested with a
   permutation null rather than by eyeballing the spread of per-player slopes. The
   spread of per-player matchup sensitivity is indistinguishable from what random
   reassignment of opponents produces, and it does not carry year to year.

2. **Do defenses have a persistent exploitable SHAPE?** ("soft deep, attack them
   there" — the scouting intuition, which is distinct from overall quality.) A
   defense's deep-minus-short yards-allowed profile does not persist across disjoint
   halves of a season, and in one season it reverses.

3. **Is game script forecastable?** Neither the pass/rush rate nor total plays. The
   team's own prior plays/game does not beat a league constant.

What this does NOT say: that defense is irrelevant. Defensive **quality** in EPA terms
is forecastable (def pass EPA allowed r=+0.368, def stuff rate +0.411, def rush YPC
+0.381, measured in PRISM_PLAYER_BY_OPPONENT_QUALITY_EXPLORATORY_V1) and does carry a
real if small effect, scaled by each player's own level. What fails is everything
finer-grained than "good or bad defense": per-player sensitivity, exploitable shape,
and script.

Known limitation, stated rather than buried: every test here treats a defense as a
stable team-season object. It is not -- personnel changes weekly with injury. That is
a plausible mechanism for the shape null and leaves defender-level availability as a
genuinely untested axis.

No sportsbook variable is read at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = (2023, 2024, 2025)
OUT_JSON = Path("data/research/opponent_nulls/opponent_signal_nulls_v1.json")
CACHE = Path("/tmp/claude-0/-home-user-imtiredofthis/1701f8f3-afc9-57ea-bbc6-cce4dc6b5fbd/scratchpad/nfl_cache")

MIN_GAMES_FOR_SLOPE = 12
MIN_DEF_PRIOR_PLAYS = 100
MIN_DEEP_TARGETS = 60
MIN_SHORT_TARGETS = 100
DEEP_AIR_YARDS = 10.0
N_PERMUTATIONS = 200
SEED = 7


def _load(kind: str, season: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{kind}_{season}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import nflreadpy as nfl
    fn = {"weekly": nfl.load_player_stats, "pbp": nfl.load_pbp}[kind]
    df = fn(seasons=[season]).to_pandas()
    df.to_parquet(p)
    return df


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.concat([_load("weekly", y).assign(_yr=y) for y in SEASONS], ignore_index=True)
    p = pd.concat([_load("pbp", y).assign(_yr=y) for y in SEASONS], ignore_index=True)
    return w, p


def _prior_pass_defense(pbp: pd.DataFrame) -> dict:
    """Strictly-prior EPA allowed per (season, defense, week). Never includes week W."""
    ps = pbp[pbp["play_type"].eq("pass") & pbp["week"].between(1, 18)]
    dq = ps.groupby(["_yr", "defteam", "week"]).agg(e=("epa", "sum"), n=("epa", "size")).reset_index()
    out: dict = {}
    for (yr, team), h in dq.groupby(["_yr", "defteam"]):
        h = h.sort_values("week")
        ce, cn = h["e"].cumsum() - h["e"], h["n"].cumsum() - h["n"]
        out[(yr, team)] = dict(zip(h["week"], np.where(cn >= MIN_DEF_PRIOR_PLAYS, ce / cn.replace(0, np.nan), np.nan)))
    return out


def per_player_slopes(df: pd.DataFrame, qcol: str = "q") -> pd.Series:
    out = {}
    for pid, g in df.groupby("pid"):
        if len(g) >= MIN_GAMES_FOR_SLOPE and g[qcol].std() > 1e-9:
            out[pid] = float(np.polyfit(g[qcol].to_numpy(float), g["y"].to_numpy(float), 1)[0])
    return pd.Series(out, dtype=float)


def null_1_player_matchup_sensitivity(w: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    prior = _prior_pass_defense(pbp)
    r = w[w["position"].astype(str).str.upper().isin(["WR", "TE"]) & w["week"].between(1, 18)].copy()
    r["y"] = pd.to_numeric(r["receiving_yards"], errors="coerce").fillna(0.0)
    r = r[pd.to_numeric(r["targets"], errors="coerce").fillna(0.0) >= 1]
    r["q"] = [prior.get((y, o), {}).get(wk, np.nan)
              for y, o, wk in zip(r["_yr"], r["opponent_team"].astype(str), r["week"])]
    r = r.dropna(subset=["q"])
    r["pid"] = r["player_id"].astype(str)

    observed = per_player_slopes(r)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(N_PERMUTATIONS):
        s = r.copy()
        # reassign each player's opponents AMONG HIS OWN GAMES: keeps his outcome
        # distribution and the opponent pool, destroys any genuine player-opponent link
        s["q"] = s.groupby("pid")["q"].transform(lambda v: rng.permutation(v.to_numpy()))
        null.append(float(per_player_slopes(s).std()))
    null = np.asarray(null)

    carry = {}
    per_season = {y: per_player_slopes(r[r["_yr"] == y]) for y in SEASONS}
    for a, b in zip(SEASONS, SEASONS[1:]):
        j = pd.concat([per_season[a].rename("a"), per_season[b].rename("b")], axis=1).dropna()
        carry[f"{a}->{b}"] = {"n": int(len(j)),
                              "r": float(np.corrcoef(j["a"], j["b"])[0, 1]) if len(j) >= 20 else None}
    p_value = float((null >= observed.std()).mean())
    return {
        "players": int(len(observed)),
        "observed_spread": float(observed.std()),
        "permutation_null_mean": float(null.mean()),
        "permutation_null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "observed_over_null": float(observed.std() / null.mean()),
        "permutation_p": p_value,
        "cross_season_carry": carry,
        "verdict": "REAL" if p_value < 0.05 else "INDISTINGUISHABLE_FROM_NOISE",
    }


def _defense_depth_profile(ps: pd.DataFrame) -> pd.DataFrame:
    g = ps.groupby(["defteam", "band"]).agg(y=("yg", "sum"), n=("yg", "size")).reset_index()
    p = g.pivot(index="defteam", columns="band", values=["y", "n"])
    out = pd.DataFrame({
        "deep": p[("y", "deep")] / p[("n", "deep")],
        "short": p[("y", "short")] / p[("n", "short")],
        "n_deep": p[("n", "deep")], "n_short": p[("n", "short")],
    })
    out["level"] = (p[("y", "deep")] + p[("y", "short")]) / (p[("n", "deep")] + p[("n", "short")])
    out["shape"] = out["deep"] - out["short"]
    return out[(out["n_deep"] >= MIN_DEEP_TARGETS) & (out["n_short"] >= MIN_SHORT_TARGETS)]


def null_2_defense_exploitable_shape(pbp: pd.DataFrame) -> dict:
    ps = pbp[pbp["play_type"].eq("pass") & pbp["week"].between(1, 18)].copy()
    ps["ay"] = pd.to_numeric(ps["air_yards"], errors="coerce")
    ps["yg"] = pd.to_numeric(ps["yards_gained"], errors="coerce").fillna(0.0)
    ps = ps.dropna(subset=["ay"])
    ps["band"] = np.where(ps["ay"] < DEEP_AIR_YARDS, "short", "deep")

    rows = {}
    for yr in SEASONS:
        d = ps[ps["_yr"] == yr]
        e, l = _defense_depth_profile(d[d["week"] <= 9]), _defense_depth_profile(d[d["week"] >= 10])
        j = e.join(l, lsuffix="_e", rsuffix="_l").dropna()
        if len(j) >= 20:
            rows[f"{yr}_disjoint_halves"] = {
                "n": int(len(j)),
                "r_level": float(np.corrcoef(j["level_e"], j["level_l"])[0, 1]),
                "r_shape": float(np.corrcoef(j["shape_e"], j["shape_l"])[0, 1]),
            }
    season = {y: _defense_depth_profile(ps[ps["_yr"] == y]) for y in SEASONS}
    for a, b in zip(SEASONS, SEASONS[1:]):
        j = season[a].join(season[b], lsuffix="_a", rsuffix="_b").dropna()
        rows[f"{a}->{b}"] = {"n": int(len(j)),
                             "r_level": float(np.corrcoef(j["level_a"], j["level_b"])[0, 1]),
                             "r_shape": float(np.corrcoef(j["shape_a"], j["shape_b"])[0, 1])}
    shapes = [v["r_shape"] for v in rows.values()]
    return {"by_window": rows, "mean_r_shape": float(np.mean(shapes)),
            "any_window_negative": bool(min(shapes) < 0),
            "verdict": "PERSISTENT" if np.mean(shapes) > 0.35 else "NOT_PERSISTENT"}


def null_3_game_script(pbp: pd.DataFrame) -> dict:
    rp = pbp[pbp["play_type"].isin(["run", "pass"]) & pbp["week"].between(1, 18)]
    tw = rp.groupby(["_yr", "posteam", "week"]).size().rename("plays").reset_index()
    frames = []
    for (yr, team), h in tw.groupby(["_yr", "posteam"]):
        h = h.sort_values("week")
        cum, idx = h["plays"].cumsum() - h["plays"], np.arange(len(h))
        frames.append(pd.DataFrame({"_yr": yr, "team": team, "week": h["week"], "plays": h["plays"],
                                    "prior": np.where(idx >= 3, cum / np.where(idx == 0, 1, idx), np.nan)}))
    d = pd.concat(frames).dropna()
    fit, hold = d[d["week"] <= 11], d[d["week"] >= 12]
    league = float(fit["plays"].mean())
    e = d[d["week"] <= 9].groupby(["_yr", "team"])["plays"].mean()
    l = d[d["week"] >= 10].groupby(["_yr", "team"])["plays"].mean()
    j = pd.concat([e.rename("e"), l.rename("l")], axis=1).dropna()
    mae_const = float(np.abs(hold["plays"] - league).mean())
    mae_prior = float(np.abs(hold["plays"] - hold["prior"]).mean())
    return {
        "holdout_team_weeks": int(len(hold)),
        "league_constant_mae_plays": mae_const,
        "own_prior_mae_plays": mae_prior,
        "own_prior_beats_constant": bool(mae_prior < mae_const),
        "realized_plays_sd": float(hold["plays"].std()),
        "disjoint_half_persistence_r": float(np.corrcoef(j["e"], j["l"])[0, 1]),
        "verdict": "FORECASTABLE" if mae_prior < mae_const else "NOT_FORECASTABLE",
    }


def main() -> int:
    w, pbp = _frames()
    report = {
        "seasons": list(SEASONS),
        "null_1_player_matchup_sensitivity": null_1_player_matchup_sensitivity(w, pbp),
        "null_2_defense_exploitable_shape": null_2_defense_exploitable_shape(pbp),
        "null_3_game_script_plays": null_3_game_script(pbp),
        "what_survives": "defensive QUALITY in EPA terms is forecastable and is NOT nulled here",
        "known_limitation": "a defense is treated as a stable team-season object; weekly "
                            "personnel/injury change is untested and is a plausible mechanism "
                            "for the shape null",
        "sportsbook_inputs_used": False,
    }
    n1 = report["null_1_player_matchup_sensitivity"]
    if n1["verdict"] == "REAL":
        raise RuntimeError(
            f"player-specific matchup sensitivity now reads REAL (p={n1['permutation_p']:.3f}); "
            "this document records a null and must be re-derived before it is cited."
        )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON}\n")
    print(f"1. player matchup sensitivity: observed {n1['observed_spread']:.2f} vs null "
          f"{n1['permutation_null_mean']:.2f}  p={n1['permutation_p']:.3f}  -> {n1['verdict']}")
    n2 = report["null_2_defense_exploitable_shape"]
    print(f"2. defense exploitable shape: mean r(shape)={n2['mean_r_shape']:+.3f} -> {n2['verdict']}")
    n3 = report["null_3_game_script_plays"]
    print(f"3. game script plays: constant {n3['league_constant_mae_plays']:.2f} vs own prior "
          f"{n3['own_prior_mae_plays']:.2f} -> {n3['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
