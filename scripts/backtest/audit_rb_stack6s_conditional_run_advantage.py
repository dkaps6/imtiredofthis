#!/usr/bin/env python3
"""RB STACK6S: no-fit conditional run-vs-pass advantage forensic audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("lead", "neutral", "trail")
CONTEXTS = ("first_down", "second_short_med", "second_long", "late_short", "late_long", "other")
SCHEMES = {"team5_shrunk": 5, "team8_shrunk": 8}
PSEUDO = 24.0
TEAM_MAP = {"JAX": "JAC", "LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC"}
EXPECTED_TEAM_GAMES = 544
EXPECTED_W6_TEAM_GAMES = 388


def num(v):
    return pd.to_numeric(v, errors="coerce")


def canon(v):
    s = str(v).strip().upper()
    return TEAM_MAP.get(s, s)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl

    p = lower(nfl.load_pbp(seasons=[2023, 2024, 2025]).to_pandas())
    if "season_type" in p.columns:
        reg = p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            p = reg
    required = {
        "season", "week", "posteam", "defteam", "rush_attempt", "qb_dropback",
        "score_differential", "down", "ydstogo", "epa", "success",
    }
    missing = required - set(p.columns)
    if missing:
        raise RuntimeError(f"STACK6S PBP missing required columns: {sorted(missing)}")

    for c in ["rush_attempt", "qb_dropback", "qb_scramble", "qb_kneel", "score_differential", "down", "ydstogo", "epa", "success"]:
        p[c] = num(p[c]) if c in p.columns else np.nan
    p["team"] = p.posteam.map(canon)
    p["opponent"] = p.defteam.map(canon)
    p["off_play"] = (p.rush_attempt.eq(1) | p.qb_dropback.eq(1)).astype(int)
    p = p.loc[p.off_play.eq(1) & p.team.ne("") & p.opponent.ne("")].copy()

    p["state"] = np.select(
        [p.score_differential.gt(3), p.score_differential.lt(-3)],
        ["lead", "trail"], default="neutral",
    )
    p["designed"] = (
        p.rush_attempt.eq(1)
        & ~p.qb_scramble.fillna(0).eq(1)
        & ~p.qb_kneel.fillna(0).eq(1)
    ).astype(int)
    p["pass_intent"] = (1 - p.designed).astype(int)

    d = p["down"]
    y = p["ydstogo"]
    conds = [
        d.eq(1),
        d.eq(2) & y.le(6),
        d.eq(2) & y.ge(7),
        d.isin([3, 4]) & y.le(3),
        d.isin([3, 4]) & y.ge(4),
    ]
    p["context"] = np.select(conds, CONTEXTS[:-1], default="other")
    p["context_count"] = 1
    return p


def build_cell_tables(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["season", "week", "state", "context"]

    def summarize(g: pd.DataFrame) -> pd.Series:
        run = g.loc[g.designed.eq(1)]
        pas = g.loc[g.pass_intent.eq(1)]
        return pd.Series({
            "plays": float(len(g)),
            "designed": float(g.designed.sum()),
            "run_epa_sum": float(num(run.epa).dropna().sum()),
            "run_epa_n": float(num(run.epa).notna().sum()),
            "pass_epa_sum": float(num(pas.epa).dropna().sum()),
            "pass_epa_n": float(num(pas.epa).notna().sum()),
            "run_success_sum": float(num(run.success).dropna().sum()),
            "run_success_n": float(num(run.success).notna().sum()),
            "pass_success_sum": float(num(pas.success).dropna().sum()),
            "pass_success_n": float(num(pas.success).notna().sum()),
        })

    off = (
        p.groupby(["season", "week", "team", "state", "context"], dropna=False)
        .apply(summarize).reset_index()
    )
    deff = (
        p.groupby(["season", "week", "opponent", "state", "context"], dropna=False)
        .apply(summarize).reset_index().rename(columns={"opponent": "defense"})
    )
    league = (
        p.groupby(keys, dropna=False).apply(summarize).reset_index()
    )
    return off, deff, league


def prior_mask(df: pd.DataFrame, season: int, week: int) -> pd.Series:
    s = num(df.season)
    w = num(df.week)
    return s.lt(season) | (s.eq(season) & w.lt(week))


def sums(g: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    if g.empty:
        return {c: 0.0 for c in cols}
    return {c: float(num(g[c]).fillna(0).sum()) for c in cols}


def branch_mean(total_sum: float, total_n: float, league_mean: float) -> float:
    return float((total_sum + PSEUDO * league_mean) / (total_n + PSEUDO))


def league_prior_means(league: pd.DataFrame, season: int, week: int, state: str, context: str) -> dict[str, float]:
    g = league.loc[
        prior_mask(league, season, week)
        & league.state.eq(state)
        & league.context.eq(context)
    ]
    q = sums(g, [
        "plays", "designed", "run_epa_sum", "run_epa_n", "pass_epa_sum", "pass_epa_n",
        "run_success_sum", "run_success_n", "pass_success_sum", "pass_success_n",
    ])
    def mean(scol, ncol, default):
        return float(q[scol] / q[ncol]) if q[ncol] > 0 else default
    return {
        "call_rate": float(q["designed"] / q["plays"]) if q["plays"] > 0 else 0.43,
        "run_epa": mean("run_epa_sum", "run_epa_n", 0.0),
        "pass_epa": mean("pass_epa_sum", "pass_epa_n", 0.0),
        "run_success": mean("run_success_sum", "run_success_n", 0.5),
        "pass_success": mean("pass_success_sum", "pass_success_n", 0.5),
    }


def build_target_cell_features(
    target_keys: pd.DataFrame,
    off: pd.DataFrame,
    deff: pd.DataFrame,
    league: pd.DataFrame,
    n_games: int,
) -> pd.DataFrame:
    rows = []
    sum_cols = [
        "plays", "designed", "run_epa_sum", "run_epa_n", "pass_epa_sum", "pass_epa_n",
        "run_success_sum", "run_success_n", "pass_success_sum", "pass_success_n",
    ]
    for _, r in target_keys.iterrows():
        season, week = int(r.season), int(r.week)
        team, opp, state, context = canon(r.team), canon(r.opponent), str(r.state), str(r.context)
        lg = league_prior_means(league, season, week, state, context)

        og = off.loc[
            off.team.eq(team)
            & off.state.eq(state)
            & off.context.eq(context)
            & prior_mask(off, season, week)
        ].sort_values(["season", "week"]).tail(n_games)
        dg = deff.loc[
            deff.defense.eq(opp)
            & deff.state.eq(state)
            & deff.context.eq(context)
            & prior_mask(deff, season, week)
        ].sort_values(["season", "week"]).tail(n_games)
        os = sums(og, sum_cols)
        ds = sums(dg, sum_cols)

        prior_call = float((os["designed"] + PSEUDO * lg["call_rate"]) / (os["plays"] + PSEUDO))

        off_run_epa = branch_mean(os["run_epa_sum"], os["run_epa_n"], lg["run_epa"])
        off_pass_epa = branch_mean(os["pass_epa_sum"], os["pass_epa_n"], lg["pass_epa"])
        def_run_epa = branch_mean(ds["run_epa_sum"], ds["run_epa_n"], lg["run_epa"])
        def_pass_epa = branch_mean(ds["pass_epa_sum"], ds["pass_epa_n"], lg["pass_epa"])

        off_run_success = branch_mean(os["run_success_sum"], os["run_success_n"], lg["run_success"])
        off_pass_success = branch_mean(os["pass_success_sum"], os["pass_success_n"], lg["pass_success"])
        def_run_success = branch_mean(ds["run_success_sum"], ds["run_success_n"], lg["run_success"])
        def_pass_success = branch_mean(ds["pass_success_sum"], ds["pass_success_n"], lg["pass_success"])

        rows.append({
            "season": season, "week": week, "team": team, "opponent": opp,
            "state": state, "context": context,
            "prior_call_prob": prior_call,
            "epa_run_advantage": (off_run_epa - off_pass_epa) + (def_run_epa - def_pass_epa),
            "success_run_advantage": (off_run_success - off_pass_success) + (def_run_success - def_pass_success),
            "off_prior_cell_plays": os["plays"],
            "def_prior_cell_plays": ds["plays"],
        })
    return pd.DataFrame(rows)


def corr(a: pd.Series, b: pd.Series) -> float:
    a, b = num(a), num(b)
    ok = a.notna() & b.notna()
    a, b = a.loc[ok], b.loc[ok]
    return float(a.corr(b)) if len(a) >= 3 and a.nunique() > 1 and b.nunique() > 1 else np.nan


def score_signal(df: pd.DataFrame, signal: str, scheme: str) -> dict:
    z = df.loc[num(df[signal]).notna() & num(df.call_residual).notna()].copy()
    full_corr = corr(z[signal], z.call_residual)
    q25 = float(num(z[signal]).quantile(0.25))
    q75 = float(num(z[signal]).quantile(0.75))
    bot = z.loc[num(z[signal]).le(q25), "call_residual"]
    top = z.loc[num(z[signal]).ge(q75), "call_residual"]
    spread = float(num(top).mean() - num(bot).mean()) if len(top) and len(bot) else np.nan
    early = corr(z.loc[z.week.between(6, 12), signal], z.loc[z.week.between(6, 12), "call_residual"])
    late = corr(z.loc[z.week.ge(13), signal], z.loc[z.week.ge(13), "call_residual"])
    qualifies = int(
        pd.notna(full_corr) and full_corr >= 0.03
        and pd.notna(spread) and spread >= 0.03
        and pd.notna(early) and early > 0
        and pd.notna(late) and late > 0
    )
    return {
        "scheme": scheme,
        "signal": signal,
        "n": int(len(z)),
        "corr_full": full_corr,
        "q25": q25,
        "q75": q75,
        "top_minus_bottom_residual_spread": spread,
        "corr_w6_12": early,
        "corr_w13_18": late,
        "scheme_signal_qualifies": qualifies,
    }


def descriptive_slices(df: pd.DataFrame, signal: str, scheme: str) -> pd.DataFrame:
    rows = []
    slices = {
        "ALL_W6_18": pd.Series(True, index=df.index),
        "POOL_OVER_5": df.pool_over_5.eq(1),
        "POOL_UNDER_5": df.pool_under_5.eq(1),
        "W6_12": df.week.between(6, 12),
        "W13_18": df.week.ge(13),
    }
    for s in STATES:
        slices[f"STATE_{s.upper()}"] = df.state.eq(s)
    for c in CONTEXTS:
        slices[f"CTX_{c.upper()}"] = df.context.eq(c)
    for name, mask in slices.items():
        z = df.loc[mask].copy()
        rows.append({
            "scheme": scheme,
            "signal": signal,
            "population": name,
            "n": int(len(z)),
            "corr_signal_residual": corr(z[signal], z.call_residual),
            "mean_signal": float(num(z[signal]).mean()) if len(z) else np.nan,
            "mean_call_residual": float(num(z.call_residual).mean()) if len(z) else np.nan,
            "actual_designed_rate": float(num(z.designed).mean()) if len(z) else np.nan,
            "mean_prior_call_prob": float(num(z.prior_call_prob).mean()) if len(z) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    h = one(a.stack6h_root, "stack6h_team_trace.csv")
    h["season"] = num(h.season).astype(int)
    h["week"] = num(h.week).astype(int)
    h["team"] = h.team.map(canon)

    p = load_pbp()
    off, deff, league = build_cell_tables(p)

    p25 = p.loc[p.season.eq(2025)].copy()
    team_games_25 = p25[["season", "week", "team"]].drop_duplicates()
    joined_tg = team_games_25.merge(
        h[["season", "week", "team", "pool_over_5", "pool_under_5"]],
        on=["season", "week", "team"], how="inner", validate="one_to_one",
    )
    if len(joined_tg) != EXPECTED_TEAM_GAMES:
        raise RuntimeError(f"STACK6S team-game join expected {EXPECTED_TEAM_GAMES}; got {len(joined_tg)}")

    p25 = p25.merge(
        h[["season", "week", "team", "pool_over_5", "pool_under_5"]],
        on=["season", "week", "team"], how="inner", validate="many_to_one",
    )
    w = p25.loc[p25.week.ge(6)].copy()
    represented_w6 = w[["season", "week", "team"]].drop_duplicates()

    target_cells = w[["season", "week", "team", "opponent", "state", "context"]].drop_duplicates().reset_index(drop=True)

    all_play_traces = []
    qualification_rows = []
    descriptive = []
    coverage_rows = []
    for scheme, n_games in SCHEMES.items():
        feats = build_target_cell_features(target_cells, off, deff, league, n_games)
        z = w.merge(
            feats,
            on=["season", "week", "team", "opponent", "state", "context"],
            how="left", validate="many_to_one",
        )
        z["call_residual"] = num(z.designed) - num(z.prior_call_prob)
        z["scheme"] = scheme
        all_play_traces.append(z)
        for signal in ["epa_run_advantage", "success_run_advantage"]:
            qualification_rows.append(score_signal(z, signal, scheme))
            descriptive.append(descriptive_slices(z, signal, scheme))
            coverage_rows.append({
                "scheme": scheme,
                "signal": signal,
                "finite_rate": float(np.isfinite(num(z[signal])).mean()),
            })

    trace = pd.concat(all_play_traces, ignore_index=True)
    quals = pd.DataFrame(qualification_rows)
    desc = pd.concat(descriptive, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)

    signal_summary = []
    for signal in ["epa_run_advantage", "success_run_advantage"]:
        q = quals.loc[quals.signal.eq(signal)]
        qualified_both = int(len(q) == 2 and q.scheme_signal_qualifies.eq(1).all())
        signal_summary.append({"signal": signal, "qualified_both_schemes": qualified_both})
    signal_summary = pd.DataFrame(signal_summary)

    context_sum = pd.Series(0, index=p.index, dtype=int)
    for c in CONTEXTS:
        context_sum += p.context.eq(c).astype(int)
    context_identity = int((context_sum == 1).all())
    success_present = int("success" in p.columns and num(p.success).notna().any())
    finite_ok = int((coverage.finite_rate >= 0.99).all())
    integrity_pass = int(
        len(joined_tg) == EXPECTED_TEAM_GAMES
        and len(represented_w6) == EXPECTED_W6_TEAM_GAMES
        and context_identity
        and success_present
        and finite_ok
    )

    any_qualified = int(signal_summary.qualified_both_schemes.eq(1).any())
    if not integrity_pass:
        disposition = "STACK6S_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    elif any_qualified:
        disposition = "CONDITIONAL_ADVANTAGE_SIGNAL_QUALIFIED"
    else:
        disposition = "CONDITIONAL_ADVANTAGE_SIGNAL_NOT_QUALIFIED"

    integrity = pd.DataFrame([{
        "pbp_rows_2023_2025": len(p),
        "team_games_2025_joined": len(joined_tg),
        "w6_18_team_games_represented": len(represented_w6),
        "w6_18_decision_plays": len(w),
        "target_cells_w6_18": len(target_cells),
        "context_identity_pass": context_identity,
        "success_source_present": success_present,
        "finite_signal_coverage_ge_99pct": finite_ok,
        "strict_prior_construction": 1,
        "fitted_models": 0,
        "feature_search": 0,
        "model_family_search": 0,
        "hyperparameter_search": 0,
        "threshold_search": 0,
        "coefficient_search": 0,
        "sportsbook_inputs": 0,
        "target_game_pbp_used_for_grading_only": 1,
        "integrity_pass": integrity_pass,
    }])
    disposition_df = pd.DataFrame([{
        "qualified_signal_count": int(signal_summary.qualified_both_schemes.sum()),
        "disposition": disposition,
        "production_change": 0,
        "predictive_model_authorized": int(integrity_pass and any_qualified),
    }])

    trace.to_csv(a.out_dir / "stack6s_play_trace.csv", index=False)
    quals.to_csv(a.out_dir / "stack6s_signal_qualification.csv", index=False)
    signal_summary.to_csv(a.out_dir / "stack6s_signal_summary.csv", index=False)
    desc.to_csv(a.out_dir / "stack6s_descriptive_slices.csv", index=False)
    coverage.to_csv(a.out_dir / "stack6s_signal_coverage.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6s_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6s_disposition.csv", index=False)

    print("=== STACK6S integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6S qualification ===")
    print(quals.to_string(index=False))
    print("=== STACK6S signal summary ===")
    print(signal_summary.to_string(index=False))
    print("=== STACK6S disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6S_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
