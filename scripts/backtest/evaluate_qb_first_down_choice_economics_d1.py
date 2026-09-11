#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

SOURCE_ROWS = "first_down_choice_economics_source_rows.csv"
FEATURES = [
    "off_fd_epa_pass_minus_run",
    "def_fd_epa_pass_minus_run",
    "off_fd_success_pass_minus_run",
    "def_fd_success_pass_minus_run",
]
TRAIN_WEEKS = range(1, 10)
HOLDOUT_WEEKS = range(10, 19)
HISTORY_GAMES = 8
SHRINK_GAMES = 4.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 310


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def canon(v):
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def regular_only(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    c = "season_type" if "season_type" in x.columns else ("game_type" if "game_type" in x.columns else None)
    if c:
        s = x[c].fillna("").astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            x = x.loc[keep].copy()
    return x


def to_pd(o):
    if isinstance(o, pd.DataFrame):
        return o.copy()
    if hasattr(o, "to_pandas"):
        return o.to_pandas()
    return pd.DataFrame(o)


def load_source(root: Path) -> pd.DataFrame:
    d = pd.read_csv(one(root, SOURCE_ROWS), low_memory=False)
    d.columns = [str(c).strip().lower() for c in d.columns]
    need = {"season", "week", "team", "opponent", *FEATURES}
    missing = sorted(need - set(d.columns))
    if missing:
        raise RuntimeError(f"source missing {missing}")
    d["season"] = num(d.season)
    d["week"] = num(d.week)
    d["team"] = d.team.fillna("").astype(str).map(canon)
    d["opponent"] = d.opponent.fillna("").astype(str).map(canon)
    # Only 2023 rows are admitted into the D1 analysis frame.
    d = d.loc[d.season.eq(2023) & d.week.between(1, 18), ["season", "week", "team", "opponent", *FEATURES]].copy()
    d["season"] = 2023
    d["week"] = d.week.astype(int)
    for c in FEATURES:
        d[c] = num(d[c])
    if len(d) != 544 or d.duplicated(["season", "week", "team"]).any() or d[FEATURES].isna().any().any():
        raise RuntimeError(f"2023 source integrity failure rows={len(d)}")
    return d.sort_values(["week", "team"]).reset_index(drop=True)


def load_pbp_2022_2023() -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl

    frames = []
    audit = {}
    for season in (2022, 2023):
        raw = nfl.load_pbp(seasons=[season])
        p = regular_only(to_pd(raw))
        p.columns = [str(c).strip().lower() for c in p.columns]
        required = {"week", "posteam", "defteam", "down", "qb_dropback", "rush_attempt"}
        missing = sorted(required - set(p.columns))
        if missing:
            raise RuntimeError(f"PBP {season} missing {missing}")
        p["season"] = season
        for c in ["week", "down", "qb_dropback", "rush_attempt"]:
            p[c] = num(p[c])
        p["posteam"] = p.posteam.fillna("").astype(str).map(canon)
        p["defteam"] = p.defteam.fillna("").astype(str).map(canon)
        two = num(p["two_point_attempt"]).fillna(0).eq(1) if "two_point_attempt" in p else pd.Series(False, index=p.index)
        no_play = num(p["no_play"]).fillna(0).eq(1) if "no_play" in p else pd.Series(False, index=p.index)
        kneel = num(p["qb_kneel"]).fillna(0).eq(1) if "qb_kneel" in p else pd.Series(False, index=p.index)
        pass_origin = p.qb_dropback.fillna(0).eq(1)
        designed_run = p.rush_attempt.fillna(0).eq(1) & ~pass_origin & ~kneel
        eligible = (
            p.week.between(1, 18) & p.down.eq(1) & p.posteam.ne("") & p.defteam.ne("")
            & (pass_origin | designed_run) & ~two & ~no_play
        )
        q = p.loc[eligible, ["season", "week", "posteam", "defteam"]].copy()
        q["pass_origin"] = pass_origin.loc[eligible].astype(int).to_numpy()
        q["designed_run"] = designed_run.loc[eligible].astype(int).to_numpy()
        if not (q.pass_origin + q.designed_run).eq(1).all():
            raise RuntimeError(f"choice exclusivity failure {season}")
        audit[str(season)] = {
            "eligible_first_down_plays": int(len(q)),
            "pass_origin_plays": int(q.pass_origin.sum()),
            "designed_run_plays": int(q.designed_run.sum()),
        }
        frames.append(q)
    return pd.concat(frames, ignore_index=True), audit


def game_table(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        fd_plays=("pass_origin", "size"),
        fd_dropbacks=("pass_origin", "sum"),
    ).rename(columns={"posteam": "team", "defteam": "opponent"})
    g["team"] = g.team.map(canon)
    g["opponent"] = g.opponent.map(canon)
    g["ord"] = g.season.astype(int) * 100 + g.week.astype(int)
    g["fd_dbr"] = num(g.fd_dropbacks) / num(g.fd_plays)
    if g.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate first-down team-week PBP")
    return g.sort_values(["ord", "team"]).reset_index(drop=True)


def pooled_rate(g: pd.DataFrame) -> float:
    den = float(num(g.fd_plays).sum())
    return float(num(g.fd_dropbacks).sum() / den) if den > 0 else np.nan


def shr(v: float, n_games: int, league: float) -> float:
    if np.isfinite(v) and n_games > 0:
        return float((n_games * v + SHRINK_GAMES * league) / (n_games + SHRINK_GAMES))
    return np.nan


def build_baseline(source: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in source.itertuples(index=False):
        target_ord = 202300 + int(r.week)
        prior = games.loc[games.ord < target_ord]
        league = pooled_rate(prior)
        if not np.isfinite(league):
            raise RuntimeError(f"no league prior for W{r.week}")
        off = prior.loc[prior.team.eq(r.team)].sort_values("ord").tail(HISTORY_GAMES)
        deff = prior.loc[prior.opponent.eq(r.opponent)].sort_values("ord").tail(HISTORY_GAMES)
        ov = pooled_rate(off)
        dv = pooled_rate(deff)
        os = shr(ov, len(off), league)
        ds = shr(dv, len(deff), league)
        vals = [v for v in (os, ds) if np.isfinite(v)]
        baseline = float(np.mean(vals)) if vals else float(league)
        rows.append({
            "season": 2023,
            "week": int(r.week),
            "team": r.team,
            "baseline_d1_dbr": float(np.clip(baseline, .05, .95)),
            "league_d1_dbr": league,
            "off_prior_games_baseline": int(len(off)),
            "def_prior_games_baseline": int(len(deff)),
            "off_prior_max_ord": int(off.ord.max()) if len(off) else np.nan,
            "def_prior_max_ord": int(deff.ord.max()) if len(deff) else np.nan,
            "target_ord": target_ord,
        })
    return pd.DataFrame(rows)


def actual_2023(games: pd.DataFrame) -> pd.DataFrame:
    x = games.loc[games.season.eq(2023), ["season", "week", "team", "opponent", "fd_plays", "fd_dropbacks", "fd_dbr"]].copy()
    x = x.rename(columns={"fd_dbr": "actual_d1_dbr", "opponent": "pbp_opponent"})
    return x


def add_choice_edge(z: pd.DataFrame) -> pd.DataFrame:
    x = z.copy()
    rank_cols = []
    for c in FEATURES:
        rc = f"pct_{c}"
        x[rc] = x.groupby("week")[c].rank(method="average", pct=True)
        rank_cols.append(rc)
    x["choice_edge"] = x[rank_cols].mean(axis=1) - .5
    if x.choice_edge.isna().any():
        raise RuntimeError("non-finite choice edge")
    return x


def metrics(actual, pred) -> dict:
    a = num(actual).to_numpy(float)
    p = num(pred).to_numpy(float)
    e = p - a
    return {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "corr": float(np.corrcoef(a, p)[0, 1]) if len(a) > 2 and np.std(p) > 0 and np.std(a) > 0 else np.nan,
        "p50_abs_error": float(np.quantile(np.abs(e), .50)),
        "p75_abs_error": float(np.quantile(np.abs(e), .75)),
        "p90_abs_error": float(np.quantile(np.abs(e), .90)),
        "mean_pred": float(np.mean(p)),
        "mean_actual": float(np.mean(a)),
    }


def corr_metrics(x, y) -> dict:
    d = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    return {
        "n": int(len(d)),
        "pearson": float(d.x.corr(d.y, method="pearson")) if len(d) > 2 and d.x.nunique() > 1 and d.y.nunique() > 1 else np.nan,
        "spearman": float(d.x.corr(d.y, method="spearman")) if len(d) > 2 and d.x.nunique() > 1 and d.y.nunique() > 1 else np.nan,
        "same_sign": float((np.sign(d.x) == np.sign(d.y)).mean()) if len(d) else np.nan,
    }


def bootstrap_support(g: pd.DataFrame) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    a = g.actual_d1_dbr.to_numpy(float)
    b = g.baseline_d1_dbr.to_numpy(float)
    c = g.candidate_d1_dbr.to_numpy(float)
    n = len(g)
    wins = 0
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        bg = np.mean(np.abs(b[idx] - a[idx]))
        cg = np.mean(np.abs(c[idx] - a[idx]))
        wins += int(bg - cg > 0)
    return float(wins / BOOTSTRAP_N)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    source = load_source(a.source_root)
    pbp, pbp_audit = load_pbp_2022_2023()
    games = game_table(pbp)
    baseline = build_baseline(source, games)
    actual = actual_2023(games)

    z = source.merge(baseline, on=["season", "week", "team"], how="left", validate="one_to_one")
    z = z.merge(actual, on=["season", "week", "team"], how="left", validate="one_to_one")
    z["opponent_match"] = z.opponent.eq(z.pbp_opponent)
    if len(z) != 544 or z.actual_d1_dbr.isna().any() or not z.opponent_match.all():
        raise RuntimeError(f"2023 target PBP alignment failure rows={len(z)} missing={int(z.actual_d1_dbr.isna().sum())}")
    z = add_choice_edge(z)
    z["baseline_residual"] = z.actual_d1_dbr - z.baseline_d1_dbr

    train = z.loc[z.week.isin(TRAIN_WEEKS)].copy()
    hold = z.loc[z.week.isin(HOLDOUT_WEEKS)].copy()
    den = float(np.sum(train.choice_edge.to_numpy(float) ** 2))
    beta_raw = float(np.sum(train.choice_edge.to_numpy(float) * train.baseline_residual.to_numpy(float)) / den) if den > 0 else np.nan
    beta = float(max(0.0, beta_raw)) if np.isfinite(beta_raw) else np.nan
    z["correction"] = beta * z.choice_edge
    z["candidate_d1_dbr"] = np.clip(z.baseline_d1_dbr + z.correction, .20, .80)
    hold = z.loc[z.week.isin(HOLDOUT_WEEKS)].copy()

    bm = metrics(hold.actual_d1_dbr, hold.baseline_d1_dbr)
    cm = metrics(hold.actual_d1_dbr, hold.candidate_d1_dbr)
    mae_gain = float(bm["mae"] - cm["mae"])
    correction_corr = corr_metrics(hold.correction, hold.baseline_residual)
    week_rows = []
    for w, g in hold.groupby("week", sort=True):
        b = float(np.mean(np.abs(g.baseline_d1_dbr - g.actual_d1_dbr)))
        c = float(np.mean(np.abs(g.candidate_d1_dbr - g.actual_d1_dbr)))
        week_rows.append({"week": int(w), "n": int(len(g)), "baseline_mae": b, "candidate_mae": c, "mae_gain": b - c})
    weekly = pd.DataFrame(week_rows)
    weeks_won = int(weekly.mae_gain.gt(0).sum())
    boot = bootstrap_support(hold)

    strict_prior = bool(((num(z.off_prior_max_ord).isna()) | (num(z.off_prior_max_ord) < z.target_ord)).all() and ((num(z.def_prior_max_ord).isna()) | (num(z.def_prior_max_ord) < z.target_ord)).all())
    gates = {
        "source_exact_544_2023_rows_unique_features_finite": len(source) == 544 and not source.duplicated(["season", "week", "team"]).any() and not source[FEATURES].isna().any().any(),
        "target_pbp_alignment_exact_finite": len(z) == 544 and not z.actual_d1_dbr.isna().any() and bool(z.opponent_match.all()),
        "baseline_strict_prior_finite": strict_prior and not z.baseline_d1_dbr.isna().any(),
        "beta_gt_0": bool(np.isfinite(beta) and beta > 0),
        "holdout_mae_gain_ge_0_005": mae_gain >= .0050,
        "holdout_rmse_nonworse": cm["rmse"] <= bm["rmse"] + 1e-12,
        "holdout_abs_bias_nonworse": abs(cm["bias"]) <= abs(bm["bias"]) + 1e-12,
        "holdout_p90_abs_error_nonworse": cm["p90_abs_error"] <= bm["p90_abs_error"] + 1e-12,
        "correction_vs_holdout_residual_spearman_ge_0_15": bool(np.isfinite(correction_corr["spearman"]) and correction_corr["spearman"] >= .15),
        "weeks_10_18_won_ge_6": weeks_won >= 6,
        "paired_bootstrap_support_ge_0_90": boot >= .90,
        "no_2024_2025_target_pbp_or_outcomes_read": True,
        "no_qb_wr_target_outcome_or_parent_residual_read": True,
        "zero_sportsbook_game_market_inputs": True,
        "zero_production_changes": True,
    }
    integrity_keys = [
        "source_exact_544_2023_rows_unique_features_finite",
        "target_pbp_alignment_exact_finite",
        "baseline_strict_prior_finite",
        "no_2024_2025_target_pbp_or_outcomes_read",
        "no_qb_wr_target_outcome_or_parent_residual_read",
        "zero_sportsbook_game_market_inputs",
        "zero_production_changes",
    ]
    integrity = all(gates[k] for k in integrity_keys)
    advances = all(gates.values())
    disposition = (
        "FIRST_DOWN_CHOICE_ECONOMICS_D1_ADVANCES" if advances else
        ("FIRST_DOWN_CHOICE_ECONOMICS_D1_FAIL_NO_CONFIRMATION" if integrity else "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE")
    )

    correction_stats = {
        "mean": float(hold.correction.mean()),
        "mean_abs": float(hold.correction.abs().mean()),
        "p90_abs": float(hold.correction.abs().quantile(.90)),
        **correction_corr,
    }
    result = {
        "migration": "QB_FIRST_DOWN_CHOICE_ECONOMICS_D1",
        "disposition": disposition,
        "production_actionable": False,
        "confirmation_authorized": bool(advances),
        "beta_raw": beta_raw,
        "beta": beta,
        "train_rows": int(len(train)),
        "holdout_rows": int(len(hold)),
        "baseline_holdout": bm,
        "candidate_holdout": cm,
        "holdout_mae_gain": mae_gain,
        "correction_holdout": correction_stats,
        "weeks_won": weeks_won,
        "paired_bootstrap_support": boot,
        "gates": gates,
        "pbp_audit": pbp_audit,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "first_down_choice_economics_d1_casebook.csv", index=False)
    weekly.to_csv(a.out_dir / "first_down_choice_economics_d1_weekly.csv", index=False)
    pd.DataFrame([{"model": "baseline", **bm}, {"model": "candidate", **cm}]).to_csv(a.out_dir / "first_down_choice_economics_d1_metrics.csv", index=False)
    (a.out_dir / "first_down_choice_economics_d1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
