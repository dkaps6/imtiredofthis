#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts._opponent_map import canon_team
import scripts.backtest.validate_qb_possession_dropback_generative_fixed as m64fix

m64 = m64fix.m

STATE_SHARE_COLS = ["neutral_share", "trailing_share", "leading_share"]
STATE_RATE_COLS = ["neutral_dropback_rate", "trailing_dropback_rate", "leading_dropback_rate"]
HIST_COLS = [
    "dropback_rate",
    "neutral_share",
    "trailing_share",
    "leading_share",
    "neutral_dropback_rate",
    "trailing_dropback_rate",
    "leading_dropback_rate",
    "drives",
    "plays_per_drive",
    "no_huddle_rate",
    "seconds_between_plays",
    "scoring_drive_rate",
]

FEATURES = [
    "team_dropback_rate",
    "oppdef_dropback_rate",
    "team_neutral_share",
    "team_trailing_share",
    "team_leading_share",
    "oppdef_neutral_share",
    "oppdef_trailing_share",
    "oppdef_leading_share",
    "team_neutral_dropback_rate",
    "team_trailing_dropback_rate",
    "team_leading_dropback_rate",
    "oppdef_neutral_dropback_rate",
    "oppdef_trailing_dropback_rate",
    "oppdef_leading_dropback_rate",
    "team_scoring_drive_rate",
    "opp_scoring_drive_rate",
    "teamdef_scoring_drive_rate",
    "oppdef_scoring_drive_rate",
    "team_drives",
    "opp_drives",
    "teamdef_drives",
    "oppdef_drives",
    "team_plays_per_drive",
    "opp_plays_per_drive",
    "team_no_huddle_rate",
    "opp_no_huddle_rate",
    "team_seconds_between_plays",
    "opp_seconds_between_plays",
]

RIDGE_ALPHA = 20.0
PRIOR_GAMES = 4.0
HISTORY_GAMES = 8


def num(v):
    return pd.to_numeric(v, errors="coerce")


def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def summarize_state_team_games(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["season", "week", "game_id", "posteam", "defteam"]
    for key, g in p.groupby(keys, dropna=False, sort=False):
        season, week, game_id, team, opp = key
        scr = g[g["_scrimmage"].eq(1)].copy()
        if scr.empty:
            continue
        known = scr[scr["_neutral"] | scr["_trailing"] | scr["_leading"]].copy()
        known_n = len(known)
        if not known_n:
            continue
        drives = int(scr["_drive"].nunique())
        plays = int(scr["_scrimmage"].sum())
        db = int(scr["_dropback"].sum())
        pa = int(scr["_pass_attempt"].sum())

        def state_share(mask: pd.Series) -> float:
            return float(mask.sum() / known_n) if known_n else np.nan

        def state_rate(mask: pd.Series) -> float:
            q = scr[mask]
            return float(q["_dropback"].mean()) if len(q) else np.nan

        drive_score = g.groupby("_drive", dropna=False)["_score_play"].max() if drives else pd.Series(dtype=float)
        rows.append({
            "season": int(season),
            "week": int(week),
            "game_id": str(game_id),
            "team": canon_team(team),
            "opponent": canon_team(opp),
            "drives": float(drives),
            "scrimmage_plays": float(plays),
            "plays_per_drive": float(plays / drives) if drives else np.nan,
            "dropbacks": float(db),
            "pass_attempts": float(pa),
            "dropback_rate": float(db / plays) if plays else np.nan,
            "attempt_conversion": float(pa / db) if db else np.nan,
            "neutral_share": state_share(known["_neutral"]),
            "trailing_share": state_share(known["_trailing"]),
            "leading_share": state_share(known["_leading"]),
            "neutral_dropback_rate": state_rate(scr["_neutral"]),
            "trailing_dropback_rate": state_rate(scr["_trailing"]),
            "leading_dropback_rate": state_rate(scr["_leading"]),
            "no_huddle_rate": float(scr["_no_huddle"].mean()),
            "seconds_between_plays": float(num(scr["_seconds_between"]).mean()),
            "scoring_drive_rate": float(num(drive_score).mean()) if len(drive_score) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("M65 state team-game summarization produced zero rows")
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("M65 state team-game summary has duplicate season/week/team rows")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def add_defensive_views(off: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "week", "game_id", "team", "opponent"] + HIST_COLS
    d = off[cols].copy()
    d = d.rename(columns={"team": "offense", "opponent": "team"})
    d = d.rename(columns={c: f"allowed_{c}" for c in HIST_COLS})
    return d


def before(frame: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    s = num(frame["season"])
    w = num(frame["week"])
    z = frame[
        frame["team"].astype(str).eq(canon_team(team))
        & ((s < int(season)) | ((s == int(season)) & (w < int(week))))
    ].copy()
    return z.sort_values(["season", "week"]).tail(HISTORY_GAMES)


def league_before(frame: pd.DataFrame, season: int, week: int, col: str) -> float:
    s = num(frame["season"])
    w = num(frame["week"])
    z = num(frame.loc[(s < int(season)) | ((s == int(season)) & (w < int(week))), col]).dropna()
    return float(z.mean()) if len(z) else np.nan


def shrunk_recent(frame: pd.DataFrame, season: int, week: int, team: str, col: str) -> float:
    hist = num(before(frame, season, week, team)[col]).dropna()
    lg = league_before(frame, season, week, col)
    if not np.isfinite(lg):
        return float(hist.mean()) if len(hist) else np.nan
    if not len(hist):
        return lg
    return float((hist.sum() + PRIOR_GAMES * lg) / (len(hist) + PRIOR_GAMES))


def pregame_feature_row(off: pd.DataFrame, defense: pd.DataFrame, season: int, week: int, team: str, opp: str) -> dict:
    team = canon_team(team)
    opp = canon_team(opp)
    row = {"season": int(season), "week": int(week), "team": team, "opponent": opp}
    for c in HIST_COLS:
        row[f"team_{c}"] = shrunk_recent(off, season, week, team, c)
        row[f"opp_{c}"] = shrunk_recent(off, season, week, opp, c)
        row[f"teamdef_{c}"] = shrunk_recent(defense, season, week, team, f"allowed_{c}")
        row[f"oppdef_{c}"] = shrunk_recent(defense, season, week, opp, f"allowed_{c}")
    return row


def build_training_feature_table(off: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = off[num(off["season"]).between(2023, 2025)].copy()
    for _, r in targets.iterrows():
        season, week = int(r["season"]), int(r["week"])
        row = pregame_feature_row(off, defense, season, week, r["team"], r["opponent"])
        for c in STATE_SHARE_COLS + STATE_RATE_COLS + ["dropback_rate"]:
            row[f"actual_{c}"] = num(pd.Series([r[c]])).iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_shares(neutral: float, trailing: float, leading: float, *, floor: float = 0.02) -> tuple[float, float, float]:
    x = np.asarray([neutral, trailing, leading], dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.clip(x, floor, 1.0)
    s = x.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.60, 0.20, 0.20
    x = x / s
    return float(x[0]), float(x[1]), float(x[2])


def recent_state_formula(feature_row: pd.Series) -> tuple[float, float, float]:
    vals = []
    for state in ("neutral", "trailing", "leading"):
        a = num(pd.Series([feature_row.get(f"team_{state}_share")])).iloc[0]
        b = num(pd.Series([feature_row.get(f"oppdef_{state}_share")])).iloc[0]
        v = np.nanmean([a, b]) if np.isfinite([a, b]).any() else np.nan
        vals.append(v)
    return normalize_shares(*vals)


def predicted_state_rates(feature_row: pd.Series) -> tuple[float, float, float]:
    overall = np.nanmean([
        num(pd.Series([feature_row.get("team_dropback_rate")])).iloc[0],
        num(pd.Series([feature_row.get("oppdef_dropback_rate")])).iloc[0],
    ])
    if not np.isfinite(overall):
        overall = 0.59
    vals = []
    for state in ("neutral", "trailing", "leading"):
        a = num(pd.Series([feature_row.get(f"team_{state}_dropback_rate")])).iloc[0]
        b = num(pd.Series([feature_row.get(f"oppdef_{state}_dropback_rate")])).iloc[0]
        finite = [v for v in (a, b) if np.isfinite(v)]
        vals.append(float(np.mean(finite)) if finite else float(overall))
    return tuple(float(np.clip(v, 0.30, 0.85)) for v in vals)


def fit_state_models(train: pd.DataFrame) -> dict[str, object]:
    models = {}
    X = train[FEATURES]
    for state in ("neutral", "trailing", "leading"):
        y = num(train[f"actual_{state}_share"])
        ok = y.notna()
        if ok.sum() < 100:
            raise RuntimeError(f"M65 insufficient state training rows for {state}: {int(ok.sum())}")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        )
        model.fit(X.loc[ok], y.loc[ok])
        models[state] = model
    return models


def metric_triplet(actual, pred) -> dict:
    z = pd.DataFrame({"a": num(actual), "p": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = z.p - z.a
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z.a.corr(z.p)) if len(z) >= 2 else np.nan,
    }


def build_predictions(m64_games: pd.DataFrame, feature_table: pd.DataFrame, off: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    out = []
    target = m64_games.copy()
    for (season, week), block in target.groupby(["season", "week"], sort=True):
        season, week = int(season), int(week)
        s = num(feature_table["season"])
        w = num(feature_table["week"])
        train = feature_table[(s < season) | ((s == season) & (w < week))].copy()
        train = train[num(train["season"]).ge(2023)].copy()
        models = fit_state_models(train)

        for _, r in block.iterrows():
            fq = feature_table[
                num(feature_table["season"]).eq(season)
                & num(feature_table["week"]).eq(week)
                & feature_table["team"].astype(str).eq(canon_team(r["team"]))
            ]
            if fq.empty:
                raise RuntimeError(f"M65 missing pregame feature row for {season} W{week} {r['team']}")
            f = fq.iloc[0]
            X = pd.DataFrame([{c: f.get(c, np.nan) for c in FEATURES}])
            pred_shares = normalize_shares(
                float(models["neutral"].predict(X)[0]),
                float(models["trailing"].predict(X)[0]),
                float(models["leading"].predict(X)[0]),
            )
            formula_shares = recent_state_formula(f)
            state_rates = predicted_state_rates(f)

            pred_dbr = float(np.dot(np.asarray(pred_shares), np.asarray(state_rates)))
            formula_dbr = float(np.dot(np.asarray(formula_shares), np.asarray(state_rates)))

            actual_shares = normalize_shares(
                float(r["m65_actual_neutral_share"]),
                float(r["m65_actual_trailing_share"]),
                float(r["m65_actual_leading_share"]),
                floor=0.0,
            )
            actual_rates = []
            for c in ("m65_actual_neutral_dropback_rate", "m65_actual_trailing_dropback_rate", "m65_actual_leading_dropback_rate"):
                v = num(pd.Series([r[c]])).iloc[0]
                actual_rates.append(float(v) if np.isfinite(v) else np.nan)
            actual_rates_fill = [
                actual_rates[i] if np.isfinite(actual_rates[i]) else state_rates[i] for i in range(3)
            ]

            oracle_occ_dbr = float(np.dot(np.asarray(actual_shares), np.asarray(state_rates)))
            oracle_rates_dbr = float(np.dot(np.asarray(pred_shares), np.asarray(actual_rates_fill)))
            oracle_all_dbr = float(np.dot(np.asarray(actual_shares), np.asarray(actual_rates_fill)))

            base_components = (
                float(r["m64_pred_drives"]),
                float(r["m64_pred_plays_per_drive"]),
                float(r["m64_pred_attempt_conversion"]),
                float(r["m64_pred_qb_attempt_share"]),
            )
            factor = float(np.prod(base_components))
            pred_att = factor * pred_dbr
            formula_att = factor * formula_dbr
            pred_pass = pred_att * float(r["ypa_contextual"])
            formula_pass = formula_att * float(r["ypa_contextual"])

            z = r.to_dict()
            for state, v in zip(("neutral", "trailing", "leading"), pred_shares):
                z[f"m65_pred_{state}_share"] = v
            for state, v in zip(("neutral", "trailing", "leading"), formula_shares):
                z[f"m65_formula_{state}_share"] = v
            for state, v in zip(("neutral", "trailing", "leading"), state_rates):
                z[f"m65_pred_{state}_dropback_rate"] = v
            z["m65_pred_dropback_rate"] = pred_dbr
            z["m65_formula_dropback_rate"] = formula_dbr
            z["m65_oracle_actual_occupancy_dropback_rate"] = oracle_occ_dbr
            z["m65_oracle_actual_state_rates_dropback_rate"] = oracle_rates_dbr
            z["m65_oracle_all_state_dropback_rate"] = oracle_all_dbr
            z["m65_attempts_state_ridge"] = pred_att
            z["m65_attempts_state_formula"] = formula_att
            z["m65_pass_state_ridge"] = pred_pass
            z["m65_pass_state_formula"] = formula_pass
            out.append(z)
    return pd.DataFrame(out)


def attach_actual_state_values(m64_games: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    actual = off[[
        "season", "week", "team",
        "neutral_share", "trailing_share", "leading_share",
        "neutral_dropback_rate", "trailing_dropback_rate", "leading_dropback_rate",
    ]].copy()
    actual = actual.rename(columns={c: f"m65_actual_{c}" for c in actual.columns if c not in {"season", "week", "team"}})
    x = m64_games.merge(actual, on=["season", "week", "team"], how="left", validate="many_to_one")
    if x["m65_actual_neutral_share"].isna().any():
        raise RuntimeError("M65 could not attach actual state shares to all target rows")
    return x


def dbr_metrics(g: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "m64_neutral": "m64_pred_dropback_rate_neutral",
        "m64_gamescript": "m64_pred_dropback_rate_gamescript",
        "m65_state_formula": "m65_formula_dropback_rate",
        "m65_state_ridge": "m65_pred_dropback_rate",
    }
    rows = []
    for season_label, q in [("2024", g[g.season.eq(2024)]), ("2025", g[g.season.eq(2025)]), ("combined", g)]:
        for label, col in candidates.items():
            rows.append({"season": season_label, "candidate": label, **metric_triplet(q["m64_actual_dropback_rate"], q[col])})
    return pd.DataFrame(rows)


def attempt_metrics(g: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "raw": "attempts_raw",
        "m64_neutral": "m64_attempts_generative_neutral",
        "m64_gamescript": "m64_attempts_generative_gamescript",
        "m65_state_formula": "m65_attempts_state_formula",
        "m65_state_ridge": "m65_attempts_state_ridge",
    }
    rows = []
    for season_label, q in [("2024", g[g.season.eq(2024)]), ("2025", g[g.season.eq(2025)]), ("combined", g)]:
        for label, col in candidates.items():
            base = metric_triplet(q["actual_pass_att"], q[col])
            err = (num(q[col]) - num(q["actual_pass_att"])).abs()
            hi = num(q["actual_pass_att"]).ge(40)
            rows.append({
                "season": season_label,
                "candidate": label,
                **base,
                "miss_8plus": int(err.ge(8).sum()),
                "miss_10plus": int(err.ge(10).sum()),
                "actual_40plus_n": int(hi.sum()),
                "actual_40plus_mae": float(err[hi].mean()) if hi.any() else np.nan,
            })
    return pd.DataFrame(rows)


def passing_metrics(g: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "raw": "m64_pass_raw_reference",
        "m64_neutral": "m64_pass_generative_neutral",
        "m64_gamescript": "m64_pass_generative_gamescript",
        "m65_state_formula": "m65_pass_state_formula",
        "m65_state_ridge": "m65_pass_state_ridge",
    }
    rows = []
    for season_label, q in [("2024", g[g.season.eq(2024)]), ("2025", g[g.season.eq(2025)]), ("combined", g)]:
        for label, col in candidates.items():
            base = metric_triplet(q["actual"], q[col])
            err = num(q[col]) - num(q["actual"])
            rows.append({
                "season": season_label,
                "candidate": label,
                **base,
                "miss_75plus": int(err.abs().ge(75).sum()),
                "miss_100plus": int(err.abs().ge(100).sum()),
                "under_100plus": int(err.le(-100).sum()),
                "over_100plus": int(err.ge(100).sum()),
            })
    return pd.DataFrame(rows)


def state_oracles(g: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "m65_state_ridge": "m65_pred_dropback_rate",
        "oracle_actual_occupancy": "m65_oracle_actual_occupancy_dropback_rate",
        "oracle_actual_state_rates": "m65_oracle_actual_state_rates_dropback_rate",
        "oracle_all_state": "m65_oracle_all_state_dropback_rate",
    }
    return pd.DataFrame([
        {"component": label, **metric_triplet(g["m64_actual_dropback_rate"], g[col])}
        for label, col in candidates.items()
    ])


def frozen_verdict(dbr: pd.DataFrame, att: pd.DataFrame, pas: pd.DataFrame) -> pd.DataFrame:
    def row(table, season, candidate):
        return table[(table["season"].eq(season)) & table["candidate"].eq(candidate)].iloc[0]

    d = row(dbr, "combined", "m65_state_ridge")
    db = row(dbr, "combined", "m64_neutral")
    a = row(att, "combined", "m65_state_ridge")
    ar = row(att, "combined", "raw")
    p = row(pas, "combined", "m65_state_ridge")
    pr = row(pas, "combined", "raw")

    dbr_nonworse = True
    pass_nonworse = True
    for season in ("2024", "2025"):
        dy = row(dbr, season, "m65_state_ridge")
        dby = row(dbr, season, "m64_neutral")
        py = row(pas, season, "m65_state_ridge")
        pry = row(pas, season, "raw")
        dbr_nonworse &= bool(float(dy["mae"]) <= float(dby["mae"]) + 1e-12)
        pass_nonworse &= bool(float(py["mae"]) <= float(pry["mae"]) + 1e-12)

    mechanism_gates = {
        "dbr_mae_gain_ge_0_010": float(db["mae"] - d["mae"]) >= 0.010,
        "dbr_rmse_gain_ge_0_010": float(db["rmse"] - d["rmse"]) >= 0.010,
        "dbr_corr_gain_ge_0_10": float(d["corr"] - db["corr"]) >= 0.10,
        "dbr_mae_nonworse_both_years": dbr_nonworse,
    }
    integration_gates = {
        "attempt_mae_gain_ge_0_25": float(ar["mae"] - a["mae"]) >= 0.25,
        "attempt_corr_gain_ge_0_05": float(a["corr"] - ar["corr"]) >= 0.05,
        "attempt_10plus_misses_reduce_5pct": int(a["miss_10plus"]) <= int(np.floor(ar["miss_10plus"] * 0.95)),
        "actual_40plus_attempt_mae_gain_ge_0_50": float(ar["actual_40plus_mae"] - a["actual_40plus_mae"]) >= 0.50,
        "pass_mae_gain_ge_0_50": float(pr["mae"] - p["mae"]) >= 0.50,
        "pass_rmse_gain_ge_1_00": float(pr["rmse"] - p["rmse"]) >= 1.00,
        "pass_corr_gain_ge_0_05": float(p["corr"] - pr["corr"]) >= 0.05,
        "pass_100plus_misses_reduce_5pct": int(p["miss_100plus"]) <= int(np.floor(pr["miss_100plus"] * 0.95)),
        "pass_mae_nonworse_both_years": pass_nonworse,
    }
    mech = bool(all(mechanism_gates.values()))
    integ = bool(mech and all(integration_gates.values()))
    interpretation = (
        "eligible_for_qb_integration"
        if integ else
        "dropback_mechanism_actionable_downstream_not_ready"
        if mech else
        "hold_dropback_state_occupancy"
    )
    return pd.DataFrame([{
        **mechanism_gates,
        **integration_gates,
        "m65_dropback_mechanism_actionable": mech,
        "m65_qb_integration_actionable": integ,
        "m64_neutral_dbr_mae": float(db["mae"]),
        "m65_dbr_mae": float(d["mae"]),
        "dbr_mae_gain": float(db["mae"] - d["mae"]),
        "m64_neutral_dbr_corr": float(db["corr"]),
        "m65_dbr_corr": float(d["corr"]),
        "dbr_corr_gain": float(d["corr"] - db["corr"]),
        "raw_attempt_mae": float(ar["mae"]),
        "m65_attempt_mae": float(a["mae"]),
        "attempt_mae_gain": float(ar["mae"] - a["mae"]),
        "raw_pass_mae": float(pr["mae"]),
        "m65_pass_mae": float(p["mae"]),
        "pass_mae_gain": float(pr["mae"] - p["mae"]),
        "raw_pass_corr": float(pr["corr"]),
        "m65_pass_corr": float(p["corr"]),
        "pass_corr_gain": float(p["corr"] - pr["corr"]),
        "interpretation": interpretation,
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m64-game-level", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    games = read(args.m64_game_level)
    games["season"] = num(games["season"]).astype(int)
    games["week"] = num(games["week"]).astype(int)
    games["team"] = games["team"].map(canon_team)
    games["opponent"] = games["opponent"].map(canon_team)

    raw = m64.load_pbp([2022, 2023, 2024, 2025])
    p = m64.prepare_pbp(raw)
    off = summarize_state_team_games(p)
    defense = add_defensive_views(off)

    games = attach_actual_state_values(games, off)
    feature_table = build_training_feature_table(off, defense)
    pred = build_predictions(games, feature_table, off, defense)

    dbr = dbr_metrics(pred)
    att = attempt_metrics(pred)
    pas = passing_metrics(pred)
    oracles = state_oracles(pred)
    verdict = frozen_verdict(dbr, att, pas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "m65_game_level.csv", index=False)
    feature_table.to_csv(args.out_dir / "m65_state_training_features.csv", index=False)
    dbr.to_csv(args.out_dir / "m65_dropback_rate_metrics.csv", index=False)
    att.to_csv(args.out_dir / "m65_attempt_metrics.csv", index=False)
    pas.to_csv(args.out_dir / "m65_passing_metrics.csv", index=False)
    oracles.to_csv(args.out_dir / "m65_state_oracles.csv", index=False)
    verdict.to_csv(args.out_dir / "m65_precommitted_interpretation.csv", index=False)

    print("[M65] interpretation")
    print(verdict.to_string(index=False))
    print("[M65] dropback metrics")
    print(dbr.to_string(index=False))
    print("[M65] attempts")
    print(att.to_string(index=False))
    print("[M65] passing")
    print(pas.to_string(index=False))
    print("[M65] state oracles")
    print(oracles.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
