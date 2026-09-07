#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
TEST_SEASONS = [2022, 2023, 2024, 2025]
FEATURES = [
    "b0_te_pool",
    "b0_total_target_pool",
    "b0_te_target_share",
    "b0_te_room_size",
    "b0_top_te_share",
    "b0_te_hhi",
    "team_te_pool_prior1",
    "team_te_pool_prior4",
    "team_te_pool_season_to_date",
    "team_total_targets_prior4",
    "team_total_targets_season_to_date",
    "team_te_share_prior4",
    "team_te_share_season_to_date",
    "team_te_rec_yards_prior4",
    "opp_te_targets_allowed_prior4",
    "opp_te_targets_allowed_season_to_date",
    "opp_total_targets_allowed_prior4",
    "opp_te_target_share_allowed_prior4",
    "opp_te_rec_yards_allowed_prior4",
    "opp_te_receptions_allowed_prior4",
]
CAP = 3.0


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, got {len(hits)}")
    return hits[0]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metric(actual: pd.Series, pred: pd.Series, thresholds: tuple[float, ...]) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    err = z["pred"] - z["actual"]
    ae = err.abs()
    out = {
        "n": int(len(z)),
        "mae": float(ae.mean()) if len(z) else np.nan,
        "rmse": float(np.sqrt(np.mean(err * err))) if len(z) else np.nan,
        "bias": float(err.mean()) if len(z) else np.nan,
        "correlation": float(z["pred"].corr(z["actual"])) if len(z) > 2 else np.nan,
        "median_abs": float(ae.median()) if len(z) else np.nan,
        "p75_abs": float(ae.quantile(.75)) if len(z) else np.nan,
        "p90_abs": float(ae.quantile(.90)) if len(z) else np.nan,
    }
    for t in thresholds:
        out[f"miss{int(t)}"] = float(ae.ge(t).mean()) if len(z) else np.nan
    return out


def rolled_prior(g: pd.DataFrame, value: str, window: int | None, season_only: bool = False) -> pd.Series:
    if season_only:
        return g.groupby(["team", "season"], sort=False)[value].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
    if window == 1:
        return g.groupby("team", sort=False)[value].shift(1)
    return g.groupby("team", sort=False)[value].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )


def make_team_table(src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    x = src.copy()
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x = x.loc[x["season"].isin(SEASONS)].copy()
    for c in ["b0_expected_targets", "targets", "receptions", "rec_yards", "b0_receptions", "b0_rec_yards"]:
        x[c] = num(x[c]).fillna(0.0).clip(lower=0)
    x["ordinal"] = x["season"] * 100 + x["week"]

    team_keys = ["season", "week", "event_id", "team"]
    allteam = x.groupby(team_keys, as_index=False).agg(
        b0_total_target_pool=("b0_expected_targets", "sum"),
        actual_total_targets=("targets", "sum"),
    )
    te = x.loc[x["position_group"].eq("TE")].copy()
    if te.empty:
        raise RuntimeError("no TE rows")

    te_team = te.groupby(team_keys, as_index=False).agg(
        b0_te_pool=("b0_expected_targets", "sum"),
        actual_te_pool=("targets", "sum"),
        actual_te_receptions=("receptions", "sum"),
        actual_te_rec_yards=("rec_yards", "sum"),
        b0_te_receptions=("b0_receptions", "sum"),
        b0_te_rec_yards=("b0_rec_yards", "sum"),
    )
    room = []
    for keys, g in te.groupby(team_keys, sort=False):
        pool = float(g["b0_expected_targets"].sum())
        sh = g["b0_expected_targets"] / pool if pool > 0 else pd.Series(np.zeros(len(g)), index=g.index)
        room.append({
            **dict(zip(team_keys, keys)),
            "b0_te_room_size": int(g["b0_expected_targets"].ge(.25).sum()),
            "b0_top_te_share": float(sh.max()) if len(sh) else 0.0,
            "b0_te_hhi": float(np.square(sh).sum()) if len(sh) else 0.0,
        })
    room = pd.DataFrame(room)
    t = allteam.merge(te_team, on=team_keys, how="inner", validate="one_to_one").merge(room, on=team_keys, how="left", validate="one_to_one")
    t["b0_te_target_share"] = np.where(t["b0_total_target_pool"] > 0, t["b0_te_pool"] / t["b0_total_target_pool"], 0.0)
    t["actual_te_share"] = np.where(t["actual_total_targets"] > 0, t["actual_te_pool"] / t["actual_total_targets"], 0.0)
    t["ordinal"] = t["season"] * 100 + t["week"]

    # Infer opponent from the other team in the same event. Exactly two teams are required.
    pairs = t[["season", "week", "event_id", "team"]].drop_duplicates()
    nteams = pairs.groupby(["season", "week", "event_id"])["team"].nunique()
    valid_events = nteams[nteams.eq(2)].index
    valid = pd.MultiIndex.from_frame(t[["season", "week", "event_id"]]).isin(valid_events)
    t = t.loc[valid].copy()
    opp_map = pairs.merge(pairs, on=["season", "week", "event_id"], suffixes=("", "_opp"))
    opp_map = opp_map.loc[opp_map["team"].ne(opp_map["team_opp"]), ["season", "week", "event_id", "team", "team_opp"]].drop_duplicates()
    t = t.merge(opp_map, on=["season", "week", "event_id", "team"], how="left", validate="one_to_one").rename(columns={"team_opp": "opponent"})
    if t["opponent"].isna().any():
        raise RuntimeError("opponent inference failed")

    t = t.sort_values(["team", "season", "week", "event_id"], kind="stable").reset_index(drop=True)
    t["team_te_pool_prior1"] = rolled_prior(t, "actual_te_pool", 1)
    t["team_te_pool_prior4"] = rolled_prior(t, "actual_te_pool", 4)
    t["team_te_pool_season_to_date"] = rolled_prior(t, "actual_te_pool", None, season_only=True)
    t["team_total_targets_prior4"] = rolled_prior(t, "actual_total_targets", 4)
    t["team_total_targets_season_to_date"] = rolled_prior(t, "actual_total_targets", None, season_only=True)
    t["team_te_share_prior4"] = rolled_prior(t, "actual_te_share", 4)
    t["team_te_share_season_to_date"] = rolled_prior(t, "actual_te_share", None, season_only=True)
    t["team_te_rec_yards_prior4"] = rolled_prior(t, "actual_te_rec_yards", 4)
    t["team_last_prior_ordinal"] = t.groupby("team", sort=False)["ordinal"].shift(1)

    # Defensive history: offensive production in a game is what that opponent allowed.
    d = t[["season", "week", "event_id", "ordinal", "team", "opponent", "actual_te_pool", "actual_total_targets", "actual_te_share", "actual_te_rec_yards", "actual_te_receptions"]].copy()
    d = d.rename(columns={
        "team": "offense",
        "opponent": "team",
        "actual_te_pool": "te_targets_allowed",
        "actual_total_targets": "total_targets_allowed",
        "actual_te_share": "te_target_share_allowed",
        "actual_te_rec_yards": "te_rec_yards_allowed",
        "actual_te_receptions": "te_receptions_allowed",
    })
    d = d.sort_values(["team", "season", "week", "event_id"], kind="stable").reset_index(drop=True)
    d["opp_te_targets_allowed_prior4"] = rolled_prior(d, "te_targets_allowed", 4)
    d["opp_te_targets_allowed_season_to_date"] = rolled_prior(d, "te_targets_allowed", None, season_only=True)
    d["opp_total_targets_allowed_prior4"] = rolled_prior(d, "total_targets_allowed", 4)
    d["opp_te_target_share_allowed_prior4"] = rolled_prior(d, "te_target_share_allowed", 4)
    d["opp_te_rec_yards_allowed_prior4"] = rolled_prior(d, "te_rec_yards_allowed", 4)
    d["opp_te_receptions_allowed_prior4"] = rolled_prior(d, "te_receptions_allowed", 4)
    d["opp_last_prior_ordinal"] = d.groupby("team", sort=False)["ordinal"].shift(1)
    dkeep = [
        "season", "week", "event_id", "team",
        "opp_te_targets_allowed_prior4", "opp_te_targets_allowed_season_to_date",
        "opp_total_targets_allowed_prior4", "opp_te_target_share_allowed_prior4",
        "opp_te_rec_yards_allowed_prior4", "opp_te_receptions_allowed_prior4",
        "opp_last_prior_ordinal",
    ]
    t = t.merge(d[dkeep], on=["season", "week", "event_id", "team"], how="left", validate="one_to_one")

    # Strict-prior audit: source ordinals, when present, must be earlier than target ordinal.
    bad_team = t["team_last_prior_ordinal"].notna() & t["team_last_prior_ordinal"].ge(t["ordinal"])
    bad_opp = t["opp_last_prior_ordinal"].notna() & t["opp_last_prior_ordinal"].ge(t["ordinal"])
    same_future = int(bad_team.sum() + bad_opp.sum())

    # B0 player-share parity reconstruction.
    te = te.merge(t[team_keys + ["b0_te_pool", "actual_te_pool", "opponent"]], on=team_keys, how="inner", validate="many_to_one")
    te["b0_te_room_share"] = np.where(te["b0_te_pool"] > 0, te["b0_expected_targets"] / te["b0_te_pool"], 0.0)
    te["b0_targets_recon"] = te["b0_te_pool"] * te["b0_te_room_share"]
    parity_gap = float((te["b0_targets_recon"] - te["b0_expected_targets"]).abs().max()) if len(te) else np.inf
    return t, te, same_future, parity_gap


def fit_predict(team: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    preds = []
    folds = []
    for test_year in TEST_SEASONS:
        train = team.loc[team["season"].lt(test_year)].copy()
        test = team.loc[team["season"].eq(test_year)].copy()
        if train.empty or test.empty:
            continue
        y_train = train["actual_te_pool"] - train["b0_te_pool"]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=20.0, fit_intercept=True)),
        ])
        pipe.fit(train[FEATURES], y_train)
        raw = pipe.predict(test[FEATURES])
        correction = np.clip(raw, -CAP, CAP)
        q = test.copy()
        q["raw_residual_prediction"] = raw
        q["te_pool_correction"] = correction
        q["candidate_te_pool"] = np.maximum(0.0, q["b0_te_pool"] + correction)
        q["test_season"] = test_year
        preds.append(q)
        folds.append({
            "test_season": int(test_year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_seasons": sorted(int(v) for v in train["season"].unique()),
        })
    if not preds:
        raise RuntimeError("no OOS predictions")
    return pd.concat(preds, ignore_index=True), folds


def translate_players(te: pd.DataFrame, team_pred: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "event_id", "team"]
    p = te.merge(team_pred[keys + ["candidate_te_pool", "te_pool_correction", "raw_residual_prediction"]], on=keys, how="inner", validate="many_to_one")
    p["candidate_targets"] = p["candidate_te_pool"] * p["b0_te_room_share"]
    p["b0_rec_per_target"] = np.where(p["b0_expected_targets"] > 0, p["b0_receptions"] / p["b0_expected_targets"], 0.0)
    p["b0_rec_yards_per_target"] = np.where(p["b0_expected_targets"] > 0, p["b0_rec_yards"] / p["b0_expected_targets"], 0.0)
    p["candidate_receptions"] = p["candidate_targets"] * p["b0_rec_per_target"]
    p["candidate_rec_yards"] = p["candidate_targets"] * p["b0_rec_yards_per_target"]
    return p


def comparison_rows(label: str, actual: pd.Series, b0: pd.Series, cand: pd.Series, thresholds: tuple[float, ...], season: str | int = "POOLED") -> list[dict]:
    mb = metric(actual, b0, thresholds)
    mc = metric(actual, cand, thresholds)
    rows = []
    for name, m in [("B0", mb), ("CANDIDATE", mc)]:
        rows.append({"scope": label, "season": season, "model": name, **m})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    src = pd.read_csv(one(args.joint_root, "joint_v1_paired_player_casebook.csv"), low_memory=False)
    src.columns = [str(c).strip().lower() for c in src.columns]
    req = {
        "season", "week", "event_id", "team", "player", "player_clean_key", "position_group",
        "b0_expected_targets", "b0_receptions", "b0_rec_yards", "targets", "receptions", "rec_yards"
    }
    miss = req - set(src.columns)
    if miss:
        raise RuntimeError(f"missing source columns: {sorted(miss)}")

    team, te, same_future, parity_gap = make_team_table(src)
    oos_team, folds = fit_predict(team)
    players = translate_players(te, oos_team)

    score_rows = []
    score_rows += comparison_rows("TEAM_TE_POOL", oos_team["actual_te_pool"], oos_team["b0_te_pool"], oos_team["candidate_te_pool"], (3, 5))
    score_rows += comparison_rows("PLAYER_TARGETS", players["targets"], players["b0_expected_targets"], players["candidate_targets"], (2, 4, 6))
    score_rows += comparison_rows("PLAYER_REC_YARDS", players["rec_yards"], players["b0_rec_yards"], players["candidate_rec_yards"], (20, 30, 40))
    for y in TEST_SEASONS:
        tg = oos_team.loc[oos_team["season"].eq(y)]
        pg = players.loc[players["season"].eq(y)]
        score_rows += comparison_rows("TEAM_TE_POOL", tg["actual_te_pool"], tg["b0_te_pool"], tg["candidate_te_pool"], (3, 5), y)
        score_rows += comparison_rows("PLAYER_TARGETS", pg["targets"], pg["b0_expected_targets"], pg["candidate_targets"], (2, 4, 6), y)
        score_rows += comparison_rows("PLAYER_REC_YARDS", pg["rec_yards"], pg["b0_rec_yards"], pg["candidate_rec_yards"], (20, 30, 40), y)
    scores = pd.DataFrame(score_rows)

    # Frozen B0 opportunity quartiles on OOS player rows.
    players["b0_opportunity_tier"] = pd.qcut(players["b0_expected_targets"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    tier_rows = []
    for tier, g in players.groupby("b0_opportunity_tier", observed=True):
        for scope, actual, b0, cand, th in [
            ("TARGETS", g["targets"], g["b0_expected_targets"], g["candidate_targets"], (2, 4, 6)),
            ("REC_YARDS", g["rec_yards"], g["b0_rec_yards"], g["candidate_rec_yards"], (20, 30, 40)),
        ]:
            mb = metric(actual, b0, th); mc = metric(actual, cand, th)
            tier_rows.append({"tier": str(tier), "scope": scope, "model": "B0", **mb})
            tier_rows.append({"tier": str(tier), "scope": scope, "model": "CANDIDATE", **mc})
    tiers = pd.DataFrame(tier_rows)

    c = oos_team["te_pool_correction"]
    correction = {
        "mean": float(c.mean()), "sd": float(c.std(ddof=0)),
        "p10": float(c.quantile(.10)), "p50": float(c.quantile(.50)), "p90": float(c.quantile(.90)),
        "fraction_positive": float(c.gt(0).mean()), "fraction_negative": float(c.lt(0).mean()),
        "cap_hit_rate": float(c.abs().ge(CAP - 1e-12).mean()), "max_abs": float(c.abs().max()),
    }

    def pick(scope: str, model: str, season="POOLED") -> dict:
        q = scores.loc[scores["scope"].eq(scope) & scores["model"].eq(model) & scores["season"].astype(str).eq(str(season))]
        if len(q) != 1:
            raise RuntimeError(f"score lookup failed {scope} {model} {season}: {len(q)}")
        return q.iloc[0].to_dict()

    pool_b, pool_c = pick("TEAM_TE_POOL", "B0"), pick("TEAM_TE_POOL", "CANDIDATE")
    targ_b, targ_c = pick("PLAYER_TARGETS", "B0"), pick("PLAYER_TARGETS", "CANDIDATE")
    yard_b, yard_c = pick("PLAYER_REC_YARDS", "B0"), pick("PLAYER_REC_YARDS", "CANDIDATE")
    season_pool_wins = sum(pick("TEAM_TE_POOL", "CANDIDATE", y)["mae"] < pick("TEAM_TE_POOL", "B0", y)["mae"] for y in TEST_SEASONS)
    season_yard_wins = sum(pick("PLAYER_REC_YARDS", "CANDIDATE", y)["mae"] < pick("PLAYER_REC_YARDS", "B0", y)["mae"] for y in TEST_SEASONS)
    g2425 = players.loc[players["season"].isin([2024, 2025])]
    y2425_b = metric(g2425["rec_yards"], g2425["b0_rec_yards"], (20, 30, 40))
    y2425_c = metric(g2425["rec_yards"], g2425["candidate_rec_yards"], (20, 30, 40))

    q4b = tiers.loc[tiers["tier"].eq("Q4") & tiers["scope"].eq("REC_YARDS") & tiers["model"].eq("B0")].iloc[0].to_dict()
    q4c = tiers.loc[tiers["tier"].eq("Q4") & tiers["scope"].eq("REC_YARDS") & tiers["model"].eq("CANDIDATE")].iloc[0].to_dict()

    integrity = {
        "oos_team_games_ge1800": bool(len(oos_team) >= 1800),
        "oos_player_games_ge3500": bool(len(players) >= 3500),
        "all_four_oos_seasons": bool(sorted(int(v) for v in oos_team["season"].unique()) == TEST_SEASONS),
        "b0_player_parity_le1e_9": bool(parity_gap <= 1e-9),
        "zero_same_future_outcomes": bool(same_future == 0),
        "sportsbook_inputs_zero": True,
    }
    scientific = {
        "team_pool_mae_improve_ge0_15": bool(pool_b["mae"] - pool_c["mae"] >= .15),
        "player_target_mae_improve_ge0_05": bool(targ_b["mae"] - targ_c["mae"] >= .05),
        "player_rec_yards_mae_improve_ge0_40": bool(yard_b["mae"] - yard_c["mae"] >= .40),
        "team_pool_mae_wins_ge3of4": bool(season_pool_wins >= 3),
        "rec_yards_mae_wins_ge3of4_and_2425": bool(season_yard_wins >= 3 and y2425_c["mae"] < y2425_b["mae"]),
        "rec_yards_p90_guard": bool(yard_c["p90_abs"] <= yard_b["p90_abs"] + .50),
        "rec_yards_miss30_guard": bool(yard_c["miss30"] <= yard_b["miss30"] + .005),
        "rec_yards_miss40_guard": bool(yard_c["miss40"] <= yard_b["miss40"] + .005),
        "high_q4_guard": bool(
            q4c["mae"] <= q4b["mae"] + .25
            and q4c["miss30"] <= q4b["miss30"] + .005
            and q4c["miss40"] <= q4b["miss40"] + .005
        ),
        "differential_corrections": bool(correction["sd"] >= .50 and correction["fraction_positive"] >= .15 and correction["fraction_negative"] >= .15),
    }
    if not all(integrity.values()):
        disposition = "TE_TARGET_POOL_CONTEXT_INTEGRITY_FAIL"
    elif all(scientific.values()):
        disposition = "TE_TARGET_POOL_CONTEXT_MODEL_ACTIONABLE"
    else:
        disposition = "TE_TARGET_POOL_CONTEXT_MODEL_FAIL"

    result = {
        "migration": "TE_R3_TARGET_POOL_CONTEXT_MODEL",
        "source_joint_run": 34081764151,
        "parent_te_r2_run": 34126026512,
        "oos_team_games": int(len(oos_team)),
        "oos_player_games": int(len(players)),
        "folds": folds,
        "feature_set": FEATURES,
        "model": "StandardScaler+Ridge(alpha=20.0)",
        "correction_cap_targets": CAP,
        "b0_player_parity_max_gap": parity_gap,
        "same_or_future_outcome_uses": same_future,
        "correction_behavior": correction,
        "pooled": {
            "team_pool_b0_mae": pool_b["mae"], "team_pool_candidate_mae": pool_c["mae"],
            "player_target_b0_mae": targ_b["mae"], "player_target_candidate_mae": targ_c["mae"],
            "player_rec_yards_b0_mae": yard_b["mae"], "player_rec_yards_candidate_mae": yard_c["mae"],
            "player_rec_yards_b0_p90": yard_b["p90_abs"], "player_rec_yards_candidate_p90": yard_c["p90_abs"],
            "player_rec_yards_b0_miss30": yard_b["miss30"], "player_rec_yards_candidate_miss30": yard_c["miss30"],
            "player_rec_yards_b0_miss40": yard_b["miss40"], "player_rec_yards_candidate_miss40": yard_c["miss40"],
        },
        "season_team_pool_mae_wins": int(season_pool_wins),
        "season_player_rec_yards_mae_wins": int(season_yard_wins),
        "combined_2024_2025_rec_yards": {"b0_mae": y2425_b["mae"], "candidate_mae": y2425_c["mae"]},
        "high_q4_rec_yards": {
            "b0_mae": q4b["mae"], "candidate_mae": q4c["mae"],
            "b0_miss30": q4b["miss30"], "candidate_miss30": q4c["miss30"],
            "b0_miss40": q4b["miss40"], "candidate_miss40": q4c["miss40"],
        },
        "integrity_gates": integrity,
        "scientific_gates": scientific,
        "sportsbook_inputs_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oos_team.to_csv(args.out_dir / "te_r3_oos_team_casebook.csv", index=False)
    players.to_csv(args.out_dir / "te_r3_oos_player_casebook.csv", index=False)
    scores.to_csv(args.out_dir / "te_r3_scorecard.csv", index=False)
    tiers.to_csv(args.out_dir / "te_r3_tier_scorecard.csv", index=False)
    pd.DataFrame(folds).to_csv(args.out_dir / "te_r3_folds.csv", index=False)
    (args.out_dir / "te_r3_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nSCORECARD")
    print(scores.to_string(index=False))
    print("\nTIERS")
    print(tiers.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
