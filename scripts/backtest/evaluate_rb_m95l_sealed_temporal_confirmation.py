"""M95L: sealed temporal confirmation of the frozen M95K RB tail architecture.

This is a research-only temporal rotation. M95K was selected using 2024 and was
repeatedly inspected on 2025, so 2025 is no longer a pristine final holdout.
M95L therefore reconstructs a 2023 late-season confirmation using only earlier
2023 information for fitted coefficients and only pregame information for every
row.

Frozen choices (no search in this script):
- M94C: mean-margin GBR, final-margin RF, plays RF, state mapper Ridge, alpha=.75
- M95F: raw logit tail scorer; 20+ Platt calibration; 25+ football calibration
- M95H: >=70% entitlement_competition, C=.03
- M95I: share calibration shrink=10; 20+ tail_share70_opportunity C=.30;
  25+ tail_share70 C=.30
- M95K: feed_compact_env, EB shrink=4, C=.03, mass-preserving rerank
- M94C remains the central carry estimate; no tail mean boost
- no sportsbook input; no production change

Confirmation target: 2023 Weeks 13-18. All fitted models use Weeks <=12 only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Import compatibility wrappers before the underlying role modules are used.
import scripts.backtest.evaluate_rb_role_availability_v5  # noqa: F401
import scripts.backtest.evaluate_rb_lead_role_entitlement_v2  # noqa: F401
import scripts.backtest.evaluate_rb_absolute_workload_distribution_v3  # noqa: F401

import scripts.backtest.evaluate_rb_absolute_workload_distribution as e
import scripts.backtest.evaluate_rb_workload_regime_calibration as f
import scripts.backtest.evaluate_rb_role_availability as g
import scripts.backtest.evaluate_rb_lead_role_entitlement as h
import scripts.backtest.evaluate_rb_deep_concentration_tail_integration as i
import scripts.backtest.evaluate_rb_feed_tendency_carry_ceiling as k
import scripts.backtest.evaluate_rb_explicit_game_state as b
import scripts.backtest.evaluate_rb_game_environment as c
import scripts.backtest.evaluate_rb_team_rush_volume as m94

SEASON = 2023
TRAIN_END = 12
CONFIRM_START = 13
CONFIRM_END = 18
TAIL_OOF_START = 5
META_START = 9
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]

# Frozen architecture constants from the already-completed migrations.
M94C_MEAN_FAMILY = "gbr"
M94C_FINAL_FAMILY = "rf"
M94C_PLAY_FAMILY = "rf"
M94C_STATE_FAMILY = "ridge"
M94C_ALPHA = 0.75
M95F_CAL20 = "platt"
M95F_CAL25 = "football"
M95H_SPEC = "entitlement_competition"
M95H_C = 0.03
M95I_SHARE_SHRINK = 10.0
M95I_META20_SPEC = "tail_share70_opportunity"
M95I_META25_SPEC = "tail_share70"
M95I_META_C = 0.30
M95K_SHRINK = 4.0
M95K_SPEC = "feed_compact_env"
M95K_C = 0.03


def num(s):
    return pd.to_numeric(s, errors="coerce")


def lower(df):
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def load_m95b_trace(root: Path) -> pd.DataFrame:
    x = lower(pd.read_csv(find_one(root, "m95b_rb_matchup_trace.csv"), low_memory=False))
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["team"] = x["team"].map(g.canon)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    x = x.loc[x["season"].eq(SEASON) & x["week"].between(1, 18)].copy()
    x["actual_carries"] = num(x["actual_carries"])
    if x.empty or x.duplicated(PLAYER_KEYS).any():
        raise RuntimeError("M95L 2023 M95B trace is empty or has duplicate player keys")
    x["actual_20plus"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus"] = x["actual_carries"].ge(25).astype(int)
    return e.add_priors(x).reset_index(drop=True)


def previous_team_leaders_all(trace: pd.DataFrame) -> pd.DataFrame:
    """M95G helper generalized to the sealed 2023 rotation."""
    z = trace.copy()
    if "actual_carries" not in z.columns:
        z["actual_carries"] = num(z["actual_rush_att"])
    rows = []
    for (season, week, team), q0 in z.groupby(TEAM_KEYS):
        q = q0.loc[num(q0["actual_carries"]).notna()].copy()
        if q.empty:
            continue
        q["actual_carries"] = num(q["actual_carries"])
        q = q.sort_values(["actual_carries", "player_clean_key"], ascending=[False, True])
        rows.append({
            "season": int(season), "week": int(week), "team": g.canon(team),
            "game_top1_key": str(q.iloc[0]["player_clean_key"]),
            "game_top1_carries": float(q.iloc[0]["actual_carries"]),
            "game_top2_key": str(q.iloc[1]["player_clean_key"]) if len(q) > 1 else "",
            "game_top2_carries": float(q.iloc[1]["actual_carries"]) if len(q) > 1 else 0.0,
        })
    game = pd.DataFrame(rows).sort_values(["season", "team", "week"])
    grp = game.groupby(["season", "team"], sort=False)
    game["prior_top1_key"] = grp["game_top1_key"].shift(1)
    game["prior_top1_carries"] = grp["game_top1_carries"].shift(1)
    game["prior_top2_key"] = grp["game_top2_key"].shift(1)
    game["prior_top2_carries"] = grp["game_top2_carries"].shift(1)
    return game[TEAM_KEYS + [
        "prior_top1_key", "prior_top1_carries", "prior_top2_key", "prior_top2_carries",
    ]]


def _m94c_week_prediction(x: pd.DataFrame, pred: pd.DataFrame, oof_margin: pd.DataFrame,
                          features: list[str], week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = x.loc[num(x["week"]).lt(week)].copy()
    test = oof_margin.loc[num(oof_margin["week"]).eq(week)].copy()
    mapper_train = oof_margin.loc[num(oof_margin["week"]).lt(week)].copy()
    if len(train) < 90 or len(mapper_train) < 90 or test.empty:
        raise RuntimeError(f"M95L insufficient M94C temporal training for week {week}")
    test = c._fit_predict_state_mapper(mapper_train, test, M94C_STATE_FAMILY)
    test = c._fit_predict_plays(train, test, features, M94C_PLAY_FAMILY)
    test["structured_team_rush_att"] = c._structured_team_rush(test)
    test["candidate_team_rush_att"] = (
        (1.0 - M94C_ALPHA) * num(test["baseline_team_rush_att"])
        + M94C_ALPHA * num(test["structured_team_rush_att"])
    ).clip(8, 50)
    rb = m94._player_candidate(pred, test, "candidate_team_rush_att")
    rb = rb.loc[num(rb["week"]).eq(week)].copy()
    return test, rb


def reconstruct_m94c_2023(m91_root: Path):
    """Rebuild the frozen M94C family on early 2023; W13-18 remain sealed."""
    state_hist = b.load_game_state_observations([SEASON])
    x, pred = b.build_state_features(m91_root, state_hist, SEASON)
    x = c._add_margin_labels(c._add_strength_edges(x, pred), state_hist)
    features = c._feature_cols(x)
    if not features:
        raise RuntimeError("M95L M94C feature set empty")

    # Expanding margin predictions are training-only mapper inputs. Families are
    # frozen from M94C; this call performs no model-family selection.
    oof_margin = c._margin_state_features(
        c._expanding_margin_oof(x, features, M94C_MEAN_FAMILY, M94C_FINAL_FAMILY)
    )

    team_parts = []
    rb_parts = []
    for week in range(META_START, TRAIN_END + 1):
        tw, rw = _m94c_week_prediction(x, pred, oof_margin, features, week)
        team_parts.append(tw); rb_parts.append(rw)

    train12 = x.loc[num(x["week"]).le(TRAIN_END)].copy()
    hold = x.loc[num(x["week"]).between(CONFIRM_START, CONFIRM_END)].copy()
    hold["pred_mean_margin"] = c._predict_margin(
        train12, hold, features, M94C_MEAN_FAMILY, "mean_score_diff", 9501
    )
    hold["pred_final_margin"] = c._predict_margin(
        train12, hold, features, M94C_FINAL_FAMILY, "final_observed_score_diff", 9502
    )
    hold = c._margin_state_features(hold)
    mapper_train = oof_margin.loc[num(oof_margin["week"]).le(TRAIN_END)].copy()
    hold = c._fit_predict_state_mapper(mapper_train, hold, M94C_STATE_FAMILY)
    hold = c._fit_predict_plays(train12, hold, features, M94C_PLAY_FAMILY)
    hold["structured_team_rush_att"] = c._structured_team_rush(hold)
    hold["candidate_team_rush_att"] = (
        (1.0 - M94C_ALPHA) * num(hold["baseline_team_rush_att"])
        + M94C_ALPHA * num(hold["structured_team_rush_att"])
    ).clip(8, 50)
    rb_hold = m94._player_candidate(pred, hold, "candidate_team_rush_att")
    rb_hold = rb_hold.loc[num(rb_hold["week"]).between(CONFIRM_START, CONFIRM_END)].copy()
    team_parts.append(hold); rb_parts.append(rb_hold)

    team = pd.concat(team_parts, ignore_index=True, sort=False)
    rb = pd.concat(rb_parts, ignore_index=True, sort=False)
    team["team"] = team["team"].map(g.canon)
    rb["team"] = rb["team"].map(g.canon)
    rb["player_clean_key"] = rb["player_clean_key"].astype(str)
    rb = rb.rename(columns={"candidate_rush_att": "m94c_rush_att"})
    return team, rb, state_hist, features


def raw_tail_oof(trace: pd.DataFrame, target: str) -> pd.DataFrame:
    pieces = []
    for week in range(TAIL_OOF_START, TRAIN_END + 1):
        tr = trace.loc[num(trace["week"]).lt(week)].copy()
        te = trace.loc[num(trace["week"]).eq(week)].copy()
        if te.empty or num(tr[target]).nunique() < 2:
            continue
        try:
            p = f.raw_tail_score(tr, te, target)
        except Exception:
            continue
        q = te.copy()
        q["raw_score"] = p
        q["actual_label"] = num(q[target]).astype(int)
        pieces.append(q)
    if not pieces:
        raise RuntimeError(f"M95L could not create raw temporal tail OOF for {target}")
    return pd.concat(pieces, ignore_index=True, sort=False)


def expanding_calibrated_oof(raw: pd.DataFrame, family: str, out_col: str) -> pd.DataFrame:
    pieces = []
    weeks = sorted(num(raw["week"]).dropna().astype(int).unique())
    for week in weeks:
        tr = raw.loc[num(raw["week"]).lt(week)].copy()
        te = raw.loc[num(raw["week"]).eq(week)].copy()
        if len(tr) < 40 or tr["actual_label"].nunique() < 2 or te.empty:
            continue
        try:
            model = f.fit_calibrator(tr, family)
            p = f.apply_calibrator(model, te, family)
        except Exception:
            continue
        q = te[PLAYER_KEYS + ["actual_label"]].copy()
        q[out_col] = p
        pieces.append(q)
    if not pieces:
        raise RuntimeError(f"M95L could not create expanding calibrated OOF {out_col}")
    return pd.concat(pieces, ignore_index=True).drop_duplicates(PLAYER_KEYS)


def build_m95f_rotation(trace: pd.DataFrame):
    raw20 = raw_tail_oof(trace, "actual_20plus")
    raw25 = raw_tail_oof(trace, "actual_25plus")
    oof20 = expanding_calibrated_oof(raw20, M95F_CAL20, "cal_prob_20")
    oof25 = expanding_calibrated_oof(raw25, M95F_CAL25, "cal_prob_25")
    oof = trace.merge(oof20[PLAYER_KEYS + ["cal_prob_20"]], on=PLAYER_KEYS, how="inner")
    oof = oof.merge(oof25[PLAYER_KEYS + ["cal_prob_25"]], on=PLAYER_KEYS, how="inner")
    oof = oof.loc[num(oof["week"]).between(META_START, TRAIN_END)].copy()

    train = trace.loc[num(trace["week"]).le(TRAIN_END)].copy()
    hold = trace.loc[num(trace["week"]).between(CONFIRM_START, CONFIRM_END)].copy()
    for target, family, raw_train, out_col in [
        ("actual_20plus", M95F_CAL20, raw20, "cal_prob_20"),
        ("actual_25plus", M95F_CAL25, raw25, "cal_prob_25"),
    ]:
        raw = f.raw_tail_score(train, hold, target)
        z = hold.copy(); z["raw_score"] = raw; z["actual_label"] = num(z[target]).astype(int)
        cal = f.fit_calibrator(raw_train, family)
        hold[out_col] = f.apply_calibrator(cal, z, family)
    hold["cal_prob_25"] = np.minimum(num(hold["cal_prob_25"]), num(hold["cal_prob_20"]))
    oof["cal_prob_25"] = np.minimum(num(oof["cal_prob_25"]), num(oof["cal_prob_20"]))
    return oof, hold, raw20, raw25


def load_role_sources():
    rosters, injuries, depth, audit = g.load_provider_sources([SEASON])
    rosters = g.add_roster_transition_features(rosters)
    depth = g.add_depth_transition_features(depth)
    return rosters, injuries, depth, audit


def enrich_role_frame(base: pd.DataFrame, trace: pd.DataFrame,
                      rosters: pd.DataFrame, injuries: pd.DataFrame,
                      depth: pd.DataFrame) -> pd.DataFrame:
    truth = h.add_entitlement_truth(base, trace)
    return h.enrich(truth, trace, rosters, injuries, depth)


def share70_oof(ent: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for train_end, start, end in [(8, 9, 10), (10, 11, 12)]:
        tr = ent.loc[num(ent["week"]).between(TAIL_OOF_START, train_end)].copy()
        te = ent.loc[num(ent["week"]).between(start, end)].copy()
        if tr.empty or te.empty or num(tr["actual_share70"]).nunique() < 2:
            continue
        p, _, _ = h.fit_predict(tr, te, "actual_share70", M95H_SPEC, M95H_C)
        q = te[PLAYER_KEYS + ["actual_share70", "prior_top1_unavailable"]].copy()
        q["p_share70_raw"] = p
        pieces.append(q)
    if not pieces:
        raise RuntimeError("M95L could not create share70 temporal OOF")
    return pd.concat(pieces, ignore_index=True).drop_duplicates(PLAYER_KEYS)


def attach_environment(base: pd.DataFrame, team_env: pd.DataFrame, rb_env: pd.DataFrame) -> pd.DataFrame:
    team_cols = TEAM_KEYS + [
        "candidate_team_rush_att", "pred_off_plays", "pred_lead_play_share",
        "pred_neutral_play_share", "pred_trail_play_share", "pred_mean_margin",
        "pred_final_margin", "gs_team_neutral_rush_rate_avg3",
        "gs_team_lead_rush_rate_avg3", "gs_team_trail_rush_rate_avg3",
    ]
    add_team = [x for x in team_cols if x in team_env.columns]
    out = base.merge(team_env[add_team].drop_duplicates(TEAM_KEYS), on=TEAM_KEYS, how="left", validate="many_to_one")
    r = rb_env[PLAYER_KEYS + ["m94c_rush_att"]].drop_duplicates(PLAYER_KEYS)
    out = out.merge(r, on=PLAYER_KEYS, how="left", validate="one_to_one")
    return out


def build_m95i_rotation(oof_base: pd.DataFrame, hold_base: pd.DataFrame,
                        trace: pd.DataFrame, team_env: pd.DataFrame, rb_env: pd.DataFrame,
                        rosters: pd.DataFrame, injuries: pd.DataFrame, depth: pd.DataFrame):
    ent_train_all = enrich_role_frame(
        trace.loc[num(trace["week"]).between(TAIL_OOF_START, TRAIN_END)].copy(),
        trace, rosters, injuries, depth,
    )
    ent_hold = enrich_role_frame(hold_base.copy(), trace, rosters, injuries, depth)

    s_oof = share70_oof(ent_train_all)
    share_cal = i.fit_regime_calibrator(s_oof, M95I_SHARE_SHRINK)
    s_oof["p_share70_cal"] = share_cal.apply(s_oof)

    ent_fit = ent_train_all.loc[num(ent_train_all["week"]).between(TAIL_OOF_START, TRAIN_END)].copy()
    p_hold_raw, _, _ = h.fit_predict(ent_fit, ent_hold, "actual_share70", M95H_SPEC, M95H_C)
    ent_hold["p_share70_raw"] = p_hold_raw
    ent_hold["p_share70_cal"] = share_cal.apply(ent_hold)

    # OOF base probabilities and share predictions are both leakage-safe row-level
    # inputs. Their meta coefficients are fit only on W9-12.
    meta_train = oof_base.merge(
        s_oof[PLAYER_KEYS + ["p_share70_cal", "prior_top1_unavailable"]],
        on=PLAYER_KEYS, how="inner", suffixes=("", "_share"), validate="one_to_one",
    )
    meta_train = attach_environment(meta_train, team_env, rb_env)

    hold = ent_hold.merge(
        hold_base[PLAYER_KEYS + ["cal_prob_20", "cal_prob_25"]].drop_duplicates(PLAYER_KEYS),
        on=PLAYER_KEYS, how="left", validate="one_to_one",
    )
    hold = attach_environment(hold, team_env, rb_env)
    hold["actual_carries"] = num(hold["actual_carries_m95h"])
    hold["actual_20plus"] = hold["actual_carries"].ge(20).astype(int)
    hold["actual_25plus"] = hold["actual_carries"].ge(25).astype(int)

    p20, _, _ = i.fit_meta(
        meta_train, hold, "actual_20plus", "cal_prob_20", M95I_META20_SPEC, M95I_META_C
    )
    p25, _, _ = i.fit_meta(
        meta_train, hold, "actual_25plus", "cal_prob_25", M95I_META25_SPEC, M95I_META_C
    )
    hold["p20_joint"] = p20
    hold["p25_joint"] = np.minimum(p25, p20)
    return meta_train, hold, ent_train_all, s_oof


def prepare_m95k_train(oof_base: pd.DataFrame, ent_train: pd.DataFrame,
                        team_env: pd.DataFrame, rb_env: pd.DataFrame) -> pd.DataFrame:
    cols = PLAYER_KEYS + ["cal_prob_20", "cal_prob_25"]
    x = ent_train.merge(oof_base[cols].drop_duplicates(PLAYER_KEYS), on=PLAYER_KEYS, how="inner", validate="one_to_one")
    x["actual_carries"] = num(x["actual_carries_m95h"])
    x = attach_environment(x, team_env, rb_env)
    # k.prep expects team environment to be merged itself. It is already present;
    # provide a key-only team frame to avoid duplicating the frozen columns.
    return k.prep(x, team_env)


def run_m95k_frozen(train_base: pd.DataFrame, hold_i: pd.DataFrame,
                     trace: pd.DataFrame, team_env: pd.DataFrame):
    train = train_base.loc[num(train_base["week"]).between(META_START, TRAIN_END)].copy()
    hold = k.prep(hold_i.copy(), team_env)
    ff = k.build_feed_features(trace, M95K_SHRINK)
    train = k.add_composites(train.merge(ff, on=PLAYER_KEYS, how="left", validate="one_to_one"))
    hold = k.add_composites(hold.merge(ff, on=PLAYER_KEYS, how="left", validate="one_to_one"))

    stable_train = train.loc[train["stable_workhorse_m95k"].eq(1)].copy()
    stable_hold = hold["stable_workhorse_m95k"].eq(1)
    vacancy = hold["vacancy_m95k"].eq(1)
    fs = k.available(stable_train, k.SPECS[M95K_SPEC])
    y = num(stable_train["actual_carries"]).ge(20).astype(int)
    if y.nunique() < 2 or stable_train.empty:
        raise RuntimeError("M95L stable workhorse training is insufficient")
    model = k.pipe(M95K_C)
    model.fit(stable_train[fs], y)

    hold["p20_m95l"] = num(hold["p20_base"])
    hold["p25_m95l"] = num(hold["p25_base"])
    if stable_hold.any():
        p20s, p25s = k.candidate_probs(model, hold.loc[stable_hold].copy(), fs)
        hold.loc[stable_hold, "p20_m95l"] = p20s
        hold.loc[stable_hold, "p25_m95l"] = p25s
    hold.loc[vacancy, "p20_m95l"] = num(hold.loc[vacancy, "p20_joint"])
    hold.loc[vacancy, "p25_m95l"] = num(hold.loc[vacancy, "p25_joint"])
    hold["p25_m95l"] = np.minimum(num(hold["p25_m95l"]), num(hold["p20_m95l"]))
    hold["m95l_rush_att"] = num(hold["m94c_rush_att"])
    return train, hold, fs


def metric_rows(z: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "all": pd.Series(True, index=z.index),
        "stable_workhorse": z["stable_workhorse_m95k"].eq(1),
        "vacancy": z["vacancy_m95k"].eq(1),
        "other": ~z["stable_workhorse_m95k"].eq(1) & ~z["vacancy_m95k"].eq(1),
    }
    rows = []
    for th in (20, 25):
        truth = num(z["actual_carries"]).ge(th).astype(int)
        for sl, mask in masks.items():
            for label, col in [("m95f_temporal", f"p{th}_base"), ("m95l_frozen", f"p{th}_m95l")]:
                q = z.loc[mask].copy()
                m = k.prob_metrics(truth.loc[mask], q[col])
                rows.append({
                    "scope": "2023_w13_18_sealed_confirmation", "target": f"actual_{th}plus",
                    "slice": sl, "model": label, **m,
                    "positive_events": int(truth.loc[mask].sum()),
                })
    return pd.DataFrame(rows)


def carry_rows(z: pd.DataFrame) -> pd.DataFrame:
    a = num(z["actual_carries"])
    masks = {
        "all_rb": pd.Series(True, index=z.index), "actual_0_5": a.between(0, 5),
        "actual_6_10": a.between(6, 10), "actual_11_14": a.between(11, 14),
        "actual_15_plus": a.ge(15), "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25),
    }
    rows = []
    for sl, mask in masks.items():
        q = z.loc[mask]
        err = num(q["m94c_rush_att"]) - num(q["actual_carries"])
        rows.append({
            "slice": sl, "n": int(len(q)), "m94c_mae": float(err.abs().mean()) if len(q) else np.nan,
            "m95l_mae": float(err.abs().mean()) if len(q) else np.nan,
            "mae_change": 0.0, "m94c_bias": float(err.mean()) if len(q) else np.nan,
            "m95l_bias": float(err.mean()) if len(q) else np.nan,
        })
    return pd.DataFrame(rows)


def get_metric(pm: pd.DataFrame, target: str, sl: str, model: str):
    q = pm.loc[pm["target"].eq(target) & pm["slice"].eq(sl) & pm["model"].eq(model)]
    if q.empty:
        raise RuntimeError(f"M95L metric missing: {target}/{sl}/{model}")
    return q.iloc[0]


def finite_auc_pair(a, b):
    return np.isfinite(float(a)) and np.isfinite(float(b))


def confirmation_disposition(pm: pd.DataFrame, z: pd.DataFrame):
    s20b = get_metric(pm, "actual_20plus", "stable_workhorse", "m95f_temporal")
    s20c = get_metric(pm, "actual_20plus", "stable_workhorse", "m95l_frozen")
    s25b = get_metric(pm, "actual_25plus", "stable_workhorse", "m95f_temporal")
    s25c = get_metric(pm, "actual_25plus", "stable_workhorse", "m95l_frozen")
    a20b = get_metric(pm, "actual_20plus", "all", "m95f_temporal")
    a20c = get_metric(pm, "actual_20plus", "all", "m95l_frozen")
    a25b = get_metric(pm, "actual_25plus", "all", "m95f_temporal")
    a25c = get_metric(pm, "actual_25plus", "all", "m95l_frozen")
    v25b = get_metric(pm, "actual_25plus", "vacancy", "m95f_temporal")
    v25c = get_metric(pm, "actual_25plus", "vacancy", "m95l_frozen")

    stable20 = int(finite_auc_pair(s20b.auc, s20c.auc) and s20c.auc > s20b.auc and s20c.brier <= s20b.brier)
    stable25 = int(finite_auc_pair(s25b.auc, s25c.auc) and s25c.auc > s25b.auc and s25c.brier <= s25b.brier)
    all20 = int((not finite_auc_pair(a20b.auc, a20c.auc) or a20c.auc >= a20b.auc - 0.005) and a20c.brier <= a20b.brier + 0.001)
    all25 = int((not finite_auc_pair(a25b.auc, a25c.auc) or a25c.auc >= a25b.auc - 0.005) and a25c.brier <= a25b.brier + 0.001)

    vpos = int(v25c.positive_events)
    if vpos >= 2 and finite_auc_pair(v25b.auc, v25c.auc):
        vacancy_status = "scored"
        vacancy_ok = int(v25c.auc >= v25b.auc - 0.02 and v25c.brier <= v25b.brier + 0.001)
    else:
        vacancy_status = "inconclusive_small_n"
        vacancy_ok = 1

    stable = z.loc[z["stable_workhorse_m95k"].eq(1)]
    mass20 = abs(float(num(stable["p20_m95l"]).mean() - num(stable["p20_base"]).mean()))
    mass25 = abs(float(num(stable["p25_m95l"]).mean() - num(stable["p25_base"]).mean()))
    mass_ok = int(mass20 < 1e-9 and mass25 < 1e-9)
    scientific = int(stable20 and stable25 and all20 and all25 and vacancy_ok and mass_ok)
    disposition = (
        "M95K_SEALED_TEMPORAL_CONFIRMATION_PASSED"
        if scientific else "M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED"
    )
    return pd.DataFrame([{
        "stable20_pass": stable20, "stable25_pass": stable25,
        "all20_nonregression_pass": all20, "all25_nonregression_pass": all25,
        "vacancy25_not_contradicted": vacancy_ok, "vacancy25_status": vacancy_status,
        "stable_mass_preserved": mass_ok, "scientific_confirmation_pass": scientific,
        "stable20_auc_gain": float(s20c.auc - s20b.auc) if finite_auc_pair(s20b.auc, s20c.auc) else np.nan,
        "stable20_brier_gain": float(s20b.brier - s20c.brier),
        "stable25_auc_gain": float(s25c.auc - s25b.auc) if finite_auc_pair(s25b.auc, s25c.auc) else np.nan,
        "stable25_brier_gain": float(s25b.brier - s25c.brier),
        "all20_auc_gain": float(a20c.auc - a20b.auc) if finite_auc_pair(a20b.auc, a20c.auc) else np.nan,
        "all20_brier_gain": float(a20b.brier - a20c.brier),
        "all25_auc_gain": float(a25c.auc - a25b.auc) if finite_auc_pair(a25b.auc, a25c.auc) else np.nan,
        "all25_brier_gain": float(a25b.brier - a25c.brier),
        "stable20_mass_abs_diff": mass20, "stable25_mass_abs_diff": mass25,
        "m94c_central_reference_preserved": 1, "sportsbook_inputs": 0,
        "production_change": 0, "confirmation_period": "2023_w13_18",
        "disposition": disposition,
    }])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m91-root", type=Path, required=True)
    ap.add_argument("--m95b-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    trace = load_m95b_trace(args.m95b_root)
    team_env, rb_env, state_hist, env_features = reconstruct_m94c_2023(args.m91_root)

    # Exact join coverage is a hard source-contract check. The confirmation is
    # not allowed to silently shrink to rows that happen to join.
    hold_keys = trace.loc[num(trace["week"]).between(CONFIRM_START, CONFIRM_END), PLAYER_KEYS]
    rb_hold_keys = rb_env.loc[num(rb_env["week"]).between(CONFIRM_START, CONFIRM_END), PLAYER_KEYS]
    coverage = hold_keys.merge(rb_hold_keys.drop_duplicates(), on=PLAYER_KEYS, how="left", indicator=True)
    join_rate = float(coverage["_merge"].eq("both").mean())
    if join_rate < 0.97:
        raise RuntimeError(f"M95L M94C player join coverage too low: {join_rate:.3%}")

    oof_base, hold_base, raw20, raw25 = build_m95f_rotation(trace)

    # Generalize the M95G prior-leader helper to the sealed season before any
    # enrichment is constructed.
    g.previous_team_leaders = previous_team_leaders_all
    rosters, injuries, depth, source_audit = load_role_sources()
    meta_train, hold_i, ent_train, share_oof_frame = build_m95i_rotation(
        oof_base, hold_base, trace, team_env, rb_env, rosters, injuries, depth
    )
    train_k = prepare_m95k_train(oof_base, ent_train, team_env, rb_env)
    train_final, result, frozen_features = run_m95k_frozen(train_k, hold_i, trace, team_env)

    pm = metric_rows(result)
    carries = carry_rows(result)
    disp = confirmation_disposition(pm, result)

    source_rows = source_audit.copy()
    source_rows["confirmation_season"] = SEASON
    extra_source = pd.DataFrame([
        {"source": "m95b_2023_pregame_trace", "season": SEASON, "rows": len(trace), "status": "ok", "confirmation_season": SEASON},
        {"source": "m91_2023_walk_forward", "season": SEASON, "rows": int(sum(1 for _ in (args.m91_root / "2023" / "pregame_universe").glob("*.csv"))), "status": "rebuilt_from_2022_history", "confirmation_season": SEASON},
        {"source": "m94c_2023_environment", "season": SEASON, "rows": len(team_env), "status": f"fixed_families;player_join={join_rate:.6f}", "confirmation_season": SEASON},
        {"source": "m95f_temporal_oof_20", "season": SEASON, "rows": len(raw20), "status": M95F_CAL20, "confirmation_season": SEASON},
        {"source": "m95f_temporal_oof_25", "season": SEASON, "rows": len(raw25), "status": M95F_CAL25, "confirmation_season": SEASON},
        {"source": "m95i_share70_oof", "season": SEASON, "rows": len(share_oof_frame), "status": "entitlement_competition_C0.03_shrink10", "confirmation_season": SEASON},
        {"source": "m95k_stable_training", "season": SEASON, "rows": int(train_final["stable_workhorse_m95k"].eq(1).sum()), "status": "feed_compact_env_k4_C0.03", "confirmation_season": SEASON},
    ])
    source = pd.concat([source_rows, extra_source], ignore_index=True, sort=False)

    architecture = pd.DataFrame([{
        "confirmation_period": "2023_w13_18", "training_cutoff": "2023_w12",
        "m94c_mean_family": M94C_MEAN_FAMILY, "m94c_final_family": M94C_FINAL_FAMILY,
        "m94c_play_family": M94C_PLAY_FAMILY, "m94c_state_family": M94C_STATE_FAMILY,
        "m94c_alpha": M94C_ALPHA, "m95f_cal20": M95F_CAL20, "m95f_cal25": M95F_CAL25,
        "m95h_share70_spec": M95H_SPEC, "m95h_C": M95H_C,
        "m95i_share_shrink": M95I_SHARE_SHRINK, "m95i_meta20_spec": M95I_META20_SPEC,
        "m95i_meta25_spec": M95I_META25_SPEC, "m95i_meta_C": M95I_META_C,
        "m95k_shrink": M95K_SHRINK, "m95k_spec": M95K_SPEC, "m95k_C": M95K_C,
        "m95k_features_used": "|".join(frozen_features), "mass_preserving": 1,
        "feature_search": 0, "coefficient_search": 0, "sportsbook_inputs": 0,
    }])

    architecture.to_csv(args.out_dir / "m95l_frozen_architecture.csv", index=False)
    source.to_csv(args.out_dir / "m95l_source_audit.csv", index=False)
    team_env.to_csv(args.out_dir / "m95l_2023_team_environment_trace.csv", index=False)
    pm.to_csv(args.out_dir / "m95l_probability_metrics.csv", index=False)
    carries.to_csv(args.out_dir / "m95l_carry_preservation.csv", index=False)
    disp.to_csv(args.out_dir / "m95l_disposition.csv", index=False)
    result.to_csv(args.out_dir / "m95l_2023_confirmation_trace.csv", index=False)

    print("[m95l] frozen architecture")
    print(architecture.to_string(index=False))
    print("\n[m95l] probability metrics")
    print(pm.to_string(index=False))
    print("\n[m95l] carry preservation")
    print(carries.to_string(index=False))
    print("\n[m95l] disposition")
    print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
