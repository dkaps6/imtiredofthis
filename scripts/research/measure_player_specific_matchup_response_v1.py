"""Player-specific matchup response: does a named player beat a positional rule?

Executes PRISM_PLAYER_SPECIFIC_MATCHUP_PREREGISTRATION_V1, frozen at da7e7c1c
before any scoring. Every threshold, feature, model form, window and decision rule
below is quoted from that document; none may be changed to produce a winner.

H1: a named receiver's week-W outcome, measured against HIS OWN strictly-prior
baseline, depends on his own prior profile interacting with the opponent's prior
defensive profile, beyond a position-level defensive main effect.

H0: the apparent effect is his own baseline + a position-level defensive main effect
+ noise. If H0 holds, generic positional rules are defensible and no player-by-defense
mechanism is warranted.

Data is authoritative and play-level: 2025 load_participation carries 22,055
coverage-labeled plays (ZONE 15,020 / MAN 7,035) plus defenders_in_box, joined to PBP
for defensive team identity and to full-season weekly outcomes.

No sportsbook variable is read at any stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_JSON = Path("data/research/player_matchup/player_specific_matchup_response_v1.json")
OUT_ROWS = Path("data/research/player_matchup/player_specific_matchup_response_v1_rows.csv")
CACHE = Path("/tmp/claude-0/-home-user-imtiredofthis/1701f8f3-afc9-57ea-bbc6-cce4dc6b5fbd/scratchpad/nfl_cache")

SEASON = 2025
# ---- frozen thresholds (preregistration section 3) ----
MIN_PRIOR_TARGETS = 20
MIN_PRIOR_GAMES_WITH_TARGET = 3
MIN_DEF_PRIOR_LABELED_PLAYS = 100
FIRST_SCORABLE_WEEK = 4
LAST_REGULAR_WEEK = 18
# ---- frozen windows (section 6) ----
FIT_WEEKS = range(4, 12)        # fit on 4-11
HOLDOUT_WEEKS = range(12, 19)   # score 12-18
PERSIST_A = range(4, 11)        # disjoint window A: 4-10
PERSIST_B = range(11, 19)       # disjoint window B: 11-18
PARTIAL_SNAP_FRACTION = 0.50    # PARTIAL_GAME: snaps below half the player's own norm
POSITIONS = ("WR", "TE", "RB")


def _load() -> dict:
    import nflreadpy as nfl
    CACHE.mkdir(parents=True, exist_ok=True)

    def cached(name: str, fn):
        p = CACHE / f"{name}_{SEASON}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        df = fn().to_pandas()
        df.to_parquet(p)
        return df

    weekly = cached("weekly", lambda: nfl.load_player_stats(seasons=[SEASON]))
    part = cached("participation", lambda: nfl.load_participation(seasons=[SEASON]))
    pbp = cached("pbp", lambda: nfl.load_pbp(seasons=[SEASON]))
    try:
        snaps = cached("snaps", lambda: nfl.load_snap_counts(seasons=[SEASON]))
    except Exception:
        snaps = None
    return {"weekly": weekly, "participation": part, "pbp": pbp, "snaps": snaps}


def build_defense_profiles(part: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (defense team, week) coverage/box counts, from play-level labels."""
    j = part[["nflverse_game_id", "play_id", "defense_man_zone_type", "defenders_in_box"]].merge(
        pbp[["game_id", "play_id", "defteam", "week"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    j = j[j["week"].between(1, LAST_REGULAR_WEEK)]
    j["is_zone"] = j["defense_man_zone_type"].eq("ZONE_COVERAGE")
    j["is_man"] = j["defense_man_zone_type"].eq("MAN_COVERAGE")
    j["labeled"] = j["is_zone"] | j["is_man"]
    box = pd.to_numeric(j["defenders_in_box"], errors="coerce")
    j["is_light_box"] = box.le(6)
    j["is_heavy_box"] = box.ge(8)
    j["box_known"] = box.notna()
    g = j.groupby(["defteam", "week"], as_index=False).agg(
        labeled_plays=("labeled", "sum"), zone_plays=("is_zone", "sum"), man_plays=("is_man", "sum"),
        box_known=("box_known", "sum"), light_box=("is_light_box", "sum"), heavy_box=("is_heavy_box", "sum"),
    )
    return g.rename(columns={"defteam": "defense"})


def defense_prior(def_week: pd.DataFrame, defense: str, week: int) -> dict | None:
    """Strictly prior: the opponent's defensive plays in weeks 1..week-1 only."""
    h = def_week[(def_week["defense"] == defense) & (def_week["week"] < week)]
    labeled = float(h["labeled_plays"].sum())
    if labeled < MIN_DEF_PRIOR_LABELED_PLAYS:
        return None
    zone = float(h["zone_plays"].sum())
    box_known = float(h["box_known"].sum())
    return {
        "def_prior_labeled_plays": labeled,
        "def_zone_rate": zone / labeled,
        "def_man_rate": 1.0 - zone / labeled,
        "def_light_box_rate": float(h["light_box"].sum()) / box_known if box_known else np.nan,
        "def_heavy_box_rate": float(h["heavy_box"].sum()) / box_known if box_known else np.nan,
    }


def state_label(row: pd.Series, snap_norm: float | None) -> str:
    """AGENTS.md anomaly gate: label before the observation becomes evidence."""
    tg = float(row.get("targets") or 0.0)
    ca = float(row.get("carries") or 0.0)
    snaps = row.get("offense_snaps")
    if snaps is None or (isinstance(snaps, float) and not np.isfinite(snaps)):
        if tg == 0.0 and ca == 0.0:
            return "UNRESOLVED_CONTEXT"
        return "HEALTHY_FULL_SNAPS_UNKNOWN"
    snaps = float(snaps)
    if snaps <= 0.0 and tg == 0.0 and ca == 0.0:
        return "DNP_INACTIVE"
    if snap_norm and snap_norm > 0 and snaps < PARTIAL_SNAP_FRACTION * snap_norm:
        return "PARTIAL_GAME"
    return "HEALTHY_FULL"


def build_panel(data: dict) -> tuple[pd.DataFrame, dict]:
    weekly, snaps = data["weekly"], data["snaps"]
    def_week = build_defense_profiles(data["participation"], data["pbp"])

    w = weekly.copy()
    w = w[w["season"].eq(SEASON) & w["week"].between(1, LAST_REGULAR_WEEK)]
    pos_col = "position" if "position" in w.columns else "position_group"
    w["pos"] = w[pos_col].astype(str).str.upper()
    w = w[w["pos"].isin(POSITIONS)]
    name_col = "player_display_name" if "player_display_name" in w.columns else "player_name"
    w["pid"] = w["player_id"].astype(str) if "player_id" in w.columns else w[name_col].astype(str)
    w["player"] = w[name_col].astype(str)
    for c in ("targets", "receiving_yards", "receiving_air_yards", "receptions", "carries"):
        w[c] = pd.to_numeric(w.get(c), errors="coerce").fillna(0.0)
    w["opponent"] = w["opponent_team"].astype(str)

    if snaps is not None and not snaps.empty:
        s = snaps.copy()
        s["week"] = pd.to_numeric(s["week"], errors="coerce")
        key = "pfr_player_id" if "pfr_player_id" in s.columns else None
        nm = "player" if "player" in s.columns else None
        if nm:
            s["_nm"] = s[nm].astype(str)
            agg = s.groupby(["_nm", "week"], as_index=False)["offense_snaps"].max()
            w = w.merge(agg, left_on=["player", "week"], right_on=["_nm", "week"], how="left").drop(
                columns=["_nm"], errors="ignore")
    if "offense_snaps" not in w.columns:
        w["offense_snaps"] = np.nan

    # own-norm snaps for PARTIAL_GAME, from the player's own season median
    norm = w.groupby("pid")["offense_snaps"].median().to_dict()
    w["state"] = [state_label(r, norm.get(r["pid"])) for _, r in w.iterrows()]

    rows = []
    # prior windows exclude DNP_INACTIVE and UNRESOLVED_CONTEXT per section 4
    usable_prior = w[~w["state"].isin(("DNP_INACTIVE", "UNRESOLVED_CONTEXT"))]
    for pid, grp in usable_prior.groupby("pid"):
        grp = grp.sort_values("week")
        for _, cur in grp.iterrows():
            week = int(cur["week"])
            if week < FIRST_SCORABLE_WEEK:
                continue
            if cur["state"] in ("PARTIAL_GAME",):          # excluded from OUTCOMES
                continue
            if float(cur["targets"]) < 1.0:                 # outcome must exist
                continue
            prior = grp[grp["week"] < week]
            pt = float(prior["targets"].sum())
            games_with_target = int((prior["targets"] > 0).sum())
            if pt < MIN_PRIOR_TARGETS or games_with_target < MIN_PRIOR_GAMES_WITH_TARGET:
                continue
            dp = defense_prior(def_week, str(cur["opponent"]), week)
            if dp is None:
                continue
            prior_games = int(len(prior))
            rec = {
                "pid": pid, "player": cur["player"], "pos": cur["pos"], "week": week,
                "opponent": cur["opponent"],
                "prior_targets": pt, "prior_games": prior_games,
                "prior_adot": float(prior["receiving_air_yards"].sum()) / pt,
                "prior_ypt": float(prior["receiving_yards"].sum()) / pt,
                "prior_tgt_per_game": pt / prior_games if prior_games else np.nan,
                "prior_catch_rate": float(prior["receptions"].sum()) / pt,
                "y_receiving_yards": float(cur["receiving_yards"]),
                "y_targets": float(cur["targets"]),
                "state": cur["state"],
                **dp,
            }
            rec["yhat_base"] = rec["prior_ypt"] * rec["prior_tgt_per_game"]
            rows.append(rec)

    panel = pd.DataFrame(rows).dropna(subset=["yhat_base", "def_zone_rate", "def_heavy_box_rate"])
    diag = {
        "state_label_counts": w["state"].value_counts().to_dict(),
        "weekly_rows_considered": int(len(w)),
        "defense_team_weeks": int(len(def_week)),
        "panel_rows": int(len(panel)),
        "panel_players": int(panel["pid"].nunique()) if len(panel) else 0,
    }
    return panel, diag


def _design(df: pd.DataFrame, model: str) -> np.ndarray:
    cols = [np.ones(len(df)), df["yhat_base"].to_numpy(float)]
    if model in ("M1", "M2"):
        for p in POSITIONS:
            d = (df["pos"] == p).to_numpy(float)
            cols.append(d * df["def_zone_rate"].to_numpy(float))
            cols.append(d * df["def_heavy_box_rate"].to_numpy(float))
    if model == "M2":
        cols.append(df["prior_adot"].to_numpy(float) * df["def_zone_rate"].to_numpy(float))
        cols.append(df["prior_adot"].to_numpy(float) * df["def_man_rate"].to_numpy(float))
    return np.column_stack(cols)


def _fit_predict(fit: pd.DataFrame, score: pd.DataFrame, model: str):
    X, y = _design(fit, model), fit["y_receiving_yards"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return _design(score, model) @ beta, beta


def run(panel: pd.DataFrame) -> dict:
    fit = panel[panel["week"].isin(list(FIT_WEEKS))]
    hold = panel[panel["week"].isin(list(HOLDOUT_WEEKS))]
    out: dict = {
        "fit_rows": int(len(fit)), "holdout_rows": int(len(hold)),
        "fit_weeks": [min(FIT_WEEKS), max(FIT_WEEKS)],
        "holdout_weeks": [min(HOLDOUT_WEEKS), max(HOLDOUT_WEEKS)],
    }
    preds = {}
    for m in ("M0", "M1", "M2"):
        yhat, beta = _fit_predict(fit, hold, m)
        preds[m] = yhat
        err = np.abs(yhat - hold["y_receiving_yards"].to_numpy(float))
        out[f"{m}_holdout_mae"] = float(err.mean())
        out[f"{m}_n_params"] = int(len(beta))

    # --- decision rule 1: player-level error, M2 vs M1 ---
    h = hold.copy()
    h["abs_err_M1"] = np.abs(preds["M1"] - h["y_receiving_yards"].to_numpy(float))
    h["abs_err_M2"] = np.abs(preds["M2"] - h["y_receiving_yards"].to_numpy(float))
    per_player = h.groupby("pid").agg(
        pos=("pos", "first"), player=("player", "first"), games=("week", "size"),
        prior_adot=("prior_adot", "mean"),
        mae_M1=("abs_err_M1", "mean"), mae_M2=("abs_err_M2", "mean"))
    per_player["improvement"] = per_player["mae_M1"] - per_player["mae_M2"]
    out["player_level"] = {
        "players_scored": int(len(per_player)),
        "players_improved": int((per_player["improvement"] > 0).sum()),
        "players_worsened": int((per_player["improvement"] < 0).sum()),
        "mean_improvement": float(per_player["improvement"].mean()),
        "median_improvement": float(per_player["improvement"].median()),
    }
    out["rule_1_M2_beats_M1_player_level"] = bool(per_player["improvement"].mean() > 0)

    # --- decision rule 3: concentrated in off-mean-aDOT quartiles ---
    per_player["adot_q"] = np.nan
    for p in POSITIONS:
        mask = per_player["pos"] == p
        if mask.sum() >= 4:
            per_player.loc[mask, "adot_q"] = pd.qcut(
                per_player.loc[mask, "prior_adot"], 4, labels=False, duplicates="drop")
    off = per_player[per_player["adot_q"].isin([0.0, 3.0])]
    mid = per_player[per_player["adot_q"].isin([1.0, 2.0])]
    out["concentration"] = {
        "off_mean_quartile_players": int(len(off)),
        "off_mean_mean_improvement": float(off["improvement"].mean()) if len(off) else np.nan,
        "middle_quartile_players": int(len(mid)),
        "middle_mean_improvement": float(mid["improvement"].mean()) if len(mid) else np.nan,
    }
    out["rule_3_concentrated_off_mean"] = bool(
        len(off) and len(mid) and off["improvement"].mean() > mid["improvement"].mean())

    # --- decision rule 2: disjoint-window sign persistence of the player interaction ---
    idx_adot_zone = -2  # order fixed by _design: [..., adot*zone, adot*man]
    coefs = {}
    for label, weeks in (("window_A_4_10", PERSIST_A), ("window_B_11_18", PERSIST_B)):
        sub = panel[panel["week"].isin(list(weeks))]
        if len(sub) <= 12:
            coefs[label] = None
            continue
        X, y = _design(sub, "M2"), sub["y_receiving_yards"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs[label] = {"rows": int(len(sub)),
                        "adot_x_zone": float(beta[idx_adot_zone]),
                        "adot_x_man": float(beta[idx_adot_zone + 1])}
    out["disjoint_persistence"] = coefs
    a, b = coefs.get("window_A_4_10"), coefs.get("window_B_11_18")
    out["rule_2_sign_persists_disjoint"] = bool(
        a and b and np.sign(a["adot_x_zone"]) == np.sign(b["adot_x_zone"]))

    out["H1_SUPPORTED"] = bool(
        out["rule_1_M2_beats_M1_player_level"]
        and out["rule_2_sign_persists_disjoint"]
        and out["rule_3_concentrated_off_mean"])
    return out, per_player


def secondary_outcomes(panel: pd.DataFrame) -> dict:
    """Preregistration section 2: targets and ypt are REPORTED, never used to pick a winner."""
    fit = panel[panel["week"].isin(list(FIT_WEEKS))]
    hold = panel[panel["week"].isin(list(HOLDOUT_WEEKS))]
    out = {}
    for label, ycol, basecol in (("targets", "y_targets", "prior_tgt_per_game"),):
        rep = {}
        for m in ("M0", "M1", "M2"):
            Xf, Xh = _design(fit, m).copy(), _design(hold, m).copy()
            Xf[:, 1] = fit[basecol].to_numpy(float)       # swap in the matching baseline
            Xh[:, 1] = hold[basecol].to_numpy(float)
            beta, *_ = np.linalg.lstsq(Xf, fit[ycol].to_numpy(float), rcond=None)
            rep[f"{m}_holdout_mae"] = float(np.abs(Xh @ beta - hold[ycol].to_numpy(float)).mean())
        rep["M2_minus_M1"] = rep["M2_holdout_mae"] - rep["M1_holdout_mae"]
        out[label] = rep
    return out


def main() -> int:
    prereg = Path("docs/research/PRISM_PLAYER_SPECIFIC_MATCHUP_PREREGISTRATION_V1.md")
    if not prereg.exists():
        raise RuntimeError("preregistration is missing; this study may not be scored unfrozen")
    text = prereg.read_text(encoding="utf-8")
    for token in ("20", "100", "12–18", "4–10"):
        if token not in text:
            raise RuntimeError(f"preregistration no longer carries frozen value {token!r}")

    data = _load()
    panel, diag = build_panel(data)
    if panel.empty:
        raise RuntimeError("panel is empty; eligibility or join failed -- failing closed")
    result, per_player = run(panel)
    secondary = secondary_outcomes(panel)
    report = {"preregistration": prereg.as_posix(), "season": SEASON,
              "diagnostics": diag, "result": result,
              "secondary_outcomes_reported_not_decisive": secondary,
              "sportsbook_inputs_used": False}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    per_player.reset_index().to_csv(OUT_ROWS, index=False)

    print(f"panel rows={diag['panel_rows']} players={diag['panel_players']}")
    print(f"state labels: {diag['state_label_counts']}")
    r = result
    print(f"\nholdout MAE  M0={r['M0_holdout_mae']:.3f}  M1={r['M1_holdout_mae']:.3f}  M2={r['M2_holdout_mae']:.3f}")
    pl = r["player_level"]
    print(f"player-level: {pl['players_improved']}/{pl['players_scored']} improved, "
          f"mean improvement {pl['mean_improvement']:+.3f} yards MAE")
    print(f"concentration: off-mean {r['concentration']['off_mean_mean_improvement']:+.3f} "
          f"vs middle {r['concentration']['middle_mean_improvement']:+.3f}")
    print(f"disjoint persistence: {json.dumps(r['disjoint_persistence'], default=str)}")
    print(f"\n  rule 1 (M2 beats M1, player level): {r['rule_1_M2_beats_M1_player_level']}")
    print(f"  rule 2 (sign persists, disjoint):   {r['rule_2_sign_persists_disjoint']}")
    print(f"  rule 3 (concentrated off-mean):     {r['rule_3_concentrated_off_mean']}")
    print(f"\nsecondary (reported, not decisive): {json.dumps(secondary)}")
    print(f"\n  H1 SUPPORTED: {r['H1_SUPPORTED']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
