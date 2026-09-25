#!/usr/bin/env python3
"""Diagnostic-only audit of WR room regime instability.

Consumes the authoritative scored room-detail artifacts from:
- Receiver Room Targets-Per-Play V1 2022-2023 discovery
- unchanged 2024-2025 confirmation

No candidate is built or scored here. Structural descriptors are attached to
already-scored WR room rows for hypothesis discovery only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.build_qb_playcaller_opening_leverage import caller_for
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION = "WR_ROOM_REGIME_INSTABILITY_AUDIT_V1"
SEASONS = (2022, 2023, 2024, 2025)
EXPECTED_WR_MAE = {
    2022: (4.761250, 4.498155),
    2023: (4.538430, 4.467731),
    2024: (4.561046, 4.724467),
    2025: (4.510085, 4.440251),
}
CONTINUOUS = (
    "prior_season_wr_tpp",
    "current_strict_prior_wr_tpp",
    "pregame_wr_rate_gap",
    "current_history_weight",
    "prior_wr_target_mass_retained",
    "current_season_games_observed",
)
BINARY = (
    "prior_top_wr_on_roster",
    "strict_prior_wr_leader_changed",
    "prior_primary_qb_on_roster",
    "strict_prior_primary_qb_changed",
    "playcaller_changed_vs_prior_season_end",
    "playcaller_changed_this_week",
)


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def clean_player(value: object) -> str:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    try:
        _, key = canonicalize_player_name_safe(raw)
        key = (key or "").strip()
        if key:
            return key
    except Exception:
        pass
    return "".join(ch.lower() for ch in raw if ch.isalnum())


def is_wr(value: object) -> bool:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    return p in {"WR", "LWR", "RWR", "SWR"} or p.startswith("WR")


def is_qb(value: object) -> bool:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    return p == "QB" or p.startswith("QB")


def load_frozen_wr_detail(a: Path, b: Path) -> pd.DataFrame:
    d1 = read(a, "2022-2023 room detail")
    d2 = read(b, "2024-2025 room detail")
    required = {
        "season", "week", "team", "room", "baseline_room_targets",
        "candidate_room_targets", "actual_room_targets",
    }
    for label, d in (("discovery", d1), ("confirmation", d2)):
        missing = required - set(d.columns)
        if missing:
            raise RuntimeError(f"{label} detail missing {sorted(missing)}")
    x = pd.concat([d1, d2], ignore_index=True, sort=False)
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["room"] = x["room"].astype(str).str.upper().str.strip()
    x = x.loc[x["season"].isin(SEASONS) & x["room"].eq("WR")].copy()
    if x.empty:
        raise RuntimeError("zero frozen WR rows")
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate frozen WR team-game rows")
    for c in ("baseline_room_targets", "candidate_room_targets", "actual_room_targets"):
        x[c] = num(x[c])
        if x[c].isna().any():
            raise RuntimeError(f"non-numeric frozen detail {c}")
    x["baseline_abs_error"] = (x["baseline_room_targets"] - x["actual_room_targets"]).abs()
    x["candidate_abs_error"] = (x["candidate_room_targets"] - x["actual_room_targets"]).abs()
    x["candidate_harm"] = x["candidate_abs_error"] - x["baseline_abs_error"]
    x["candidate_shift"] = x["candidate_room_targets"] - x["baseline_room_targets"]
    x["candidate_closer"] = (x["candidate_abs_error"] < x["baseline_abs_error"]).astype(int)

    for season, (exp_b, exp_c) in EXPECTED_WR_MAE.items():
        s = x.loc[x["season"].eq(season)]
        if s.empty:
            raise RuntimeError(f"missing frozen WR season {season}")
        got_b = float(s["baseline_abs_error"].mean())
        got_c = float(s["candidate_abs_error"].mean())
        if abs(got_b - exp_b) > 5e-5 or abs(got_c - exp_c) > 5e-5:
            raise RuntimeError(
                f"frozen artifact authority mismatch {season}: "
                f"{got_b:.6f}->{got_c:.6f} expected {exp_b:.6f}->{exp_c:.6f}"
            )
    return x.sort_values(["season", "week", "team"]).reset_index(drop=True)


def prep_logs(logs: pd.DataFrame) -> pd.DataFrame:
    x = logs.copy()
    req = {"season", "week", "team", "position", "targets", "pass_att"}
    missing = req - set(x.columns)
    if missing:
        raise RuntimeError(f"player logs missing {sorted(missing)}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["targets"] = num(x["targets"]).fillna(0.0)
    x["pass_att"] = num(x["pass_att"]).fillna(0.0)
    if "player_clean_key" not in x.columns:
        if "player" not in x.columns:
            raise RuntimeError("player logs missing player_clean_key/player")
        x["player_clean_key"] = x["player"].map(clean_player)
    else:
        x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str)
    x["position"] = x["position"].fillna("").astype(str).str.upper().str.strip()
    return x


def prep_team_weekly(team: pd.DataFrame) -> pd.DataFrame:
    x = team.copy()
    req = {"season", "week", "team", "plays_est"}
    missing = req - set(x.columns)
    if missing:
        raise RuntimeError(f"team weekly missing {sorted(missing)}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["plays_est"] = num(x["plays_est"])
    x = x.dropna(subset=["season", "week", "team", "plays_est"])
    x = x.drop_duplicates(["season", "week", "team"], keep="last")
    return x


def load_universe(path: Path) -> pd.DataFrame:
    u = read(path, f"pregame universe {path.name}")
    req = {"team", "position", "player"}
    missing = req - set(u.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    u["team"] = u["team"].map(canon_team)
    u["position"] = u["position"].fillna("").astype(str).str.upper().str.strip()
    u["player_key"] = u["player"].map(clean_player)
    return u


def season_team_wr_stats(logs: pd.DataFrame, team: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wr = logs.loc[logs["position"].map(is_wr)].copy()
    wr_team = (
        wr.groupby(["season", "team"], as_index=False)
        .agg(wr_targets=("targets", "sum"))
    )
    plays = (
        team.groupby(["season", "team"], as_index=False)
        .agg(plays=("plays_est", "sum"), games=("week", "nunique"))
    )
    rate = wr_team.merge(plays, on=["season", "team"], how="outer")
    rate["wr_targets"] = num(rate["wr_targets"]).fillna(0.0)
    rate["wr_tpp"] = rate["wr_targets"] / num(rate["plays"]).replace(0, np.nan)

    wr_player = (
        wr.groupby(["season", "team", "player_clean_key"], as_index=False)
        .agg(wr_targets=("targets", "sum"))
    )
    return rate, wr_player


def season_team_qb_stats(logs: pd.DataFrame) -> pd.DataFrame:
    q = logs.loc[logs["position"].map(is_qb)].copy()
    return (
        q.groupby(["season", "team", "player_clean_key"], as_index=False)
        .agg(pass_att=("pass_att", "sum"))
    )


def prior_leader(frame: pd.DataFrame, season: int, team: str, value: str) -> tuple[str, float]:
    z = frame.loc[frame["season"].eq(int(season)) & frame["team"].eq(team)].copy()
    if z.empty:
        return "", 0.0
    z = z.sort_values([value, "player_clean_key"], ascending=[False, True])
    r = z.iloc[0]
    return str(r["player_clean_key"]), float(r[value])


def strict_prior_leader(
    logs: pd.DataFrame, *, season: int, week: int, team: str, position_test, value: str
) -> str:
    z = logs.loc[
        logs["season"].eq(int(season))
        & logs["week"].lt(int(week))
        & logs["team"].eq(team)
        & logs["position"].map(position_test)
    ].copy()
    if z.empty:
        return ""
    q = (
        z.groupby("player_clean_key", as_index=False)
        .agg(metric=(value, "sum"))
        .sort_values(["metric", "player_clean_key"], ascending=[False, True])
    )
    return str(q.iloc[0]["player_clean_key"]) if len(q) else ""


def attach_structural_state(
    detail: pd.DataFrame,
    logs: pd.DataFrame,
    team: pd.DataFrame,
    wr_season: pd.DataFrame,
    wr_player: pd.DataFrame,
    qb_player: pd.DataFrame,
    universe_dirs: dict[int, Path],
) -> pd.DataFrame:
    rows = []
    universe_cache: dict[tuple[int, int], pd.DataFrame] = {}

    rate_lookup = {
        (int(r.season), str(r.team)): (float(r.wr_tpp) if pd.notna(r.wr_tpp) else np.nan)
        for r in wr_season.itertuples(index=False)
    }
    games_lookup = {
        (int(r.season), str(r.team)): int(r.games) if pd.notna(r.games) else 0
        for r in wr_season.itertuples(index=False)
    }

    for r in detail.itertuples(index=False):
        season, week, team_name = int(r.season), int(r.week), canon_team(r.team)
        prior = season - 1
        key = (season, week)
        if key not in universe_cache:
            universe_cache[key] = load_universe(
                universe_dirs[season] / f"{season}_week_{week:02d}.csv"
            )
        u = universe_cache[key]
        tu = u.loc[u["team"].eq(team_name)].copy()
        active_wr = set(tu.loc[tu["position"].map(is_wr), "player_key"].astype(str))
        active_qb = set(tu.loc[tu["position"].map(is_qb), "player_key"].astype(str))

        prior_wr = wr_player.loc[
            wr_player["season"].eq(prior) & wr_player["team"].eq(team_name)
        ].copy()
        prior_wr_total = float(prior_wr["wr_targets"].sum()) if len(prior_wr) else 0.0
        retained = float(
            prior_wr.loc[prior_wr["player_clean_key"].isin(active_wr), "wr_targets"].sum()
        ) if len(prior_wr) else 0.0
        retained_frac = retained / prior_wr_total if prior_wr_total > 0 else np.nan

        prior_top_wr, _ = prior_leader(wr_player, prior, team_name, "wr_targets")
        current_wr_leader = strict_prior_leader(
            logs, season=season, week=week, team=team_name,
            position_test=is_wr, value="targets"
        )

        prior_qb, _ = prior_leader(qb_player, prior, team_name, "pass_att")
        current_qb = strict_prior_leader(
            logs, season=season, week=week, team=team_name,
            position_test=is_qb, value="pass_att"
        )

        prior_wr_tpp = rate_lookup.get((prior, team_name), np.nan)
        cur_hist_logs = logs.loc[
            logs["season"].eq(season)
            & logs["week"].lt(week)
            & logs["team"].eq(team_name)
            & logs["position"].map(is_wr)
        ]
        cur_hist_targets = float(cur_hist_logs["targets"].sum()) if len(cur_hist_logs) else 0.0
        cur_games = team.loc[
            team["season"].eq(season)
            & team["week"].lt(week)
            & team["team"].eq(team_name)
        ]
        cur_hist_plays = float(cur_games["plays_est"].sum()) if len(cur_games) else 0.0
        cur_hist_tpp = cur_hist_targets / cur_hist_plays if cur_hist_plays > 0 else np.nan
        cur_games_n = int(cur_games["week"].nunique()) if len(cur_games) else 0

        prior_games_n = int(games_lookup.get((prior, team_name), 0))
        history_weight = (
            cur_games_n / (cur_games_n + prior_games_n)
            if (cur_games_n + prior_games_n) > 0 else np.nan
        )

        current_caller = caller_for(season, week, team_name)
        prev_season_caller = caller_for(prior, 18, team_name)
        prev_week_caller = caller_for(season, max(1, week - 1), team_name)
        caller_prior_coverage = bool(current_caller and prev_season_caller)
        caller_week_coverage = bool(current_caller and prev_week_caller and week > 1)

        current_full = rate_lookup.get((season, team_name), np.nan)
        realized_drift = (
            float(current_full - prior_wr_tpp)
            if np.isfinite(current_full) and np.isfinite(prior_wr_tpp) else np.nan
        )

        rec = r._asdict()
        rec.update({
            "prior_season_wr_tpp": prior_wr_tpp,
            "current_strict_prior_wr_tpp": cur_hist_tpp,
            "pregame_wr_rate_gap": (
                float(cur_hist_tpp - prior_wr_tpp)
                if np.isfinite(cur_hist_tpp) and np.isfinite(prior_wr_tpp) else np.nan
            ),
            "current_season_games_observed": cur_games_n,
            "prior_season_games": prior_games_n,
            "current_history_weight": history_weight,
            "prior_wr_target_mass_retained": retained_frac,
            "prior_top_wr_on_roster": (
                float(prior_top_wr in active_wr) if prior_top_wr else np.nan
            ),
            "strict_prior_wr_leader_changed": (
                float(current_wr_leader != prior_top_wr)
                if current_wr_leader and prior_top_wr else np.nan
            ),
            "prior_primary_qb_on_roster": (
                float(prior_qb in active_qb) if prior_qb else np.nan
            ),
            "strict_prior_primary_qb_changed": (
                float(current_qb != prior_qb)
                if current_qb and prior_qb else np.nan
            ),
            "playcaller_current_name": current_caller,
            "playcaller_prior_season_end_name": prev_season_caller,
            "playcaller_changed_vs_prior_season_end": (
                float(current_caller != prev_season_caller)
                if caller_prior_coverage else np.nan
            ),
            "playcaller_changed_this_week": (
                float(current_caller != prev_week_caller)
                if caller_week_coverage else np.nan
            ),
            "realized_full_season_wr_tpp": current_full,
            "realized_full_season_wr_rate_drift": realized_drift,
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def spearman(x: pd.Series, y: pd.Series) -> tuple[int, float | None]:
    q = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(q) < 3 or q["x"].nunique() < 2 or q["y"].nunique() < 2:
        return int(len(q)), None
    rx = q["x"].rank(method="average")
    ry = q["y"].rank(method="average")
    return int(len(q)), float(rx.corr(ry))


def association_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, d in [("pooled", detail)] + [
        (str(s), detail.loc[detail["season"].eq(s)]) for s in SEASONS
    ]:
        for feature in CONTINUOUS:
            n, rho = spearman(d[feature], d["candidate_harm"])
            rows.append({
                "scope": scope,
                "feature": feature,
                "n": n,
                "spearman_vs_candidate_harm": rho,
            })
    return pd.DataFrame(rows)


def transition_group_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        d = detail.loc[detail["season"].eq(season)]
        for feature in BINARY:
            for state in (0.0, 1.0):
                g = d.loc[num(d[feature]).eq(state)]
                if g.empty:
                    continue
                rows.append({
                    "season": season,
                    "feature": feature,
                    "state": int(state),
                    "n": int(len(g)),
                    "mean_candidate_harm": float(g["candidate_harm"].mean()),
                    "median_candidate_harm": float(g["candidate_harm"].median()),
                    "candidate_closer_rate": float(g["candidate_closer"].mean()),
                })
    return pd.DataFrame(rows)


def season_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        d = detail.loc[detail["season"].eq(season)]
        team = (
            d.groupby("team", as_index=False)
            .agg(total_harm=("candidate_harm", "sum"), mean_harm=("candidate_harm", "mean"))
        )
        positive = team.loc[team["total_harm"] > 0, "total_harm"]
        pos_total = float(positive.sum())
        top8 = float(positive.nlargest(8).sum()) if len(positive) else 0.0
        rows.append({
            "season": season,
            "n_rows": int(len(d)),
            "baseline_mae": float(d["baseline_abs_error"].mean()),
            "candidate_mae": float(d["candidate_abs_error"].mean()),
            "mean_candidate_harm": float(d["candidate_harm"].mean()),
            "candidate_closer_rate": float(d["candidate_closer"].mean()),
            "harmful_teams": int((team["total_harm"] > 0).sum()),
            "helpful_teams": int((team["total_harm"] < 0).sum()),
            "zero_teams": int((team["total_harm"].abs() <= 1e-12).sum()),
            "positive_harm_top8_share": (top8 / pos_total if pos_total > 0 else np.nan),
            "positive_harm_total": pos_total,
        })
    return pd.DataFrame(rows)


def team_season_summary(detail: pd.DataFrame) -> pd.DataFrame:
    agg = (
        detail.groupby(["season", "team"], as_index=False)
        .agg(
            n_games=("week", "nunique"),
            total_candidate_harm=("candidate_harm", "sum"),
            mean_candidate_harm=("candidate_harm", "mean"),
            candidate_closer_rate=("candidate_closer", "mean"),
            prior_season_wr_tpp=("prior_season_wr_tpp", "first"),
            realized_full_season_wr_tpp=("realized_full_season_wr_tpp", "first"),
            realized_full_season_wr_rate_drift=("realized_full_season_wr_rate_drift", "first"),
            mean_prior_wr_target_mass_retained=("prior_wr_target_mass_retained", "mean"),
            min_prior_wr_target_mass_retained=("prior_wr_target_mass_retained", "min"),
            mean_current_history_weight=("current_history_weight", "mean"),
            mean_abs_pregame_wr_rate_gap=("pregame_wr_rate_gap", lambda s: float(num(s).abs().mean())),
            prior_top_wr_retained_rate=("prior_top_wr_on_roster", "mean"),
            prior_primary_qb_retained_rate=("prior_primary_qb_on_roster", "mean"),
            caller_changed_vs_prior_season_rate=("playcaller_changed_vs_prior_season_end", "mean"),
        )
    )
    return agg.sort_values(["season", "total_candidate_harm"], ascending=[True, False])


def source_coverage(detail: pd.DataFrame) -> pd.DataFrame:
    cols = list(CONTINUOUS) + list(BINARY) + ["realized_full_season_wr_rate_drift"]
    rows = []
    for season in SEASONS:
        d = detail.loc[detail["season"].eq(season)]
        for c in cols:
            rows.append({
                "season": season,
                "feature": c,
                "n": int(len(d)),
                "non_null": int(d[c].notna().sum()),
                "coverage": float(d[c].notna().mean()) if len(d) else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--detail-2022-2023", type=Path, required=True)
    p.add_argument("--detail-2024-2025", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    for season in SEASONS:
        p.add_argument(f"--universe-{season}", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    frozen = load_frozen_wr_detail(a.detail_2022_2023, a.detail_2024_2025)
    logs = prep_logs(read(a.player_logs, "historical player logs"))
    team = prep_team_weekly(read(a.team_weekly, "historical team weekly"))
    wr_season, wr_player = season_team_wr_stats(logs, team)
    qb_player = season_team_qb_stats(logs)
    universe_dirs = {s: getattr(a, f"universe_{s}") for s in SEASONS}

    detail = attach_structural_state(
        frozen, logs, team, wr_season, wr_player, qb_player, universe_dirs
    )
    seasons = season_summary(detail)
    teams = team_season_summary(detail)
    assoc = association_summary(detail)
    groups = transition_group_summary(detail)
    coverage = source_coverage(detail)
    week = (
        detail.groupby(["season", "week"], as_index=False)
        .agg(
            n=("team", "size"),
            mean_candidate_harm=("candidate_harm", "mean"),
            candidate_closer_rate=("candidate_closer", "mean"),
            mean_current_history_weight=("current_history_weight", "mean"),
            mean_abs_pregame_wr_rate_gap=("pregame_wr_rate_gap", lambda s: float(num(s).abs().mean())),
        )
    )

    s24 = seasons.loc[seasons["season"].eq(2024)].iloc[0]
    top24 = teams.loc[teams["season"].eq(2024)].head(10).copy()

    payload = {
        "version": VERSION,
        "disposition": "WR_ROOM_REGIME_INSTABILITY_AUDIT_V1_COMPLETE",
        "candidate_variants_scored": 0,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "production_mutations": 0,
        "rows": int(len(detail)),
        "seasons": {
            str(int(r.season)): {
                "baseline_mae": float(r.baseline_mae),
                "candidate_mae": float(r.candidate_mae),
                "mean_candidate_harm": float(r.mean_candidate_harm),
                "candidate_closer_rate": float(r.candidate_closer_rate),
                "harmful_teams": int(r.harmful_teams),
                "helpful_teams": int(r.helpful_teams),
                "positive_harm_top8_share": (
                    None if pd.isna(r.positive_harm_top8_share)
                    else float(r.positive_harm_top8_share)
                ),
            }
            for r in seasons.itertuples(index=False)
        },
        "year_2024": {
            "harmful_teams": int(s24.harmful_teams),
            "helpful_teams": int(s24.helpful_teams),
            "positive_harm_top8_share": (
                None if pd.isna(s24.positive_harm_top8_share)
                else float(s24.positive_harm_top8_share)
            ),
            "top_harm_teams": [
                {
                    "team": str(r.team),
                    "total_harm": float(r.total_candidate_harm),
                    "mean_harm": float(r.mean_candidate_harm),
                    "realized_wr_tpp_drift": (
                        None if pd.isna(r.realized_full_season_wr_rate_drift)
                        else float(r.realized_full_season_wr_rate_drift)
                    ),
                    "mean_prior_wr_target_mass_retained": (
                        None if pd.isna(r.mean_prior_wr_target_mass_retained)
                        else float(r.mean_prior_wr_target_mass_retained)
                    ),
                    "prior_primary_qb_retained_rate": (
                        None if pd.isna(r.prior_primary_qb_retained_rate)
                        else float(r.prior_primary_qb_retained_rate)
                    ),
                    "caller_changed_vs_prior_season_rate": (
                        None if pd.isna(r.caller_changed_vs_prior_season_rate)
                        else float(r.caller_changed_vs_prior_season_rate)
                    ),
                }
                for r in top24.itertuples(index=False)
            ],
        },
        "interpretation_status": "REQUIRES_POST_RUN_SCIENTIFIC_DISPOSITION",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "wr_room_regime_detail.csv", index=False)
    teams.to_csv(a.out_dir / "team_season_regime_summary.csv", index=False)
    seasons.to_csv(a.out_dir / "season_summary.csv", index=False)
    assoc.to_csv(a.out_dir / "association_summary.csv", index=False)
    groups.to_csv(a.out_dir / "transition_group_summary.csv", index=False)
    coverage.to_csv(a.out_dir / "source_coverage.csv", index=False)
    week.to_csv(a.out_dir / "week_summary.csv", index=False)
    (a.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# WR Room Regime-Instability Audit V1",
        "",
        "Disposition: **WR_ROOM_REGIME_INSTABILITY_AUDIT_V1_COMPLETE**",
        "",
        "- candidate variants scored: **0**",
        "- parameters fit: **0**",
        "- sportsbook inputs: **0**",
        "",
        "## Frozen WR result reproduction",
        "",
    ]
    for r in seasons.itertuples(index=False):
        lines.append(
            f"- {int(r.season)}: MAE {r.baseline_mae:.6f} -> {r.candidate_mae:.6f}; "
            f"harmful/helpful teams {int(r.harmful_teams)}/{int(r.helpful_teams)}; "
            f"top-8 positive-harm share "
            f"{r.positive_harm_top8_share:.3f}"
            if pd.notna(r.positive_harm_top8_share)
            else f"- {int(r.season)}: MAE {r.baseline_mae:.6f} -> {r.candidate_mae:.6f}"
        )
    lines += [
        "",
        "## 2024 highest excess-harm teams",
        "",
    ]
    for r in top24.itertuples(index=False):
        lines.append(
            f"- {r.team}: total excess absolute error {r.total_candidate_harm:.3f}; "
            f"mean {r.mean_candidate_harm:.3f}; "
            f"realized WR TPP drift {r.realized_full_season_wr_rate_drift:.5f}; "
            f"mean retained prior-WR target mass {r.mean_prior_wr_target_mass_retained:.3f}"
        )
    lines += [
        "",
        "## Scientific status",
        "",
        "This artifact is diagnostic only. A separate post-run scientific disposition "
        "must decide whether a leakage-safe structural hypothesis is warranted. "
        "No candidate is authorized by this run alone.",
    ]
    (a.out_dir / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
