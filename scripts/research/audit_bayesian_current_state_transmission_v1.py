#!/usr/bin/env python3
"""Bayesian Current-State Transmission V1.

Read-only audit frozen in:
docs/research/BAYESIAN_CURRENT_STATE_TRANSMISSION_V1_PLAN.md

The audit compares the exact production PlayerForm prior/current blend against
the exact production empirical-Bayes posterior for the three opportunity
metrics already shown by Current-Season State Persistence V1 to update quickly.

No parameter is fit or changed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.bayesian_v2 import (
    GROUP_STRENGTH,
    PRIOR_PLAYER_CAP,
    build_bayesian_baseline,
)
from scripts.player_form_v2 import (
    _blend,
    _dedupe_resolved_universe,
    _load_weekly,
    _normalize_weekly,
    _season_totals,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.player_identity_v3 import build_identity_registry, resolve_slate_identities

VERSION = "BAYESIAN_CURRENT_STATE_TRANSMISSION_V1"
EVALS = ((2024, 2023), (2025, 2024))
OFF_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "LWR", "RWR", "SWR", "TE"}
ALLOWED_ROSTER_STATUS = {"ACT", "INA"}
PRIMARY = (
    ("RB", "rush_share", "rush_share_game"),
    ("WR", "tgt_share", "tgt_share_game"),
    ("TE", "tgt_share", "tgt_share_game"),
)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260927


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _key(value: object) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _team(value: object) -> str:
    try:
        return canon_team(value)
    except Exception:
        return ""


def _load_normalized_logs(seasons: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in sorted(seasons):
        raw = _load_weekly(int(season))
        x = _normalize_weekly(raw, int(season))
        x = x.loc[pd.to_numeric(x["week"], errors="coerce").between(1, 18)].copy()
        frames.append(x)
    out = pd.concat(frames, ignore_index=True, sort=False)
    if out.empty:
        raise RuntimeError("normalized historical weekly logs are empty")
    if out["player_identity_key"].astype(str).eq("").any():
        raise RuntimeError("normalized historical weekly logs contain blank stable identity")
    return out


def _load_rosters(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = _to_pandas(nfl.load_rosters_weekly(int(season)))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {"season", "week", "team", "position"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"weekly roster missing columns: {sorted(missing)}")
    name_col = "full_name" if "full_name" in raw.columns else "football_name" if "football_name" in raw.columns else None
    if name_col is None:
        raise RuntimeError("weekly roster missing full_name/football_name")
    raw["_player_name"] = raw[name_col].astype("string").fillna("").str.strip()
    raw["team"] = raw["team"].map(_team)
    raw["position"] = raw["position"].astype("string").fillna("").str.upper().str.strip()
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce")
    raw = raw.loc[
        raw["season"].eq(int(season))
        & raw["week"].between(1, 18)
        & raw["position"].isin(OFF_POSITIONS)
        & raw["team"].ne("")
        & raw["_player_name"].ne("")
    ].copy()
    if "status" in raw.columns:
        allowed = raw["status"].astype(str).str.upper().isin(ALLOWED_ROSTER_STATUS)
        if allowed.any():
            raw = raw.loc[allowed].copy()
    return raw


def _pregame_universe(rosters: pd.DataFrame, season: int, week: int, registry: pd.DataFrame) -> pd.DataFrame:
    q = rosters.loc[
        pd.to_numeric(rosters["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(rosters["week"], errors="coerce").eq(int(week))
    ].copy()
    if q.empty:
        raise RuntimeError(f"weekly roster has zero target-week rows season={season} week={week}")
    u = pd.DataFrame({
        "player": q["_player_name"].astype(str),
        "team": q["team"].astype(str),
        "position": q["position"].astype(str),
        "role": "",
        "season": int(season),
        "week": int(week),
        "opponent": "",
    })
    u["player_clean_key"] = u["player"].map(_key)
    u = resolve_slate_identities(
        u,
        registry,
        name_col="player",
        team_col="team",
        strict_ambiguous=True,
        allow_temporary=True,
    )
    u = _dedupe_resolved_universe(u)
    if u.duplicated(["team", "player_identity_key"]).any():
        raise RuntimeError("resolved target-week roster contains duplicate team/player identity")
    return u


def _feature_frame(
    logs: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    season: int,
    prior_season: int,
    week: int,
) -> tuple[pd.DataFrame, dict]:
    s = pd.to_numeric(logs["season"], errors="coerce")
    w = pd.to_numeric(logs["week"], errors="coerce")
    prior_logs = logs.loc[s.eq(int(prior_season))].copy()
    current_logs = logs.loc[s.eq(int(season)) & w.lt(int(week))].copy()

    if not current_logs.empty and int(pd.to_numeric(current_logs["week"], errors="coerce").max()) >= int(week):
        raise RuntimeError("current feature history contains target/future week")

    eligible_identity_logs = pd.concat([prior_logs, current_logs], ignore_index=True, sort=False)
    registry = build_identity_registry(eligible_identity_logs)
    universe = _pregame_universe(rosters, season, week, registry)

    prior_totals = _season_totals(prior_logs)
    current_totals = _season_totals(current_logs)
    form = _blend(prior_totals, current_totals, universe)
    if form.empty:
        raise RuntimeError(f"PlayerForm blend empty season={season} week={week}")

    # Preserve current roster position as the production Bayes grouping source.
    form["position"] = form["position"].astype("string").fillna("").str.upper().str.strip()
    bayes = build_bayesian_baseline(form)

    keep = [
        "player_identity_key", "player", "team", "position", "prior_games", "current_games",
        "tgt_share", "rush_share", "tgt_share_prior", "tgt_share_current",
        "rush_share_prior", "rush_share_current",
    ]
    frame = form[[c for c in keep if c in form.columns]].copy()
    bkeep = [
        "team", "player_clean_key", "bayes_tgt_share", "bayes_rush_share",
        "bayes_tgt_share_effective_n", "bayes_rush_share_effective_n",
        "bayes_evidence_state",
    ]
    # PlayerForm and Bayes share the canonical player_clean_key/team grain.
    frame["player_clean_key"] = form["player_clean_key"].astype(str)
    frame = frame.merge(
        bayes[bkeep],
        on=["team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    if frame[["bayes_tgt_share", "bayes_rush_share"]].isna().all(axis=1).any():
        raise RuntimeError("Bayesian posterior unavailable on one or more resolved pregame rows")

    temp = int(frame["player_identity_key"].astype(str).str.startswith("temp:").sum())
    audit = {
        "season": int(season),
        "week": int(week),
        "pregame_roster_rows": int(len(universe)),
        "feature_rows": int(len(frame)),
        "temporary_identity_rows": temp,
        "prior_feature_max_week": None,
        "current_feature_max_week": int(pd.to_numeric(current_logs["week"], errors="coerce").max()) if not current_logs.empty else None,
    }
    return frame, audit


def _target_actuals(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    q = logs.loc[
        pd.to_numeric(logs["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(logs["week"], errors="coerce").eq(int(week))
    ].copy()
    if q.empty:
        return q
    if q.duplicated(["player_identity_key"]).any():
        sample = q.loc[q.duplicated(["player_identity_key"], keep=False), ["player_identity_key", "player", "team"]].head(20)
        raise RuntimeError(f"target actuals contain duplicate player identity rows: {sample.to_dict('records')}")
    return q


def _family(value: object) -> str:
    p = str(value or "").upper().strip()
    if p in {"WR", "LWR", "RWR", "SWR"}:
        return "WR"
    if p == "TE":
        return "TE"
    if p == "RB":
        return "RB"
    return p


def build_panel(logs: pd.DataFrame, rosters_by_season: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    audits: list[dict] = []

    for season, prior in EVALS:
        rosters = rosters_by_season[int(season)]
        for week in range(2, 19):
            frame, audit = _feature_frame(
                logs,
                rosters,
                season=int(season),
                prior_season=int(prior),
                week=int(week),
            )
            audits.append(audit)
            actual = _target_actuals(logs, int(season), int(week))
            if actual.empty:
                continue

            actual_cols = ["player_identity_key", "team", "tgt_share_game", "rush_share_game"]
            joined = frame.merge(
                actual[actual_cols],
                on=["player_identity_key", "team"],
                how="inner",
                validate="one_to_one",
            )
            joined["position_family"] = joined["position"].map(_family)

            for pos, metric, actual_col in PRIMARY:
                q = joined.loc[joined["position_family"].eq(pos)].copy()
                if q.empty:
                    continue
                pf_col = metric
                bayes_col = f"bayes_{metric}"
                needed = ["prior_games", "current_games", pf_col, bayes_col, actual_col]
                for col in needed:
                    q[col] = pd.to_numeric(q[col], errors="coerce")
                q = q.dropna(subset=needed)
                q = q.loc[q["prior_games"].ge(1) & q["current_games"].ge(1)].copy()
                if q.empty:
                    continue
                out = pd.DataFrame({
                    "season": int(season),
                    "target_week": int(week),
                    "player_identity_key": q["player_identity_key"].astype(str),
                    "player": q["player"].astype(str),
                    "team": q["team"].astype(str),
                    "position": pos,
                    "metric": metric,
                    "prior_games": q["prior_games"].astype(int),
                    "current_games": q["current_games"].astype(int),
                    "playerform_value": q[pf_col].astype(float),
                    "bayes_value": q[bayes_col].astype(float),
                    "actual_value": q[actual_col].astype(float),
                    "bayes_effective_n": pd.to_numeric(
                        q[f"bayes_{metric}_effective_n"], errors="coerce"
                    ).astype(float),
                    "bayes_evidence_state": q["bayes_evidence_state"].astype(str),
                })
                out["playerform_abs_error"] = (out["playerform_value"] - out["actual_value"]).abs()
                out["bayes_abs_error"] = (out["bayes_value"] - out["actual_value"]).abs()
                out["paired_ae_delta_bayes_minus_playerform"] = (
                    out["bayes_abs_error"] - out["playerform_abs_error"]
                )
                rows.append(out)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if panel.empty:
        raise RuntimeError("transmission audit produced zero scored rows")
    if panel.duplicated(["season", "target_week", "player_identity_key", "metric"]).any():
        raise RuntimeError("transmission panel contains duplicate player-week-metric rows")
    return panel, pd.DataFrame(audits)


def _corr(a: pd.Series, b: pd.Series, method: str) -> float:
    q = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(q) < 3 or q["a"].nunique() < 2 or q["b"].nunique() < 2:
        return np.nan
    return float(q["a"].corr(q["b"], method=method))


def _summary(q: pd.DataFrame, label: str) -> dict:
    actual = q["actual_value"].astype(float)
    out: dict[str, object] = {
        "sample": label,
        "position": str(q["position"].iloc[0]),
        "metric": str(q["metric"].iloc[0]),
        "n": int(len(q)),
        "unique_players": int(q["player_identity_key"].nunique()),
        "unique_weeks": int(q[["season", "target_week"]].drop_duplicates().shape[0]),
    }
    for name in ("playerform", "bayes"):
        pred = q[f"{name}_value"].astype(float)
        err = pred - actual
        out[f"{name}_mae"] = float(np.abs(err).mean())
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
        out[f"{name}_bias"] = float(err.mean())
        out[f"{name}_pearson"] = _corr(pred, actual, "pearson")
        out[f"{name}_spearman"] = _corr(pred, actual, "spearman")
    delta = q["paired_ae_delta_bayes_minus_playerform"].astype(float)
    out["mean_ae_delta_bayes_minus_playerform"] = float(delta.mean())
    out["playerform_closer_rate"] = float((delta > 0).mean())
    out["bayes_closer_rate"] = float((delta < 0).mean())
    out["tie_rate"] = float((delta == 0).mean())
    return out


def _player_cluster_bootstrap(q: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> tuple[float, float, float]:
    if q.empty:
        return np.nan, np.nan, np.nan
    grouped = {
        k: g["paired_ae_delta_bayes_minus_playerform"].astype(float).to_numpy()
        for k, g in q.groupby("player_identity_key", sort=False)
    }
    players = np.array(list(grouped.keys()), dtype=object)
    if len(players) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(players, size=len(players), replace=True)
        parts = [grouped[p] for p in sample]
        vals[i] = float(np.concatenate(parts).mean())
    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
        float((vals > 0).mean()),
    )


def summarize(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season in (2024, 2025):
        for (_, _), q in panel.loc[panel["season"].eq(season)].groupby(["position", "metric"], sort=True):
            r = _summary(q, str(season))
            lo, hi, p = _player_cluster_bootstrap(q)
            r.update({
                "bootstrap_delta_ci_low": lo,
                "bootstrap_delta_ci_high": hi,
                "bootstrap_p_playerform_better": p,
            })
            rows.append(r)

    # Pooled Week-3 analogue: exactly two completed current-season games.
    two = panel.loc[panel["current_games"].eq(2)].copy()
    for (_, _), q in two.groupby(["position", "metric"], sort=True):
        r = _summary(q, "WEEK3_ANALOGUE_CURRENT_GAMES_2_POOLED")
        lo, hi, p = _player_cluster_bootstrap(q)
        r.update({
            "bootstrap_delta_ci_low": lo,
            "bootstrap_delta_ci_high": hi,
            "bootstrap_p_playerform_better": p,
        })
        rows.append(r)

    # Veteran two-game subset isolates the exact 33% vs ~18% architecture case.
    vet = two.loc[panel.loc[two.index, "prior_games"].ge(6)].copy()
    for (_, _), q in vet.groupby(["position", "metric"], sort=True):
        r = _summary(q, "WEEK3_ANALOGUE_CURRENT_GAMES_2_PRIOR6PLUS")
        lo, hi, p = _player_cluster_bootstrap(q)
        r.update({
            "bootstrap_delta_ci_low": lo,
            "bootstrap_delta_ci_high": hi,
            "bootstrap_p_playerform_better": p,
        })
        rows.append(r)

    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> dict:
    per_metric: list[dict] = []
    for pos, metric, _ in PRIMARY:
        q = summary.loc[
            summary["position"].eq(pos)
            & summary["metric"].eq(metric)
            & summary["sample"].isin(["2024", "2025"])
        ].copy()
        if set(q["sample"]) != {"2024", "2025"}:
            raise RuntimeError(f"missing season summary for {pos} {metric}")
        by = q.set_index("sample")
        deltas = {
            s: float(by.loc[s, "mean_ae_delta_bayes_minus_playerform"])
            for s in ("2024", "2025")
        }
        if all(v > 0 for v in deltas.values()):
            status = "PLAYERFORM_BETTER"
        elif all(v < 0 for v in deltas.values()):
            status = "BAYES_BETTER"
        else:
            status = "MIXED"

        w3 = summary.loc[
            summary["position"].eq(pos)
            & summary["metric"].eq(metric)
            & summary["sample"].eq("WEEK3_ANALOGUE_CURRENT_GAMES_2_POOLED")
        ]
        if len(w3) != 1:
            raise RuntimeError(f"missing Week-3 analogue summary for {pos} {metric}")
        week3_delta = float(w3.iloc[0]["mean_ae_delta_bayes_minus_playerform"])
        per_metric.append({
            "position": pos,
            "metric": metric,
            "season_status": status,
            "delta_2024": deltas["2024"],
            "delta_2025": deltas["2025"],
            "week3_analogue_delta": week3_delta,
        })

    systemic = all(
        x["season_status"] == "PLAYERFORM_BETTER" and x["week3_analogue_delta"] > 0
        for x in per_metric
    )
    any_pf = any(x["season_status"] == "PLAYERFORM_BETTER" for x in per_metric)
    if systemic:
        disposition = "BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED"
    elif any_pf:
        disposition = "BAYESIAN_CURRENT_STATE_TRANSMISSION_METRIC_SPECIFIC_MISMATCH"
    else:
        disposition = "BAYESIAN_CURRENT_STATE_TRANSMISSION_NO_MISMATCH"
    return {"disposition": disposition, "per_metric": per_metric}


def _weight_audit() -> dict:
    out = {}
    for current_games in (1, 2, 3, 4, 5, 8):
        pf = current_games / (current_games + 4.0)
        bayes_veteran = current_games / (
            GROUP_STRENGTH["tgt_share"]
            + PRIOR_PLAYER_CAP["tgt_share"]
            + current_games
        )
        out[str(current_games)] = {
            "playerform_current_weight": float(pf),
            "bayes_current_weight_prior6plus": float(bayes_veteran),
            "bayes_to_playerform_weight_ratio": float(bayes_veteran / pf),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/research/bayesian_current_state_transmission_v1"))
    a = p.parse_args()

    seasons = {2023, 2024, 2025}
    logs = _load_normalized_logs(seasons)
    rosters = {season: _load_rosters(season) for season in (2024, 2025)}

    panel, universe_audit = build_panel(logs, rosters)
    summary = summarize(panel)
    result = classify(summary)
    result.update({
        "version": VERSION,
        "plan": "docs/research/BAYESIAN_CURRENT_STATE_TRANSMISSION_V1_PLAN.md",
        "sportsbook_inputs_used": 0,
        "outcomes_2026_used": 0,
        "parameters_fit": 0,
        "candidate_variants": 0,
        "production_changed": 0,
        "production_group_strength_tgt_share": float(GROUP_STRENGTH["tgt_share"]),
        "production_group_strength_rush_share": float(GROUP_STRENGTH["rush_share"]),
        "production_prior_cap_tgt_share": float(PRIOR_PLAYER_CAP["tgt_share"]),
        "production_prior_cap_rush_share": float(PRIOR_PLAYER_CAP["rush_share"]),
        "current_weight_audit": _weight_audit(),
        "panel_rows": int(len(panel)),
        "temporary_identity_rows_across_weekly_universes": int(universe_audit["temporary_identity_rows"].sum()),
    })

    # Integrity: the exact comparison implementations must remain the production constants.
    integrity = {
        "version": VERSION,
        "group_strength_tgt_share_exact": bool(float(GROUP_STRENGTH["tgt_share"]) == 3.0),
        "group_strength_rush_share_exact": bool(float(GROUP_STRENGTH["rush_share"]) == 3.0),
        "prior_cap_tgt_share_exact": bool(float(PRIOR_PLAYER_CAP["tgt_share"]) == 6.0),
        "prior_cap_rush_share_exact": bool(float(PRIOR_PLAYER_CAP["rush_share"]) == 6.0),
        "zero_2026_outcomes": True,
        "zero_sportsbook_inputs": True,
        "candidate_variants": 0,
        "parameters_fit": 0,
        "same_row_comparison": True,
        "target_week_outcomes_joined_after_prediction": True,
    }
    if not all(v is True for k, v in integrity.items() if isinstance(v, bool)):
        raise RuntimeError(f"integrity gate failed: {integrity}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(a.out_dir / "bayesian_current_state_transmission_panel.csv", index=False)
    summary.to_csv(a.out_dir / "bayesian_current_state_transmission_summary.csv", index=False)
    universe_audit.to_csv(a.out_dir / "bayesian_current_state_transmission_universe_audit.csv", index=False)
    (a.out_dir / "bayesian_current_state_transmission_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (a.out_dir / "bayesian_current_state_transmission_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
