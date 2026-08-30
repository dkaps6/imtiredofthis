#!/usr/bin/env python3
"""Canonical pregame Team Context v3 for 2026 production.

Team Context v3 is the downstream authority for team-level football context.
Upstream collectors may remain heterogeneous during the 2026 overhaul, but the
model consumes one 32-row artifact with explicit provenance and timing.

No sportsbook/player-prop data is read here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_season, resolve_week

DATA = Path("data")
TEAM_FORM = DATA / "team_form.csv"
QB_PROMOTED = DATA / "qb_promoted_team_context.csv"
COVERAGE = DATA / "cb_coverage_team.csv"
OUT = DATA / "team_context_v3.csv"
PROVENANCE = DATA / "team_context_v3_provenance.csv"
VERSION = "TEAM_CONTEXT_V3"

# Fields whose semantics were explicitly corrected/confirmed through M89/M90.
PROMOTED_FIELDS = (
    "true_proe",
    "neutral_pace_true",
    "pressure_rate_allowed",
    "pressure_rate_generated",
    "hit_sack_pressure_rate_allowed",
    "hit_sack_pressure_rate_generated",
    "pass_attempts_per_dropback",
    "pass_rate_off",
    "pass_rate_faced",
    "def_pass_epa_allowed",
    "def_pass_success_allowed",
    "def_ypa_allowed",
    "off_ypa",
    "off_pass_epa",
    "plays_est",
)

# These fields are supplied by the guarded TeamForm/PBP production path.  The
# P0 wrapper stamps which PBP season actually supplied them.
GUARDED_PBP_FIELDS = (
    "success_rate_off",
    "success_rate_def",
    "success_rate_diff",
    "explosive_play_rate_allowed",
)

# Useful context that is retained from the legacy merged team-form collection
# until each provider is independently certified during the provider-readiness
# phase.  Their provenance intentionally says so rather than overstating trust.
LEGACY_CONTEXT_FIELDS = (
    "def_rush_epa",
    "neutral_pace_last5",
    "seconds_per_play_last5",
    "sec_per_play_last5",
    "light_box_rate",
    "heavy_box_rate",
    "middle_open_rate",
    "rz_rate",
    "ay_per_att",
)


def _read(path: Path, *, required: bool) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"Team Context v3 required input missing/empty: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"Team Context v3 required input has zero rows: {path}")
    return df


def _canon_teams(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    x = df.copy()
    team_col = "team" if "team" in x.columns else "team_abbr" if "team_abbr" in x.columns else None
    if not team_col:
        raise RuntimeError(f"{label} missing team/team_abbr")
    x["team"] = x[team_col].map(canon_team)
    if x["team"].eq("").any():
        raise RuntimeError(f"{label} contains unresolvable team identity")
    if x.duplicated("team").any():
        dup = sorted(x.loc[x.duplicated("team", keep=False), "team"].unique().tolist())
        raise RuntimeError(f"{label} contains duplicate team rows: {dup}")
    return x


def _num(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            if pd.notna(value) and np.isfinite(float(value)):
                return float(value)
    return np.nan


def _text(row: pd.Series, names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            value = str(row.get(name)).strip()
            if value and value.lower() not in {"nan", "none", "<na>"}:
                return value
    return default


def _coverage_lookup(coverage: pd.DataFrame) -> dict[str, pd.Series]:
    if coverage.empty:
        return {}
    x = _canon_teams(coverage, label="cb_coverage_team")
    return {str(row["team"]): row for _, row in x.iterrows()}


def _provenance_row(
    *,
    team: str,
    feature: str,
    value,
    source: str,
    source_seasons: str,
    history_games: int | None,
    freshness_state: str,
    active_season: int,
    target_week: int,
) -> dict:
    return {
        "team": team,
        "feature": feature,
        "value": value,
        "source": source,
        "source_seasons": source_seasons,
        "history_games": history_games,
        "freshness_state": freshness_state,
        "active_season": int(active_season),
        "target_week": int(target_week),
        "team_context_version": VERSION,
    }


def build_team_context_v3(
    season: int,
    week: int,
    *,
    team_form: pd.DataFrame | None = None,
    qb_context: pd.DataFrame | None = None,
    coverage: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf = _canon_teams(
        _read(TEAM_FORM, required=True) if team_form is None else team_form,
        label="team_form",
    )
    qb = _canon_teams(
        _read(QB_PROMOTED, required=True) if qb_context is None else qb_context,
        label="qb_promoted_team_context",
    )
    cov = _read(COVERAGE, required=False) if coverage is None else coverage.copy()
    cov_lookup = _coverage_lookup(cov)

    if len(tf) != 32 or tf["team"].nunique() != 32:
        raise RuntimeError(f"Team Context v3 requires 32 TeamForm teams; got rows={len(tf)} teams={tf['team'].nunique()}")
    if len(qb) != 32 or qb["team"].nunique() != 32:
        raise RuntimeError(f"Team Context v3 requires 32 promoted-QB teams; got rows={len(qb)} teams={qb['team'].nunique()}")
    if set(tf["team"]) != set(qb["team"]):
        raise RuntimeError("TeamForm and promoted QB context team sets differ")

    qb_by_team = {str(row["team"]): row for _, row in qb.iterrows()}
    context_rows: list[dict] = []
    provenance_rows: list[dict] = []

    for _, tf_row in tf.iterrows():
        team = str(tf_row["team"])
        q = qb_by_team[team]
        q_season = int(pd.to_numeric(q.get("season"), errors="raise"))
        q_week = int(pd.to_numeric(q.get("week"), errors="raise"))
        if q_season != int(season) or q_week != int(week):
            raise RuntimeError(
                f"Promoted QB context runtime mismatch team={team} got={q_season}/W{q_week} expected={season}/W{week}"
            )

        hist_games = int(pd.to_numeric(q.get("qb_context_history_games"), errors="coerce") or 0)
        current_games = int(pd.to_numeric(q.get("qb_context_current_games"), errors="coerce") or 0)
        prior_games = int(pd.to_numeric(q.get("qb_context_prior_games"), errors="coerce") or 0)
        source_seasons = _text(q, ["qb_context_source_seasons"], "")
        latest_season = int(pd.to_numeric(q.get("qb_context_latest_season"), errors="coerce"))
        latest_week = int(pd.to_numeric(q.get("qb_context_latest_week"), errors="coerce"))
        if hist_games <= 0 or current_games + prior_games != hist_games:
            raise RuntimeError(
                f"Invalid promoted history accounting team={team} history={hist_games} current={current_games} prior={prior_games}"
            )
        freshness = "current_season_history" if current_games > 0 else "prior_only_preseason"

        rec: dict[str, object] = {
            "team": team,
            "team_abbr": team,
            "season": int(season),
            "week": int(week),
            "team_context_version": VERSION,
            "context_history_games": hist_games,
            "context_current_games": current_games,
            "context_prior_games": prior_games,
            "context_latest_season": latest_season,
            "context_latest_week": latest_week,
            "context_source_seasons": source_seasons,
            "context_freshness_state": freshness,
        }

        # M89/M90 promoted fields are mandatory and override any legacy aliases.
        for feature in PROMOTED_FIELDS:
            value = _num(q, [feature])
            if not np.isfinite(value):
                raise RuntimeError(f"Promoted Team Context v3 feature missing team={team} feature={feature}")
            rec[feature] = value
            provenance_rows.append(
                _provenance_row(
                    team=team,
                    feature=feature,
                    value=value,
                    source="M89_M90_PROMOTED_ROLLING_8",
                    source_seasons=source_seasons,
                    history_games=hist_games,
                    freshness_state=freshness,
                    active_season=season,
                    target_week=week,
                )
            )

        # Canonical aliases consumed by the existing rule/context contracts.
        rec["proe"] = rec["true_proe"]
        rec["neutral_pace"] = rec["neutral_pace_true"]
        rec["def_pass_epa"] = rec["def_pass_epa_allowed"]

        pbp_source = _text(tf_row, ["pbp_feature_source"], "team_form_guarded_pbp")
        pbp_season_num = pd.to_numeric(pd.Series([tf_row.get("pbp_feature_season")]), errors="coerce").iloc[0]
        pbp_season = str(int(pbp_season_num)) if pd.notna(pbp_season_num) else ""
        pbp_prior_used = int(_num(tf_row, ["pbp_prior_used"])) if np.isfinite(_num(tf_row, ["pbp_prior_used"])) else 0
        pbp_freshness = "guarded_prior_pbp" if pbp_prior_used else "guarded_current_pbp"
        for feature in GUARDED_PBP_FIELDS:
            value = _num(tf_row, [feature])
            if not np.isfinite(value):
                raise RuntimeError(f"Guarded TeamForm feature missing team={team} feature={feature}")
            rec[feature] = value
            provenance_rows.append(
                _provenance_row(
                    team=team,
                    feature=feature,
                    value=value,
                    source=pbp_source,
                    source_seasons=pbp_season,
                    history_games=None,
                    freshness_state=pbp_freshness,
                    active_season=season,
                    target_week=week,
                )
            )

        # Coverage v2 gets precedence over the legacy merged coverage aliases.
        cov_row = cov_lookup.get(team, pd.Series(dtype="object"))
        for target, names in {
            "coverage_man_rate": ["man_rate", "coverage_man_rate"],
            "coverage_zone_rate": ["zone_rate", "coverage_zone_rate"],
        }.items():
            value = _num(cov_row, names) if not cov_row.empty else np.nan
            source = "coverage_v2_current_slate"
            state = "current_pipeline_provider_semantics_pending_audit"
            if not np.isfinite(value):
                value = _num(tf_row, [target, "man_rate" if "man" in target else "zone_rate"])
                source = "team_form_legacy_merged"
                state = "legacy_merged_needs_provider_audit"
            rec[target] = value
            provenance_rows.append(
                _provenance_row(
                    team=team,
                    feature=target,
                    value=value,
                    source=source,
                    source_seasons=str(season),
                    history_games=None,
                    freshness_state=state,
                    active_season=season,
                    target_week=week,
                )
            )

        # Other retained fields remain explicitly marked as legacy until the
        # provider-readiness phase certifies each upstream source.
        for feature in LEGACY_CONTEXT_FIELDS:
            if feature in rec:
                continue
            aliases = [feature]
            if feature == "sec_per_play_last5":
                aliases.append("seconds_per_play_last5")
            if feature == "seconds_per_play_last5":
                aliases.append("sec_per_play_last5")
            value = _num(tf_row, aliases)
            rec[feature] = value
            provenance_rows.append(
                _provenance_row(
                    team=team,
                    feature=feature,
                    value=value,
                    source="team_form_legacy_merged",
                    source_seasons=str(season),
                    history_games=None,
                    freshness_state="legacy_merged_needs_provider_audit",
                    active_season=season,
                    target_week=week,
                )
            )

        context_rows.append(rec)

    context = pd.DataFrame(context_rows).sort_values("team").reset_index(drop=True)
    provenance = pd.DataFrame(provenance_rows).sort_values(["team", "feature"]).reset_index(drop=True)

    conv = pd.to_numeric(context["pass_attempts_per_dropback"], errors="coerce")
    if conv.isna().any() or not conv.between(0.50, 1.0, inclusive="both").all():
        raise RuntimeError("Team Context v3 has invalid official-attempt conversion")

    if context[["team", "season", "week"]].duplicated().any():
        raise RuntimeError("Team Context v3 contains duplicate team/runtime rows")
    if context["team"].nunique() != 32:
        raise RuntimeError("Team Context v3 does not contain all 32 canonical teams")

    forbidden = [
        c for c in context.columns
        if any(token in c.lower() for token in ("sportsbook", "bookmaker", "prop_line", "market_line", "over_odds", "under_odds"))
    ]
    if forbidden:
        raise RuntimeError(f"Sportsbook fields are prohibited from Team Context v3: {forbidden}")

    return context, provenance


def materialize(season: int | None = None, week: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_season = int(season if season is not None else resolve_season())
    target_week = int(week if week is not None else resolve_week())
    context, provenance = build_team_context_v3(active_season, target_week)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    context.to_csv(OUT, index=False)
    provenance.to_csv(PROVENANCE, index=False)
    legacy = int(provenance["freshness_state"].astype(str).str.contains("legacy_merged").sum())
    print(
        f"[team_context_v3] season={active_season} week={target_week} teams={len(context)} "
        f"provenance_rows={len(provenance)} legacy_pending_audit={legacy}"
    )
    print(f"[team_context_v3] wrote {OUT} and {PROVENANCE}")
    return context, provenance


def main() -> int:
    materialize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())