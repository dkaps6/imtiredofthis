#!/usr/bin/env python3
"""Audit FantasyAlarm WR-CB archive timing, coverage, and stable identity.

Source-quality only. No model fitting, no outcomes, no sportsbook inputs.
Missing rows are never interpreted as zero exposure or no matchup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe


DEF_POSITIONS = {"CB", "DB", "S", "FS", "SS"}
WR_POSITIONS = {"WR", "LWR", "RWR", "SWR"}



def _content_version_timing(x: pd.DataFrame) -> pd.Series:
    """Classify article-version timing; metadata is not archived-table proof."""
    out = pd.Series("UNVERIFIED_NO_MODIFICATION_METADATA", index=x.index, dtype="string")
    pub = pd.to_datetime(x["published_at_utc"], utc=True, errors="coerce")
    kickoff = pd.to_datetime(x["kickoff_utc"], utc=True, errors="coerce")
    modified = pd.to_datetime(
        x["modified_at_utc"] if "modified_at_utc" in x else pd.Series("", index=x.index),
        utc=True, errors="coerce",
    )
    meta = (
        x["modification_metadata_status"].astype("string").fillna("")
        if "modification_metadata_status" in x
        else pd.Series("", index=x.index, dtype="string")
    )
    out.loc[meta.eq("CONFLICTING_MODIFICATION_METADATA")] = "UNVERIFIED_CONFLICTING_MODIFICATION_METADATA"
    out.loc[meta.eq("INVALID_MODIFICATION_METADATA")] = "UNVERIFIED_INVALID_MODIFICATION_METADATA"
    valid = (
        meta.eq("UNAMBIGUOUS_MODIFICATION_METADATA")
        & pub.notna() & kickoff.notna() & modified.notna()
    )
    out.loc[valid & modified.lt(pub)] = "UNVERIFIED_MODIFIED_BEFORE_PUBLICATION"
    out.loc[valid & modified.ge(pub) & modified.le(kickoff)] = "METADATA_PRE_KICKOFF_COMPATIBLE_NOT_SNAPSHOT_PROOF"
    out.loc[valid & modified.ge(pub) & modified.gt(kickoff)] = "MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED"
    out.loc[kickoff.isna()] = "MISSING_KICKOFF"
    out.loc[pub.isna()] = "MISSING_PUBLICATION_TIME"
    return out


def _provider_bridge(x: pd.DataFrame, source_col: str, gsis_col: str) -> tuple[dict[str, str], set[str]]:
    """Use only pre-kickoff, schedule-consistent anchor identities.

    Quarantined page rows must not teach provider-ID mappings used to declare
    other pre-kickoff source rows ready. Identity metadata, not football input.
    """
    anchor = x.loc[
        x["schedule_match"]
        & x["publication_timing_status"].eq("PRE_KICKOFF")
        & x[source_col].astype(str).str.len().gt(0)
        & x[gsis_col].astype(str).str.len().gt(0),
        [source_col, gsis_col],
    ].drop_duplicates()
    grouped = anchor.groupby(source_col)[gsis_col].agg(
        lambda z: tuple(sorted(set(str(v) for v in z if str(v))))
    )
    collisions = {str(k) for k, ids in grouped.items() if len(ids) != 1}
    mapping = {str(k): ids[0] for k, ids in grouped.items() if len(ids) == 1}
    return mapping, collisions



def _to_pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _first(df: pd.DataFrame, names: list[str], default="") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _name_key(value) -> str:
    return str(canonicalize_player_name_safe(value)[1] or "").strip()


def _position(value) -> str:
    x = "" if value is None or pd.isna(value) else str(value).strip().upper()
    if x.startswith("WR"):
        return "WR"
    if x in {"LWR", "RWR", "SWR"}:
        return "WR"
    if x.startswith("CB"):
        return "CB"
    return x


def _load_schedule(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    frames: list[pd.DataFrame] = []
    for season in seasons:
        raw = _lower(_to_pd(nfl.load_schedules(int(season))))
        if raw.empty:
            raise RuntimeError(f"schedule source empty season={season}")
        if "game_type" in raw.columns:
            raw = raw.loc[raw["game_type"].astype(str).str.upper().eq("REG")].copy()
        raw["season"] = pd.to_numeric(_first(raw, ["season"], season), errors="coerce").fillna(season).astype(int)
        raw["week"] = pd.to_numeric(_first(raw, ["week"]), errors="coerce")

        kickoff = None
        for col in ["game_datetime", "datetime", "start_time"]:
            if col in raw.columns:
                candidate = pd.to_datetime(raw[col], utc=True, errors="coerce")
                if candidate.notna().any():
                    kickoff = candidate
                    break
        if kickoff is None:
            if "gameday" not in raw.columns or "gametime" not in raw.columns:
                raise RuntimeError(f"schedule lacks kickoff timestamp fields season={season}")
            local = pd.to_datetime(
                raw["gameday"].astype(str).str.strip()
                + " "
                + raw["gametime"].astype(str).str.strip(),
                errors="coerce",
            )
            kickoff = local.dt.tz_localize(
                ZoneInfo("America/New_York"),
                ambiguous="raise",
                nonexistent="raise",
            ).dt.tz_convert("UTC")
        raw["kickoff_utc"] = kickoff
        home = _first(raw, ["home_team", "home"]).map(canon_team)
        away = _first(raw, ["away_team", "away"]).map(canon_team)
        base = raw[["season", "week", "kickoff_utc"]].copy()
        h = base.copy()
        h["team"] = home
        h["scheduled_opponent"] = away
        a = base.copy()
        a["team"] = away
        a["scheduled_opponent"] = home
        frames.extend([h, a])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.loc[
        out["week"].notna()
        & out["kickoff_utc"].notna()
        & out["team"].astype(str).str.len().gt(0)
    ].copy()
    out["week"] = out["week"].astype(int)
    dup = out.groupby(["season", "week", "team"])["kickoff_utc"].nunique()
    bad = dup.loc[dup.ne(1)]
    if not bad.empty:
        raise RuntimeError(f"non-unique regular-season team/week kickoff: {bad.head(20).to_dict()}")
    return out.drop_duplicates(["season", "week", "team"], keep="last")


def _load_rosters(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    frames: list[pd.DataFrame] = []
    for season in seasons:
        raw = _lower(_to_pd(nfl.load_rosters_weekly(int(season))))
        if raw.empty:
            raise RuntimeError(f"weekly roster source empty season={season}")
        raw["season"] = pd.to_numeric(_first(raw, ["season"], season), errors="coerce").fillna(season).astype(int)
        raw["week"] = pd.to_numeric(_first(raw, ["week"]), errors="coerce")
        raw["team"] = _first(raw, ["team", "team_abbr", "club_code"]).map(canon_team)
        raw["position"] = _first(raw, ["position", "pos"]).map(_position)
        raw["player_name"] = _first(
            raw, ["full_name", "football_name", "player_name", "player", "name"]
        ).astype("string").fillna("").str.strip()
        raw["name_key"] = raw["player_name"].map(_name_key)
        raw["player_id"] = _first(raw, ["gsis_id", "player_id"]).astype("string").fillna("").str.strip()
        raw = raw.loc[
            raw["week"].between(1, 18, inclusive="both")
            & raw["team"].astype(str).str.len().gt(0)
            & raw["name_key"].astype(str).str.len().gt(0)
            & raw["player_id"].astype(str).str.len().gt(0)
        ].copy()
        raw["week"] = raw["week"].astype(int)
        frames.append(raw[["season", "week", "team", "position", "name_key", "player_id"]])
    return pd.concat(frames, ignore_index=True, sort=False)


def _identity_lookup(rosters: pd.DataFrame, *, positions: set[str]) -> dict[tuple[int, int, str, str], tuple[str, str]]:
    eligible = rosters.loc[rosters["position"].isin(positions)].copy()
    grouped = eligible.groupby(["season", "week", "team", "name_key"])["player_id"].agg(
        lambda s: tuple(sorted(set(str(x) for x in s if str(x))))
    )
    season_grouped = eligible.groupby(["season", "team", "name_key"])["player_id"].agg(
        lambda s: tuple(sorted(set(str(x) for x in s if str(x))))
    )
    out: dict[tuple[int, int, str, str], tuple[str, str]] = {}
    for key, ids in grouped.items():
        if len(ids) == 1:
            out[key] = (ids[0], "WEEK_EXACT")
        elif len(ids) > 1:
            out[key] = ("", "COLLISION")
    # Fill only missing exact-week keys from a season-wide unique person identity.
    for (season, team, name), ids in season_grouped.items():
        if len(ids) != 1:
            continue
        weeks = eligible.loc[
            eligible["season"].eq(season)
            & eligible["team"].eq(team)
            & eligible["name_key"].eq(name),
            "week",
        ].unique()
        for week in range(1, 19):
            key = (int(season), int(week), str(team), str(name))
            if key not in out:
                out[key] = (ids[0], "SEASON_UNIQUE_NAME_TEAM")
    return out


def audit(assignments: pd.DataFrame, page_audit: pd.DataFrame, out_dir: Path) -> dict:
    if assignments.empty:
        raise RuntimeError("WR-CB source audit has zero assignment rows")
    x = assignments.copy()
    for c in ["season", "week"]:
        x[c] = pd.to_numeric(x[c], errors="raise").astype(int)
    seasons = sorted(x["season"].unique().tolist())

    schedule = _load_schedule(seasons)
    schedule = schedule.rename(columns={"team": "wr_team"})
    x = x.merge(schedule, on=["season", "week", "wr_team"], how="left", validate="many_to_one")
    x["schedule_match"] = (
        x["scheduled_opponent"].astype(str).str.len().gt(0)
        & x["opponent"].astype(str).eq(x["scheduled_opponent"].astype(str))
    )
    x["published_utc"] = pd.to_datetime(x["published_at_utc"], utc=True, errors="coerce")
    x["content_version_timing_status"] = _content_version_timing(x)
    x["content_version_metadata_pregame_compatible"] = x["content_version_timing_status"].eq(
        "METADATA_PRE_KICKOFF_COMPATIBLE_NOT_SNAPSHOT_PROOF"
    )
    # Neither original publication nor site-reported last edit proves archived
    # factual tables existed unchanged before kickoff. Requires independent
    # contemporaneous snapshot evidence; NOT available in this acquisition.
    x["historical_content_snapshot_verified"] = False
    x["publication_timing_status"] = "PRE_KICKOFF"
    x.loc[x["published_utc"].isna(), "publication_timing_status"] = "MISSING_PUBLICATION_TIME"
    x.loc[x["kickoff_utc"].isna(), "publication_timing_status"] = "MISSING_KICKOFF"
    valid = x["published_utc"].notna() & x["kickoff_utc"].notna()
    x.loc[valid & x["published_utc"].gt(x["kickoff_utc"]), "publication_timing_status"] = "AFTER_KICKOFF"

    rosters = _load_rosters(seasons)
    wr_lookup = _identity_lookup(rosters, positions=WR_POSITIONS)
    cb_lookup = _identity_lookup(rosters, positions=DEF_POSITIONS)

    wr_ids, wr_methods, cb_ids, cb_methods = [], [], [], []
    for row in x.itertuples(index=False):
        wr_key = (int(row.season), int(row.week), str(row.wr_team), str(row.wr_clean_key))
        cb_key = (int(row.season), int(row.week), str(row.opponent), str(row.cb_clean_key))
        wid, wm = wr_lookup.get(wr_key, ("", "UNRESOLVED"))
        cid, cm = cb_lookup.get(cb_key, ("", "UNRESOLVED"))
        wr_ids.append(wid); wr_methods.append(wm)
        cb_ids.append(cid); cb_methods.append(cm)
    x["wr_gsis_id"] = wr_ids
    x["wr_identity_method"] = wr_methods
    x["cb_gsis_id"] = cb_ids
    x["cb_identity_method"] = cb_methods

    # Provider-stable player IDs are identity metadata, never football features.
    # Learn a bridge ONLY from schedule-consistent rows where our roster join
    # already identified exactly one GSIS person. Then propagate that identity
    # only when one FantasyAlarm ID maps to exactly one GSIS ID.
    for col in ["wr_source_player_id", "cb_source_player_id"]:
        if col not in x.columns:
            x[col] = ""
        x[col] = x[col].astype("string").fillna("").str.strip()

    wr_provider_map, wr_provider_collisions = _provider_bridge(x, "wr_source_player_id", "wr_gsis_id")
    cb_provider_map, cb_provider_collisions = _provider_bridge(x, "cb_source_player_id", "cb_gsis_id")

    for source_col, gsis_col, method_col, mapping, collisions in [
        ("wr_source_player_id", "wr_gsis_id", "wr_identity_method", wr_provider_map, wr_provider_collisions),
        ("cb_source_player_id", "cb_gsis_id", "cb_identity_method", cb_provider_map, cb_provider_collisions),
    ]:
        unresolved = x[gsis_col].astype(str).str.len().eq(0)
        bridged = x[source_col].map(mapping).astype("string").fillna("")
        eligible = (
            unresolved
            & x[source_col].astype(str).str.len().gt(0)
            & ~x[source_col].isin(collisions)
            & bridged.str.len().gt(0)
        )
        x.loc[eligible, gsis_col] = bridged.loc[eligible]
        x.loc[eligible, method_col] = "FANTASYALARM_STABLE_ID_BRIDGE"

    x["stable_identity_ready"] = x["wr_gsis_id"].astype(str).str.len().gt(0) & x["cb_gsis_id"].astype(str).str.len().gt(0)
    # This is a SOURCE-quality eligibility flag only. It is not a model input
    # and does not authorize a matchup candidate. Rows must be explicitly
    # observed, pre-kickoff, schedule-consistent, stably identified, and aligned.
    x["source_quality_row_ready"] = (
        x["publication_timing_status"].eq("PRE_KICKOFF")
        & x["schedule_match"]
        & x["stable_identity_ready"]
        & x["alignment_bucket"].ne("UNKNOWN_ALIGNMENT")
    )

    # Person-key stability: a canonical source name may never map to >1 stable ID.
    wr_collision = x.loc[x["wr_gsis_id"].astype(str).str.len().gt(0)].groupby("wr_clean_key")["wr_gsis_id"].nunique()
    cb_collision = x.loc[x["cb_gsis_id"].astype(str).str.len().gt(0)].groupby("cb_clean_key")["cb_gsis_id"].nunique()
    wr_bad = wr_collision.loc[wr_collision.gt(1)]
    cb_bad = cb_collision.loc[cb_collision.gt(1)]

    page_rows = []
    for (season, week), g in x.groupby(["season", "week"], sort=True):
        counts = g["alignment_bucket"].value_counts().to_dict()
        page_rows.append({
            "season": int(season),
            "week": int(week),
            "rows": int(len(g)),
            "outside_rows": int(counts.get("LWR_VS_RCB", 0) + counts.get("RWR_VS_LCB", 0)),
            "slot_rows": int(counts.get("SWR_VS_SCB", 0)),
            "unknown_alignment_rows": int(counts.get("UNKNOWN_ALIGNMENT", 0)),
            "unique_wr_teams": int(g["wr_team"].nunique()),
            "unique_opponent_teams": int(g["opponent"].nunique()),
            "pre_kickoff_rows": int(g["publication_timing_status"].eq("PRE_KICKOFF").sum()),
            "after_kickoff_rows": int(g["publication_timing_status"].eq("AFTER_KICKOFF").sum()),
            "missing_timing_rows": int(g["publication_timing_status"].isin(["MISSING_PUBLICATION_TIME", "MISSING_KICKOFF"]).sum()),
            "schedule_match_rows": int(g["schedule_match"].sum()),
            "schedule_mismatch_rows": int((~g["schedule_match"]).sum()),
            "stable_identity_rows": int(g["stable_identity_ready"].sum()),
            "stable_identity_rate": float(g["stable_identity_ready"].mean()),
            "source_quality_ready_rows": int(g["source_quality_row_ready"].sum()),
            "content_version_metadata_pregame_compatible_rows": int(g["content_version_metadata_pregame_compatible"].sum()),
            "modified_after_game_kickoff_rows": int(g["content_version_timing_status"].eq("MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED").sum()),
            "source_quality_ready_rate": float(g["source_quality_row_ready"].mean()),
        })
    pages = pd.DataFrame(page_rows)

    expected = {int(s): 18 for s in seasons}
    if 2026 in expected:
        expected[2026] = max(3, int(x.loc[x["season"].eq(2026), "week"].max()))
    manifest_weeks = pages.groupby("season")["week"].nunique().to_dict()
    season_rows = []
    for season in seasons:
        g = x.loc[x["season"].eq(season)]
        season_rows.append({
            "season": int(season),
            "observed_weeks": int(manifest_weeks.get(season, 0)),
            "expected_weeks_through_scope": int(expected[season]),
            "week_coverage_rate": float(manifest_weeks.get(season, 0) / expected[season]),
            "rows": int(len(g)),
            "outside_rows": int(g["alignment_bucket"].isin(["LWR_VS_RCB", "RWR_VS_LCB"]).sum()),
            "slot_rows": int(g["alignment_bucket"].eq("SWR_VS_SCB").sum()),
            "stable_identity_rate": float(g["stable_identity_ready"].mean()),
            "schedule_mismatch_rows": int((~g["schedule_match"]).sum()),
            "source_quality_ready_rows": int(g["source_quality_row_ready"].sum()),
            "source_quality_ready_rate": float(g["source_quality_row_ready"].mean()),
            "content_version_metadata_pregame_compatible_rows": int(g["content_version_metadata_pregame_compatible"].sum()),
            "modified_after_game_kickoff_rows": int(g["content_version_timing_status"].eq("MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED").sum()),
            "after_kickoff_rows": int(g["publication_timing_status"].eq("AFTER_KICKOFF").sum()),
            "missing_timing_rows": int(g["publication_timing_status"].isin(["MISSING_PUBLICATION_TIME", "MISSING_KICKOFF"]).sum()),
        })
    season_df = pd.DataFrame(season_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(out_dir / "fantasyalarm_wr_cb_source_quality_rows.csv", index=False)
    pages.to_csv(out_dir / "fantasyalarm_wr_cb_source_quality_pages.csv", index=False)
    season_df.to_csv(out_dir / "fantasyalarm_wr_cb_source_quality_seasons.csv", index=False)

    full_week_coverage = bool(
        season_df.loc[season_df["season"].isin([2021, 2022, 2023, 2024, 2025]), "week_coverage_rate"].eq(1.0).all()
    )
    summary = {
        "contract": "WR_CB_FREE_HISTORICAL_ARCHIVE_SOURCE_QUALITY_V1",
        "disposition": "SOURCE_QUALITY_AUDIT_IN_PROGRESS",
        "source_gate_cleared": False,
        "rows": int(len(x)),
        "pages": int(len(pages)),
        "seasons": seasons,
        "historical_2021_2025_full_week_coverage": full_week_coverage,
        "pre_kickoff_rows": int(x["publication_timing_status"].eq("PRE_KICKOFF").sum()),
        "after_kickoff_rows": int(x["publication_timing_status"].eq("AFTER_KICKOFF").sum()),
        "missing_timing_rows": int(x["publication_timing_status"].isin(["MISSING_PUBLICATION_TIME", "MISSING_KICKOFF"]).sum()),
        "stable_identity_rows": int(x["stable_identity_ready"].sum()),
        "stable_identity_rate": float(x["stable_identity_ready"].mean()),
        "schedule_match_rows": int(x["schedule_match"].sum()),
        "schedule_mismatch_rows": int((~x["schedule_match"]).sum()),
        "source_quality_ready_rows": int(x["source_quality_row_ready"].sum()),
        "source_quality_ready_means_publication_eligibility_only": True,
        "content_version_status_counts": x["content_version_timing_status"].value_counts().to_dict(),
        "content_version_metadata_pregame_compatible_rows": int(x["content_version_metadata_pregame_compatible"].sum()),
        "historical_content_snapshot_verified_rows": 0,
        "source_quality_ready_rate": float(x["source_quality_row_ready"].mean()),
        "wr_name_to_stable_id_collisions": int(len(wr_bad)),
        "cb_name_to_stable_id_collisions": int(len(cb_bad)),
        "wr_provider_id_to_gsis_collisions": int(len(wr_provider_collisions)),
        "cb_provider_id_to_gsis_collisions": int(len(cb_provider_collisions)),
        "wr_provider_id_bridge_rows": int(x["wr_identity_method"].eq("FANTASYALARM_STABLE_ID_BRIDGE").sum()),
        "cb_provider_id_bridge_rows": int(x["cb_identity_method"].eq("FANTASYALARM_STABLE_ID_BRIDGE").sum()),
        "unknown_alignment_rows": int(x["alignment_bucket"].eq("UNKNOWN_ALIGNMENT").sum()),
        "editorial_matchup_used_as_model_feature": False,
        "sportsbook_inputs_used": False,
        "target_game_outcomes_used": False,
        "model_candidates_scored": 0,
        "parameters_fit": 0,
        "missing_rows_interpreted_as_zero": False,
        "note": "Publication-eligible source-quality rows are provisional: source dateModified can be post-kickoff, and even pre-kickoff metadata alone does not prove archived table contents. Historical snapshot proof + coverage/identity/parsing required. Gate CLOSED.",
    }
    (out_dir / "fantasyalarm_wr_cb_source_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", type=Path, required=True)
    ap.add_argument("--page-audit", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    assignments = pd.read_csv(args.assignments, low_memory=False)
    page_audit = pd.read_csv(args.page_audit, low_memory=False)
    audit(assignments, page_audit, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
