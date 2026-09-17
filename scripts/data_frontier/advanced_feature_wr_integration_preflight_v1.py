"""WR integration preflight for Advanced Feature Materializer V1.

Purpose:
- certify a BDB2026 tracking-id -> nflverse/GSIS identity bridge;
- join strict-prior receiver history snapshots to 2023 weekly receiving outcomes;
- report coverage only.

This script does NOT fit/evaluate a predictive model and does not touch production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.data_frontier.advanced_feature_materializer_common_v1 import (
    ALGORITHM_VERSION,
    assert_strict_prior,
    sha256_file,
)

VERSION = "NFL_ADVANCED_WR_INTEGRATION_PREFLIGHT_V1"


def clean_name(value: object) -> str:
    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("’", "'")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?$", "", s).strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = set(columns)
    return next((c for c in candidates if c in cols), None)


def height_inches(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        x = float(s)
        return x if x > 20 else x * 12.0
    m = re.fullmatch(r"(\d+)\s*[-']\s*(\d+)", s)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    return np.nan


def normalize_position(value: object) -> str:
    s = "" if value is None else str(value).upper().strip()
    mapping = {"HB": "RB", "FB": "RB"}
    return mapping.get(s, s)


def extract_bdb_target_identities(corpus_dir: Path) -> tuple[pd.DataFrame, dict]:
    records = []
    schema = {}
    identity_candidates = [
        "player_name",
        "display_name",
        "name",
        "player_position",
        "position",
        "player_height",
        "height",
        "player_weight",
        "weight",
        "player_birth_date",
        "birth_date",
        "team",
        "club",
        "team_abbr",
    ]
    for week in range(1, 19):
        path = next(corpus_dir.rglob(f"input_2023_w{week:02d}.csv"))
        header = pd.read_csv(path, nrows=0)
        schema[str(week)] = list(header.columns)
        needed = ["game_id", "play_id", "nfl_id", "player_role"] + [
            c for c in identity_candidates if c in header.columns
        ]
        df = pd.read_csv(path, usecols=list(dict.fromkeys(needed)))
        t = df.loc[df["player_role"].eq("Targeted Receiver")].drop_duplicates()
        t["week"] = week
        records.append(t)

    raw = pd.concat(records, ignore_index=True)
    if raw.empty:
        raise RuntimeError("BDB2026 identity preflight found zero Targeted Receiver rows")

    cols = list(raw.columns)
    name_col = first_existing(cols, ["player_name", "display_name", "name"])
    position_col = first_existing(cols, ["player_position", "position"])
    height_col = first_existing(cols, ["player_height", "height"])
    weight_col = first_existing(cols, ["player_weight", "weight"])
    birth_col = first_existing(cols, ["player_birth_date", "birth_date"])
    team_col = first_existing(cols, ["team", "club", "team_abbr"])

    if name_col is None:
        raise RuntimeError(f"BDB2026 Targeted Receiver rows expose no usable name column: {cols}")

    keep = ["nfl_id", name_col]
    for c in [position_col, height_col, weight_col, birth_col, team_col]:
        if c and c not in keep:
            keep.append(c)
    identity_rows = raw[keep].drop_duplicates().copy()

    rename = {name_col: "bdb_name"}
    if position_col:
        rename[position_col] = "bdb_position"
    if height_col:
        rename[height_col] = "bdb_height"
    if weight_col:
        rename[weight_col] = "bdb_weight"
    if birth_col:
        rename[birth_col] = "bdb_birth_date"
    if team_col:
        rename[team_col] = "bdb_team"
    identity_rows = identity_rows.rename(columns=rename)

    # Collapse stable source metadata at nfl_id grain and surface conflicts.
    out = []
    conflicts = {}
    for nfl_id, g in identity_rows.groupby("nfl_id", sort=False):
        row = {"bdb_nfl_id": str(nfl_id)}
        for c in ["bdb_name", "bdb_position", "bdb_height", "bdb_weight", "bdb_birth_date", "bdb_team"]:
            if c not in g.columns:
                row[c] = np.nan
                continue
            vals = [x for x in g[c].dropna().astype(str).str.strip().unique().tolist() if x]
            if len(vals) > 1:
                conflicts[c] = conflicts.get(c, 0) + 1
            row[c] = vals[0] if len(vals) == 1 else (vals[0] if vals else np.nan)
        out.append(row)
    identities = pd.DataFrame(out)
    identities["bdb_name_key"] = identities["bdb_name"].map(clean_name)
    identities["bdb_position_norm"] = identities["bdb_position"].map(normalize_position)
    identities["bdb_height_inches"] = identities["bdb_height"].map(height_inches)
    identities["bdb_weight_num"] = pd.to_numeric(identities["bdb_weight"], errors="coerce")
    identities["bdb_birth_date_norm"] = pd.to_datetime(
        identities["bdb_birth_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    audit = {
        "targeted_receiver_rows": int(len(raw)),
        "unique_bdb_nfl_ids": int(identities["bdb_nfl_id"].nunique()),
        "identity_columns_available": {
            "name": name_col,
            "position": position_col,
            "height": height_col,
            "weight": weight_col,
            "birth_date": birth_col,
            "team": team_col,
        },
        "identity_metadata_conflicts_by_field": conflicts,
        "input_schema_union": sorted({c for cols in schema.values() for c in cols}),
    }
    return identities, audit


def prepare_nflverse_players(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, low_memory=False)
    p.columns = [str(c).strip().lower() for c in p.columns]
    if "gsis_id" not in p.columns:
        raise RuntimeError("nflverse players table missing gsis_id")
    name_cols = [
        c
        for c in [
            "display_name",
            "football_name",
            "common_first_name",
            "first_name",
            "last_name",
            "short_name",
        ]
        if c in p.columns
    ]
    if "display_name" not in p.columns:
        raise RuntimeError("nflverse players table missing display_name")

    p["position_norm"] = (
        p[first_existing(p.columns, ["position", "ngs_position", "position_group"]) or "position"]
        .map(normalize_position)
    )
    p["height_inches"] = (
        pd.to_numeric(p["height"], errors="coerce") if "height" in p.columns else np.nan
    )
    p["weight_num"] = (
        pd.to_numeric(p["weight"], errors="coerce") if "weight" in p.columns else np.nan
    )
    p["birth_date_norm"] = (
        pd.to_datetime(p["birth_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "birth_date" in p.columns
        else ""
    )
    p["nfl_id_str"] = p["nfl_id"].astype(str).str.strip() if "nfl_id" in p.columns else ""

    alias_rows = []
    for idx, row in p.iterrows():
        aliases = set()
        for c in name_cols:
            value = row.get(c)
            if pd.notna(value):
                key = clean_name(value)
                if key:
                    aliases.add(key)
        # first + last combination is useful when display_name is missing/odd.
        if "first_name" in p.columns and "last_name" in p.columns:
            key = clean_name(f"{row.get('first_name', '')} {row.get('last_name', '')}")
            if key:
                aliases.add(key)
        for alias in aliases:
            alias_rows.append(
                {
                    "player_row": idx,
                    "name_key": alias,
                    "gsis_id": str(row["gsis_id"]).strip(),
                    "display_name": row.get("display_name"),
                    "position_norm": row["position_norm"],
                    "height_inches": row["height_inches"],
                    "weight_num": row["weight_num"],
                    "birth_date_norm": row["birth_date_norm"],
                    "nfl_id_str": row["nfl_id_str"],
                }
            )
    return pd.DataFrame(alias_rows)


def certify_crosswalk(bdb: pd.DataFrame, nfl_alias: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    method_counts = {}
    for _, b in bdb.iterrows():
        b_id = str(b["bdb_nfl_id"])
        direct = nfl_alias.loc[
            nfl_alias["nfl_id_str"].eq(b_id) & nfl_alias["gsis_id"].ne("")
        ].drop_duplicates("gsis_id")

        method = "UNRESOLVED"
        gsis = ""
        candidates = 0
        if len(direct) == 1:
            method = "DIRECT_NFL_ID"
            gsis = str(direct.iloc[0]["gsis_id"])
            candidates = 1
        else:
            cand = nfl_alias.loc[
                nfl_alias["name_key"].eq(b["bdb_name_key"]) & nfl_alias["gsis_id"].ne("")
            ].drop_duplicates("gsis_id")
            bpos = b.get("bdb_position_norm")
            if pd.notna(bpos) and str(bpos).strip():
                pos = cand.loc[cand["position_norm"].eq(str(bpos).strip())]
                if not pos.empty:
                    cand = pos

            # Strongest corroboration: birth date.
            dob = str(b.get("bdb_birth_date_norm") or "")
            if dob and dob != "nan":
                dob_c = cand.loc[cand["birth_date_norm"].eq(dob)]
                if len(dob_c) == 1:
                    method = "NAME_DOB_POSITION"
                    gsis = str(dob_c.iloc[0]["gsis_id"])
                    candidates = 1

            # Next: unique name/position plus physical corroboration.
            if not gsis and len(cand):
                h = pd.to_numeric(pd.Series([b.get("bdb_height_inches")]), errors="coerce").iloc[0]
                w = pd.to_numeric(pd.Series([b.get("bdb_weight_num")]), errors="coerce").iloc[0]
                phys = cand.copy()
                if pd.notna(h):
                    phys = phys.loc[pd.to_numeric(phys["height_inches"], errors="coerce").eq(float(h))]
                if pd.notna(w):
                    phys = phys.loc[(pd.to_numeric(phys["weight_num"], errors="coerce") - float(w)).abs().le(3)]
                phys = phys.drop_duplicates("gsis_id")
                if len(phys) == 1 and (pd.notna(h) or pd.notna(w)):
                    method = "NAME_POSITION_PHYSICAL"
                    gsis = str(phys.iloc[0]["gsis_id"])
                    candidates = 1

            if not gsis:
                candidates = int(cand["gsis_id"].nunique()) if len(cand) else 0
                if candidates == 1:
                    method = "NAME_POSITION_UNIQUE_REVIEW_ONLY"

        certified = method in {
            "DIRECT_NFL_ID",
            "NAME_DOB_POSITION",
            "NAME_POSITION_PHYSICAL",
        }
        method_counts[method] = method_counts.get(method, 0) + 1
        rows.append(
            {
                "bdb_nfl_id": b_id,
                "bdb_name": b.get("bdb_name"),
                "bdb_position": b.get("bdb_position_norm"),
                "gsis_id": gsis,
                "identity_method": method,
                "identity_certified": certified,
                "candidate_count": candidates,
            }
        )
    x = pd.DataFrame(rows)
    audit = {
        "unique_bdb_ids": int(len(x)),
        "certified_ids": int(x["identity_certified"].sum()),
        "certified_rate": float(x["identity_certified"].mean()) if len(x) else 0.0,
        "review_only_ids": int(x["identity_method"].eq("NAME_POSITION_UNIQUE_REVIEW_ONLY").sum()),
        "unresolved_ids": int(x["identity_method"].eq("UNRESOLVED").sum()),
        "method_counts": dict(sorted(method_counts.items())),
        "direct_id_match_rate": float(x["identity_method"].eq("DIRECT_NFL_ID").mean()) if len(x) else 0.0,
    }
    return x, audit


def weekly_outcomes_2023() -> tuple[pd.DataFrame, dict]:
    w = load_weekly_player_stats(2023).copy()
    w.columns = [str(c).strip().lower() for c in w.columns]
    id_col = first_existing(w.columns, ["player_id", "gsis_id", "player_gsis_id"])
    rec_col = first_existing(w.columns, ["receiving_yards", "rec_yards"])
    tgt_col = first_existing(w.columns, ["targets"])
    recn_col = first_existing(w.columns, ["receptions"])
    pos_col = first_existing(w.columns, ["position", "position_group"])
    name_col = first_existing(w.columns, ["player_display_name", "player_name", "display_name", "player"])
    if id_col is None or rec_col is None:
        raise RuntimeError(f"weekly stats missing identity/receiving columns: {list(w.columns)}")
    out = pd.DataFrame(
        {
            "week": pd.to_numeric(w["week"], errors="coerce"),
            "gsis_id": w[id_col].astype(str).str.strip(),
            "rec_yards": pd.to_numeric(w[rec_col], errors="coerce"),
            "targets": pd.to_numeric(w[tgt_col], errors="coerce") if tgt_col else np.nan,
            "receptions": pd.to_numeric(w[recn_col], errors="coerce") if recn_col else np.nan,
            "position": w[pos_col].map(normalize_position) if pos_col else "",
            "player_name": w[name_col] if name_col else "",
        }
    )
    out = out.loc[out["week"].notna() & out["gsis_id"].ne("")].copy()
    out["week"] = out["week"].astype(int)
    out = out.loc[out["week"].between(1, 18)].copy()
    audit = {
        "weekly_rows": int(len(out)),
        "unique_gsis_ids": int(out["gsis_id"].nunique()),
        "weeks": sorted(map(int, out["week"].unique().tolist())),
        "identity_column": id_col,
        "receiving_yards_column": rec_col,
        "position_column": pos_col,
    }
    return out, audit


def build_integration_panel(
    history_path: Path,
    crosswalk: pd.DataFrame,
    weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    hist = pd.read_csv(history_path, low_memory=False)
    hist["nfl_id"] = hist["nfl_id"].astype(str).str.strip()
    assert_strict_prior(hist)
    cert = crosswalk.loc[crosswalk["identity_certified"] & crosswalk["gsis_id"].ne("")].copy()
    x = hist.merge(cert[["bdb_nfl_id", "gsis_id"]], left_on="nfl_id", right_on="bdb_nfl_id", how="left")
    x["identity_certified"] = x["gsis_id"].notna() & x["gsis_id"].astype(str).ne("")
    joined = x.merge(
        weekly,
        left_on=["target_week", "gsis_id"],
        right_on=["week", "gsis_id"],
        how="left",
        suffixes=("", "_outcome"),
    )
    advanced_fields = [
        "hist_receiver_release_nearest_defender_median_yards",
        "hist_receiver_release_second_defender_median_yards",
        "hist_receiver_release_crowding_2yd_rate",
        "hist_receiver_release_crowding_3yd_rate",
    ]
    joined["advanced_history_ready"] = joined[advanced_fields].notna().all(axis=1)
    joined["receiving_outcome_present"] = joined["rec_yards"].notna()
    joined["wr_outcome"] = joined["position"].eq("WR")
    joined["wr_candidate_ready"] = (
        joined["identity_certified"]
        & joined["advanced_history_ready"]
        & joined["receiving_outcome_present"]
        & joined["wr_outcome"]
    )

    per_week = (
        joined.groupby("target_week", dropna=False)
        .agg(
            history_rows=("nfl_id", "size"),
            certified_identity_rows=("identity_certified", "sum"),
            advanced_history_ready_rows=("advanced_history_ready", "sum"),
            receiving_outcome_rows=("receiving_outcome_present", "sum"),
            wr_candidate_ready_rows=("wr_candidate_ready", "sum"),
        )
        .reset_index()
    )

    audit = {
        "history_snapshot_rows": int(len(hist)),
        "history_unique_bdb_ids": int(hist["nfl_id"].nunique()),
        "certified_identity_history_rows": int(joined["identity_certified"].sum()),
        "advanced_history_ready_rows": int(joined["advanced_history_ready"].sum()),
        "receiving_outcome_join_rows": int(joined["receiving_outcome_present"].sum()),
        "wr_candidate_ready_rows": int(joined["wr_candidate_ready"].sum()),
        "wr_candidate_unique_players": int(joined.loc[joined["wr_candidate_ready"], "gsis_id"].nunique()),
        "wr_candidate_weeks": sorted(
            map(int, joined.loc[joined["wr_candidate_ready"], "target_week"].dropna().unique().tolist())
        ),
        "per_week": per_week.to_dict(orient="records"),
        "predictive_metrics_calculated": False,
    }
    return joined, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--materialized-dir", type=Path, required=True)
    p.add_argument("--nflverse-players", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    out = args.out_dir
    private = out / "private"
    sanitized = out / "sanitized"
    private.mkdir(parents=True, exist_ok=True)
    sanitized.mkdir(parents=True, exist_ok=True)

    bdb_ids, bdb_audit = extract_bdb_target_identities(args.corpus_dir)
    nfl_alias = prepare_nflverse_players(args.nflverse_players)
    crosswalk, crosswalk_audit = certify_crosswalk(bdb_ids, nfl_alias)
    weekly, weekly_audit = weekly_outcomes_2023()

    history_path = args.materialized_dir / "private" / "bdb2026_receiver_history_snapshots_v1.csv"
    if not history_path.exists():
        raise RuntimeError(f"missing materialized receiver history {history_path}")
    panel, panel_audit = build_integration_panel(history_path, crosswalk, weekly)

    # Private derived rows are hashed for reproducibility but not uploaded.
    crosswalk_path = private / "bdb2026_to_gsis_crosswalk_v1.csv"
    crosswalk.sort_values("bdb_nfl_id").to_csv(crosswalk_path, index=False)
    panel_path = private / "wr_advanced_feature_integration_panel_2023_v1.csv"
    panel.sort_values(["target_week", "gsis_id", "nfl_id"], na_position="last").to_csv(panel_path, index=False)

    weekly_path = private / "nflverse_weekly_receiving_outcomes_2023_v1.csv"
    weekly.sort_values(["week", "gsis_id"]).to_csv(weekly_path, index=False)

    report = {
        "version": VERSION,
        "materializer_version": ALGORITHM_VERSION,
        "bdb_source_season": 2023,
        "nflverse_players_file_sha256": sha256_file(args.nflverse_players),
        "bdb_identity_audit": bdb_audit,
        "crosswalk_audit": crosswalk_audit,
        "weekly_outcome_audit": weekly_audit,
        "integration_panel_audit": panel_audit,
        "private_ephemeral_files": {
            "crosswalk": {
                "rows": int(len(crosswalk)),
                "sha256": sha256_file(crosswalk_path),
                "uploaded": False,
            },
            "weekly_outcomes": {
                "rows": int(len(weekly)),
                "sha256": sha256_file(weekly_path),
                "uploaded": False,
            },
            "integration_panel": {
                "rows": int(len(panel)),
                "sha256": sha256_file(panel_path),
                "uploaded": False,
            },
        },
        "identity_certification_rule": {
            "certified_methods": [
                "DIRECT_NFL_ID",
                "NAME_DOB_POSITION",
                "NAME_POSITION_PHYSICAL",
            ],
            "review_only_method": "NAME_POSITION_UNIQUE_REVIEW_ONLY",
            "unresolved_rows_are_not_joined": True,
        },
        "temporal_contract": {
            "strict_prior_history_only": True,
            "target_game_observation_used": False,
            "same_game_partial_history_used": False,
        },
        "predictive_experiment": False,
        "predictive_metrics_calculated": False,
        "production_changed": False,
        "sportsbook_inputs_used": False,
        "disposition": "NFL_ADVANCED_WR_INTEGRATION_PREFLIGHT_V1_COMPLETE",
    }
    target = sanitized / "nfl_advanced_wr_integration_preflight_v1.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
