#!/usr/bin/env python3
"""Source-only Phase 4B identity audit using prior-roster GSIS aliases.

Purpose
-------
Test whether the Phase 4B WR identity gap is caused by name-key drift without
loading receiving-yard outcomes.  Names are used only as deterministic aliases
to a stable GSIS ID.  Actual targets are then counted from official nflverse PBP
by that stable ID.

Resolution order (strictly prior roster rows only):
1. current-team exact full-name alias
2. current-team suffix-insensitive alias
3. globally unique exact full-name alias
4. globally unique suffix-insensitive alias

Every available roster name field is retained as an alias (full_name,
football_name, player_name, player, name, short_name).  Ambiguity fails closed.
No fuzzy matching and no target-week roster row is used to resolve identity.

This is a source audit only.  It never reads mc_rec_yards or actual_rec_yards.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.player_identity_v3 import clean_player_id, player_name_key

CANDIDATE = "WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_CANDIDATE_ROWS = 4193
EXPECTED_CANDIDATE_BY_SEASON = {2023: 2076, 2024: 2117}
EXPECTED_FEATURE_ROWS = 5321
EXPECTED_TEAM_GAMES = 1088
EXPECTED_ANCHOR_GAMES = 1026
TG = ["season", "week", "team"]
IDENT = TG + ["player_clean_key"]
NAME_FIELDS = ["full_name", "football_name", "player_name", "player", "name", "short_name"]


def _to_pd(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _first(frame: pd.DataFrame, candidates: Iterable[str], default="") -> pd.Series:
    for c in candidates:
        if c in frame.columns:
            return frame[c]
    return pd.Series(default, index=frame.index)


def _text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _load_structures(predictions: Path, features: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_cols = [
        "variant", "event_id", "team", "player_clean_key", "player", "wr_rank",
        "entitlement_tgt_share", "pred_targets", "season", "week", "actual_targets",
    ]
    pred = pd.read_csv(predictions, usecols=pred_cols, low_memory=False)
    pred = pred.loc[pred["variant"].astype(str).eq(CANDIDATE)].copy()
    if len(pred) != EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError(f"candidate row count drift {len(pred)} != {EXPECTED_CANDIDATE_ROWS}")
    for c in ["wr_rank", "entitlement_tgt_share", "pred_targets", "season", "week", "actual_targets"]:
        pred[c] = pd.to_numeric(pred[c], errors="raise")
    pred["season"] = pred["season"].astype(int)
    pred["week"] = pred["week"].astype(int)
    pred["wr_rank"] = pred["wr_rank"].astype(int)
    pred["team"] = pred["team"].map(canon_team)
    pred["player_clean_key"] = pred["player_clean_key"].astype(str)
    if pred.duplicated(IDENT).any():
        raise RuntimeError("duplicate candidate identity")
    if pred.groupby("season").size().to_dict() != EXPECTED_CANDIDATE_BY_SEASON:
        raise RuntimeError("candidate season-count drift")
    if len(pred[TG].drop_duplicates()) != EXPECTED_TEAM_GAMES:
        raise RuntimeError("candidate team-game count drift")

    feat_cols = [
        "event_id", "player", "player_clean_key", "team", "season", "week",
        "baseline_wr_rank", "baseline_entitlement_tgt_share", "candidate_entitlement_tgt_share",
    ]
    feat = pd.read_csv(features, usecols=feat_cols, low_memory=False)
    for c in ["season", "week", "baseline_wr_rank", "baseline_entitlement_tgt_share", "candidate_entitlement_tgt_share"]:
        feat[c] = pd.to_numeric(feat[c], errors="raise")
    feat["season"] = feat["season"].astype(int)
    feat["week"] = feat["week"].astype(int)
    feat["baseline_wr_rank"] = feat["baseline_wr_rank"].astype(int)
    feat["team"] = feat["team"].map(canon_team)
    feat["player_clean_key"] = feat["player_clean_key"].astype(str)
    feat = feat.loc[feat["baseline_wr_rank"].ge(2)].copy()
    if len(feat) != EXPECTED_FEATURE_ROWS:
        raise RuntimeError(f"WR2+ feature row count drift {len(feat)} != {EXPECTED_FEATURE_ROWS}")
    if feat.duplicated(IDENT).any():
        raise RuntimeError("duplicate WR2+ feature identity")

    anchors = pred.loc[pred["wr_rank"].eq(1), IDENT + ["player"]].copy()
    if len(anchors) != EXPECTED_ANCHOR_GAMES or anchors.duplicated(TG).any():
        raise RuntimeError("WR1 anchor grain drift")
    feature_on_anchor_games = feat.merge(anchors[TG].drop_duplicates(), on=TG, how="inner", validate="many_to_one")
    canonical = pd.concat(
        [
            anchors[IDENT + ["player"]],
            feature_on_anchor_games[IDENT + ["player"]],
        ],
        ignore_index=True,
    ).drop_duplicates(IDENT)
    return pred, feat, canonical


def load_roster_aliases(seasons: Iterable[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    rows: list[dict] = []
    for season in sorted({int(s) for s in seasons}):
        raw = _to_pd(nfl.load_rosters_weekly(int(season)))
        if raw.empty:
            raise RuntimeError(f"weekly roster source returned zero rows season={season}")
        x = raw.copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        x["season"] = pd.to_numeric(_first(x, ["season"], season), errors="coerce").fillna(season).astype(int)
        x["week"] = pd.to_numeric(_first(x, ["week"]), errors="coerce")
        x["team"] = _first(x, ["team", "team_abbr", "club_code"]).map(canon_team)
        x["player_id"] = _first(x, ["gsis_id", "player_id"]).map(clean_player_id)
        x = x.loc[
            x["season"].eq(season)
            & x["week"].between(1, 22, inclusive="both")
            & x["team"].astype(str).str.len().gt(0)
            & x["player_id"].astype(str).str.len().gt(0)
        ].copy()
        if x.empty:
            raise RuntimeError(f"weekly roster source has zero stable identities season={season}")
        present_name_fields = [c for c in NAME_FIELDS if c in x.columns]
        if not present_name_fields:
            raise RuntimeError(f"weekly roster source has no usable name fields season={season}")
        for r in x.itertuples(index=False):
            d = r._asdict()
            for field in present_name_fields:
                alias = _text(d.get(field))
                if not alias:
                    continue
                full = player_name_key(alias)
                base = player_name_key(alias, strip_suffix=True)
                if not full or not base:
                    continue
                rows.append({
                    "season": int(d["season"]),
                    "week": int(d["week"]),
                    "team": str(d["team"]),
                    "player_id": clean_player_id(d["player_id"]),
                    "alias": alias,
                    "alias_field": field,
                    "full_key": full,
                    "base_key": base,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("prior-roster alias source produced zero rows")
    return out.drop_duplicates(
        ["season", "week", "team", "player_id", "alias_field", "full_key", "base_key"],
        keep="last",
    ).reset_index(drop=True)


def _index_aliases(aliases: pd.DataFrame, key_col: str) -> dict[str, list[tuple[int, int, str, str]]]:
    out: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    for r in aliases[["season", "week", "team", "player_id", key_col]].itertuples(index=False):
        key = str(getattr(r, key_col))
        if key:
            out[key].append((int(r.season), int(r.week), str(r.team), clean_player_id(r.player_id)))
    return out


def _prior_ids(
    index: dict[str, list[tuple[int, int, str, str]]],
    key: str,
    season: int,
    week: int,
    team: str | None,
) -> set[str]:
    vals = index.get(str(key), [])
    ids: set[str] = set()
    for s, w, tm, pid in vals:
        if not (s < int(season) or (s == int(season) and w < int(week))):
            continue
        if team is not None and tm != str(team):
            continue
        if pid:
            ids.add(pid)
    return ids


def resolve_prior_gsis(
    player: str,
    team: str,
    season: int,
    week: int,
    full_index: dict[str, list[tuple[int, int, str, str]]],
    base_index: dict[str, list[tuple[int, int, str, str]]],
) -> dict:
    full = player_name_key(player)
    base = player_name_key(player, strip_suffix=True)
    criteria = [
        ("team_exact_full_alias", full_index, full, str(team)),
        ("team_suffix_insensitive_alias", base_index, base, str(team)),
        ("global_unique_full_alias", full_index, full, None),
        ("global_unique_suffix_alias", base_index, base, None),
    ]
    for method, index, key, team_filter in criteria:
        if not key:
            continue
        ids = _prior_ids(index, key, season, week, team_filter)
        if len(ids) == 1:
            return {
                "identity_status": "RESOLVED_GSIS",
                "identity_method": method,
                "resolved_player_id": next(iter(ids)),
                "identity_candidate_count": 1,
                "identity_full_key": full,
                "identity_base_key": base,
            }
        if len(ids) > 1:
            return {
                "identity_status": "AMBIGUOUS_IDENTITY",
                "identity_method": f"ambiguous_{method}",
                "resolved_player_id": "",
                "identity_candidate_count": int(len(ids)),
                "identity_full_key": full,
                "identity_base_key": base,
            }
    return {
        "identity_status": "UNRESOLVED_IDENTITY",
        "identity_method": "no_prior_roster_alias",
        "resolved_player_id": "",
        "identity_candidate_count": 0,
        "identity_full_key": full,
        "identity_base_key": base,
    }


def load_pbp_targets(seasons: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    raw = _to_pd(nfl.load_pbp(seasons=sorted({int(s) for s in seasons})))
    if raw.empty:
        raise RuntimeError("PBP source returned zero rows")
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "season_type", "posteam", "pass_attempt", "two_point_attempt",
        "receiver_player_id",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"PBP source missing required columns: {missing}")
    if "no_play" not in x.columns:
        x["no_play"] = 0
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["team"] = x["posteam"].map(canon_team)
    x["receiver_id"] = x["receiver_player_id"].map(clean_player_id)
    regular = x["season_type"].astype(str).str.upper().eq("REG") & x["week"].between(1, 18)
    team_games = x.loc[regular & x["team"].astype(str).str.len().gt(0), ["season", "week", "team"]].drop_duplicates()
    official = regular
    official &= pd.to_numeric(x["pass_attempt"], errors="coerce").fillna(0).eq(1)
    official &= ~pd.to_numeric(x["two_point_attempt"], errors="coerce").fillna(0).eq(1)
    official &= ~pd.to_numeric(x["no_play"], errors="coerce").fillna(0).eq(1)
    official &= x["receiver_id"].ne("")
    t = x.loc[official, ["season", "week", "team", "receiver_id"]].copy()
    t["season"] = t["season"].astype(int)
    t["week"] = t["week"].astype(int)
    counts = (
        t.groupby(["season", "week", "team", "receiver_id"], sort=True)
        .size().rename("pbp_targets").reset_index()
    )
    team_games["season"] = team_games["season"].astype(int)
    team_games["week"] = team_games["week"].astype(int)
    return counts, team_games


def resolve_frame(
    frame: pd.DataFrame,
    full_index,
    base_index,
    target_counts: pd.DataFrame,
    pbp_team_games: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for r in frame.itertuples(index=False):
        rec = {c: getattr(r, c) for c in frame.columns}
        audit = resolve_prior_gsis(
            str(rec["player"]), str(rec["team"]), int(rec["season"]), int(rec["week"]),
            full_index, base_index,
        )
        rec.update(audit)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out.merge(
        pbp_team_games.assign(pbp_team_game_present=True),
        on=TG, how="left", validate="many_to_one",
    )
    out["pbp_team_game_present"] = out["pbp_team_game_present"].fillna(False).astype(bool)
    c = target_counts.rename(columns={"receiver_id": "resolved_player_id"})
    out = out.merge(c, on=TG + ["resolved_player_id"], how="left", validate="many_to_one")
    resolved = out["identity_status"].eq("RESOLVED_GSIS")
    out.loc[resolved & out["pbp_team_game_present"] & out["pbp_targets"].isna(), "pbp_targets"] = 0.0
    out.loc[~resolved, "pbp_targets"] = np.nan
    return out


def _summary(frame: pd.DataFrame) -> dict:
    vc = frame["identity_status"].value_counts(dropna=False).to_dict()
    methods = frame["identity_method"].value_counts(dropna=False).to_dict()
    resolved = int(vc.get("RESOLVED_GSIS", 0))
    return {
        "rows": int(len(frame)),
        "resolved_gsis": resolved,
        "resolved_pct": float(resolved / len(frame)) if len(frame) else np.nan,
        "ambiguous": int(vc.get("AMBIGUOUS_IDENTITY", 0)),
        "unresolved": int(vc.get("UNRESOLVED_IDENTITY", 0)),
        "pbp_team_game_missing": int((~frame["pbp_team_game_present"]).sum()),
        "resolved_zero_targets": int((frame["identity_status"].eq("RESOLVED_GSIS") & frame["pbp_targets"].eq(0)).sum()),
        "resolved_positive_targets": int((frame["identity_status"].eq("RESOLVED_GSIS") & frame["pbp_targets"].gt(0)).sum()),
        "identity_method_counts": {str(k): int(v) for k, v in methods.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    pred, feat, canonical = _load_structures(args.predictions, args.features)
    aliases = load_roster_aliases([2022, 2023, 2024])
    full_index = _index_aliases(aliases, "full_key")
    base_index = _index_aliases(aliases, "base_key")
    counts, team_games = load_pbp_targets([2023, 2024])

    l4 = resolve_frame(feat[IDENT + ["player"]].copy(), full_index, base_index, counts, team_games)
    l23 = resolve_frame(canonical[IDENT + ["player"]].copy(), full_index, base_index, counts, team_games)
    cand = resolve_frame(
        pred[IDENT + ["player", "actual_targets", "wr_rank"]].copy(),
        full_index, base_index, counts, team_games,
    )
    cand_res = cand.loc[cand["identity_status"].eq("RESOLVED_GSIS") & cand["pbp_targets"].notna()].copy()
    cand_res["target_delta"] = pd.to_numeric(cand_res["pbp_targets"], errors="raise") - pd.to_numeric(cand_res["actual_targets"], errors="raise")
    parity = cand_res["target_delta"].abs().le(1e-12)

    known = l4.loc[
        l4["player"].astype(str).isin(["Chris Godwin", "Velus Jones", "Nathaniel Dell"]),
        IDENT + ["player", "identity_status", "identity_method", "resolved_player_id", "pbp_targets"],
    ].copy()

    result = {
        "specification": "WR_PHASE4B_PRIOR_ROSTER_GSIS_ALIAS_V2",
        "identity_source": "strictly_prior_weekly_roster_all_deterministic_aliases_to_stable_gsis",
        "target_source": "nflverse_pbp_receiver_player_id",
        "pbp_target_rule": "REG week1-18 AND pass_attempt==1 AND two_point_attempt!=1 AND no_play!=1 AND receiver_player_id nonnull",
        "target_week_roster_used_for_resolution": False,
        "fuzzy_matching": False,
        "layer4": _summary(l4),
        "layer23": _summary(l23),
        "candidate_authority_validation": {
            "rows": int(len(cand)),
            "resolved_rows": int(len(cand_res)),
            "exact_target_parity_rows": int(parity.sum()),
            "target_parity_fail_rows": int((~parity).sum()),
            "max_abs_target_delta": float(cand_res["target_delta"].abs().max()) if len(cand_res) else np.nan,
        },
        "known_drift_examples_resolved": known.to_dict("records"),
        "receiving_yard_fields_loaded": False,
        "zero_imputation_performed": False,
        "sportsbook_inputs": 0,
        "challenger_model_authorized": False,
        "production_change": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    l4.to_csv(args.out_dir / "layer4_identity_audit.csv", index=False)
    l23.to_csv(args.out_dir / "layer23_identity_audit.csv", index=False)
    cand.loc[~cand["identity_status"].eq("RESOLVED_GSIS")].to_csv(
        args.out_dir / "candidate_unresolved_identity_audit.csv", index=False
    )
    known.to_csv(args.out_dir / "known_key_drift_examples.csv", index=False)
    (args.out_dir / "prior_roster_gsis_alias_v2.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
