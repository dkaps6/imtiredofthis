from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.rb_receiving_identity_runtime_v1 import (
    EPS,
    FEATURES,
    attach_identity,
    identity_atlas,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

VERSION = "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_V1"
SEASON = 2026
WEEK = 1
HISTORY_START = 2013
HISTORY_THROUGH = 2025
RB_FAMILIES = {"RB", "FB"}

MODEL_PATH = Path("data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json")
ROOM_STATE_PATH = Path("data/models/rb_r26_production_v1/rb_r26_week1_room_state_v1.csv")
AUDIT_JSON = Path("data/rb_r26_receptions_production_audit.json")
TRACE_CSV = Path("data/rb_r26_receptions_production_trace.csv")
ARRAY_AUDIT_CSV = Path("data/rb_r26_receptions_production_array_audit.csv")

EXPECTED_MODEL_SHA256 = "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba"
EXPECTED_ROOM_STATE_SHA256 = "27ad7bad8fcdfa6b0b1090994e0c0d2bdc4c0e45209d2409abc9574b5d354258"
EXPECTED_VACANCY_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CLE", "DAL", "DEN", "DET", "GB", "HOU",
    "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
    "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}
CONTROL_TEAM = "CIN"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _position_family(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("position_family", frame.get("position", pd.Series("", index=frame.index)))
    p = source.fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
    p = p.where(~p.str.startswith("RB"), "RB")
    p = p.where(~p.str.startswith("FB"), "FB")
    p = p.where(~p.str.startswith("QB"), "QB")
    p = p.where(~p.str.startswith("WR"), "WR")
    p = p.where(~p.str.startswith("TE"), "TE")
    return p


def _object_key_copy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("player_clean_key", "team"):
        if col in out.columns:
            before = out[col].astype("string").fillna("<NA>").tolist()
            out[col] = out[col].astype(object)
            after = out[col].astype("string").fillna("<NA>").tolist()
            if before != after:
                raise RuntimeError(f"R26 production dtype normalization changed identity values in {col}")
    return out


def _manual_r9(payload: dict, frame: pd.DataFrame) -> np.ndarray:
    required = list(payload["feature_order"])
    if required != list(FEATURES):
        raise RuntimeError("R26 production R9 feature order drift")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"R26 production strict-prior feature frame missing {missing}")
    x = frame[required].to_numpy(float)
    mean = np.asarray(payload["scaler_mean"], float)
    scale = np.asarray(payload["scaler_scale"], float)
    coef = np.asarray(payload["coefficients"], float)
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coef):
        raise RuntimeError("R26 production serialized R9 dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coef).all():
        raise RuntimeError("R26 production nonfinite serialized R9 model/input")
    if np.any(scale <= 0):
        raise RuntimeError("R26 production serialized R9 scaler has nonpositive scale")
    pred = ((x - mean) / scale) @ coef + float(payload["intercept"])
    return np.clip(pred, -float(payload["prediction_clip"]), float(payload["prediction_clip"]))


def _load_contract(model_path: Path, room_state_path: Path) -> tuple[dict, pd.DataFrame, dict]:
    if not model_path.is_file() or not room_state_path.is_file():
        raise RuntimeError(f"R26 production assets missing model={model_path} room_state={room_state_path}")
    model_sha = _sha256_file(model_path)
    room_sha = _sha256_file(room_state_path)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"R26 production R19 model hash drift: {model_sha}")
    if room_sha != EXPECTED_ROOM_STATE_SHA256:
        raise RuntimeError(f"R26 production room-state hash drift: {room_sha}")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    r9 = model.get("models", {}).get("r8_r9_identity", {})
    contract_ok = bool(
        model.get("candidate") == "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1"
        and model.get("status") == "SHADOW_ONLY"
        and int(model.get("fit_for_season", -1)) == SEASON
        and list(r9.get("feature_order", [])) == list(FEATURES)
        and r9.get("model") == "StandardScaler+Ridge"
        and float(r9.get("alpha", -1)) == 20.0
        and float(r9.get("train_clip", -1)) == 2.0
        and float(r9.get("prediction_clip", -1)) == 1.0
        and int(r9.get("training_season", -1)) == HISTORY_THROUGH
        and float(r9.get("r9_reliability", -1)) == 1.0
        and int(model.get("sportsbook_inputs_added", -1)) == 0
        and int(model.get("production_parameters_changed", -1)) == 0
    )
    if not contract_ok:
        raise RuntimeError("R26 production serialized R9 contract drift")

    room = pd.read_csv(room_state_path, low_memory=False)
    required = {"season", "week", "team"}
    if required - set(room.columns):
        raise RuntimeError(f"R26 production room-state missing columns {sorted(required-set(room.columns))}")
    room = room.loc[
        pd.to_numeric(room["season"], errors="coerce").eq(SEASON)
        & pd.to_numeric(room["week"], errors="coerce").eq(WEEK)
    ].copy()
    if room["team"].duplicated().any():
        raise RuntimeError("R26 production room-state has duplicate teams")
    observed = set(room["team"].astype(str).str.upper().str.strip())
    if observed != EXPECTED_VACANCY_TEAMS or CONTROL_TEAM in observed:
        raise RuntimeError(f"R26 production vacancy-team contract drift: observed={sorted(observed)}")
    return model, room, {"model_sha256": model_sha, "room_state_sha256": room_sha}


def _same_array(a, b) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)
    return aa.shape == bb.shape and np.array_equal(aa, bb)


def apply_rb_r26_receptions_production(
    result,
    metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
    iterations=None,
    seed=None,
    model_path: Path = MODEL_PATH,
    room_state_path: Path = ROOM_STATE_PATH,
):
    if int(season) != SEASON or int(week) != WEEK:
        raise RuntimeError(f"{VERSION} is qualified only for 2026 Week 1, got season={season} week={week}")

    model, _room, asset_audit = _load_contract(model_path, room_state_path)
    frame = metrics.copy()
    required = {"event_id", "team", "player_clean_key", "entitlement_tgt_share", "position"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"R26 production football frame missing columns: {sorted(missing)}")
    if frame.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("R26 production football frame has duplicate event/team/player keys")

    frame["position_family"] = _position_family(frame)
    frame["team"] = frame["team"].astype(str).str.upper().str.strip()
    frame["entitlement_tgt_share"] = pd.to_numeric(frame["entitlement_tgt_share"], errors="coerce")
    if frame["entitlement_tgt_share"].isna().any() or (frame["entitlement_tgt_share"] < 0).any():
        raise RuntimeError("R26 production baseline entitlement missing/nonfinite/negative")

    rb = frame.loc[frame.position_family.isin(RB_FAMILIES)].copy()
    if rb.empty or rb["team"].nunique() != 32:
        raise RuntimeError(f"R26 production RB/FB coverage invalid rows={len(rb)} teams={rb['team'].nunique()}")
    rb["season"] = SEASON
    rb["week"] = WEEK

    states, prev = identity_atlas(HISTORY_START, HISTORY_THROUGH)
    state_time = pd.to_numeric(states.get("time_key", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(states) == 0 or len(state_time) == 0 or int(state_time.max()) >= SEASON * 100 + WEEK:
        raise RuntimeError("R26 production strict-prior identity history reaches current/future Week 1")

    rb_for_identity = _object_key_copy(rb)
    states_for_identity = _object_key_copy(states)
    prev_for_identity = _object_key_copy(prev)
    rb = attach_identity(rb_for_identity, SEASON, WEEK, states_for_identity, prev_for_identity)
    if rb[list(FEATURES)].isna().any().any() or not np.isfinite(rb[list(FEATURES)].to_numpy(float)).all():
        raise RuntimeError("R26 production strict-prior R9 features incomplete/nonfinite")

    r9 = model["models"]["r8_r9_identity"]
    rb["r9_raw_residual"] = _manual_r9(r9, rb)
    rb["r9_reliability"] = float(r9["r9_reliability"])
    rb["r9_calibrated_residual"] = rb.r9_reliability * rb.r9_raw_residual
    rb["vacancy_active"] = rb["team"].isin(EXPECTED_VACANCY_TEAMS).astype(int)
    rb["baseline_entitlement_tgt_share"] = rb["entitlement_tgt_share"].astype(float)
    rb["baseline_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    if (rb.baseline_rb_pool <= 0).any():
        raise RuntimeError("R26 production encountered nonpositive RB+FB entitlement pool")
    rb["baseline_rb_within_share"] = rb.baseline_entitlement_tgt_share / rb.baseline_rb_pool
    rb["r9_score"] = np.log(rb.baseline_rb_within_share.clip(lower=0.0) + float(EPS)) + rb.r9_calibrated_residual
    rb["candidate_rb_within_share"] = rb.baseline_rb_within_share.astype(float)

    room_rows: list[dict] = []
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=True).groups.items():
        g = rb.loc[idx]
        baseline = g.baseline_rb_within_share.to_numpy(float)
        if str(team) in EXPECTED_VACANCY_TEAMS:
            score = g.r9_score.to_numpy(float)
            weight = np.exp(score - np.max(score))
            candidate = weight / weight.sum() if weight.sum() > 0 else baseline.copy()
        else:
            candidate = baseline.copy()
        if len(candidate):
            candidate[int(np.argmax(candidate))] += 1.0 - float(candidate.sum())
        rb.loc[idx, "candidate_rb_within_share"] = candidate
        pool = float(g.baseline_rb_pool.iloc[0])
        room_rows.append({
            "event_id": str(event_id),
            "team": str(team),
            "vacancy_active": int(str(team) in EXPECTED_VACANCY_TEAMS),
            "players": int(len(g)),
            "baseline_rb_pool": pool,
            "candidate_rb_pool": float(pool * candidate.sum()),
            "rb_pool_gap": float(pool * candidate.sum() - pool),
        })

    rb["candidate_entitlement_tgt_share"] = rb.baseline_rb_pool * rb.candidate_rb_within_share
    rb["entitlement_delta"] = rb.candidate_entitlement_tgt_share - rb.baseline_entitlement_tgt_share

    candidate_metrics = frame.copy()
    original_non_entitlement = candidate_metrics.drop(columns=["entitlement_tgt_share"]).copy()
    rb_idx = rb.set_index(["event_id", "team", "player_clean_key"])
    mask = candidate_metrics.position_family.isin(RB_FAMILIES)
    keys = pd.MultiIndex.from_frame(candidate_metrics.loc[mask, ["event_id", "team", "player_clean_key"]])
    candidate_metrics.loc[mask, "entitlement_tgt_share"] = rb_idx.loc[keys, "candidate_entitlement_tgt_share"].to_numpy(float)
    if not original_non_entitlement.equals(candidate_metrics.drop(columns=["entitlement_tgt_share"])):
        raise RuntimeError("R26 production candidate metrics changed non-entitlement football columns")

    team_before = frame.groupby(["event_id", "team"])["entitlement_tgt_share"].sum().sort_index()
    team_after = candidate_metrics.groupby(["event_id", "team"])["entitlement_tgt_share"].sum().sort_index()
    max_team_gap = float((team_after - team_before).abs().max())
    max_room_gap = float(pd.DataFrame(room_rows)["rb_pool_gap"].abs().max())
    nonrb = ~candidate_metrics.position_family.isin(RB_FAMILIES)
    max_nonrb_delta = float((
        candidate_metrics.loc[nonrb, "entitlement_tgt_share"].to_numpy(float)
        - frame.loc[nonrb, "entitlement_tgt_share"].to_numpy(float)
    ).max(initial=0.0)) if nonrb.any() else 0.0
    max_nonrb_abs_delta = float(np.max(np.abs(
        candidate_metrics.loc[nonrb, "entitlement_tgt_share"].to_numpy(float)
        - frame.loc[nonrb, "entitlement_tgt_share"].to_numpy(float)
    ))) if nonrb.any() else 0.0
    cin = rb["team"].eq(CONTROL_TEAM)
    max_cin_entitlement_delta = float(rb.loc[cin, "entitlement_delta"].abs().max()) if cin.any() else float("inf")
    if max_room_gap > 1e-12 or max_team_gap > 1e-12 or max_nonrb_abs_delta > 1e-12 or max_cin_entitlement_delta > 1e-12:
        raise RuntimeError(
            "R26 production entitlement conservation failure "
            f"room={max_room_gap} team={max_team_gap} nonrb={max_nonrb_abs_delta} cin={max_cin_entitlement_delta}"
        )

    candidate_result = explicit_simulate(candidate_metrics, iterations=iterations, seed=seed)
    final = copy.deepcopy(result)
    before_keys = set(result.values)
    if set(candidate_result.values) != before_keys:
        raise RuntimeError("R26 production candidate simulation key universe differs from protected result")

    pos_map = {
        (str(r.event_id), str(r.player_clean_key)): str(r.position_family)
        for r in frame[["event_id", "player_clean_key", "position_family"]].itertuples(index=False)
    }
    team_map = {
        (str(r.event_id), str(r.player_clean_key)): str(r.team)
        for r in frame[["event_id", "player_clean_key", "team"]].itertuples(index=False)
    }
    player_map = {
        (str(r.event_id), str(r.player_clean_key)): str(getattr(r, "player", ""))
        for r in frame[["event_id", "player_clean_key", "player"]].itertuples(index=False)
    } if "player" in frame.columns else {}

    trace_rows = []
    allowed_changed: set[tuple[str, str, str]] = set()
    rb_lookup = rb.set_index(["event_id", "team", "player_clean_key"])
    for row in rb.itertuples(index=False):
        game = str(row.event_id)
        pkey = str(row.player_clean_key)
        team = str(row.team)
        key = (game, pkey, "receptions")
        if key not in result.values or key not in candidate_result.values:
            raise RuntimeError(f"R26 production missing receptions array {key}")
        baseline = np.asarray(result.values[key], dtype=float)
        candidate = np.asarray(candidate_result.values[key], dtype=float)
        if baseline.shape != candidate.shape or len(candidate) == 0:
            raise RuntimeError(f"R26 production reception draw-count mismatch {key}")
        if not np.isfinite(candidate).all() or (candidate < 0).any() or np.max(np.abs(candidate - np.rint(candidate))) > 1e-12:
            raise RuntimeError(f"R26 production invalid candidate receptions array {key}")
        applied = team in EXPECTED_VACANCY_TEAMS
        if applied:
            final.values[key] = candidate.copy()
            allowed_changed.add(key)
        else:
            final.values[key] = baseline.copy()
        final_arr = np.asarray(final.values[key], dtype=float)
        ix = (row.event_id, row.team, row.player_clean_key)
        rr = rb_lookup.loc[ix]
        trace_rows.append({
            "event_id": game,
            "team": team,
            "player": player_map.get((game, pkey), ""),
            "player_clean_key": pkey,
            "position_family": str(row.position_family),
            "vacancy_active": int(row.vacancy_active),
            "rb_r26_receptions_applied": bool(applied),
            "rb_r26_receptions_version": VERSION if applied else "",
            "baseline_entitlement_tgt_share": float(rr.baseline_entitlement_tgt_share),
            "candidate_entitlement_tgt_share": float(rr.candidate_entitlement_tgt_share),
            "entitlement_delta": float(rr.entitlement_delta),
            "r9_raw_residual": float(rr.r9_raw_residual),
            "r9_calibrated_residual": float(rr.r9_calibrated_residual),
            "baseline_receptions_mean": float(baseline.mean()),
            "candidate_receptions_mean": float(candidate.mean()),
            "final_receptions_mean": float(final_arr.mean()),
            "final_minus_baseline_receptions_mean": float(final_arr.mean() - baseline.mean()),
        })

    array_rows = []
    forbidden = 0
    changed = 0
    for key in sorted(before_keys):
        old = np.asarray(result.values[key])
        new = np.asarray(final.values[key])
        is_changed = not _same_array(old, new)
        allowed = key in allowed_changed
        if is_changed:
            changed += 1
        if is_changed and not allowed:
            forbidden += 1
        pos = pos_map.get((str(key[0]), str(key[1])), "")
        team = team_map.get((str(key[0]), str(key[1])), "")
        array_rows.append({
            "event_id": str(key[0]),
            "team": team,
            "player_clean_key": str(key[1]),
            "position_family": pos,
            "market": str(key[2]),
            "changed": bool(is_changed),
            "allowed_to_change": bool(allowed),
            "forbidden_change": bool(is_changed and not allowed),
            "mean_delta": float(np.asarray(new, float).mean() - np.asarray(old, float).mean()) if len(new) else 0.0,
        })

    trace = pd.DataFrame(trace_rows).sort_values(["event_id", "team", "player_clean_key"], kind="mergesort").reset_index(drop=True)
    array_audit = pd.DataFrame(array_rows)
    if set(final.values) != before_keys:
        raise RuntimeError("R26 production changed final simulation key universe")
    if forbidden != 0:
        bad = array_audit.loc[array_audit.forbidden_change].head(20).to_dict("records")
        raise RuntimeError(f"R26 production changed forbidden arrays: {bad}")
    cin_trace = trace.loc[trace.team.eq(CONTROL_TEAM)]
    if cin_trace.empty or cin_trace.rb_r26_receptions_applied.any() or cin_trace.final_minus_baseline_receptions_mean.abs().max() > 1e-12:
        raise RuntimeError("R26 production CIN control did not remain exact baseline")

    for market in ("rec_yards", "rush_rec_yards", "rush_yards", "rush_att"):
        bad = array_audit.loc[
            array_audit.position_family.isin(RB_FAMILIES)
            & array_audit.market.eq(market)
            & array_audit.changed
        ]
        if not bad.empty:
            raise RuntimeError(f"R26 production changed protected RB market={market}")

    nonrb_changed = array_audit.loc[~array_audit.position_family.isin(RB_FAMILIES) & array_audit.changed]
    if not nonrb_changed.empty:
        raise RuntimeError("R26 production changed non-RB simulation arrays")

    TRACE_CSV.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(TRACE_CSV, index=False)
    array_audit.to_csv(ARRAY_AUDIT_CSV, index=False)
    payload = {
        "candidate": VERSION,
        "disposition": "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_ADAPTER_PASS",
        "integration_valid": True,
        "season": SEASON,
        "week": WEEK,
        "history_start": HISTORY_START,
        "history_through": HISTORY_THROUGH,
        "strict_prior_max_time_key": int(state_time.max()),
        "football_rb_fb_rows": int(len(rb)),
        "football_teams": int(frame.team.nunique()),
        "vacancy_teams": int(len(EXPECTED_VACANCY_TEAMS)),
        "control_team": CONTROL_TEAM,
        "applied_rb_fb_rows": int(trace.rb_r26_receptions_applied.sum()),
        "changed_reception_arrays": int(changed),
        "forbidden_changed_arrays": int(forbidden),
        "max_rb_pool_gap": max_room_gap,
        "max_team_entitlement_gap": max_team_gap,
        "max_non_rb_entitlement_delta": max_nonrb_abs_delta,
        "max_cin_entitlement_delta": max_cin_entitlement_delta,
        "max_abs_receptions_mean_delta": float(trace.final_minus_baseline_receptions_mean.abs().max()),
        "model_assets": asset_audit,
        "r9_refit": False,
        "same_week_outcomes_used": 0,
        "sportsbook_inputs_used": 0,
        "receiving_yards_changed": False,
        "rush_receiving_yards_changed": False,
        "rushing_changed": False,
        "non_rb_changed": False,
        "baseline_retained_for_audit_only": True,
        "final_receptions_authority": VERSION,
        "trace": str(TRACE_CSV),
        "array_audit": str(ARRAY_AUDIT_CSV),
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final.rb_r26_receptions_audit = payload
    return final, trace, payload
