#!/usr/bin/env python3
"""Frozen RB-PD2 forward/shadow confirmation primitives.

Research-only implementation of the already-frozen forward confirmation plan:
- section 3 historical difficulty state;
- section 8 exact mean-neutral empirical width transform;
- section 9 pregame lock integrity helpers.

This module does not change production pricing science, does not consume
sportsbook prices/lines in candidate construction, and does not grade outcomes.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import CANON_TEAM_CODES, canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

HISTORY_WINDOW = 8
MIN_PRIOR_GAMES = 4
REFERENCE_MIN = 100
WIDTH_ONSET = 0.50
WIDTH_CAP = 0.30
MEAN_TOLERANCE = 1e-8

RUSH_YARDS_MC_WEIGHT = 0.5569542426070742
RUSH_YARDS_ML_WEIGHT = 0.4430457573929258
RUSH_YARDS_STATE_WEIGHT = 0.0
RUSH_YARDS_FIT_SCOPE = "all_2024_oos_frozen_for_2025"
RUSH_YARDS_PROMOTION_LINEAGE = "RB_STACK1_RUN_33535308110_FOR_P3"

IDENTITY_FINGERPRINT_FIELDS = (
    "manual_name_overrides_sha256",
    "canonical_names_py_sha256",
)
ELIGIBLE_POSITIONS = {"RB", "HB", "FB"}
DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _ord(season: int, week: int) -> int:
    return int(season) * 100 + int(week)


def _canonical_player_key(value: Any) -> str:
    _, key = canonicalize_player_name_safe(value)
    return str(key or "").strip()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return ""


def identity_source_fingerprints(root: Path = Path(".")) -> dict[str, str]:
    root = Path(root)
    return {
        "manual_name_overrides_sha256": _file_sha256(root / "data/manual_name_overrides.csv"),
        "canonical_names_py_sha256": _file_sha256(root / "scripts/utils/canonical_names.py"),
        # Diagnostic only. Ourlads is a mutable weekly enrichment source, not a
        # stable identity-contract file. Actual target identity still has to
        # resolve to the canonical key present in the frozen history state.
        "roles_ourlads_sha256": _file_sha256(root / "data/roles_ourlads.csv"),
    }


def _valid_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", text))


def assert_identity_fingerprints_match(
    history_manifest: dict[str, Any],
    capture_receipt: dict[str, Any],
) -> None:
    """Fail closed unless both sides prove the same non-empty identity sources."""
    capture_prov = capture_receipt.get("provenance") or {}
    for field in IDENTITY_FINGERPRINT_FIELDS:
        left = str(history_manifest.get(field) or "").strip().lower()
        right = str(capture_prov.get(field) or "").strip().lower()
        if not _valid_digest(left) or not _valid_digest(right):
            raise RuntimeError(
                f"unprovable identity fingerprint {field}: history={left!r} capture={right!r}"
            )
        if left != right:
            raise RuntimeError(
                f"identity fingerprint mismatch {field}: history={left} capture={right}"
            )


def verify_frozen_2025_weights(weights: pd.DataFrame) -> dict[str, Any]:
    x = weights.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    q = x.loc[x["market"].astype(str).str.lower().eq("rush_yards")].copy()
    if len(q) != 1:
        raise RuntimeError(f"expected one rush_yards weight row, found {len(q)}")
    row = q.iloc[0]
    checks = {
        "mc_weight": float(row["mc_weight"]) == RUSH_YARDS_MC_WEIGHT,
        "ml_weight": float(row["ml_weight"]) == RUSH_YARDS_ML_WEIGHT,
        "state_weight": float(row["state_weight"]) == RUSH_YARDS_STATE_WEIGHT,
        "fit_scope": str(row.get("fit_scope", "")) == RUSH_YARDS_FIT_SCOPE,
        "promotion_lineage": str(row.get("promotion_lineage", "")) == RUSH_YARDS_PROMOTION_LINEAGE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen 2025 rush_yards weight contract drifted: {checks}")
    return checks


def build_history_state(history: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen last-8 rushing-yard error state from certified rows.

    Required input columns:
      season, week, team, player_clean_key, position,
      projection_mean, actual_rush_yards, pregame_lineage_certified

    Outcomes are used only to grade already-completed rows. The function refuses
    any row whose pregame projection lineage was not explicitly certified.
    """
    required = {
        "season", "week", "team", "player_clean_key", "position",
        "projection_mean", "actual_rush_yards", "pregame_lineage_certified",
    }
    missing = required - set(history.columns)
    if missing:
        raise RuntimeError(f"history missing required columns: {sorted(missing)}")

    x = history.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["position"] = x["position"].fillna("").astype(str).str.upper().str.strip()
    x = x.loc[x["position"].isin(ELIGIBLE_POSITIONS)].copy()
    if x.empty:
        return pd.DataFrame()

    if not x["season"].notna().all() or not x["week"].notna().all():
        raise RuntimeError("history contains non-numeric season/week")
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    if not x["week"].between(1, 18).all():
        raise RuntimeError("history contains week outside 1..18")

    certified = x["pregame_lineage_certified"].astype(bool)
    if not certified.all():
        raise RuntimeError("uncertified pregame projection lineage in history")

    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(_canonical_player_key)
    if x["player_clean_key"].eq("").any():
        raise RuntimeError("blank canonical player_clean_key in history")
    if (~x["team"].isin(CANON_TEAM_CODES)).any():
        bad = sorted(set(x.loc[~x["team"].isin(CANON_TEAM_CODES), "team"].tolist()))
        raise RuntimeError(f"non-canonical team code in history: {bad}")

    x["projection_mean"] = pd.to_numeric(x["projection_mean"], errors="coerce")
    x["actual_rush_yards"] = pd.to_numeric(x["actual_rush_yards"], errors="coerce")
    if not np.isfinite(x["projection_mean"]).all() or not np.isfinite(x["actual_rush_yards"]).all():
        raise RuntimeError("non-finite projection/outcome in completed history")

    identity = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(identity).any():
        raise RuntimeError("duplicate historical football identity")

    x = x.sort_values(["season", "week", "player_clean_key"], kind="stable").reset_index(drop=True)

    player_hist: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    for r in x.itertuples(index=False):
        player = str(r.player_clean_key)
        prior = player_hist.get(player, [])[-HISTORY_WINDOW:]
        target_ord = _ord(r.season, r.week)
        error = float(r.projection_mean) - float(r.actual_rush_yards)
        abs_error = abs(error)

        rec = r._asdict()
        rec.update({
            "target_ord": target_ord,
            "target_yard_error": error,
            "target_yard_abs_error": abs_error,
            "prior_games": len(prior),
            "prior8_yard_mae": float(np.mean([p["abs_error"] for p in prior])) if prior else np.nan,
            "last_prior_ord": int(prior[-1]["ord"]) if prior else np.nan,
        })
        rows.append(rec)
        player_hist.setdefault(player, []).append({"ord": target_ord, "abs_error": abs_error})

    out = pd.DataFrame(rows)
    has_prior = out["last_prior_ord"].notna()
    if (out.loc[has_prior, "last_prior_ord"] >= out.loc[has_prior, "target_ord"]).any():
        raise RuntimeError("same/future game entered player history")

    return strict_prior_difficulty_scores(out)


def strict_prior_difficulty_scores(state: pd.DataFrame) -> pd.DataFrame:
    """Exact frozen same-week-deferred empirical percentile mapping."""
    if state.empty:
        return state.copy()
    out = state.copy().sort_values(
        ["season", "week", "player_clean_key"], kind="stable"
    ).reset_index(drop=True)
    out["difficulty_score"] = np.nan
    out["difficulty_reference_n"] = 0
    out["difficulty_reference_max_ord"] = np.nan

    ref: list[float] = []
    last_ref_ord = np.nan
    for (season, week), idx in out.groupby(["season", "week"], sort=True).groups.items():
        ids = list(idx)
        ref_n = len(ref)
        for i in ids:
            out.at[i, "difficulty_reference_n"] = ref_n
            out.at[i, "difficulty_reference_max_ord"] = last_ref_ord
            if int(out.at[i, "prior_games"]) < MIN_PRIOR_GAMES or ref_n < REFERENCE_MIN:
                continue
            value = float(out.at[i, "prior8_yard_mae"])
            if np.isfinite(value):
                out.at[i, "difficulty_score"] = bisect.bisect_right(ref, value) / ref_n
        for i in ids:
            if int(out.at[i, "prior_games"]) < MIN_PRIOR_GAMES:
                continue
            value = float(out.at[i, "prior8_yard_mae"])
            if np.isfinite(value):
                bisect.insort(ref, value)
        last_ref_ord = _ord(int(season), int(week))

    scored = out["difficulty_score"].notna()
    if scored.any():
        if (out.loc[scored, "difficulty_reference_n"] < REFERENCE_MIN).any():
            raise RuntimeError("difficulty score emitted below reference floor")
        bad = (
            pd.to_numeric(out.loc[scored, "difficulty_reference_max_ord"], errors="coerce")
            >= pd.to_numeric(out.loc[scored, "target_ord"], errors="coerce")
        )
        if bad.any():
            raise RuntimeError("same/future week contaminated difficulty reference")
    return out


def target_difficulty_state(
    history_state: pd.DataFrame,
    *,
    player_clean_key: str,
    season: int,
    week: int,
) -> dict[str, Any]:
    """Compute the frozen state for one future target using only earlier rows."""
    target_ord = _ord(season, week)
    key = _canonical_player_key(player_clean_key)
    if not key:
        raise RuntimeError("blank target player_clean_key")

    if history_state is None or history_state.empty:
        prior = pd.DataFrame()
        ref_rows = pd.DataFrame()
    else:
        h = history_state.copy()
        h["target_ord"] = pd.to_numeric(h["target_ord"], errors="coerce")
        prior_all = h.loc[
            h["player_clean_key"].astype(str).eq(key) & h["target_ord"].lt(target_ord)
        ].sort_values("target_ord", kind="stable")
        prior = prior_all.tail(HISTORY_WINDOW)
        ref_rows = h.loc[
            h["target_ord"].lt(target_ord)
            & pd.to_numeric(h["prior_games"], errors="coerce").ge(MIN_PRIOR_GAMES)
            & pd.to_numeric(h["prior8_yard_mae"], errors="coerce").notna()
        ].copy()

    prior_games = int(len(prior))
    prior8_mae = (
        float(pd.to_numeric(prior["target_yard_abs_error"], errors="coerce").mean())
        if prior_games else np.nan
    )
    ref = sorted(
        pd.to_numeric(ref_rows["prior8_yard_mae"], errors="coerce")
        .dropna().astype(float).tolist()
    )
    ref_n = len(ref)
    ref_max_ord = (
        int(pd.to_numeric(ref_rows["target_ord"], errors="coerce").max())
        if ref_n else np.nan
    )
    if np.isfinite(ref_max_ord) and ref_max_ord >= target_ord:
        raise RuntimeError("target reference includes same/future week")

    score = np.nan
    if prior_games >= MIN_PRIOR_GAMES and ref_n >= REFERENCE_MIN and np.isfinite(prior8_mae):
        score = bisect.bisect_right(ref, float(prior8_mae)) / ref_n

    return {
        "prior_games": prior_games,
        "prior8_yard_mae": prior8_mae,
        "difficulty_score": float(score) if np.isfinite(score) else np.nan,
        "difficulty_reference_n": int(ref_n),
        "difficulty_reference_max_ord": ref_max_ord,
    }


def width_multiplier(score: float) -> float:
    if not np.isfinite(score):
        return 1.0
    return float(
        1.0
        + WIDTH_CAP
        * np.clip((float(score) - WIDTH_ONSET) / (1.0 - WIDTH_ONSET), 0.0, 1.0)
    )


def widen_mean_neutral(draws: np.ndarray, multiplier: float) -> np.ndarray:
    """Exact frozen section-8 transform."""
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError("invalid baseline draw array")
    mu = float(np.mean(x))
    raw = np.maximum(0.0, mu + float(multiplier) * (x - mu))
    raw_mean = float(np.mean(raw))
    if not np.isfinite(raw_mean) or raw_mean <= 0.0:
        raise RuntimeError("candidate width transform produced invalid mean")
    out = np.asarray(raw * (mu / raw_mean), dtype=np.float64)
    if out.size != x.size or not np.isfinite(out).all() or (out < 0).any():
        raise RuntimeError("candidate width transform produced invalid draws")
    if abs(float(np.mean(out)) - mu) > MEAN_TOLERANCE:
        raise RuntimeError("candidate transform violated mean-neutrality")
    return out


def draw_digest(draws: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(draws, dtype=np.float64))
    return hashlib.sha256(x.tobytes()).hexdigest()


def distribution_summary(draws: np.ndarray) -> dict[str, Any]:
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError("invalid empirical array")
    q05, q10, q50, q90, q95 = [
        float(v) for v in np.quantile(x, [0.05, 0.10, 0.50, 0.90, 0.95], method="linear")
    ]
    return {
        "draw_count": int(x.size),
        "draw_digest_sha256": draw_digest(x),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q05": q05,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "q95": q95,
    }


def build_candidate(draws: np.ndarray, difficulty_score: float) -> tuple[np.ndarray, float]:
    mult = width_multiplier(difficulty_score)
    candidate = widen_mean_neutral(draws, mult)
    return candidate, mult


def normalize_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "week", "home", "away", "kickoff_utc"}
    missing = required - set(schedule.columns)
    if missing:
        raise RuntimeError(f"schedule missing kickoff contract columns: {sorted(missing)}")
    s = schedule.copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s["home"] = s["home"].map(canon_team)
    s["away"] = s["away"].map(canon_team)

    bad_team = ~(s["home"].isin(CANON_TEAM_CODES) & s["away"].isin(CANON_TEAM_CODES))
    if bad_team.any():
        raise RuntimeError("schedule contains uncanonicalizable team alias")

    raw = s["kickoff_utc"]
    date_only = raw.map(lambda v: bool(DATE_ONLY_RE.fullmatch(str(v).strip())))
    if date_only.any():
        raise RuntimeError("date-only kickoff value cannot satisfy kickoff contract")
    s["kickoff_utc"] = pd.to_datetime(raw, utc=True, errors="coerce")
    if s["kickoff_utc"].isna().any():
        raise RuntimeError("schedule contains missing/malformed kickoff_utc")
    return s


def resolve_kickoff_utc(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    opponent: str | None = None,
) -> pd.Timestamp:
    s = normalize_schedule(schedule)
    t = canon_team(team)
    if t not in CANON_TEAM_CODES:
        raise RuntimeError(f"target team cannot canonicalize: {team!r}")
    q = s.loc[
        s["season"].eq(int(season))
        & s["week"].eq(int(week))
        & (s["home"].eq(t) | s["away"].eq(t))
    ].copy()
    if len(q) != 1:
        raise RuntimeError(
            f"schedule match must be exactly one row season={season} week={week} team={t}; found={len(q)}"
        )
    row = q.iloc[0]
    if opponent is not None:
        opp = canon_team(opponent)
        if opp not in CANON_TEAM_CODES:
            raise RuntimeError(f"target opponent cannot canonicalize: {opponent!r}")
        other = str(row["away"] if str(row["home"]) == t else row["home"])
        if other != opp:
            raise RuntimeError(
                f"schedule opponent mismatch season={season} week={week} team={t}: "
                f"capture={opp} schedule={other}"
            )
    return pd.Timestamp(row["kickoff_utc"])


def schedule_row_sha256(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
) -> str:
    s = normalize_schedule(schedule)
    t = canon_team(team)
    q = s.loc[
        s["season"].eq(int(season))
        & s["week"].eq(int(week))
        & (s["home"].eq(t) | s["away"].eq(t))
    ].copy()
    if len(q) != 1:
        raise RuntimeError("cannot fingerprint ambiguous/missing schedule row")
    row = q.iloc[0]
    payload = {
        "season": int(row["season"]),
        "week": int(row["week"]),
        "home": str(row["home"]),
        "away": str(row["away"]),
        "kickoff_utc": pd.Timestamp(row["kickoff_utc"]).isoformat(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def history_state_digest(history_state: pd.DataFrame) -> str:
    if history_state is None:
        raise RuntimeError("history_state is required")
    cols = sorted(history_state.columns)
    x = history_state[cols].copy().sort_values(
        [c for c in ("season", "week", "team", "player_clean_key") if c in cols],
        kind="stable",
    )
    payload = x.to_csv(index=False, na_rep="").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_history_manifest(
    history_state: pd.DataFrame,
    *,
    root: Path = Path("."),
    source_label: str,
) -> dict[str, Any]:
    fp = identity_source_fingerprints(root)
    return {
        "version": "RB_PD2_FORWARD_HISTORY_STATE_V1",
        "source_label": str(source_label),
        "row_count": int(len(history_state)),
        "history_state_sha256": history_state_digest(history_state),
        "history_window": HISTORY_WINDOW,
        "minimum_prior_games": MIN_PRIOR_GAMES,
        "reference_min": REFERENCE_MIN,
        "rush_yards_mc_weight": RUSH_YARDS_MC_WEIGHT,
        "rush_yards_ml_weight": RUSH_YARDS_ML_WEIGHT,
        "rush_yards_state_weight": RUSH_YARDS_STATE_WEIGHT,
        "rush_yards_fit_scope": RUSH_YARDS_FIT_SCOPE,
        "rush_yards_promotion_lineage": RUSH_YARDS_PROMOTION_LINEAGE,
        **fp,
    }


def _utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise RuntimeError(f"invalid {label}: {value!r}")
    return pd.Timestamp(ts)


def lock_row(
    *,
    capture_record: dict[str, Any],
    capture_receipt: dict[str, Any],
    baseline_draws: np.ndarray,
    history_state: pd.DataFrame,
    history_manifest: dict[str, Any],
    schedule: pd.DataFrame,
    prospective_start_utc: Any,
    lock_timestamp_utc: Any | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Assemble one immutable prospective lock or fail closed.

    The lock uses capture time only to prove the distribution existed pregame.
    The artifact itself must also be persisted pregame, so lock_timestamp_utc is
    independently checked against kickoff.
    """
    if not capture_receipt.get("valid"):
        raise RuntimeError("capture session is invalid")
    if not capture_record.get("baseline_lock_eligible"):
        raise RuntimeError("capture row is baseline-lock-ineligible")
    if capture_record.get("outcome_present_at_lock") is not False:
        raise RuntimeError("target-game outcome presence is forbidden at lock")
    if capture_record.get("sportsbook_inputs_used_in_candidate") is not False:
        raise RuntimeError("sportsbook input contamination flag is not false")
    if capture_record.get("production_output_mutated") is not False:
        raise RuntimeError("production mutation flag is not false")

    assert_identity_fingerprints_match(history_manifest, capture_receipt)

    provenance = capture_receipt.get("provenance") or {}
    code_sha = str(provenance.get("code_sha") or "").strip()
    run_id = str(provenance.get("workflow_run_id") or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", code_sha):
        raise RuntimeError("capture code SHA is missing/unprovable")
    if not run_id:
        raise RuntimeError("capture workflow run ID is missing/unprovable")

    captured = _utc_timestamp(capture_record.get("captured_at_utc"), label="capture timestamp")
    start = _utc_timestamp(prospective_start_utc, label="prospective start")
    locked = _utc_timestamp(
        lock_timestamp_utc if lock_timestamp_utc is not None else datetime.now(timezone.utc),
        label="lock timestamp",
    )
    if captured < start or locked < start:
        raise RuntimeError("row predates frozen prospective start boundary")
    if locked < captured:
        raise RuntimeError("lock timestamp precedes capture timestamp")

    season = int(capture_record["season"])
    week = int(capture_record["week"])
    team = canon_team(capture_record["team"])
    kickoff = resolve_kickoff_utc(
        schedule,
        season=season,
        week=week,
        team=team,
        opponent=capture_record.get("opponent"),
    )
    if captured >= kickoff:
        raise RuntimeError("baseline capture was not pregame")
    if locked >= kickoff:
        raise RuntimeError("lock artifact was not persisted pregame")

    baseline = np.asarray(baseline_draws, dtype=np.float64)
    base_summary = distribution_summary(baseline)
    expected_digest = str((capture_record.get("baseline") or {}).get("draw_digest_sha256") or "")
    if base_summary["draw_digest_sha256"] != expected_digest:
        raise RuntimeError("baseline array digest does not match capture manifest")

    state = target_difficulty_state(
        history_state,
        player_clean_key=capture_record["player_clean_key"],
        season=season,
        week=week,
    )
    # Frozen section 5 population requires a scoreable difficulty state.
    if int(state["prior_games"]) < MIN_PRIOR_GAMES:
        raise RuntimeError("target is outside scientific population: insufficient prior games")
    if int(state["difficulty_reference_n"]) < REFERENCE_MIN or not np.isfinite(state["difficulty_score"]):
        raise RuntimeError("target is outside scientific population: invalid difficulty score/reference floor")

    candidate, mult = build_candidate(baseline, state["difficulty_score"])
    cand_summary = distribution_summary(candidate)
    if abs(cand_summary["mean"] - base_summary["mean"]) > MEAN_TOLERANCE:
        raise RuntimeError("candidate mean differs from baseline mean")
    if cand_summary["draw_count"] != base_summary["draw_count"]:
        raise RuntimeError("candidate draw count differs from baseline")

    hist_digest = str(history_manifest.get("history_state_sha256") or "")
    if not _valid_digest(hist_digest):
        raise RuntimeError("history manifest digest is missing/unprovable")

    rec = {
        "lock_version": "RB_PD2_FORWARD_LOCK_V1",
        "season": season,
        "week": week,
        "event_id": str(capture_record.get("event_id") or ""),
        "team": team,
        "opponent": canon_team(capture_record.get("opponent")),
        "player": str(capture_record.get("player") or ""),
        "player_clean_key": str(capture_record.get("player_clean_key") or ""),
        "position": str(capture_record.get("position") or ""),
        "market": "rush_yards",
        "session_id": str(capture_record.get("session_id") or ""),
        "capture_timestamp_utc": captured.isoformat(),
        "lock_timestamp_utc": locked.isoformat(),
        "kickoff_utc": kickoff.isoformat(),
        "schedule_source": "scripts/build/_schedule_utils.py:get_nfl_schedule",
        "schedule_row_sha256": schedule_row_sha256(schedule, season=season, week=week, team=team),
        "production_code_sha": code_sha,
        "workflow_run_id": run_id,
        "workflow_run_attempt": str(provenance.get("workflow_run_attempt") or ""),
        "workflow_job": str(provenance.get("workflow_job") or ""),
        "history_state_sha256": hist_digest,
        "manual_name_overrides_sha256": str(history_manifest["manual_name_overrides_sha256"]),
        "roles_ourlads_sha256": str(history_manifest["roles_ourlads_sha256"]),
        "target_mean": float(capture_record["target_mean"]),
        "prior_games": int(state["prior_games"]),
        "prior8_yard_mae": float(state["prior8_yard_mae"]) if np.isfinite(state["prior8_yard_mae"]) else None,
        "difficulty_score": float(state["difficulty_score"]) if np.isfinite(state["difficulty_score"]) else None,
        "difficulty_reference_n": int(state["difficulty_reference_n"]),
        "difficulty_reference_max_ord": (
            int(state["difficulty_reference_max_ord"])
            if np.isfinite(state["difficulty_reference_max_ord"]) else None
        ),
        "width_multiplier": float(mult),
        "baseline": base_summary,
        "candidate": cand_summary,
        "sportsbook_inputs_used_in_candidate": False,
        "production_output_mutated": False,
        "outcome_present_at_lock": False,
    }
    forbidden = {"actual", "actual_rush_yards", "target_outcome", "realized_rush_yards"}
    if forbidden.intersection(rec):
        raise RuntimeError("realized target outcome leaked into lock row")
    return rec, candidate
