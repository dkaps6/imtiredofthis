#!/usr/bin/env python3
"""Observational capture of the production RB rush-yards empirical draw array.

Implements section 7 of
`docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md` and nothing else.
This module captures the exact production `adjusted_outcomes` array for eligible
RB/HB/FB `rush_yards` rows so the frozen shadow candidate can be built *outside*
the production process.

It deliberately does NOT compute the difficulty score, the width multiplier or
the candidate distribution. Those need the reconstructed generic-ensemble parent
history (plan section 3), which does not exist inside the pricing loop; pulling
it in would put historical reconstruction on the certified production path,
which is precisely what section 7's isolation requirement exists to prevent.

Four contracts govern this module.

**1. Sportsbook-independent grain.** `scripts/materialize_pricing_offers_v1.py`
expands the compact live layer to one row per book/line, so the same football
array reaches `capture()` several times for one player-game-market. The frozen
section 5 population is one player-game, so captures are deduplicated at
`(season, week, event_id, team, opponent, player_clean_key, market)`. A repeat
must agree exactly on draw digest, draw count, `target_mean` and `mc_proj`;
disagreement is a shadow integrity error. No book, line, side, price, edge or
decision field enters the key or the artifact.

**2. The exact array survives.** Sections 8 and 9 need the draws elementwise --
a digest cannot be inverted and quantiles cannot produce a candidate or an
empirical CRPS. The float64 draws are persisted losslessly to an NPZ keyed by
`array_key`; the JSONL carries the manifest and the digest that reads verify.

**3. Live-safe failure.** Capture runs with the flag ON inside the authoritative
pregame pricing invocation -- that is the whole point of same-process capture,
and it is why this module must never be able to abort production. `capture()`
and `finalize()` therefore never raise. A failure records a research-scoped
sentinel, pricing continues untouched, and the session is marked invalid so no
row from it can become a section 9 prospective lock. Research callers opt into
loudness with `assert_session_valid()`.

**4. Session-scoped output.** Every pricing invocation opens a session and
writes to its own directory, so repeated pregame runs cannot accumulate into one
ambiguous file and a crash between capture and finalize cannot leak stale rows
into a later session in the same interpreter.

Nothing here writes to `outputs/props_priced_clean.csv`, the workbook, or any
production artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.utils.canonical_names import canonicalize_player_name_safe

FLAG = "RB_PD2_SHADOW_CAPTURE"
DEFAULT_ROOT = Path("data/research/rb_pd2_shadow")

ELIGIBLE_POSITIONS = {"RB", "HB", "FB"}
ELIGIBLE_MARKET = "rush_yards"

DRAWS_FILE = "baseline_draws.npz"
MANIFEST_FILE = "baseline_capture.jsonl"
RECEIPT_FILE = "session_receipt.json"

# Identity fields that must be present and non-blank for a capture to be usable
# as a section 9 join key.
REQUIRED_IDENTITY = ("event_id", "team", "opponent", "player_clean_key")

_SESSION: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------
def capture_enabled() -> bool:
    """True only when the research flag is explicitly turned on."""
    return str(os.getenv(FLAG, "")).strip().lower() in {"1", "true", "yes", "on"}


def is_eligible(position: str, market: str) -> bool:
    return str(position or "").upper().strip() in ELIGIBLE_POSITIONS and str(market) == ELIGIBLE_MARKET


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _provenance() -> dict:
    """Immutable run identity, so section 9 can prove which code produced a lock."""
    return {
        "code_sha": os.getenv("GITHUB_SHA", ""),
        "workflow": os.getenv("GITHUB_WORKFLOW", ""),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "workflow_job": os.getenv("GITHUB_JOB", ""),
        "runner_ref": os.getenv("GITHUB_REF", ""),
        # Player identity is resolved through a mutable alias table. If it
        # changes between the section 3 history build and a capture, the same
        # back gets two keys -- so section 9 can compare these fingerprints and
        # refuse to join across a table it cannot prove was identical.
        "manual_name_overrides_sha256": _file_digest(Path("data/manual_name_overrides.csv")),
        "roles_ourlads_sha256": _file_digest(Path("data/roles_ourlads.csv")),
    }


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def begin_session(*, season: int, out_root: Path | None = None) -> str:
    """Open a capture session, discarding any state left by an earlier one.

    Unconditionally replacing `_SESSION` is what prevents stale-buffer carryover:
    if a previous invocation raised after capturing but before finalizing, its
    records die here rather than contaminating this session's artifact.
    """
    global _SESSION
    session_id = f"{int(season)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    root = Path(out_root) if out_root is not None else DEFAULT_ROOT
    _SESSION = {
        "session_id": session_id,
        "season": int(season),
        "started_at_utc": _utc_now(),
        "dir": root / session_id,
        "provenance": _provenance(),
        "expected": set(),      # football keys noted AT the capture-eligible seam
        "records": {},          # football key -> manifest record
        "arrays": {},           # array_key -> exact float64 draws
        "sentinels": [],        # research-scoped failures
        "pricing_rows_seen": 0,
        "duplicate_rows_collapsed": 0,
        "finalized": False,
    }
    return session_id


def session_id() -> str:
    return "" if _SESSION is None else str(_SESSION["session_id"])


def pending() -> int:
    return 0 if _SESSION is None else len(_SESSION["records"])


def sentinels() -> list[dict]:
    return [] if _SESSION is None else list(_SESSION["sentinels"])


def reset() -> None:
    """Drop session state without writing. Test support only."""
    global _SESSION
    _SESSION = None


def _sentinel(kind: str, football_key: str, detail: str) -> None:
    if _SESSION is None:
        return
    _SESSION["sentinels"].append({
        "kind": kind,
        "football_key": football_key,
        "detail": detail,
        "at_utc": _utc_now(),
    })


# ---------------------------------------------------------------------------
# Array handling
# ---------------------------------------------------------------------------
def _digest(draws: np.ndarray) -> str:
    """Stable digest of the exact draw array, for lock-artifact reproducibility."""
    return hashlib.sha256(np.ascontiguousarray(draws, dtype=np.float64).tobytes()).hexdigest()


def _summary(draws: np.ndarray) -> dict:
    q05, q10, q50, q90, q95 = (float(v) for v in np.quantile(draws, [0.05, 0.10, 0.50, 0.90, 0.95]))
    return {
        "draw_count": int(draws.size),
        "draw_digest_sha256": _digest(draws),
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0,
        "q05": q05, "q10": q10, "q50": q50, "q90": q90, "q95": q95,
    }


def _football_key(record_id: dict) -> str:
    return "|".join(str(record_id[field]) for field in (
        "season", "week", "event_id", "team", "opponent", "player_clean_key", "market",
    ))


def _array_key(football_key: str) -> str:
    """NPZ-safe stable name. The readable key stays in the manifest."""
    return "draws_" + hashlib.sha256(football_key.encode("utf-8")).hexdigest()[:32]


def _num(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


# ---------------------------------------------------------------------------
# Expected-set accumulation (at the seam, independent of capture)
# ---------------------------------------------------------------------------
def _identity_text(value, *, upper: bool = False) -> str:
    """Normalize an identity scalar while treating pandas/NumPy missing as blank."""
    if value is None or bool(pd.isna(value)):
        return ""
    text = str(value).strip()
    return text.upper() if upper else text


def canonical_player_key(row) -> tuple[str, str]:
    """Return (canonical key, the raw source value it was derived from).

    `metrics_ready.player_clean_key` is NOT already canonical. Every consumer
    canonicalizes it defensively -- `ml_v2.apply_ml_to_metrics`,
    `state_v2.apply_state_to_metrics` and `backtest/component_predictions.py`
    all map the column through `canonicalize_player_name_safe` before joining --
    because `canonicalize_player_name()` is a remapper, not a normalizer: it
    resolves through a manual map and the Ourlads roles lookup, and every entry
    in that manual map changes the key. Several are running backs
    (`zonovanknight` -> `bamknight`, `chrisrodriguez` -> `chrisrodriguezjr`,
    `lequintallen` -> `lequintallenjr`).

    Storing the raw value would give section 7 a different identity system from
    the section 3 history built off the component path, so the section 9 join
    would silently drop or split those player-games. The same source precedence
    as the adapters is used deliberately, so one player has one key everywhere.
    """
    source_value = row.get("player_clean_key") if "player_clean_key" in row else row.get("player")
    source = _identity_text(source_value)
    if not source:
        return "", source
    _, canonical = canonicalize_player_name_safe(source)
    return str(canonical or "").strip(), source


def _identity(row, *, season: int, week: int) -> dict:
    canonical, _source = canonical_player_key(row)
    return {
        "season": int(season),
        "week": int(week),
        "event_id": _identity_text(row.get("event_id")),
        "team": _identity_text(row.get("team"), upper=True),
        "opponent": _identity_text(row.get("opponent"), upper=True),
        "player_clean_key": canonical,
        "market": ELIGIBLE_MARKET,
    }


def note_expected(*, row, market: str, position: str, season: int, week: int) -> bool:
    """Record that one eligible player-game reached the capture-eligible seam.

    Called from the pricing loop immediately before `capture()`, so a key is
    noted only once the row has a valid `base_outcomes`, a resolved `mc_proj`
    and final football `target_mean`, and an exact `adjusted_outcomes` baseline.
    That is the frozen section 5 denominator: a row the pricing loop dropped
    earlier never had a baseline distribution and was never a forward
    observation, so it must not appear here.

    Deliberately a separate call from `capture()` rather than a side effect of
    it. If `capture()` fails to store a scientifically eligible key -- by
    raising, by sentinel, or by silently returning -- this set still holds the
    key, and finalization reports it missing. Deriving the expected set from
    anything `capture()` produced would make that class undetectable.
    """
    if _SESSION is None or not is_eligible(position, market):
        return False
    identity = _identity(row, season=season, week=week)
    if any(not identity[f] for f in REQUIRED_IDENTITY):
        # `capture()` raises its own `blank_identity` sentinel for this row.
        return False
    _SESSION["expected"].add(_football_key(identity))
    return True


def noted_expected_keys() -> set[str]:
    return set() if _SESSION is None else set(_SESSION["expected"])


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def capture(
    *,
    row,
    adjusted_outcomes,
    target_mean: float,
    market: str,
    position: str,
    season: int,
    week: int,
    mc_proj: float = float("nan"),
) -> bool:
    """Record one baseline row. Returns True only when a new capture was stored.

    Never raises. A production pricing run may have the flag ON, so any failure
    here becomes a sentinel that invalidates the shadow session while leaving
    the canonical priced output completely untouched.
    """
    if _SESSION is None or not is_eligible(position, market):
        return False

    football_key = ""
    try:
        _SESSION["pricing_rows_seen"] += 1

        identity = _identity(row, season=season, week=week)
        _, _raw_player_key = canonical_player_key(row)
        football_key = _football_key(identity)

        missing = [f for f in REQUIRED_IDENTITY if not identity[f]]
        if missing:
            _sentinel("blank_identity", football_key, f"missing identity fields: {missing}")
            return False

        draws = np.array(adjusted_outcomes, dtype=np.float64, copy=True)
        if draws.ndim != 1 or draws.size == 0:
            _sentinel("bad_array_shape", football_key, f"shape={draws.shape}")
            return False
        if not np.isfinite(draws).all():
            _sentinel("non_finite_draw", football_key, "draw array contains a non-finite value")
            return False
        if (draws < 0).any():
            _sentinel("negative_draw", football_key, "rushing yard draws must be nonnegative")
            return False

        baseline = _summary(draws)
        target = _num(target_mean)
        mc = _num(mc_proj)

        prior = _SESSION["records"].get(football_key)
        if prior is not None:
            # A repeated book/line row for one player-game. The football
            # distribution must be identical; anything else means pricing state
            # changed mid-loop and the capture set cannot be trusted.
            mismatches = []
            if baseline["draw_digest_sha256"] != prior["baseline"]["draw_digest_sha256"]:
                mismatches.append("draw_digest_sha256")
            if baseline["draw_count"] != prior["baseline"]["draw_count"]:
                mismatches.append("draw_count")
            if not _same_float(target, prior["target_mean"]):
                mismatches.append("target_mean")
            if not _same_float(mc, prior["mc_proj"]):
                mismatches.append("mc_proj")
            if mismatches:
                _sentinel(
                    "duplicate_football_key_mismatch", football_key,
                    f"repeated pricing row disagrees on {mismatches}",
                )
                return False
            prior["duplicate_pricing_rows"] += 1
            _SESSION["duplicate_rows_collapsed"] += 1
            return False

        array_key = _array_key(football_key)
        # `target_mean` is non-finite when the ensemble mean is unavailable; that
        # is a legitimate production fallback (`adjusted_outcomes = base_outcomes`),
        # not a capture failure, so the row is marked lock-ineligible rather than
        # invalidating the whole session.
        eligible = np.isfinite(target)
        record = {
            "session_id": _SESSION["session_id"],
            "captured_at_utc": _utc_now(),
            "football_key": football_key,
            "array_key": array_key,
            "array_file": DRAWS_FILE,
            **identity,
            "player": str(row.get("player") or ""),
            # Section 3 of the lineage preflight requires an alias remap to be
            # auditable rather than inferred on the fly, so the pre-canonical
            # value travels with the record.
            "player_clean_key_source": _raw_player_key,
            "player_clean_key_remapped": bool(_raw_player_key and _raw_player_key != identity["player_clean_key"]),
            "position": str(position).upper().strip(),
            "target_mean": target,
            "mc_proj": mc,
            "commence_time": str(row.get("commence_time") or ""),
            "baseline": baseline,
            "duplicate_pricing_rows": 1,
            "baseline_lock_eligible": bool(eligible),
            "baseline_lock_ineligible_reason": "" if eligible else "non_finite_target_mean",
            # Explicit section 9 integrity flags, asserted at the point of capture.
            "sportsbook_inputs_used_in_candidate": False,
            "production_output_mutated": False,
            "outcome_present_at_lock": False,
        }
        _SESSION["records"][football_key] = record
        _SESSION["arrays"][array_key] = draws
        return True
    except Exception as exc:  # pragma: no cover - defensive; proven by the failure test
        _sentinel("capture_exception", football_key, f"{type(exc).__name__}: {exc}")
        return False


def _same_float(a: float, b: float) -> bool:
    """Exact equality, treating two NaNs as agreeing."""
    a_nan, b_nan = bool(np.isnan(a)), bool(np.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return a == b


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
def expected_keys(metrics, *, season: int) -> set[str]:
    """Eligible football keys present in the pricing frame, before the seam.

    This is a *diagnostic* population, not the validity gate. It counts every
    RB/HB/FB `rush_yards` player-game in `metrics_ready`, including rows the
    pricing loop later drops (`run_pricing_v2.py:226-229` drops a row whose
    simulation lookup returned nothing). Such a row never obtained an empirical
    baseline, so it is not part of the frozen section 5 forward population and
    must not invalidate a session -- `note_expected()` supplies the gate.

    It is still worth measuring: a slate where simulation quietly misses several
    eligible backs yields a smaller study population than the frame implies, and
    a forward confirmation that cannot see its own population shrink is not
    trustworthy. `finalize()` reports the difference as
    `dropped_before_seam_keys` without failing the session.

    Production's own `_position_family` / `_runtime_week` / `MARKET_MAP` do the
    resolving, so this cannot drift away from the eligibility test the hook
    applies. Keys are sportsbook-independent, so book/line expansion collapses
    here exactly as it does in `capture()`.
    """
    # Imported lazily: `run_pricing_v2` imports this module from inside `price()`,
    # so it is fully loaded by the time this runs, and no import cycle forms.
    from scripts.run_pricing_v2 import _position_family, _runtime_week
    from scripts.simulation_v2 import MARKET_MAP

    keys: set[str] = set()
    for _, row in metrics.iterrows():
        raw_market = str(row.get("market", "") or "").lower()
        market = MARKET_MAP.get(raw_market, raw_market)
        if not is_eligible(_position_family(row), market):
            continue
        identity = _identity(row, season=season, week=_runtime_week(row))
        if any(not identity[f] for f in REQUIRED_IDENTITY):
            continue
        keys.add(_football_key(identity))
    return keys


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------
def finalize(
    *,
    expected_football_keys: set[str] | None = None,
    pre_seam_eligible_keys: set[str] | None = None,
) -> dict:
    """Write the session artifacts and return a receipt. Never raises.

    `expected_football_keys` is the gate: it must be the set accumulated at the
    capture-eligible seam by `note_expected()`. `valid` is False when any
    sentinel fired, when one of those keys is missing, or when writing failed.
    A section 9 prospective lock may only be assembled from a valid receipt.

    `pre_seam_eligible_keys` is reported, never gated on. The difference between
    it and the gate set is how many eligible player-games the pricing loop
    dropped before they could obtain a baseline -- real information about
    population shrink, but not a capture defect.
    """
    if _SESSION is None:
        return {"session_id": "", "valid": False, "rows_written": 0,
                "sentinels": [{"kind": "no_session", "football_key": "", "detail": "finalize without begin_session"}]}

    receipt: dict[str, Any] = {
        "session_id": _SESSION["session_id"],
        "season": _SESSION["season"],
        "started_at_utc": _SESSION["started_at_utc"],
        "finalized_at_utc": _utc_now(),
        "provenance": _SESSION["provenance"],
        "pricing_rows_seen": _SESSION["pricing_rows_seen"],
        "duplicate_rows_collapsed": _SESSION["duplicate_rows_collapsed"],
        "rows_written": 0,
        "lock_eligible_rows": 0,
        "missing_expected_keys": [],
        "expected_key_count": 0 if expected_football_keys is None else len(expected_football_keys),
        "pre_seam_eligible_count": 0 if pre_seam_eligible_keys is None else len(pre_seam_eligible_keys),
        "dropped_before_seam_keys": [],
        "sentinels": list(_SESSION["sentinels"]),
        "valid": False,
        "dir": str(_SESSION["dir"]),
    }

    try:
        if pre_seam_eligible_keys is not None:
            # Diagnostic only: these never reached the seam, so they were never
            # forward observations. Reported so population shrink is visible.
            gate = set(expected_football_keys or ())
            receipt["dropped_before_seam_keys"] = sorted(set(pre_seam_eligible_keys) - gate)

        if expected_football_keys is not None:
            missing = sorted(set(expected_football_keys) - set(_SESSION["records"]))
            receipt["missing_expected_keys"] = missing
            if missing:
                _sentinel("missing_expected_key", "", f"{len(missing)} expected football keys were never captured")
                receipt["sentinels"] = list(_SESSION["sentinels"])

        out_dir = Path(_SESSION["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lossless exact draws. `allow_pickle` stays off: these are plain float64.
        np.savez_compressed(out_dir / DRAWS_FILE, **_SESSION["arrays"])

        with (out_dir / MANIFEST_FILE).open("w", encoding="utf-8") as fh:
            for key in sorted(_SESSION["records"]):
                # allow_nan=False keeps a silent NaN out of the manifest; a
                # non-finite value that reached here is a real integrity problem.
                fh.write(json.dumps(_scrub(_SESSION["records"][key]), sort_keys=True, allow_nan=False) + "\n")

        receipt["rows_written"] = len(_SESSION["records"])
        receipt["lock_eligible_rows"] = sum(
            1 for r in _SESSION["records"].values() if r["baseline_lock_eligible"]
        )
        receipt["valid"] = not _SESSION["sentinels"]
    except Exception as exc:
        _sentinel("finalize_exception", "", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        receipt["sentinels"] = list(_SESSION["sentinels"])
        receipt["valid"] = False

    try:
        Path(_SESSION["dir"]).mkdir(parents=True, exist_ok=True)
        (Path(_SESSION["dir"]) / RECEIPT_FILE).write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception:
        pass

    _SESSION["finalized"] = True
    return receipt


def _scrub(record: dict) -> dict:
    """Replace non-finite floats with None so JSON stays strict-parseable."""
    def fix(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        return value
    return {k: fix(v) for k, v in record.items()}


# ---------------------------------------------------------------------------
# Research-side readers (never called from production)
# ---------------------------------------------------------------------------
def assert_session_valid(receipt: dict) -> None:
    """Loud counterpart to the live-safe capture path. Research callers only."""
    if not receipt.get("valid"):
        raise RuntimeError(
            f"shadow capture session {receipt.get('session_id')!r} is invalid: "
            f"sentinels={receipt.get('sentinels')} "
            f"missing_expected_keys={receipt.get('missing_expected_keys')}"
        )


def load_session(session_dir: Path) -> tuple[list[dict], dict]:
    """Return (manifest records, receipt) for a finalized session."""
    session_dir = Path(session_dir)
    records = [
        json.loads(line)
        for line in (session_dir / MANIFEST_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    receipt = json.loads((session_dir / RECEIPT_FILE).read_text(encoding="utf-8"))
    return records, receipt


def load_draws(session_dir: Path, record: dict) -> np.ndarray:
    """Load one record's exact draws, verifying the digest before returning."""
    with np.load(Path(session_dir) / record["array_file"], allow_pickle=False) as npz:
        draws = np.asarray(npz[record["array_key"]], dtype=np.float64)
    digest = _digest(draws)
    if digest != record["baseline"]["draw_digest_sha256"]:
        raise RuntimeError(
            f"shadow draw digest mismatch for {record['football_key']}: "
            f"stored={record['baseline']['draw_digest_sha256']} loaded={digest}"
        )
    if int(draws.size) != int(record["baseline"]["draw_count"]):
        raise RuntimeError(f"shadow draw count mismatch for {record['football_key']}")
    return draws
