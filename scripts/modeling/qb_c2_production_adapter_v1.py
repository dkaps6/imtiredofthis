"""Mean-neutral production adapter for the frozen Phase-J QB C2 selector.

This adapter sits entirely inside the football simulation layer.  It takes the
exact state-capturing canonical simulation produced from the current 469-player
full-roster universe, creates the already-qualified C2 QB passing distribution,
and exposes that C2 shape only for teams selected by
``QB_DISTRIBUTION_STATE_SELECTOR_V1``.

Hard contracts:
- M89/M90 remains the point-mean authority downstream.
- C2 is anchored to each canonical raw QB simulation mean before selection, so
  the MC/ensemble mean is unchanged.
- only the selected starting QB's ``pass_yards`` array may change.
- starter selection is football-only: Ourlads is the fallback and a versioned
  official-team authority row may supersede stale depth ordering.
- selector inputs are the frozen six Phase-J football features only.
- sportsbook fields are forbidden from starter selection, selector features,
  candidate generation, and distribution replacement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.qb_distribution_state_v1 import load_artifact, select_c2
from scripts.modeling.qb_pass_synthesis_v1 import attempt_conversion, load_team_context
from scripts.simulation_c2_qb_candidate import StateSimulationResult, apply_c2
from scripts.simulation_v2 import _player_key
from scripts.utils.player_identity_v3 import player_name_key

DATA = Path("data")
STATE_CONTEXT = DATA / "qb_distribution_state_context.csv"
STARTER_AUTHORITY = Path("config/qb_starter_authority_v1.csv")
AUDIT_CSV = DATA / "qb_c2_production_integration_audit.csv"
AUDIT_JSON = DATA / "qb_c2_production_integration_audit.json"
STARTER_AUDIT = DATA / "qb_c2_production_starter_audit.csv"

FORBIDDEN_SELECTOR_FIELDS = {
    "line", "source_line", "over_odds", "under_odds", "book", "book_title",
    "vegas_line", "vegas_odds", "market_prob", "edge_pct", "edge_abs",
    "home_wp", "away_wp", "team_wp",
}


def _f(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _key(value: object) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def _role_rank(value: object) -> int:
    text = str(value or "").upper().replace(" ", "")
    if text in {"QB1", "QB01"} or "START" in text or "FIRST" in text:
        return 1
    if text in {"QB2", "QB02"} or "SECOND" in text:
        return 2
    if text in {"QB3", "QB03"} or "THIRD" in text:
        return 3
    if text in {"QB4", "QB04"} or "FOURTH" in text:
        return 4
    return 99


def _load_starter_authority(season: int, week: int, path: Path = STARTER_AUTHORITY) -> pd.DataFrame:
    required = {
        "season", "week", "team", "starter", "authority_type",
        "authority_date", "source_url", "reason",
    }
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame(columns=sorted(required | {"starter_key"}))
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"QB starter authority missing columns: {missing}")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["team"] = frame["team"].map(canon_team)
    frame["starter_key"] = frame["starter"].map(_key)
    frame = frame.loc[
        frame["season"].eq(int(season)) & frame["week"].eq(int(week))
    ].copy()
    if frame.duplicated("team").any():
        sample = frame.loc[frame.duplicated("team", keep=False), ["team", "starter"]].to_dict("records")
        raise RuntimeError(f"duplicate QB starter authority rows: {sample}")
    allowed = {"official_team_announcement", "official_team_depth_chart"}
    bad = frame.loc[~frame["authority_type"].isin(allowed)]
    if not bad.empty:
        raise RuntimeError(f"unsupported QB starter authority type: {bad['authority_type'].tolist()}")
    if frame["starter_key"].eq("").any():
        raise RuntimeError("QB starter authority contains blank canonical starter")
    return frame


def annotate_primary_qbs(
    metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
    authority_path: Path = STARTER_AUTHORITY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach one football-only primary QB per team and return its provenance audit."""
    frame = metrics.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    need = {"event_id", "team", "player", "player_clean_key", "position", "depth_role"}
    missing = sorted(need - set(frame.columns))
    if missing:
        raise RuntimeError(f"QB C2 production universe missing starter fields: {missing}")

    leaked = sorted(FORBIDDEN_SELECTOR_FIELDS & set(frame.columns))
    if leaked:
        raise RuntimeError(f"sportsbook fields leaked into QB C2 football universe: {leaked}")

    authority = _load_starter_authority(int(season), int(week), authority_path)
    by_team = authority.set_index("team", drop=False) if not authority.empty else None

    frame["qb_projection_eligible"] = 0
    frame["qb_role_score"] = np.nan
    frame["qb_role_source"] = "not_qb"
    qb_mask = frame["position"].astype(str).str.upper().str.strip().eq("QB")
    rows: list[dict[str, Any]] = []

    for team, part in frame.loc[qb_mask].groupby("team", sort=True):
        team = canon_team(team)
        ranked = part.copy()
        ranked["_depth_rank"] = ranked["depth_role"].map(_role_rank)
        ranked["_identity_key"] = ranked["player"].map(_key)
        ranked = ranked.sort_values(["_depth_rank", "player_clean_key"], kind="mergesort")
        if ranked.empty or int(ranked.iloc[0]["_depth_rank"]) != 1:
            raise RuntimeError(f"team={team} has no football-only Ourlads QB1 fallback")
        if int(ranked["_depth_rank"].eq(1).sum()) != 1:
            sample = ranked.loc[ranked["_depth_rank"].eq(1), ["player", "depth_role"]].to_dict("records")
            raise RuntimeError(f"team={team} has ambiguous Ourlads QB1 fallback: {sample}")

        fallback_idx = ranked.index[0]
        primary_idx = fallback_idx
        source = "ourlads_depth_role_fallback"
        authority_date = ""
        source_url = ""
        reason = "No newer versioned official starter authority row for this season/week"

        if by_team is not None and team in by_team.index:
            ar = by_team.loc[team]
            if isinstance(ar, pd.DataFrame):
                raise RuntimeError(f"duplicate QB starter authority after index team={team}")
            wanted = str(ar["starter_key"])
            matches = ranked.loc[ranked["_identity_key"].eq(wanted)]
            if len(matches) != 1:
                sample = ranked[["player", "depth_role"]].to_dict("records")
                raise RuntimeError(
                    f"official QB starter not uniquely present in football roster team={team} "
                    f"starter={ar['starter']} roster={sample}"
                )
            primary_idx = matches.index[0]
            source = str(ar["authority_type"])
            authority_date = str(ar["authority_date"])
            source_url = str(ar["source_url"])
            reason = str(ar["reason"])

        frame.loc[ranked.index, "qb_role_score"] = -ranked["_depth_rank"].astype(float).to_numpy()
        frame.loc[ranked.index, "qb_role_source"] = "ourlads_depth_role"
        frame.at[primary_idx, "qb_projection_eligible"] = 1
        frame.at[primary_idx, "qb_role_score"] = 0.0
        frame.at[primary_idx, "qb_role_source"] = source
        rows.append({
            "event_id": str(frame.at[primary_idx, "event_id"]),
            "season": int(season),
            "week": int(week),
            "team": team,
            "primary_player": str(frame.at[primary_idx, "player"]),
            "primary_player_clean_key": str(frame.at[primary_idx, "player_clean_key"]),
            "primary_player_identity_key": _key(frame.at[primary_idx, "player"]),
            "primary_depth_role": str(frame.at[primary_idx, "depth_role"]),
            "ourlads_qb1_player": str(frame.at[fallback_idx, "player"]),
            "authority_source": source,
            "authority_date": authority_date,
            "authority_url": source_url,
            "authority_reason": reason,
            "authority_overrode_ourlads": int(primary_idx != fallback_idx),
            "sportsbook_inputs_used": 0,
        })

    audit = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    if len(audit) != 32 or audit["team"].nunique() != 32:
        raise RuntimeError(f"QB C2 starter authority must cover 32 teams, got rows={len(audit)}")
    if not audit["sportsbook_inputs_used"].eq(0).all():
        raise RuntimeError("QB C2 starter authority sportsbook leakage flag")
    STARTER_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(STARTER_AUDIT, index=False)
    return frame, audit


def _load_state_context(season: int, week: int, path: Path = STATE_CONTEXT) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"QB C2 production state context missing: {path}")
    ctx = pd.read_csv(path, low_memory=False)
    ctx.columns = [str(c).strip().lower() for c in ctx.columns]
    required = {
        "season", "week", "team", "opponent",
        "pass_opportunity_spot", "pass_efficiency_spot",
        "rush_opportunity_spot", "rush_efficiency_spot", "sportsbook_inputs_used",
    }
    missing = sorted(required - set(ctx.columns))
    if missing:
        raise RuntimeError(f"QB C2 state context missing columns: {missing}")
    ctx["season"] = pd.to_numeric(ctx["season"], errors="coerce")
    ctx["week"] = pd.to_numeric(ctx["week"], errors="coerce")
    ctx["team"] = ctx["team"].map(canon_team)
    ctx = ctx.loc[ctx["season"].eq(int(season)) & ctx["week"].eq(int(week))].copy()
    if len(ctx) != 32 or ctx["team"].nunique() != 32 or ctx.duplicated("team").any():
        raise RuntimeError(f"QB C2 state context must cover exactly 32 teams, got rows={len(ctx)}")
    if not pd.to_numeric(ctx["sportsbook_inputs_used"], errors="coerce").eq(0).all():
        raise RuntimeError("QB C2 state context sportsbook leakage flag")
    spots = [
        "pass_opportunity_spot", "pass_efficiency_spot",
        "rush_opportunity_spot", "rush_efficiency_spot",
    ]
    vals = ctx[spots].apply(pd.to_numeric, errors="coerce")
    if vals.isna().any().any() or not np.isfinite(vals.to_numpy(float)).all():
        raise RuntimeError("QB C2 state context contains non-finite spot features")
    return ctx


def apply_qb_c2_selector(
    base_state: StateSimulationResult,
    metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> tuple[StateSimulationResult, pd.DataFrame, dict]:
    """Return canonical simulation with C2 substituted only for selected starter QBs."""
    frame, starter_audit = annotate_primary_qbs(metrics, season=int(season), week=int(week))
    ctx = _load_state_context(int(season), int(week))
    ctx_by_team = ctx.set_index("team", drop=False)
    m89_context = load_team_context()
    selector_artifact = load_artifact()
    if int(selector_artifact.get("sportsbook_inputs_used", 1)) != 0:
        raise RuntimeError("QB C2 selector artifact sportsbook leakage flag")

    primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()
    if len(primary) != 32 or primary["team"].nunique() != 32:
        raise RuntimeError("QB C2 production adapter did not resolve exactly one primary QB per team")

    anchors: dict[tuple[str, str], float] = {}
    for r in primary.itertuples(index=False):
        game = str(getattr(r, "event_id")); team = canon_team(getattr(r, "team"))
        key = (game, str(getattr(r, "player_clean_key")), "pass_yards")
        values = base_state.values.get(key)
        if values is None or len(values) == 0:
            raise RuntimeError(f"canonical QB pass-yards array missing primary team={team} player={getattr(r,'player')}")
        mean = float(np.mean(np.asarray(values, float)))
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid canonical raw QB mean team={team} mean={mean}")
        anchors[(game, team)] = mean

    candidate = apply_c2(base_state, frame, anchor_map=anchors, seed=5601)
    selected_values = {k: np.asarray(v).copy() for k, v in base_state.values.items()}
    rows: list[dict[str, Any]] = []
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_changed: set[tuple[str, str, str]] = set()

    for _, row in primary.sort_values("team").iterrows():
        game = str(row["event_id"]); team = canon_team(row["team"]); pkey = str(row["player_clean_key"])
        cr = ctx_by_team.loc[team]
        if isinstance(cr, pd.DataFrame):
            raise RuntimeError(f"duplicate QB C2 context team={team}")

        plays = _f(row.get("rules_plays_est"))
        pass_rate = _f(row.get("rules_pass_rate"))
        conv = attempt_conversion(row, m89_context)
        qb_share = _f(row.get("qb_pass_att_share"), 1.0)
        qb_share = float(np.clip(qb_share, 0.0, 1.0)) if np.isfinite(qb_share) else 1.0
        pred_attempts = plays * pass_rate * conv * qb_share
        if not np.isfinite(pred_attempts) or pred_attempts <= 0:
            raise RuntimeError(
                f"QB C2 selector cannot construct promoted predicted attempts team={team} "
                f"plays={plays} pass_rate={pass_rate} conversion={conv} share={qb_share}"
            )

        features = {
            "pass_opportunity_spot": _f(cr.get("pass_opportunity_spot")),
            "pass_efficiency_spot": _f(cr.get("pass_efficiency_spot")),
            "rush_opportunity_spot": _f(cr.get("rush_opportunity_spot")),
            "rush_efficiency_spot": _f(cr.get("rush_efficiency_spot")),
            "pred_qb_attempts": pred_attempts,
            "week": float(week),
        }
        selected, delta, version = select_c2(features, selector_artifact)
        key = (game, pkey, "pass_yards")
        canonical = np.asarray(base_state.values[key], dtype=float)
        c2 = np.asarray(candidate.values[key], dtype=float)
        if canonical.shape != c2.shape or not np.isfinite(c2).all():
            raise RuntimeError(f"QB C2 candidate array invalid team={team} player={row.get('player')}")
        mean_gap = float(c2.mean() - canonical.mean())
        if abs(mean_gap) > 1e-10:
            raise RuntimeError(f"QB C2 raw mean-neutrality failed team={team} gap={mean_gap}")
        if selected:
            selected_values[key] = c2.copy()
            allowed_changed.add(key)

        record = {
            "event_id": game,
            "season": int(season),
            "week": int(week),
            "team": team,
            "opponent": canon_team(row.get("opponent")),
            "player": str(row.get("player")),
            "player_clean_key": pkey,
            "selector_version": version,
            "selector_delta_pass_attempts": float(delta),
            "selector_c2_selected": int(selected),
            "pred_qb_attempts": float(pred_attempts),
            "canonical_raw_mean": float(canonical.mean()),
            "c2_raw_mean": float(c2.mean()),
            "raw_mean_gap": mean_gap,
            "canonical_raw_sd": float(np.std(canonical, ddof=1)),
            "c2_raw_sd": float(np.std(c2, ddof=1)),
            "canonical_p10": float(np.quantile(canonical, .10)),
            "canonical_p50": float(np.quantile(canonical, .50)),
            "canonical_p90": float(np.quantile(canonical, .90)),
            "c2_p10": float(np.quantile(c2, .10)),
            "c2_p50": float(np.quantile(c2, .50)),
            "c2_p90": float(np.quantile(c2, .90)),
            "starter_authority_source": str(row.get("qb_role_source", "")),
            "sportsbook_inputs_to_selector": 0,
        }
        rows.append(record)
        meta[(team, pkey)] = {
            "applied": int(selected),
            "version": version,
            "delta_pass_attempts": float(delta),
            "pred_qb_attempts": float(pred_attempts),
            "raw_mean_gap": mean_gap,
            "starter_authority_source": str(row.get("qb_role_source", "")),
        }

    changed: set[tuple[str, str, str]] = set()
    max_nonselected_element_gap = 0.0
    for key in base_state.values:
        a = np.asarray(base_state.values[key], dtype=float)
        b = np.asarray(selected_values[key], dtype=float)
        if a.shape != b.shape:
            raise RuntimeError(f"QB C2 integration changed simulation shape key={key}")
        gap = float(np.max(np.abs(a - b))) if len(a) else 0.0
        if gap > 0:
            changed.add(key)
        if key not in allowed_changed:
            max_nonselected_element_gap = max(max_nonselected_element_gap, gap)
    illegal = changed - allowed_changed
    if illegal or max_nonselected_element_gap > 0:
        raise RuntimeError(
            "QB C2 integration changed non-selected/non-QB arrays; "
            f"illegal={list(illegal)[:20]} max_nonselected_gap={max_nonselected_element_gap}"
        )

    audit = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    selected_n = int(audit["selector_c2_selected"].sum())
    max_mean_gap = float(audit["raw_mean_gap"].abs().max())
    payload = {
        "disposition": "QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS",
        "selector_version": str(selector_artifact["version"]),
        "football_qb_rows": int(len(audit)),
        "selected_qb_rows": selected_n,
        "changed_simulation_keys": int(len(changed)),
        "all_changed_keys_are_selected_qb_pass_yards": bool(not illegal),
        "max_raw_qb_mean_gap": max_mean_gap,
        "max_nonselected_element_gap": max_nonselected_element_gap,
        "m89_m90_mean_authority_preserved_by_contract": True,
        "mc_ensemble_mean_preserved_by_contract": True,
        "starter_authority_overrides": int(starter_audit["authority_overrode_ourlads"].sum()),
        "sportsbook_inputs_to_starter_selection": 0,
        "sportsbook_inputs_to_selector": 0,
        "sportsbook_inputs_to_c2_generation": 0,
        "receiver_outputs_replaced": 0,
        "rb_outputs_replaced": 0,
        "production_distribution_specialist": "C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1",
        "audit_csv": str(AUDIT_CSV),
        "starter_audit_csv": str(STARTER_AUDIT),
    }
    if selected_n <= 0:
        raise RuntimeError("QB C2 production selector selected zero QBs")
    if max_mean_gap > 1e-10:
        raise RuntimeError(f"QB C2 production mean-neutral gate failed max_gap={max_mean_gap}")

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)
    AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_result = StateSimulationResult(selected_values, base_state.iterations, base_state.team_states)
    # Duck-typed metadata is intentionally attached to the simulation object so
    # run_pricing_v2 can expose per-QB lineage without using sportsbook identity.
    selected_result.qb_distribution_meta = meta
    selected_result.qb_distribution_audit = payload
    print("[qb_c2_production] " + json.dumps(payload, sort_keys=True))
    return selected_result, audit, payload
