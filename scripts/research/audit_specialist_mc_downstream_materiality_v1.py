#!/usr/bin/env python3
"""Frozen Week-3 specialist finite-MC downstream materiality audit.

No Week-3 outcomes are loaded. No sportsbook acquisition occurs. The audit
replays the preserved paid Week-3 football state and asks whether protected
finite-MC path drift changes downstream probabilities / EV / decisions more
than ordinary 25,000-draw resampling noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team\nfrom scripts.modeling.discrete_count_alignment_v1 import align_prealigned_outcomes
from scripts.modeling.ensemble_v2 import apply_ensemble, load_weights
from scripts.modeling.qb_pass_synthesis_v1 import (
    build_feature_dict,
    load_artifact as load_qb_synthesis_artifact,
    load_player_logs as load_qb_player_logs,
    load_team_context as load_qb_team_context,
    predict_correction as predict_qb_synthesis,
)
from scripts.operations.grade_market_track_record_v1 import _ev_roi, select_model_bet
from scripts.simulation_c2_qb_candidate import StateSimulationResult, apply_c2, simulate_with_states
from scripts.simulation_v2 import MARKET_MAP

ITERATIONS = 25000
PRODUCTION_SEED = 42
C2_SEED = 5601
ALT_SEEDS = [1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042, 10042, 11042, 12042]
TOL = 1e-12
MEAN_REPLAY_TOL = 1e-9
TARGET_REPLAY_TOL = 1e-8
PROB_REPLAY_TOL = 1.0 / ITERATIONS + 1e-12
SUPPORTED = {"pass_yards", "rush_yards", "rec_yards", "receptions"}
KEY = ["event_id", "player_clean_key", "market"]
PM_KEY = ["season", "week", "event_id", "player", "market"]
QUOTE_KEY = ["season", "week", "event_id", "player_clean_key", "market", "book", "vegas_line"]


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _unique(df: pd.DataFrame, cols: list[str], label: str) -> None:
    dup = df.duplicated(cols, keep=False)
    if dup.any():
        raise RuntimeError(f"{label} duplicate keys {cols}: {df.loc[dup, cols].head(20).to_dict('records')}")


def _canon_market(value: object) -> str:
    raw = str(value or "").lower().strip()
    return MARKET_MAP.get(raw, raw)


def _finite(value: object, default=np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _build_entitlement_state(root: Path) -> pd.DataFrame:
    target = _read_csv(root / "data/target_entitlement_v1_trace.csv", "target entitlement trace")
    te = _read_csv(root / "data/te_r5p_full_slate_entitlement_trace.csv", "TE trace")
    wr = _read_csv(root / "data/wr_r15_full_slate_entitlement_trace.csv", "WR trace")
    keys = ["event_id", "team", "player_clean_key"]
    _unique(target, keys, "target trace")
    _unique(te, keys, "TE trace")
    _unique(wr, keys, "WR trace")

    need = {"m38_explicit_entitlement_tgt_share", "entitlement_tgt_share", "wr_r15_anchor"}
    missing = need - set(target.columns)
    if missing:
        raise RuntimeError(f"target trace missing {sorted(missing)}")

    s = target[keys + ["position", "m38_explicit_entitlement_tgt_share", "entitlement_tgt_share", "wr_r15_anchor"]].copy()
    s = s.rename(columns={
        "m38_explicit_entitlement_tgt_share": "m38_entitlement",
        "entitlement_tgt_share": "final_entitlement",
    })
    t = te[keys + ["te_r5p_entitlement_tgt_share"]].copy()
    s = s.merge(t, on=keys, how="left", validate="one_to_one")
    s["m38_entitlement"] = pd.to_numeric(s["m38_entitlement"], errors="raise").astype(float)
    s["final_entitlement"] = pd.to_numeric(s["final_entitlement"], errors="raise").astype(float)
    s["te_r5p_entitlement_tgt_share"] = pd.to_numeric(s["te_r5p_entitlement_tgt_share"], errors="coerce")
    s["te_entitlement"] = s["te_r5p_entitlement_tgt_share"].combine_first(s["m38_entitlement"]).astype(float)
    s["te_delta"] = s["te_entitlement"] - s["m38_entitlement"]
    s["wr_delta"] = s["final_entitlement"] - s["te_entitlement"]
    s["te_protected"] = s["te_delta"].abs().le(TOL)
    s["wr_protected"] = s["wr_delta"].abs().le(TOL)

    w = wr[keys + ["wr_r15_entitlement_tgt_share"]].merge(
        s[keys + ["final_entitlement"]], on=keys, how="left", validate="one_to_one"
    )
    gap = (
        pd.to_numeric(w["wr_r15_entitlement_tgt_share"], errors="raise")
        - pd.to_numeric(w["final_entitlement"], errors="raise")
    ).abs()
    if len(gap) and float(gap.max()) > TOL:
        raise RuntimeError(f"WR trace/final entitlement mismatch max={float(gap.max())}")
    return s


def _provider_aliases(paid: pd.DataFrame) -> dict[str, str]:
    p = paid.copy()
    p["team"] = p["team"].map(canon_team)
    p["opponent"] = p["opponent"].map(canon_team)
    def canonical(r):
        a, b = sorted([str(r["team"]), str(r["opponent"])])
        return f"{int(r['season'])}_{int(r['week']):02d}_{a}_{b}"
    p["_canonical_event_id"] = p.apply(canonical, axis=1)
    aliases = {}
    for cg, g in p.groupby("_canonical_event_id", sort=False):
        ids = sorted(set(g["event_id"].astype(str)))
        if len(ids) != 1:
            raise RuntimeError(f"provider event identity ambiguous canonical={cg}: {ids}")
        aliases[str(cg)] = ids[0]
    return aliases


def _install_provider_aliases(result: StateSimulationResult, aliases: dict[str, str]) -> None:
    additions = {}
    for (game, pkey, market), values in list(result.values.items()):
        provider = aliases.get(str(game))
        if provider:
            additions[(provider, pkey, market)] = values
    result.values.update(additions)


def _stage_metrics(universe: pd.DataFrame, state: pd.DataFrame, entitlement_col: str, starters: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_id", "team", "player_clean_key"]
    x = universe.copy()
    x = x.merge(state[keys + [entitlement_col]], on=keys, how="left", validate="one_to_one")
    if x[entitlement_col].isna().any():
        raise RuntimeError(f"stage universe missing {entitlement_col}")
    x["entitlement_tgt_share"] = pd.to_numeric(x[entitlement_col], errors="raise").astype(float)

    # Replay the exact frozen 30 Week-3 primary-QB decisions from the paid artifact.
    x["qb_projection_eligible"] = 0
    x["qb_role_score"] = -999.0
    for r in starters.itertuples(index=False):
        mask = (
            x["event_id"].astype(str).eq(str(r.event_id))
            & x["team"].astype(str).eq(str(r.team))
            & x["player_clean_key"].astype(str).eq(str(r.primary_player_clean_key))
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"starter replay mismatch team={r.team} player={r.primary_player_clean_key} count={int(mask.sum())}")
        x.loc[mask, "qb_projection_eligible"] = 1
        x.loc[mask, "qb_role_score"] = 0.0
    return x


def _mean_map(result: StateSimulationResult) -> dict[tuple[str, str, str], float]:
    return {tuple(map(str, k)): float(np.mean(np.asarray(v, dtype=float))) for k, v in result.values.items()}


def _simulate_stage(metrics: pd.DataFrame, starters: pd.DataFrame, *, seed: int) -> tuple[StateSimulationResult, StateSimulationResult, pd.DataFrame]:
    base = simulate_with_states(metrics, iterations=ITERATIONS, seed=int(seed))
    anchor_map: dict[tuple[str, str], float] = {}
    c2_rows = []
    for r in starters.itertuples(index=False):
        key = (str(r.event_id), str(r.primary_player_clean_key), "pass_yards")
        arr = base.values.get(key)
        if arr is None or len(arr) != ITERATIONS:
            raise RuntimeError(f"missing primary QB base pass array key={key}")
        raw = np.asarray(arr, dtype=float)
        mean = float(raw.mean())
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid primary QB base mean key={key} mean={mean}")
        anchor_map[(str(r.event_id), str(r.team))] = mean
    selected = apply_c2(base, metrics, anchor_map=anchor_map, seed=C2_SEED)
    for r in starters.itertuples(index=False):
        key = (str(r.event_id), str(r.primary_player_clean_key), "pass_yards")
        a = np.asarray(base.values[key], dtype=float)
        b = np.asarray(selected.values[key], dtype=float)
        c2_rows.append({
            "event_id": str(r.event_id),
            "team": str(r.team),
            "player_clean_key": str(r.primary_player_clean_key),
            "canonical_raw_mean": float(a.mean()),
            "c2_raw_mean": float(b.mean()),
            "raw_mean_gap": float(b.mean() - a.mean()),
            "canonical_raw_sd": float(np.std(a, ddof=1)),
            "c2_raw_sd": float(np.std(b, ddof=1)),
            "canonical_p10": float(np.quantile(a, .10)),
            "canonical_p50": float(np.quantile(a, .50)),
            "canonical_p90": float(np.quantile(a, .90)),
            "c2_p10": float(np.quantile(b, .10)),
            "c2_p50": float(np.quantile(b, .50)),
            "c2_p90": float(np.quantile(b, .90)),
        })
    return base, selected, pd.DataFrame(c2_rows)


def _validate_parent_means(
    root: Path,
    stage_means: dict[str, dict[tuple[str, str, str], float]],
) -> dict[str, float]:
    te = _read_csv(root / "data/te_r5p_full_slate_simulation_delta.csv", "TE simulation delta")
    wr = _read_csv(root / "data/wr_r15_full_slate_simulation_delta.csv", "WR simulation delta")
    maxima = {}
    checks = [
        ("m38", te, "m38_baseline_mean"),
        ("te", te, "te_r5p_mean"),
        ("te_wrfile", wr, "te_r5p_mean"),
        ("wr", wr, "wr_r15_mean"),
    ]
    for label, frame, col in checks:
        source_name = "te" if label == "te_wrfile" else label
        m = stage_means[source_name]
        gaps = []
        for r in frame.itertuples(index=False):
            key = (str(r.event_id), str(r.player_clean_key), str(r.market))
            if key not in m:
                raise RuntimeError(f"missing reconstructed simulation key {key} for {label}")
            gaps.append(abs(float(m[key]) - float(getattr(r, col))))
        maxima[label] = float(max(gaps) if gaps else 0.0)
    return maxima


def _validate_c2_final(replayed: pd.DataFrame, frozen: pd.DataFrame) -> dict[str, float]:
    keys = ["event_id", "team", "player_clean_key"]
    cols = [
        "canonical_raw_mean", "c2_raw_mean", "canonical_raw_sd", "c2_raw_sd",
        "canonical_p10", "canonical_p50", "canonical_p90", "c2_p10", "c2_p50", "c2_p90",
    ]
    m = replayed.merge(frozen[keys + cols], on=keys, how="inner", validate="one_to_one", suffixes=("_replay", "_frozen"))
    if len(m) != len(frozen):
        raise RuntimeError(f"C2 frozen population mismatch replay={len(m)} frozen={len(frozen)}")
    return {
        col: float((pd.to_numeric(m[f"{col}_replay"]) - pd.to_numeric(m[f"{col}_frozen"])).abs().max())
        for col in cols
    }


def _representative_rule_rows(root: Path) -> dict[tuple[str, str, str], pd.Series]:
    r = _read_csv(root / "data/model_rule_simulation_inputs.csv", "model rule simulation inputs")
    r["_canonical_market"] = r["market"].map(_canon_market)
    out = {}
    for _, row in r.sort_values(["event_id", "player_clean_key", "_canonical_market", "book"], kind="mergesort").iterrows():
        key = (str(row["event_id"]), str(row["player_clean_key"]), str(row["_canonical_market"]))
        out.setdefault(key, row)
    return out


def _target_mean_full(
    *,
    market: str,
    mc_proj: float,
    paid_meta: pd.Series,
    rule_row: pd.Series | None,
    weights: pd.DataFrame,
    qb_bundle: dict,
) -> tuple[float, float]:
    comp = pd.DataFrame([{
        "market": market,
        "mc_proj": mc_proj,
        "ml_proj": paid_meta.get("ml_proj"),
        "state_proj": paid_meta.get("state_proj"),
    }])
    ens = apply_ensemble(comp, weights=weights).iloc[0]
    ensemble = float(ens["ensemble_proj"])
    if market != "pass_yards":
        return ensemble, ensemble
    if rule_row is None:
        raise RuntimeError(f"missing QB rule row key={(paid_meta.get('event_id'), paid_meta.get('player_clean_key'), market)}")
    features = build_feature_dict(
        rule_row,
        base_proj=ensemble,
        mc_proj=mc_proj,
        team_context=qb_bundle["team_context"],
        player_logs=qb_bundle["player_logs"],
        weather=qb_bundle["weather"],
        season=2026,
        week=3,
    )
    synth, _, _ = predict_qb_synthesis(features, artifact=qb_bundle["artifact"])
    return float(synth), ensemble


def _price_stage(
    result: StateSimulationResult,
    paid: pd.DataFrame,
    rule_rows: dict[tuple[str, str, str], pd.Series],
    weights: pd.DataFrame,
    qb_bundle: dict,
) -> dict[str, pd.DataFrame]:
    meta = paid.drop_duplicates(KEY, keep="first").copy()
    paid_by_key = {tuple(map(str, r)): g.copy() for r, g in paid.groupby(KEY, sort=False)}
    rows = {"SHAPE_ONLY_FIXED_FINAL_MEAN": [], "FULL_DOWNSTREAM_PROPAGATION": []}

    for mr in meta.itertuples(index=False):
        key = (str(mr.event_id), str(mr.player_clean_key), str(mr.market))
        arr = result.values.get(key)
        if arr is None or len(arr) != ITERATIONS:
            raise RuntimeError(f"missing simulated pricing key={key}")
        base = np.asarray(arr, dtype=float)
        if str(mr.market) == "pass_yards":
            conv = _finite(getattr(mr, "qb_attempt_conversion", np.nan))
            share = _finite(getattr(mr, "qb_pass_att_share", 1.0), 1.0)
            if not np.isfinite(conv):
                raise RuntimeError(f"missing QB attempt conversion key={key}")
            base = base * conv * share

        mc = float(base.mean())
        pm = paid_by_key[key].iloc[0]
        shape_target = float(pm["model_proj"])
        rule = rule_rows.get(key)
        full_target, ensemble = _target_mean_full(
            market=str(mr.market), mc_proj=mc, paid_meta=pm,
            rule_row=rule, weights=weights, qb_bundle=qb_bundle,
        )

        for surface, target in [
            ("SHAPE_ONLY_FIXED_FINAL_MEAN", shape_target),
            ("FULL_DOWNSTREAM_PROPAGATION", full_target),
        ]:
            eligible = bool(np.isfinite(mc) and mc > 0 and np.isfinite(target))
            adjusted = base * max(0.0, target / mc) if eligible else base.copy()
            adjusted, _ = align_prealigned_outcomes(
                adjusted, market=str(mr.market), eligible=eligible, target_mean=target
            )
            group = paid_by_key[key]
            over_by_line = {
                float(line): float(np.mean(adjusted > float(line)))
                for line in pd.to_numeric(group["vegas_line"], errors="raise").unique()
            }
            for rr in group.itertuples(index=False):
                line = float(rr.vegas_line)
                po = over_by_line[line]
                prob = po if str(rr.side).upper() == "OVER" else 1.0 - po
                rec = rr._asdict()
                rec["fair_prob"] = float(prob)
                rec["stage_mc_proj"] = mc
                rec["stage_ensemble_proj"] = ensemble
                rec["stage_target_mean"] = float(target)
                rec["stage_model_proj"] = float(np.mean(adjusted))
                rec["stage_ev_roi"] = float(_ev_roi(prob, rr.vegas_odds))
                rows[surface].append(rec)

    return {k: pd.DataFrame(v) for k, v in rows.items()}


def _quote_state(board: pd.DataFrame) -> pd.DataFrame:
    b = board.copy()
    b["_ev"] = [
        _ev_roi(p, o) for p, o in zip(pd.to_numeric(b["fair_prob"], errors="coerce"), pd.to_numeric(b["vegas_odds"], errors="coerce"))
    ]
    b["_side_rank"] = b["side"].astype(str).str.upper().map({"OVER": 0, "UNDER": 1}).fillna(9)
    keys = [c for c in QUOTE_KEY if c in b.columns]
    q = b.sort_values(keys + ["_ev", "_side_rank"], ascending=[True] * len(keys) + [False, True], kind="mergesort")
    q = q.drop_duplicates(keys, keep="first").copy()
    q["quote_has_edge"] = q["_ev"].gt(0)
    return q


def _best_ev_table(board: pd.DataFrame) -> pd.DataFrame:
    q = _quote_state(board)
    key = [c for c in PM_KEY if c in q.columns]
    q["_book_key"] = q.get("book", "").astype("string").fillna("").str.lower()
    q = q.sort_values(key + ["_ev", "_book_key", "vegas_line"], ascending=[True] * len(key) + [False, True, True], kind="mergesort")
    out = q.drop_duplicates(key, keep="first").copy()
    out["best_ev"] = out["_ev"]
    return out


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(float)
    rb = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ra) <= 0 or np.std(rb) <= 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _q(series: pd.Series, p: float) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(p)) if len(series) else float("nan")


def _compare_boards(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protected_keys: set[tuple[str, str]],
    *,
    comparison: str,
    surface: str,
    kind: str,
    seed: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    l = left.loc[
        [(str(e), str(p)) in protected_keys for e, p in zip(left["event_id"], left["player_clean_key"])]
    ].copy()
    r = right.loc[
        [(str(e), str(p)) in protected_keys for e, p in zip(right["event_id"], right["player_clean_key"])]
    ].copy()

    if set(l["paid_row_id"]) != set(r["paid_row_id"]):
        raise RuntimeError(f"paid-row universe mismatch {comparison} {surface}")
    m = l[["paid_row_id", "fair_prob", "stage_ev_roi", "event_id", "player", "player_clean_key", "team", "market", "book", "vegas_line", "side"]].merge(
        r[["paid_row_id", "fair_prob", "stage_ev_roi"]],
        on="paid_row_id", how="inner", validate="one_to_one", suffixes=("_left", "_right")
    )
    m["abs_prob_delta"] = (m["fair_prob_right"] - m["fair_prob_left"]).abs()
    m["abs_ev_delta"] = (m["stage_ev_roi_right"] - m["stage_ev_roi_left"]).abs()

    ql = _quote_state(l)
    qr = _quote_state(r)
    qkeys = [c for c in QUOTE_KEY if c in ql.columns]
    qm = ql[qkeys + ["side", "_ev", "quote_has_edge"]].merge(
        qr[qkeys + ["side", "_ev", "quote_has_edge"]],
        on=qkeys, how="inner", validate="one_to_one", suffixes=("_left", "_right")
    )
    quote_side_flips = int(qm["side_left"].astype(str).ne(qm["side_right"].astype(str)).sum())
    quote_edge_flips = int(qm["quote_has_edge_left"].ne(qm["quote_has_edge_right"]).sum())

    sl = select_model_bet(l)
    sr = select_model_bet(r)
    pkeys = [c for c in PM_KEY if c in l.columns]
    all_pm = l[pkeys].drop_duplicates().merge(r[pkeys].drop_duplicates(), on=pkeys, how="outer")
    def _sel_map(s: pd.DataFrame) -> dict[tuple, tuple]:
        out = {}
        for rr in s.itertuples(index=False):
            k = tuple(getattr(rr, c) for c in pkeys)
            out[k] = (
                str(getattr(rr, "side", "")),
                str(getattr(rr, "book", "")),
                float(getattr(rr, "vegas_line")),
                float(getattr(rr, "vegas_odds")),
            )
        return out
    lm, rm = _sel_map(sl), _sel_map(sr)
    bet_pass = side_flip = identity = 0
    for rr in all_pm.itertuples(index=False):
        k = tuple(getattr(rr, c) for c in pkeys)
        a, b = lm.get(k), rm.get(k)
        if (a is None) != (b is None):
            bet_pass += 1
        if a is not None and b is not None:
            if a[0] != b[0]:
                side_flip += 1
            if a != b:
                identity += 1

    bl = _best_ev_table(l)
    br = _best_ev_table(r)
    bm = bl[pkeys + ["best_ev"]].merge(br[pkeys + ["best_ev"]], on=pkeys, how="inner", suffixes=("_left", "_right"))
    corr = _rank_corr(bm["best_ev_left"], bm["best_ev_right"]) if len(bm) else float("nan")
    best_abs = (bm["best_ev_right"] - bm["best_ev_left"]).abs() if len(bm) else pd.Series(dtype=float)

    def _top_turnover(frame_a: pd.DataFrame, frame_b: pd.DataFrame, k: int) -> int:
        aa = frame_a.sort_values("best_ev", ascending=False).head(k)
        bb = frame_b.sort_values("best_ev", ascending=False).head(k)
        ka = {tuple(x) for x in aa[pkeys].itertuples(index=False, name=None)}
        kb = {tuple(x) for x in bb[pkeys].itertuples(index=False, name=None)}
        return int(min(len(ka), len(kb)) - len(ka & kb))

    summary = {
        "kind": kind,
        "comparison": comparison,
        "surface": surface,
        "seed": int(seed) if seed is not None else np.nan,
        "protected_side_rows": int(len(m)),
        "protected_quotes": int(len(qm)),
        "protected_player_markets": int(len(all_pm)),
        "mean_abs_prob_delta": float(m["abs_prob_delta"].mean()) if len(m) else np.nan,
        "median_abs_prob_delta": float(m["abs_prob_delta"].median()) if len(m) else np.nan,
        "p90_abs_prob_delta": _q(m["abs_prob_delta"], .90),
        "p95_abs_prob_delta": _q(m["abs_prob_delta"], .95),
        "p99_abs_prob_delta": _q(m["abs_prob_delta"], .99),
        "max_abs_prob_delta": float(m["abs_prob_delta"].max()) if len(m) else np.nan,
        "mean_abs_ev_delta": float(m["abs_ev_delta"].mean()) if len(m) else np.nan,
        "p95_abs_ev_delta": _q(m["abs_ev_delta"], .95),
        "p99_abs_ev_delta": _q(m["abs_ev_delta"], .99),
        "max_abs_ev_delta": float(m["abs_ev_delta"].max()) if len(m) else np.nan,
        "quote_preferred_side_flips": quote_side_flips,
        "quote_has_edge_pass_flips": quote_edge_flips,
        "best_snapshot_bet_pass_flips": int(bet_pass),
        "best_snapshot_side_flips": int(side_flip),
        "best_snapshot_identity_changes": int(identity),
        "mean_abs_best_ev_delta": float(best_abs.mean()) if len(best_abs) else np.nan,
        "max_abs_best_ev_delta": float(best_abs.max()) if len(best_abs) else np.nan,
        "best_ev_spearman": corr,
        "top10_turnover": _top_turnover(bl, br, 10),
        "top25_turnover": _top_turnover(bl, br, 25),
    }
    m["kind"] = kind
    m["comparison"] = comparison
    m["surface"] = surface
    m["seed"] = int(seed) if seed is not None else np.nan
    return summary, m


def _protected_sets(state: pd.DataFrame, aliases: dict[str, str] | None = None) -> dict[str, set[tuple[str, str]]]:
    aliases = aliases or {}
    event = state["event_id"].astype(str).map(lambda x: aliases.get(x, x))
    return {
        "TE_R5P_PROTECTED": set(
            zip(
                event.loc[state["te_protected"]],
                state.loc[state["te_protected"], "player_clean_key"].astype(str),
            )
        ),
        "WR_R15_PROTECTED": set(
            zip(
                event.loc[state["wr_protected"]],
                state.loc[state["wr_protected"], "player_clean_key"].astype(str),
            )
        ),
    }


def _envelope(resampling: pd.DataFrame, scope: str, surface: str) -> dict[str, float]:
    g = resampling.loc[
        resampling["comparison"].eq(scope) & resampling["surface"].eq(surface)
    ]
    metrics = [
        "p99_abs_prob_delta", "p99_abs_ev_delta", "quote_preferred_side_flips",
        "quote_has_edge_pass_flips", "best_snapshot_bet_pass_flips",
        "best_snapshot_identity_changes", "top10_turnover", "top25_turnover",
        "max_abs_best_ev_delta",
    ]
    return {m: float(pd.to_numeric(g[m], errors="coerce").max()) for m in metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-run-id", default="36293274478")
    ap.add_argument("--source-artifact-id", default="10923570170")
    ap.add_argument("--source-artifact-digest", default="sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480")
    args = ap.parse_args()
    root = args.root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _build_entitlement_state(root)
    universe = _read_csv(root / "data/football_simulation_universe.csv", "football universe")
    starters = _read_csv(root / "data/qb_c2_production_starter_audit.csv", "QB C2 starter audit")
    c2_frozen = _read_csv(root / "data/qb_c2_production_integration_audit.csv", "QB C2 integration audit")
    c2_payload = _read_json(root / "data/qb_c2_production_integration_audit.json", "QB C2 integration payload")
    if len(starters) != 30 or int(c2_payload.get("selected_qb_rows", -1)) != 30 or len(c2_frozen) != 30:
        raise RuntimeError("frozen Week-3 QB C2 paid authority is not exact 30-row selected slate")

    paid = _read_csv(root / "outputs/props_priced_clean.csv", "paid priced board")
    paid = paid.loc[paid["market"].isin(sorted(SUPPORTED))].copy().reset_index(drop=True)
    paid["paid_row_id"] = np.arange(len(paid), dtype=int)
    aliases = _provider_aliases(paid)
    if paid.empty:
        raise RuntimeError("no supported paid board rows")
    rule_rows = _representative_rule_rows(root)
    weights = load_weights(Path("data/model_ensemble_weights.csv"))
    if weights.empty:
        raise RuntimeError("working data/model_ensemble_weights.csv unavailable")

    qb_bundle = {
        "artifact": load_qb_synthesis_artifact(),
        "team_context": load_qb_team_context(),
        "player_logs": load_qb_player_logs(),
        "weather": pd.read_csv("data/weather_week.csv", low_memory=False) if Path("data/weather_week.csv").exists() else pd.DataFrame(),
    }

    metrics = {
        "m38": _stage_metrics(universe, state, "m38_entitlement", starters),
        "te": _stage_metrics(universe, state, "te_entitlement", starters),
        "wr": _stage_metrics(universe, state, "final_entitlement", starters),
    }

    stage_boards: dict[str, dict[str, pd.DataFrame]] = {}
    stage_means: dict[str, dict[tuple[str, str, str], float]] = {}
    c2_final_replay = None
    for name in ("m38", "te", "wr"):
        base, selected, c2_diag = _simulate_stage(metrics[name], starters, seed=PRODUCTION_SEED)
        stage_means[name] = _mean_map(base)
        _install_provider_aliases(selected, aliases)
        stage_boards[name] = _price_stage(selected, paid, rule_rows, weights, qb_bundle)
        if name == "wr":
            c2_final_replay = c2_diag
        del base, selected

    parent_mean_gaps = _validate_parent_means(root, stage_means)
    c2_gaps = _validate_c2_final(c2_final_replay, c2_frozen)

    final_full = stage_boards["wr"]["FULL_DOWNSTREAM_PROPAGATION"]
    final_shape = stage_boards["wr"]["SHAPE_ONLY_FIXED_FINAL_MEAN"]
    pay = paid.set_index("paid_row_id")
    ff = final_full.set_index("paid_row_id")
    fs = final_shape.set_index("paid_row_id")
    max_mc_gap = float((pd.to_numeric(ff["stage_mc_proj"]) - pd.to_numeric(pay["mc_proj"])).abs().max())
    max_target_gap = float((pd.to_numeric(ff["stage_target_mean"]) - pd.to_numeric(pay["model_proj"])).abs().max())
    max_full_prob_gap = float((pd.to_numeric(ff["fair_prob"]) - pd.to_numeric(pay["fair_prob"])).abs().max())
    max_shape_prob_gap = float((pd.to_numeric(fs["fair_prob"]) - pd.to_numeric(pay["fair_prob"])).abs().max())

    integrity = {
        "source_run_id": str(args.source_run_id),
        "source_artifact_id": str(args.source_artifact_id),
        "source_artifact_digest": str(args.source_artifact_digest),
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "production_changed": False,
        "paid_supported_rows": int(len(paid)),
        "parent_stage_mean_max_gaps": parent_mean_gaps,
        "c2_final_replay_max_gaps": c2_gaps,
        "final_mc_proj_max_gap": max_mc_gap,
        "final_target_mean_max_gap": max_target_gap,
        "final_full_fair_prob_max_gap": max_full_prob_gap,
        "final_shape_fair_prob_max_gap": max_shape_prob_gap,
        "parent_stage_mean_replay_pass": bool(max(parent_mean_gaps.values()) <= MEAN_REPLAY_TOL),
        "c2_replay_pass": bool(max(c2_gaps.values()) <= 1e-8),
        "final_mc_replay_pass": bool(max_mc_gap <= MEAN_REPLAY_TOL),
        "final_target_replay_pass": bool(max_target_gap <= TARGET_REPLAY_TOL),
        "final_probability_replay_pass": bool(max(max_full_prob_gap, max_shape_prob_gap) <= PROB_REPLAY_TOL),
        "code_identity_verified_by_workflow": os.getenv("CODE_IDENTITY_VERIFIED", "") == "1",
    }
    integrity["all_integrity_gates_pass"] = bool(
        integrity["parent_stage_mean_replay_pass"]
        and integrity["c2_replay_pass"]
        and integrity["final_mc_replay_pass"]
        and integrity["final_target_replay_pass"]
        and integrity["final_probability_replay_pass"]
        and integrity["code_identity_verified_by_workflow"]
    )
    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not integrity["all_integrity_gates_pass"]:
        payload = {
            "version": "WEEK3_SPECIALIST_MC_DOWNSTREAM_MATERIALITY_V1",
            "disposition": "SPECIALIST_MC_DOWNSTREAM_MATERIALITY_INTEGRITY_FAILURE",
            "integrity": integrity,
            "production_changed": False,
            "week3_outcomes_used": False,
            "repair_authorized": False,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(payload["disposition"])

    protected = _protected_sets(state, aliases)
    specialist_rows = []
    specialist_detail = []
    comparisons = [
        ("M38_TO_TE_R5P", "m38", "te", "TE_R5P_PROTECTED"),
        ("TE_R5P_TO_WR_R15", "te", "wr", "WR_R15_PROTECTED"),
    ]
    for cname, left, right, scope in comparisons:
        for surface in ("SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION"):
            s, d = _compare_boards(
                stage_boards[left][surface], stage_boards[right][surface], protected[scope],
                comparison=cname, surface=surface, kind="SPECIALIST",
            )
            s["protected_scope"] = scope
            specialist_rows.append(s)
            d["protected_scope"] = scope
            specialist_detail.append(d)

    # Ordinary finite-MC resampling benchmark: identical final football state, alternate seeds.
    resampling_rows = []
    for seed in ALT_SEEDS:
        _, selected, _ = _simulate_stage(metrics["wr"], starters, seed=seed)
        _install_provider_aliases(selected, aliases)
        alt_boards = _price_stage(selected, paid, rule_rows, weights, qb_bundle)
        del selected
        for scope in ("TE_R5P_PROTECTED", "WR_R15_PROTECTED"):
            for surface in ("SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION"):
                s, _ = _compare_boards(
                    stage_boards["wr"][surface], alt_boards[surface], protected[scope],
                    comparison=scope, surface=surface, kind="RESAMPLING", seed=seed,
                )
                s["protected_scope"] = scope
                resampling_rows.append(s)

    specialist_summary = pd.DataFrame(specialist_rows)
    specialist_detail_df = pd.concat(specialist_detail, ignore_index=True) if specialist_detail else pd.DataFrame()
    resampling_summary = pd.DataFrame(resampling_rows)

    primary_metrics = [
        "p99_abs_prob_delta", "p99_abs_ev_delta", "quote_preferred_side_flips",
        "quote_has_edge_pass_flips", "best_snapshot_bet_pass_flips",
        "best_snapshot_identity_changes", "top10_turnover", "top25_turnover",
        "max_abs_best_ev_delta",
    ]
    any_material = False
    exceeds = False
    envelope_payload = {}
    for _, row in specialist_summary.iterrows():
        scope = str(row["protected_scope"])
        surface = str(row["surface"])
        env = _envelope(resampling_summary, scope, surface)
        envelope_payload[f"{row['comparison']}|{surface}"] = env
        row_material = any(
            float(row[m]) > 0
            for m in [
                "quote_preferred_side_flips", "quote_has_edge_pass_flips",
                "best_snapshot_bet_pass_flips", "best_snapshot_identity_changes",
                "top10_turnover", "top25_turnover",
            ]
        )
        any_material = any_material or row_material
        core_decision = (
            float(row["best_snapshot_bet_pass_flips"]) > 0
            or float(row["best_snapshot_identity_changes"]) > 0
        )
        metric_exceeds = any(
            np.isfinite(float(row[m])) and float(row[m]) > float(env[m]) + 1e-15
            for m in primary_metrics
        )
        if core_decision and metric_exceeds:
            exceeds = True

    if exceeds:
        disposition = "SPECIALIST_MC_MATERIALITY_EXCEEDS_ORDINARY_RESAMPLING_NOISE"
        repair_authorized = False
        rng_isolation_design_authorized = True
    elif any_material:
        disposition = "SPECIALIST_MC_MATERIAL_BUT_WITHIN_ORDINARY_RESAMPLING_NOISE"
        repair_authorized = False
        rng_isolation_design_authorized = False
    else:
        disposition = "SPECIALIST_MC_DOWNSTREAM_NOT_MATERIAL"
        repair_authorized = False
        rng_isolation_design_authorized = False

    specialist_summary.to_csv(out_dir / "specialist_summary.csv", index=False)
    specialist_detail_df.to_csv(out_dir / "specialist_detail.csv", index=False)
    resampling_summary.to_csv(out_dir / "resampling_summary.csv", index=False)
    top = (
        specialist_detail_df.sort_values(["abs_prob_delta", "abs_ev_delta"], ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    top.to_csv(out_dir / "top_material_moves.csv", index=False)

    payload = {
        "version": "WEEK3_SPECIALIST_MC_DOWNSTREAM_MATERIALITY_V1",
        "disposition": disposition,
        "source_run_id": str(args.source_run_id),
        "source_artifact_id": str(args.source_artifact_id),
        "source_artifact_digest": str(args.source_artifact_digest),
        "iterations": ITERATIONS,
        "production_seed": PRODUCTION_SEED,
        "resampling_seeds": ALT_SEEDS,
        "supported_markets": sorted(SUPPORTED),
        "excluded_markets": {
            "anytime_td": "not dedicated-science certified",
            "rush_rec_yards": "RB Rush+Receiving Conservation V2 separate pathwise semantics",
            "rush_att": "no paid Week-3 sportsbook rows",
        },
        "integrity": integrity,
        "specialist": specialist_summary.to_dict("records"),
        "ordinary_resampling_envelope": envelope_payload,
        "production_changed": False,
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "repair_authorized": repair_authorized,
        "rng_isolation_design_authorized": rng_isolation_design_authorized,
        "next_if_ordinary_noise_material": (
            "SEPARATELY_FROZEN_MC_CONVERGENCE_DECISION_STABILITY_STUDY"
            if disposition == "SPECIALIST_MC_MATERIAL_BUT_WITHIN_ORDINARY_RESAMPLING_NOISE"
            else ""
        ),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
