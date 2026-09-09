from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_r8_receiving_identity_v1 as r8
from scripts.modeling.rb_r17_tail_distribution_adapter_v1 import ResidualPools, adapt_rb_receiving_tail

VERSION = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1"
SEASON = 2026
WEEK = 1
HISTORY_START = 2013
HISTORY_THROUGH = 2025
ADAPTER_SEED = 918

MODEL_PATH = Path("data/models/rb_r19_runtime/rb_r19_tail_scorer_model_v1.json")
POOLS_PATH = Path("data/models/rb_r19_runtime/rb_r19_residual_pools_v1.npz")
AUDIT_JSON = Path("data/rb_receiving_tail_production_audit.json")
TRACE_CSV = Path("data/rb_receiving_tail_production_trace.csv")

EXPECTED_MODEL_SHA256 = "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba"
EXPECTED_POOLS_SHA256 = "c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362"
EXPECTED_R19 = {
    "run_id": 34288244770,
    "artifact_id": 10080377483,
    "digest": "sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7",
    "head_sha": "6ac1342f737f142acac6a3e4b459f442faf1442a",
}
EXPECTED_POOL_SEMANTICS = {
    "non_tail": (3890, "f677e91cd25cdbd6db044e9decccec6312943b99f8ffc25a827527e54d4d7b1d"),
    "tail_30_49": (174, "3da7bf656fcab3c111c8d5fb60e38fb7f735d7fcce639dabad899431f613c225"),
    "tail_50_plus": (79, "ee203d3ffa01b8687e7774d817d0dce6cdc8b56ca62321a5583d5287dfd5ee43"),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_f64_sorted(values: np.ndarray) -> str:
    x = np.sort(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(x.tobytes()).hexdigest()


def _position_family(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("position_family", frame.get("position", pd.Series("", index=frame.index)))
    return source.fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})


def _manual_linear(payload: dict, frame: pd.DataFrame, *, logistic: bool) -> np.ndarray:
    feats = list(payload["feature_order"])
    missing = [c for c in feats if c not in frame.columns]
    if missing:
        raise RuntimeError(f"R22 scorer missing frozen features: {missing}")
    x = frame[feats].to_numpy(float)
    mean = np.asarray(payload["scaler_mean"], float)
    scale = np.asarray(payload["scaler_scale"], float)
    coef = np.asarray(payload["coefficients"], float)
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coef):
        raise RuntimeError("R22 serialized model dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coef).all():
        raise RuntimeError("R22 nonfinite serialized model or feature input")
    if np.any(scale <= 0):
        raise RuntimeError("R22 serialized scaler has nonpositive scale")
    eta = ((x - mean) / scale) @ coef + float(payload["intercept"])
    if logistic:
        eta = np.clip(eta, -700, 700)
        return 1.0 / (1.0 + np.exp(-eta))
    return eta


def _load_assets(model_path: Path = MODEL_PATH, pools_path: Path = POOLS_PATH) -> tuple[dict, dict[str, np.ndarray], dict]:
    if not model_path.is_file() or not pools_path.is_file():
        raise RuntimeError(
            f"R22 exact R19 runtime assets missing. model={model_path} pools={pools_path}. "
            "Production must download the pinned R19 artifact before pricing."
        )
    model_sha = _sha256_file(model_path)
    pools_sha = _sha256_file(pools_path)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"R22 R19 model hash drift: {model_sha}")
    if pools_sha != EXPECTED_POOLS_SHA256:
        raise RuntimeError(f"R22 R19 pool-file hash drift: {pools_sha}")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    contract_ok = bool(
        model.get("candidate") == "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1"
        and int(model.get("version", -1)) == 1
        and model.get("status") == "SHADOW_ONLY"
        and int(model.get("fit_for_season", -1)) == SEASON
        and model.get("git_sha") == EXPECTED_R19["head_sha"]
        and int(model.get("sportsbook_inputs_added", -1)) == 0
        and int(model.get("production_parameters_changed", -1)) == 0
    )
    if not contract_ok:
        raise RuntimeError("R22 R19 model contract drift")

    with np.load(pools_path) as z:
        if set(z.files) != set(EXPECTED_POOL_SEMANTICS):
            raise RuntimeError(f"R22 residual pool members drifted: {z.files}")
        pools = {k: np.asarray(z[k], dtype=float).copy() for k in z.files}
    pool_audit = {}
    for name, (expected_n, expected_hash) in EXPECTED_POOL_SEMANTICS.items():
        vals = pools[name]
        observed_hash = _sha_f64_sorted(vals)
        ok = bool(
            len(vals) == expected_n
            and observed_hash == expected_hash
            and np.isfinite(vals).all()
            and (len(vals) < 2 or np.all(vals[:-1] <= vals[1:]))
            and int(model["residual_pools"][name]["count"]) == expected_n
            and model["residual_pools"][name]["sha256_f64_sorted"] == expected_hash
        )
        pool_audit[name] = {
            "count": int(len(vals)),
            "expected_count": expected_n,
            "sha256_f64_sorted": observed_hash,
            "expected_sha256_f64_sorted": expected_hash,
            "pass": ok,
        }
        if not ok:
            raise RuntimeError(f"R22 residual pool semantic drift: {name} {pool_audit[name]}")
    return model, pools, {
        "model_sha256": model_sha,
        "pools_file_sha256": pools_sha,
        "pool_audit": pool_audit,
        "r19_source": EXPECTED_R19,
    }


def apply_rb_receiving_tail_production(
    result,
    metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
    model_path: Path = MODEL_PATH,
    pools_path: Path = POOLS_PATH,
    seed: int = ADAPTER_SEED,
):
    if int(season) != SEASON or int(week) != WEEK:
        raise RuntimeError(f"{VERSION} is qualified only for 2026 Week 1, got season={season} week={week}")

    model, pool_values, asset_audit = _load_assets(model_path, pools_path)
    frame = metrics.copy()
    required = {
        "event_id", "team", "player_clean_key", "entitlement_tgt_share",
        "rules_plays_est", "rules_pass_rate", "rules_ypt",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"R22 live football frame missing columns: {sorted(missing)}")
    if frame.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError("R22 live football frame has duplicate event/team/player keys")

    frame["position_family"] = _position_family(frame)
    for c in ["entitlement_tgt_share", "rules_plays_est", "rules_pass_rate", "rules_ypt"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    rb = frame.loc[frame.position_family.isin({"RB", "FB"})].copy()
    if len(rb) == 0 or rb.position_family.eq("RB").sum() == 0:
        raise RuntimeError("R22 live football frame has zero RB rows")
    numeric = rb[["entitlement_tgt_share", "rules_plays_est", "rules_pass_rate", "rules_ypt"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (rb.entitlement_tgt_share < 0).any():
        raise RuntimeError("R22 live RB inputs are missing/nonfinite/negative")

    states, prev = r8._identity_atlas(HISTORY_START, HISTORY_THROUGH)
    current_time_key = SEASON * 100 + WEEK
    state_time = pd.to_numeric(states.get("time_key", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(states) == 0 or len(state_time) == 0 or int(state_time.max()) >= current_time_key:
        raise RuntimeError("R22 strict-prior identity history source is not clean through 2025")

    rb["season"] = SEASON
    rb["week"] = WEEK
    rb = r8._attach_identity(rb, SEASON, WEEK, states, prev)
    r8_payload = model["models"]["r8_r9_identity"]
    r8_features = list(r8_payload["feature_order"])
    if r8_features != list(r8.FEATURES):
        raise RuntimeError("R22 R8 identity feature order drift")
    if rb[r8_features].isna().any().any() or not np.isfinite(rb[r8_features].to_numpy(float)).all():
        raise RuntimeError("R22 strict-prior R8 feature frame incomplete/nonfinite")

    team_attempts = frame.groupby(["event_id", "team"], as_index=False).agg(
        rules_plays_est=("rules_plays_est", "mean"),
        rules_pass_rate=("rules_pass_rate", "mean"),
    )
    team_attempts["team_pass_attempt_projection"] = team_attempts.rules_plays_est * team_attempts.rules_pass_rate
    rb = rb.merge(
        team_attempts[["event_id", "team", "team_pass_attempt_projection"]],
        on=["event_id", "team"], how="left", validate="many_to_one",
    )
    rb["baseline_entitlement_tgt_share"] = rb.entitlement_tgt_share.astype(float)
    rb["baseline_pred_targets"] = rb.team_pass_attempt_projection * rb.baseline_entitlement_tgt_share

    means = []
    for row in rb.itertuples(index=False):
        key = (str(row.event_id), str(row.player_clean_key), "rec_yards")
        if key not in result.values:
            raise RuntimeError(f"R22 missing canonical RB/FB rec_yards array: {key}")
        means.append(float(np.asarray(result.values[key], float).mean()))
    rb["baseline_pred_rec_yards"] = means
    rb["frozen_ypt"] = rb.rules_ypt.astype(float)

    raw = _manual_linear(r8_payload, rb, logistic=False)
    raw = np.clip(raw, -float(r8_payload["prediction_clip"]), float(r8_payload["prediction_clip"]))
    reliability = float(r8_payload["r9_reliability"])
    rb["r9_raw_r8_residual"] = raw
    rb["r9_reliability"] = reliability
    rb["r9_calibrated_residual"] = reliability * rb.r9_raw_r8_residual
    rb["r9_shadow_entitlement_tgt_share"] = rb.baseline_entitlement_tgt_share.astype(float)

    pool_gaps = []
    for (_, _), idx in rb.groupby(["event_id", "team"], sort=False).groups.items():
        g = rb.loc[idx]
        base_pool = float(g.baseline_entitlement_tgt_share.sum())
        if base_pool <= 0:
            raise RuntimeError("R22 encountered nonpositive governed RB target pool")
        within = g.baseline_entitlement_tgt_share.to_numpy(float) / base_pool
        score = np.log(np.clip(within, 0.0, None) + float(r8.EPS)) + g.r9_calibrated_residual.to_numpy(float)
        weights = np.exp(score - np.max(score))
        weights = weights / weights.sum()
        candidate = base_pool * weights
        candidate[int(np.argmax(weights))] += base_pool - float(candidate.sum())
        rb.loc[idx, "r9_shadow_entitlement_tgt_share"] = candidate
        pool_gaps.append(abs(float(candidate.sum()) - base_pool))
    max_r9_pool_gap = float(max(pool_gaps)) if pool_gaps else float("inf")
    if max_r9_pool_gap > 1e-12:
        raise RuntimeError(f"R22 R9 shadow feature pool conservation failed: {max_r9_pool_gap}")

    rb["r9_shadow_pred_targets"] = rb.team_pass_attempt_projection * rb.r9_shadow_entitlement_tgt_share
    rb["target_delta"] = rb.r9_shadow_pred_targets - rb.baseline_pred_targets
    rb["identity_pct"] = rb.groupby(["season", "week"])["prior_rb_room_share"].rank(pct=True, method="average")
    rb["identity_top20"] = rb.identity_pct.gt(0.80).astype(float)
    rb["state_probability"] = 0.0
    top = rb.identity_top20.eq(1.0)
    if not top.any():
        raise RuntimeError("R22 current slate produced zero TOP20 receiving-identity rows")
    rb.loc[top, "state_probability"] = _manual_linear(model["models"]["r11_high5"], rb.loc[top], logistic=True)
    rb["p30"] = _manual_linear(model["models"]["r16_cat30"], rb, logistic=True)
    rb["p50"] = _manual_linear(model["models"]["r16_cat50"], rb, logistic=True)
    rb["nested_p50"] = np.minimum(rb.p30, rb.p50)

    probs = rb[["state_probability", "p30", "p50", "nested_p50"]].to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise RuntimeError("R22 scorer produced invalid probabilities")
    if not (rb.loc[~top, "state_probability"] == 0.0).all():
        raise RuntimeError("R22 REST80 state probability is not exactly zero")

    risk = rb[["event_id", "player_clean_key", "p30", "p50"]].copy()
    pools = ResidualPools(
        non_tail=pool_values["non_tail"],
        tail_30_49=pool_values["tail_30_49"],
        tail_50_plus=pool_values["tail_50_plus"],
    )
    before = {k: np.asarray(v).copy() for k, v in result.values.items()}
    adapted, raw_adapter_audit = adapt_rb_receiving_tail(result, frame, risk, pools, seed=seed)
    adapted_2, _ = adapt_rb_receiving_tail(result, frame, risk, pools, seed=seed)

    posmap = {
        (str(r.event_id), str(r.player_clean_key)): str(r.position_family).upper()
        for r in frame.itertuples(index=False)
    }
    max_mean_delta = 0.0
    min_spearman = 1.0
    non_rb_exact = True
    rb_other_exact = True
    fb_exact = True
    receptions_exact = True
    deterministic = True
    finite_nonnegative = True
    rush_rec_identity = True
    trace_rows = []

    rb_probs = rb.set_index(["event_id", "player_clean_key"])
    for key, old in before.items():
        if key not in adapted.values or key not in adapted_2.values:
            raise RuntimeError(f"R22 adapted result dropped key: {key}")
        new = np.asarray(adapted.values[key], float)
        new2 = np.asarray(adapted_2.values[key], float)
        deterministic &= np.array_equal(new, new2)
        game, pkey, market = key
        pos = posmap.get((str(game), str(pkey)), "")
        is_rb = pos == "RB"
        is_fb = pos == "FB"
        if not is_rb:
            non_rb_exact &= np.array_equal(old, new)
        if is_fb:
            fb_exact &= np.array_equal(old, new)
        if is_rb and market not in {"rec_yards", "rush_rec_yards"}:
            rb_other_exact &= np.array_equal(old, new)
        if market == "receptions":
            receptions_exact &= np.array_equal(old, new)
        if is_rb and market == "rec_yards":
            finite_nonnegative &= bool(np.isfinite(new).all() and (new >= 0.0).all())
            delta = abs(float(new.mean()) - float(np.asarray(old, float).mean()))
            max_mean_delta = max(max_mean_delta, delta)
            rho = 1.0
            oldf = np.asarray(old, float)
            if len(oldf) and not np.allclose(oldf, oldf[0], atol=0, rtol=0):
                rho = float(pd.Series(oldf).corr(pd.Series(new), method="spearman"))
                if not np.isfinite(rho):
                    rho = -1.0
            min_spearman = min(min_spearman, rho)
            ix = (str(game), str(pkey))
            r = rb_probs.loc[ix]
            trace_rows.append({
                "event_id": str(game), "team": str(r.team),
                "player_clean_key": str(pkey), "position_family": "RB",
                "rb_receiving_tail_applied": True,
                "rb_receiving_tail_version": VERSION,
                "canonical_mean": float(oldf.mean()), "adapted_mean": float(new.mean()),
                "mean_delta": float(new.mean() - oldf.mean()), "spearman": rho,
                "canonical_q50": float(np.quantile(oldf, 0.50)), "adapted_q50": float(np.quantile(new, 0.50)),
                "canonical_q75": float(np.quantile(oldf, 0.75)), "adapted_q75": float(np.quantile(new, 0.75)),
                "canonical_q90": float(np.quantile(oldf, 0.90)), "adapted_q90": float(np.quantile(new, 0.90)),
                "canonical_q95": float(np.quantile(oldf, 0.95)), "adapted_q95": float(np.quantile(new, 0.95)),
                "state_probability": float(r.state_probability), "p30": float(r.p30), "p50": float(r.p50),
                "baseline_pred_targets": float(r.baseline_pred_targets),
                "baseline_pred_rec_yards": float(r.baseline_pred_rec_yards),
                "prior_rb_room_share": float(r.prior_rb_room_share),
                "identity_top20": float(r.identity_top20),
            })

    for row in frame.loc[frame.position_family.eq("RB")].itertuples(index=False):
        game, pkey = str(row.event_id), str(row.player_clean_key)
        rush_key = (game, pkey, "rush_yards")
        rec_key = (game, pkey, "rec_yards")
        combo_key = (game, pkey, "rush_rec_yards")
        if all(k in adapted.values for k in [rush_key, rec_key, combo_key]):
            rush_rec_identity &= bool(np.allclose(
                np.asarray(adapted.values[combo_key], float),
                np.asarray(adapted.values[rush_key], float) + np.asarray(adapted.values[rec_key], float),
                atol=1e-10, rtol=0,
            ))

    trace = pd.DataFrame(trace_rows).sort_values(["event_id", "player_clean_key"], kind="mergesort").reset_index(drop=True)
    adapted_count = int(len(trace))
    expected_rb_count = int(frame.position_family.eq("RB").sum())
    if adapted_count != expected_rb_count:
        raise RuntimeError(f"R22 failed to adapt every RB: adapted={adapted_count} expected={expected_rb_count}")

    gates = {
        "qualified_week": True,
        "r19_assets_exact": True,
        "strict_prior_history": True,
        "r9_shadow_pool_conservation": max_r9_pool_gap <= 1e-12,
        "all_rb_adapted": adapted_count == expected_rb_count,
        "fb_exact": bool(fb_exact),
        "mean_parity": max_mean_delta <= 1e-8,
        "rank_preservation": min_spearman >= 0.9999,
        "finite_nonnegative_rec_yards": bool(finite_nonnegative),
        "deterministic_replay": bool(deterministic),
        "non_rb_exact": bool(non_rb_exact),
        "rb_nonreceiving_markets_exact": bool(rb_other_exact),
        "receptions_exact": bool(receptions_exact),
        "rush_rec_identity": bool(rush_rec_identity),
        "sportsbook_inputs_upstream_zero": True,
        "current_or_future_outcomes_zero": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"R22 production adapter gate failure: {gates}")

    TRACE_CSV.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(TRACE_CSV, index=False)
    payload = {
        "candidate": VERSION,
        "disposition": "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS",
        "integration_valid": True,
        "season": SEASON,
        "week": WEEK,
        "history_start": HISTORY_START,
        "history_through": HISTORY_THROUGH,
        "adapter_seed": int(seed),
        "football_rb_rows": expected_rb_count,
        "adapted_rb_rows": adapted_count,
        "fb_rows": int(frame.position_family.eq("FB").sum()),
        "max_mean_delta": max_mean_delta,
        "min_spearman": min_spearman,
        "max_r9_shadow_pool_gap": max_r9_pool_gap,
        "state_probability_min": float(rb.state_probability.min()),
        "state_probability_max": float(rb.state_probability.max()),
        "p30_min": float(rb.p30.min()), "p30_max": float(rb.p30.max()),
        "p50_min": float(rb.p50.min()), "p50_max": float(rb.p50.max()),
        "asset_audit": asset_audit,
        "raw_adapter_rows": int(len(raw_adapter_audit)),
        "trace": str(TRACE_CSV),
        "gates": gates,
        "sportsbook_inputs_added": 0,
        "current_or_future_outcomes_used": 0,
        "production_mean_parameters_changed": 0,
        "governance_note": "Qualified only for 2026 Week 1 RB receiving-yard distribution shape. Receptions/targets/means remain canonical.",
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    adapted.rb_receiving_tail_audit = payload
    return adapted, trace, payload
