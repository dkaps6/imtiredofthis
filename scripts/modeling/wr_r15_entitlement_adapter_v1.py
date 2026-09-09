"""Production-safe WR-R15 redistribution inside the conserved WR room.

Scientific authorization is frozen by WR-R15 OOS run 34238301577.  The final
coefficient contract is the all-history refit from run 34240496725.  This adapter
implements exactly that authorized mechanism:

- upstream explicit finite target entitlement already contains M38;
- the highest-entitlement WR in each team-game is the immutable M38 WR1 anchor;
- only the residual WR2+ pool is redistributed;
- redistribution uses strictly-prior offensive participation only;
- total WR-room mass, total team player mass, WR1 entitlement, non-WR
  entitlement, and the residual/unmodeled target bucket are immutable;
- sportsbook data is forbidden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.te_r5p_entitlement_adapter_v1 import _key, _load_snaps, _team

MODEL_PATH = Path("data/models/wr_r15_production_model_v1/wr_r15_production_model_v1.json")
EXPECTED_VERSION = "WR_R15_PRODUCTION_MODEL_V1"
EXPECTED_OOS_RUN = 34238301577
EXPECTED_OOS_ARTIFACT = 10061328722
EXPECTED_FINAL_FIT_RUN = 34240496725
EXPECTED_FINAL_FIT_ARTIFACT = 10062104621
WR_POS = {"WR", "LWR", "RWR", "SWR"}


def _load_model() -> dict:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size <= 0:
        raise RuntimeError(f"WR-R15 frozen production model missing: {MODEL_PATH}")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("model_version") != EXPECTED_VERSION:
        raise RuntimeError(f"unexpected WR-R15 model version: {model.get('model_version')}")
    if int(model.get("authorized_by_run", -1)) != EXPECTED_OOS_RUN:
        raise RuntimeError("WR-R15 model does not point to frozen OOS authorization run")
    if int(model.get("authorized_by_artifact", -1)) != EXPECTED_OOS_ARTIFACT:
        raise RuntimeError("WR-R15 model does not point to frozen OOS artifact")
    if int(model.get("source_final_fit_run", -1)) != EXPECTED_FINAL_FIT_RUN:
        raise RuntimeError("WR-R15 model does not point to frozen final-fit run")
    if int(model.get("source_final_fit_artifact", -1)) != EXPECTED_FINAL_FIT_ARTIFACT:
        raise RuntimeError("WR-R15 model does not point to frozen final-fit artifact")
    if model.get("training_seasons") != [2022, 2023, 2024, 2025]:
        raise RuntimeError("WR-R15 training seasons drifted")
    if model.get("scientific_confirmation_seasons") != [2023, 2024]:
        raise RuntimeError("WR-R15 scientific confirmation seasons drifted")
    if model.get("scientific_confirmation_2025_used") is not False:
        raise RuntimeError("WR-R15 contract incorrectly claims 2025 scientific confirmation")
    if int(model.get("sportsbook_inputs_used", -1)) != 0:
        raise RuntimeError("WR-R15 frozen model is not sportsbook independent")
    if float(model.get("ridge_alpha", np.nan)) != 20.0:
        raise RuntimeError("WR-R15 ridge alpha drifted")
    if float(model.get("eps", np.nan)) != 0.02:
        raise RuntimeError("WR-R15 epsilon drifted")
    if model.get("prediction_clip") != [-1.0, 1.0]:
        raise RuntimeError("WR-R15 prediction clip drifted")
    features = model.get("features", [])
    if not isinstance(features, list) or len(features) != 15:
        raise RuntimeError("WR-R15 feature contract is not length 15")
    for field in ("scaler_mean", "scaler_scale", "ridge_coef"):
        vals = np.asarray(model.get(field, []), dtype=float)
        if len(vals) != 15 or not np.isfinite(vals).all():
            raise RuntimeError(f"WR-R15 invalid parameter vector: {field}")
    scale = np.asarray(model["scaler_scale"], dtype=float)
    if np.any(scale <= 0):
        raise RuntimeError("WR-R15 scaler contains non-positive scale")
    if not np.isfinite(float(model.get("ridge_intercept", np.nan))):
        raise RuntimeError("WR-R15 invalid ridge intercept")
    return model


def _strict_prior_features(sec: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach participation observations strictly before each target game.

    ``_source_row_index`` is carried by value rather than reconstructed from a
    subset index.  This keeps production row identity explicit and avoids the
    historical row-index ambiguity discovered during RB-R6 research.
    """
    out = sec.copy().reset_index(drop=True)
    out["snap_player_key"] = out["player_clean_key"].map(_key)
    out["snap_team_key"] = out["team"].map(_team)
    season = pd.to_numeric(out["season"], errors="coerce")
    week = pd.to_numeric(out["week"], errors="coerce")
    if season.isna().any() or week.isna().any():
        raise RuntimeError("WR-R15 current slate missing season/week")
    out["ordinal"] = season * 100 + week

    any_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby("player_key", sort=False)}
    same_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby(["player_key", "team"], sort=False)}
    rows: list[dict] = []
    future_violations = 0

    def last_num(frame: pd.DataFrame, col: str) -> float:
        if frame.empty:
            return np.nan
        z = pd.to_numeric(frame[col], errors="coerce")
        return float(z.iloc[-1]) if len(z) and pd.notna(z.iloc[-1]) else np.nan

    def mean3(frame: pd.DataFrame, col: str) -> float:
        if len(frame) < 3:
            return np.nan
        z = pd.to_numeric(frame[col], errors="coerce")
        return float(z.mean()) if z.notna().any() else np.nan

    for _, r in out.iterrows():
        pk = str(r["snap_player_key"])
        tm = str(r["snap_team_key"])
        ordinal = float(r["ordinal"])
        ah = any_maps.get(pk, pd.DataFrame())
        sh = same_maps.get((pk, tm), pd.DataFrame())
        if len(ah):
            ah = ah.loc[pd.to_numeric(ah["ordinal"], errors="coerce").lt(ordinal)]
        if len(sh):
            sh = sh.loc[pd.to_numeric(sh["ordinal"], errors="coerce").lt(ordinal)]
        if len(ah) and float(pd.to_numeric(ah["ordinal"], errors="coerce").max()) >= ordinal:
            future_violations += 1
        if len(sh) and float(pd.to_numeric(sh["ordinal"], errors="coerce").max()) >= ordinal:
            future_violations += 1
        a1, a3, s1, s3 = ah.tail(1), ah.tail(3), sh.tail(1), sh.tail(3)
        rows.append({
            "prior_count_anyteam": int(len(ah)),
            "prior_count_same_team": int(len(sh)),
            "prior1_anyteam": bool(len(a1) >= 1),
            "prior3_anyteam": bool(len(a3) >= 3),
            "prior1_same_team": bool(len(s1) >= 1),
            "prior3_same_team": bool(len(s3) >= 3),
            "prior1_anyteam_offense_pct": last_num(a1, "offense_pct"),
            "prior1_anyteam_offense_snaps": last_num(a1, "offense_snaps"),
            "prior3_anyteam_offense_pct": mean3(a3, "offense_pct"),
            "prior3_anyteam_offense_snaps": mean3(a3, "offense_snaps"),
            "prior1_same_team_offense_pct": last_num(s1, "offense_pct"),
            "prior1_same_team_offense_snaps": last_num(s1, "offense_snaps"),
        })
    feat = pd.concat([out, pd.DataFrame(rows)], axis=1)
    return feat, int(future_violations)


def apply_wr_r15_entitlement(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply frozen WR-R15 to WR2+ while preserving the M38 WR1 anchor."""
    if metrics is None or metrics.empty:
        raise RuntimeError("WR-R15 cannot consume empty entitlement frame")
    out = metrics.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if not out.index.is_unique:
        raise RuntimeError("WR-R15 requires unique source row index")

    required = {
        "event_id", "season", "week", "team", "player", "player_clean_key",
        "position", "entitlement_tgt_share", "entitlement_residual_share",
    }
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"WR-R15 entitlement input missing columns: {sorted(missing)}")

    model = _load_model()
    baseline = pd.to_numeric(out["entitlement_tgt_share"], errors="coerce")
    if baseline.isna().any() or not np.isfinite(baseline.to_numpy(float)).all() or baseline.lt(0).any():
        raise RuntimeError("WR-R15 received invalid baseline entitlement")
    out["wr_r15_baseline_entitlement_tgt_share"] = baseline.astype(float)
    out["wr_r15_applied"] = False
    out["wr_r15_anchor"] = False
    out["wr_r15_model_version"] = ""
    out["wr_r15_route"] = ""

    pos = out["position"].astype("string").fillna("").str.upper().str.strip()
    wr_mask = pos.isin(WR_POS)
    wr = out.loc[wr_mask].copy()
    if wr.empty:
        raise RuntimeError("WR-R15 found zero wide receivers on Full Slate")

    anchor_indices: list[int] = []
    secondary_indices: list[int] = []
    for (_, _), g in wr.groupby(["event_id", "team"], sort=False):
        vals = pd.to_numeric(g["wr_r15_baseline_entitlement_tgt_share"], errors="raise")
        anchor = vals.idxmax()
        anchor_indices.append(anchor)
        secondary_indices.extend([i for i in g.index if i != anchor])

    if len(anchor_indices) != wr.groupby(["event_id", "team"]).ngroups:
        raise RuntimeError("WR-R15 failed to establish one WR1 anchor per team-game")
    if not secondary_indices:
        raise RuntimeError("WR-R15 found zero WR2+ rows")

    out.loc[anchor_indices, "wr_r15_anchor"] = True
    out.loc[anchor_indices, "wr_r15_model_version"] = EXPECTED_VERSION
    out.loc[anchor_indices, "wr_r15_route"] = "M38_WR1_ANCHOR"

    sec = out.loc[secondary_indices].copy()
    sec["_source_row_index"] = sec.index.astype(int)
    sec["b0_secondary_pool"] = sec.groupby(["event_id", "team"])["wr_r15_baseline_entitlement_tgt_share"].transform("sum")
    sec["b0_secondary_room_share"] = np.where(
        sec["b0_secondary_pool"].gt(0),
        sec["wr_r15_baseline_entitlement_tgt_share"] / sec["b0_secondary_pool"],
        0.0,
    )
    sec["log_b0_secondary_pool"] = np.log1p(sec["b0_secondary_pool"].clip(lower=0.0))
    sec["secondary_room_size"] = sec.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)

    snaps, dup_rate, source_seasons = _load_snaps()
    feat, future_violations = _strict_prior_features(sec, snaps)
    if future_violations != 0:
        raise RuntimeError(f"WR-R15 strict-prior construction used same/future rows: {future_violations}")

    feat["prior1_same_team_available"] = feat["prior1_same_team"].fillna(False).astype(float)
    feat["prior3_same_team_available"] = feat["prior3_same_team"].fillna(False).astype(float)
    feat["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(feat["prior_count_same_team"], errors="coerce").fillna(0).clip(lower=0))
    feat["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(feat["prior_count_anyteam"], errors="coerce").fillna(0).clip(lower=0))
    for src, dst in (
        ("prior1_same_team_offense_pct", "secondary_snap_share_prior1_same_team"),
        ("prior3_anyteam_offense_pct", "secondary_snap_share_prior3_anyteam"),
    ):
        z = pd.to_numeric(feat[src], errors="coerce").fillna(0.0).clip(lower=0.0)
        den = z.groupby([feat["event_id"], feat["team"]]).transform("sum")
        feat[dst] = np.where(den.gt(0), z / den, 0.0)

    features = list(model["features"])
    for c in features:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    x = feat[features].to_numpy(float)
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coef = np.asarray(model["ridge_coef"], dtype=float)
    residual = ((x - mean) / scale) @ coef + float(model["ridge_intercept"])
    lo, hi = [float(v) for v in model["prediction_clip"]]
    residual = np.clip(residual, lo, hi)
    eps = float(model["eps"])
    feat["wr_r15_residual"] = residual
    feat["wr_r15_score"] = np.log(feat["b0_secondary_room_share"].clip(lower=0.0).to_numpy(float) + eps) + residual
    feat["wr_r15_secondary_room_share"] = 0.0
    feat["wr_r15_entitlement_tgt_share"] = feat["wr_r15_baseline_entitlement_tgt_share"].astype(float)

    for (event_id, team), idx in feat.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(feat.loc[idx, "b0_secondary_pool"].iloc[0])
        score = feat.loc[idx, "wr_r15_score"].to_numpy(float)
        if pool <= 0:
            room_share = np.zeros(len(idx), dtype=float)
            candidate = np.zeros(len(idx), dtype=float)
        else:
            stable = score - float(np.max(score))
            w = np.exp(stable)
            room_share = w / float(w.sum())
            candidate = pool * room_share
            gap = pool - float(candidate.sum())
            if len(candidate):
                candidate[int(np.argmax(room_share))] += gap
        feat.loc[idx, "wr_r15_secondary_room_share"] = room_share
        feat.loc[idx, "wr_r15_entitlement_tgt_share"] = candidate

    final = feat.set_index("_source_row_index")["wr_r15_entitlement_tgt_share"]
    out.loc[final.index, "entitlement_tgt_share"] = final.astype(float)
    out.loc[final.index, "wr_r15_applied"] = True
    out.loc[final.index, "wr_r15_model_version"] = EXPECTED_VERSION
    out.loc[final.index, "wr_r15_route"] = "R15_WR2PLUS"

    baseline_ent = out["wr_r15_baseline_entitlement_tgt_share"].astype(float)
    final_ent = out["entitlement_tgt_share"].astype(float)
    anchor_delta = (final_ent.loc[anchor_indices] - baseline_ent.loc[anchor_indices]).abs()
    non_wr_delta = (final_ent.loc[~wr_mask] - baseline_ent.loc[~wr_mask]).abs()

    before_secondary = out.loc[secondary_indices].groupby(["event_id", "team"])["wr_r15_baseline_entitlement_tgt_share"].sum()
    after_secondary = out.loc[secondary_indices].groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    secondary_gap = (after_secondary - before_secondary).abs()
    before_wr = out.loc[wr_mask].groupby(["event_id", "team"])["wr_r15_baseline_entitlement_tgt_share"].sum()
    after_wr = out.loc[wr_mask].groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    wr_gap = (after_wr - before_wr).abs()
    before_team = out.groupby(["event_id", "team"])["wr_r15_baseline_entitlement_tgt_share"].sum()
    after_team = out.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    team_gap = (after_team - before_team).abs()

    residual_per_team = out.groupby(["event_id", "team"])["entitlement_residual_share"].nunique(dropna=False)
    if int(residual_per_team.max()) != 1:
        raise RuntimeError("WR-R15 found inconsistent residual share inside a team")

    max_anchor_delta = float(anchor_delta.max()) if len(anchor_delta) else 0.0
    max_non_wr_delta = float(non_wr_delta.max()) if len(non_wr_delta) else 0.0
    max_secondary_gap = float(secondary_gap.max()) if len(secondary_gap) else 0.0
    max_wr_gap = float(wr_gap.max()) if len(wr_gap) else 0.0
    max_team_gap = float(team_gap.max()) if len(team_gap) else 0.0
    if max_anchor_delta > 0:
        raise RuntimeError(f"WR-R15 changed M38 WR1 anchor entitlement: {max_anchor_delta}")
    if max_non_wr_delta > 0:
        raise RuntimeError(f"WR-R15 changed non-WR entitlement: {max_non_wr_delta}")
    if max_secondary_gap > 1e-12:
        raise RuntimeError(f"WR-R15 changed WR2+ pool mass: {max_secondary_gap}")
    if max_wr_gap > 1e-12:
        raise RuntimeError(f"WR-R15 changed total WR-room mass: {max_wr_gap}")
    if max_team_gap > 1e-12:
        raise RuntimeError(f"WR-R15 changed total team player entitlement: {max_team_gap}")

    feat["entitlement_delta"] = (
        feat["wr_r15_entitlement_tgt_share"].astype(float)
        - feat["wr_r15_baseline_entitlement_tgt_share"].astype(float)
    )
    trace_cols = [
        "event_id", "season", "week", "team", "player", "player_clean_key",
        "wr_r15_baseline_entitlement_tgt_share", "b0_secondary_pool",
        "b0_secondary_room_share", "secondary_room_size", "prior_count_anyteam",
        "prior_count_same_team", "prior1_same_team_offense_pct",
        "prior1_same_team_offense_snaps", "prior1_anyteam_offense_pct",
        "prior3_anyteam_offense_pct", "prior1_anyteam_offense_snaps",
        "prior3_anyteam_offense_snaps", "secondary_snap_share_prior1_same_team",
        "secondary_snap_share_prior3_anyteam", "wr_r15_residual", "wr_r15_score",
        "wr_r15_secondary_room_share", "wr_r15_entitlement_tgt_share",
        "entitlement_delta",
    ]
    trace = feat[trace_cols].copy()
    audit = {
        "disposition": "WR_R15_FULL_SLATE_ENTITLEMENT_READY",
        "model_version": EXPECTED_VERSION,
        "authorized_by_run": EXPECTED_OOS_RUN,
        "authorized_by_artifact": EXPECTED_OOS_ARTIFACT,
        "source_final_fit_run": EXPECTED_FINAL_FIT_RUN,
        "source_final_fit_artifact": EXPECTED_FINAL_FIT_ARTIFACT,
        "scientific_confirmation_seasons": [2023, 2024],
        "scientific_confirmation_2025_used": False,
        "training_seasons": model["training_seasons"],
        "training_rows": int(model["training_rows"]),
        "current_wr_rows": int(wr_mask.sum()),
        "current_wr1_anchor_rows": int(len(anchor_indices)),
        "current_wr2plus_rows": int(len(secondary_indices)),
        "teams_with_wr_rows": int(wr["team"].nunique()),
        "sportsbook_inputs_used": False,
        "current_or_future_outcomes_used": False,
        "strict_prior_participation_only": True,
        "m38_wr1_anchor_preserved": True,
        "wr2plus_pool_preserved": True,
        "wr_room_mass_preserved": True,
        "non_wr_entitlement_preserved": True,
        "team_total_player_entitlement_preserved": True,
        "target_residual_bucket_preserved": True,
        "max_wr1_anchor_entitlement_delta": max_anchor_delta,
        "max_wr2plus_pool_gap": max_secondary_gap,
        "max_wr_room_mass_gap": max_wr_gap,
        "max_non_wr_entitlement_delta": max_non_wr_delta,
        "max_team_player_entitlement_gap": max_team_gap,
        "prior1_anyteam_coverage": float(feat["prior1_anyteam"].mean()),
        "prior3_anyteam_coverage": float(feat["prior3_anyteam"].mean()),
        "prior1_same_team_coverage": float(feat["prior1_same_team"].mean()),
        "prior3_same_team_coverage": float(feat["prior3_same_team"].mean()),
        "raw_snap_duplicate_rate": float(dup_rate),
        "snap_source_seasons": source_seasons,
    }
    return out, trace, audit
