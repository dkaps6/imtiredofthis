"""Production-safe TE-R5P redistribution inside the conserved receiving pool.

TE-R5P is allowed to redistribute only the target mass that the projection-neutral
finite entitlement seam already assigned to a team's tight-end room. It may not
create target opportunity, alter any non-TE player's entitlement, change the team
residual bucket, or consume sportsbook information.

The frozen model parameters are the exact successful final fit from GitHub run
34153141322 / artifact 10030047802, trained on 2022-2025 only. Strict-prior snap
features reproduce TE-R4's source contract using nflreadpy snap counts from
2020-2025 and observations strictly before the target game.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path("data/models/te_r5p_production_model_v1/te_r5p_production_model_v1.json")
SOURCE_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
TEAM_MAP = {"OAK":"LV","SD":"LAC","STL":"LAR","LA":"LAR","JAC":"JAX","ARZ":"ARI","WSH":"WAS"}
EXPECTED_VERSION = "TE_R5P_PRODUCTION_MODEL_V1"
EXPECTED_FINAL_FIT_RUN = 34153141322
EXPECTED_FINAL_FIT_ARTIFACT = 10030047802


def _pdx(v) -> pd.DataFrame:
    if isinstance(v, pd.DataFrame):
        return v.copy()
    if hasattr(v, "to_pandas"):
        return v.to_pandas()
    if hasattr(v, "to_dicts"):
        return pd.DataFrame(v.to_dicts())
    return pd.DataFrame(v)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _team(v) -> str:
    s = str(v).strip().upper() if not pd.isna(v) else ""
    if s in {"", "NAN", "NONE", "<NA>"}:
        return ""
    return TEAM_MAP.get(s, s)


def _first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(pd.NA, index=df.index)


def _load_model() -> dict:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size <= 0:
        raise RuntimeError(f"TE-R5P frozen production model missing: {MODEL_PATH}")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("model_version") != EXPECTED_VERSION:
        raise RuntimeError(f"unexpected TE-R5P model version: {model.get('model_version')}")
    if int(model.get("source_final_fit_run", -1)) != EXPECTED_FINAL_FIT_RUN:
        raise RuntimeError("TE-R5P model does not point to frozen final-fit run")
    if int(model.get("source_final_fit_artifact", -1)) != EXPECTED_FINAL_FIT_ARTIFACT:
        raise RuntimeError("TE-R5P model does not point to frozen final-fit artifact")
    if int(model.get("sportsbook_inputs_used", -1)) != 0:
        raise RuntimeError("TE-R5P frozen model is not sportsbook independent")
    if model.get("training_seasons") != [2022, 2023, 2024, 2025]:
        raise RuntimeError("TE-R5P frozen training seasons drifted")
    features = model.get("features", [])
    if not isinstance(features, list) or len(features) != 16:
        raise RuntimeError("TE-R5P feature contract is not length 16")
    for k in ("scaler_mean", "scaler_scale", "ridge_coef"):
        vals = np.asarray(model.get(k, []), dtype=float)
        if len(vals) != 16 or not np.isfinite(vals).all():
            raise RuntimeError(f"TE-R5P invalid parameter vector: {k}")
    if not np.isfinite(float(model.get("ridge_intercept", np.nan))):
        raise RuntimeError("TE-R5P invalid intercept")
    return model


def _load_snaps() -> tuple[pd.DataFrame, float, list[int]]:
    import nflreadpy as nfl

    q = _lower(_pdx(nfl.load_snap_counts(seasons=SOURCE_SEASONS)))
    if "season" not in q.columns or "week" not in q.columns:
        raise RuntimeError("TE-R5P snap source missing season/week")
    q["season"] = pd.to_numeric(q["season"], errors="coerce")
    q["week"] = pd.to_numeric(q["week"], errors="coerce")
    q = q.loc[q["season"].isin(SOURCE_SEASONS) & q["week"].between(1, 18)].copy()
    q["team"] = _first(q, ["team", "team_abbr", "club"]).map(_team)
    q["player_key"] = _first(q, ["player", "player_name", "full_name"]).map(_key)
    q["offense_pct"] = pd.to_numeric(_first(q, ["offense_pct", "offense_percentage"]), errors="coerce")
    q["offense_snaps"] = pd.to_numeric(_first(q, ["offense_snaps"]), errors="coerce")
    q["ordinal"] = q["season"] * 100 + q["week"]
    q = q.loc[q["team"].ne("") & q["player_key"].ne("")].copy()
    keys = ["season", "week", "team", "player_key"]
    dup_rate = float(q.duplicated(keys, keep=False).mean()) if len(q) else 1.0
    q = q.sort_values(keys, kind="stable").drop_duplicates(keys, keep="last").reset_index(drop=True)
    seasons = sorted(int(v) for v in q["season"].dropna().unique())
    if seasons != SOURCE_SEASONS:
        raise RuntimeError(f"TE-R5P snap source seasons drifted: {seasons}")
    if dup_rate > 0.01:
        raise RuntimeError(f"TE-R5P raw snap duplicate rate too high: {dup_rate}")
    return q, dup_rate, seasons


def _strict_prior_features(te: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    out = te.copy()
    out["player_key"] = out["player_clean_key"].map(_key)
    out["team_key"] = out["team"].map(_team)
    season = pd.to_numeric(out["season"], errors="coerce")
    week = pd.to_numeric(out["week"], errors="coerce")
    if season.isna().any() or week.isna().any():
        raise RuntimeError("TE-R5P current slate missing season/week")
    out["ordinal"] = season * 100 + week

    any_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby("player_key", sort=False)}
    same_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby(["player_key", "team"], sort=False)}
    rows: list[dict] = []
    same_future = 0
    for _, r in out.iterrows():
        o = float(r["ordinal"])
        pk = str(r["player_key"])
        tm = str(r["team_key"])
        ah = any_maps.get(pk, pd.DataFrame())
        if len(ah):
            ah = ah.loc[ah["ordinal"].lt(o)]
        sh = same_maps.get((pk, tm), pd.DataFrame())
        if len(sh):
            sh = sh.loc[sh["ordinal"].lt(o)]
        if len(ah) and float(ah["ordinal"].max()) >= o:
            same_future += 1
        if len(sh) and float(sh["ordinal"].max()) >= o:
            same_future += 1
        a1, a3, s1, s3 = ah.tail(1), ah.tail(3), sh.tail(1), sh.tail(3)

        def last_num(frame: pd.DataFrame, col: str) -> float:
            if not len(frame):
                return np.nan
            z = pd.to_numeric(frame[col], errors="coerce")
            return float(z.iloc[-1]) if len(z) and pd.notna(z.iloc[-1]) else np.nan

        def mean3(frame: pd.DataFrame, col: str) -> float:
            if len(frame) < 3:
                return np.nan
            z = pd.to_numeric(frame[col], errors="coerce")
            return float(z.mean()) if z.notna().any() else np.nan

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
    if same_future != 0:
        raise RuntimeError(f"TE-R5P strict prior construction used same/future rows: {same_future}")
    return pd.concat([out.reset_index(drop=False).rename(columns={"index":"_row_index"}), pd.DataFrame(rows)], axis=1)


def apply_te_r5p_entitlement(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Redistribute each team's existing TE target pool using frozen TE-R5P."""
    if metrics is None or metrics.empty:
        raise RuntimeError("TE-R5P cannot consume empty entitlement frame")
    out = metrics.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {"event_id", "season", "week", "team", "player", "player_clean_key", "position", "entitlement_tgt_share", "entitlement_residual_share"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"TE-R5P entitlement input missing columns: {sorted(missing)}")

    model = _load_model()
    baseline = pd.to_numeric(out["entitlement_tgt_share"], errors="coerce")
    if baseline.isna().any() or not np.isfinite(baseline.to_numpy(float)).all() or baseline.lt(0).any():
        raise RuntimeError("TE-R5P received invalid baseline entitlement")
    out["baseline_entitlement_tgt_share"] = baseline.astype(float)
    out["te_r5p_applied"] = False
    out["te_r5p_model_version"] = ""

    te_mask = out["position"].astype("string").fillna("").str.upper().eq("TE")
    te = out.loc[te_mask].copy()
    if te.empty:
        raise RuntimeError("TE-R5P found zero tight ends on Full Slate")

    snaps, dup_rate, source_seasons = _load_snaps()
    feat = _strict_prior_features(te, snaps)
    feat["b0_te_pool"] = feat.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    feat["b0_te_room_share"] = np.where(
        feat["b0_te_pool"].gt(0),
        feat["baseline_entitlement_tgt_share"] / feat["b0_te_pool"],
        0.0,
    )
    feat["log_b0_te_pool"] = np.log1p(feat["b0_te_pool"].clip(lower=0))
    feat["pool_ratio"] = 1.0
    feat["room_size"] = feat.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)
    feat["prior1_same_team_available"] = feat["prior1_same_team"].fillna(False).astype(float)
    feat["prior3_same_team_available"] = feat["prior3_same_team"].fillna(False).astype(float)
    feat["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(feat["prior_count_same_team"], errors="coerce").fillna(0).clip(lower=0))
    feat["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(feat["prior_count_anyteam"], errors="coerce").fillna(0).clip(lower=0))
    for src, dst in [
        ("prior1_same_team_offense_pct", "snap_share_prior1_same_team"),
        ("prior3_anyteam_offense_pct", "snap_share_prior3_anyteam"),
    ]:
        z = pd.to_numeric(feat[src], errors="coerce").fillna(0).clip(lower=0)
        den = z.groupby([feat["event_id"], feat["team"]]).transform("sum")
        feat[dst] = np.where(den.gt(0), z / den, 0.0)

    features = list(model["features"])
    for c in features:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    x = feat[features].to_numpy(dtype=float)
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coef = np.asarray(model["ridge_coef"], dtype=float)
    z = (x - mean) / scale
    residual = z @ coef + float(model["ridge_intercept"])
    lo, hi = [float(v) for v in model["prediction_clip"]]
    residual = np.clip(residual, lo, hi)
    eps = float(model["eps"])
    feat["te_r5p_residual"] = residual
    feat["te_r5p_score"] = np.log(feat["b0_te_room_share"].clip(lower=0).to_numpy(float) + eps) + residual
    feat["te_r5p_room_share"] = 0.0
    feat["te_r5p_entitlement_tgt_share"] = 0.0

    for (event_id, team), idx in feat.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(feat.loc[idx, "b0_te_pool"].iloc[0])
        if pool <= 0:
            candidate = np.zeros(len(idx), dtype=float)
            room_share = np.zeros(len(idx), dtype=float)
        else:
            score = feat.loc[idx, "te_r5p_score"].to_numpy(float)
            stable = score - float(np.max(score))
            w = np.exp(stable)
            room_share = w / float(w.sum())
            candidate = pool * room_share
            # Force exact room conservation to floating precision without changing
            # the football model: assign only the arithmetic remainder to the
            # largest TE probability.
            gap = pool - float(candidate.sum())
            if len(candidate):
                candidate[int(np.argmax(room_share))] += gap
        feat.loc[idx, "te_r5p_room_share"] = room_share
        feat.loc[idx, "te_r5p_entitlement_tgt_share"] = candidate

    final = feat.set_index("_row_index")["te_r5p_entitlement_tgt_share"]
    out.loc[final.index, "entitlement_tgt_share"] = final.astype(float)
    out.loc[final.index, "te_r5p_applied"] = True
    out.loc[final.index, "te_r5p_model_version"] = EXPECTED_VERSION

    non_te_delta = (
        out.loc[~te_mask, "entitlement_tgt_share"].astype(float)
        - out.loc[~te_mask, "baseline_entitlement_tgt_share"].astype(float)
    ).abs()
    before_te = out.loc[te_mask].groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].sum()
    after_te = out.loc[te_mask].groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    te_pool_gap = (after_te - before_te).abs()
    before_team = out.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].sum()
    after_team = out.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    team_gap = (after_team - before_team).abs()
    residual_per_team = out.groupby(["event_id", "team"])["entitlement_residual_share"].nunique(dropna=False)
    if int(residual_per_team.max()) != 1:
        raise RuntimeError("TE-R5P found inconsistent residual share inside a team")

    max_non_te_delta = float(non_te_delta.max()) if len(non_te_delta) else 0.0
    max_te_pool_gap = float(te_pool_gap.max()) if len(te_pool_gap) else 0.0
    max_team_gap = float(team_gap.max()) if len(team_gap) else 0.0
    if max_non_te_delta > 0:
        raise RuntimeError(f"TE-R5P changed non-TE entitlement: {max_non_te_delta}")
    if max_te_pool_gap > 1e-12:
        raise RuntimeError(f"TE-R5P changed team TE-room mass: {max_te_pool_gap}")
    if max_team_gap > 1e-12:
        raise RuntimeError(f"TE-R5P changed total player entitlement: {max_team_gap}")

    feat["baseline_entitlement_tgt_share"] = pd.to_numeric(feat["baseline_entitlement_tgt_share"], errors="coerce").fillna(0.0)
    feat["entitlement_delta"] = feat["te_r5p_entitlement_tgt_share"] - feat["baseline_entitlement_tgt_share"]
    trace_cols = [
        "event_id", "season", "week", "team", "player", "player_clean_key",
        "baseline_entitlement_tgt_share", "b0_te_pool", "b0_te_room_share",
        "prior_count_anyteam", "prior_count_same_team", "prior1_same_team_offense_pct",
        "prior1_same_team_offense_snaps", "prior1_anyteam_offense_pct",
        "prior3_anyteam_offense_pct", "prior1_anyteam_offense_snaps",
        "prior3_anyteam_offense_snaps", "te_r5p_residual", "te_r5p_score",
        "te_r5p_room_share", "te_r5p_entitlement_tgt_share", "entitlement_delta",
    ]
    trace = feat[trace_cols].copy()
    audit = {
        "disposition": "TE_R5P_FULL_SLATE_ENTITLEMENT_READY",
        "model_version": EXPECTED_VERSION,
        "source_final_fit_run": EXPECTED_FINAL_FIT_RUN,
        "source_final_fit_artifact": EXPECTED_FINAL_FIT_ARTIFACT,
        "authorized_by_run": int(model["authorized_by_run"]),
        "authorized_by_artifact": int(model["authorized_by_artifact"]),
        "training_seasons": model["training_seasons"],
        "training_rows": int(model["training_rows"]),
        "snap_source": "nflreadpy.load_snap_counts",
        "snap_source_seasons": source_seasons,
        "raw_snap_duplicate_rate": dup_rate,
        "current_te_rows": int(te_mask.sum()),
        "teams_with_te_rows": int(te["team"].nunique()),
        "sportsbook_inputs_used": False,
        "current_or_future_outcomes_used": False,
        "team_te_pool_preserved": True,
        "non_te_entitlement_preserved": True,
        "team_total_player_entitlement_preserved": True,
        "max_te_pool_gap": max_te_pool_gap,
        "max_non_te_entitlement_delta": max_non_te_delta,
        "max_team_player_entitlement_gap": max_team_gap,
        "prior1_anyteam_coverage": float(feat["prior1_anyteam"].mean()),
        "prior3_anyteam_coverage": float(feat["prior3_anyteam"].mean()),
        "prior1_same_team_coverage": float(feat["prior1_same_team"].mean()),
        "prior3_same_team_coverage": float(feat["prior3_same_team"].mean()),
    }
    return out, trace, audit
