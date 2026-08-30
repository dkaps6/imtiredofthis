"""Production adapter for the M89/M90-promoted QB passing-yards synthesis.

The promoted model is a football-only residual correction on top of the canonical
MC/ML/State ensemble. Sportsbook lines are never features. The artifact is a
transparent JSON export of the exact M89 preprocessing + Ridge architecture,
refit after validation on all corrected 2023-2025 canonical rows for 2026
prospective deployment.

Player Identity v3 changes only how the live eight-game QB history is retrieved;
the M89/M90 feature contract, coefficients, preprocessing, and residual cap are
unchanged. Team Context v3 is the authoritative live team-level source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

MODEL_PATH = Path("model/qb_pass_synthesis_v1.json")
TEAM_CONTEXT_PATH = Path("data/team_context_v3.csv")
HISTORY_GAMES = 8


def _pkey(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _num(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def load_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"QB synthesis artifact missing: {path}")
    obj = json.loads(path.read_text())
    required = {
        "version", "feature_contract", "residual_cap", "imputer_statistics",
        "indicator_feature_indices", "scaler_mean", "scaler_scale",
        "ridge_coef", "ridge_intercept",
    }
    missing = required - set(obj)
    if missing:
        raise RuntimeError(f"QB synthesis artifact missing fields: {sorted(missing)}")
    if len(obj["feature_contract"]) != 21:
        raise RuntimeError("QB synthesis feature contract must contain exactly 21 M89 features")
    expanded = len(obj["feature_contract"]) + len(obj["indicator_feature_indices"])
    if not (len(obj["scaler_mean"]) == len(obj["scaler_scale"]) == len(obj["ridge_coef"]) == expanded):
        raise RuntimeError("QB synthesis artifact preprocessing dimensions are inconsistent")
    return obj


def predict_correction(features: dict, artifact: dict | None = None) -> tuple[float, float, str]:
    """Return (final_projection, capped_residual_correction, artifact_version)."""
    art = artifact or load_artifact()
    names = list(art["feature_contract"])
    x = np.array([_num(features.get(name)) for name in names], dtype=float)
    missing = ~np.isfinite(x)
    med = np.array(art["imputer_statistics"], dtype=float)
    if len(med) != len(x):
        raise RuntimeError("QB synthesis imputer dimension mismatch")
    x = np.where(missing, med, x)
    indicator_idx = np.array(art["indicator_feature_indices"], dtype=int)
    if len(indicator_idx):
        x = np.concatenate([x, missing[indicator_idx].astype(float)])
    mean = np.array(art["scaler_mean"], dtype=float)
    scale = np.array(art["scaler_scale"], dtype=float)
    coef = np.array(art["ridge_coef"], dtype=float)
    safe_scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    z = (x - mean) / safe_scale
    raw = float(art["ridge_intercept"] + np.dot(z, coef))
    cap = float(art["residual_cap"])
    correction = float(np.clip(raw, -cap, cap))
    base = _num(features.get("base_proj"))
    if not np.isfinite(base):
        raise RuntimeError("QB synthesis requires finite base_proj")
    return float(base + correction), correction, str(art["version"])


def load_team_context(path: Path = TEAM_CONTEXT_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Team Context v3 missing: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {
        "team", "team_context_version", "pass_attempts_per_dropback",
        "true_proe", "neutral_pace_true", "pass_rate_off", "pass_rate_faced",
        "def_pass_epa_allowed", "def_pass_success_allowed", "def_ypa_allowed",
        "hit_sack_pressure_rate_allowed", "hit_sack_pressure_rate_generated",
        "plays_est",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Team Context v3 missing promoted QB fields: {sorted(missing)}")
    df["team"] = df["team"].map(canon_team)
    if df["team"].eq("").any() or df.duplicated("team").any():
        raise RuntimeError("Team Context v3 has invalid/duplicate team keys")
    if not df["team_context_version"].astype(str).eq("TEAM_CONTEXT_V3").all():
        raise RuntimeError("Unexpected Team Context version for promoted QB pricing")
    return df


def load_player_logs(path: Path = Path("data/player_game_logs.csv")) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "player_identity_key" in df.columns:
        df["_identity_key"] = df["player_identity_key"].astype("string").fillna("").str.strip()
    else:
        df["_identity_key"] = ""
    if "player_clean_key" in df.columns:
        df["_pkey"] = df["player_clean_key"].map(_pkey)
    elif "player" in df.columns:
        df["_pkey"] = df["player"].map(_pkey)
    else:
        df["_pkey"] = ""
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
    return df


def qb_prior(
    logs: pd.DataFrame,
    player_key: str,
    season: int,
    week: int,
    *,
    player_identity_key: str = "",
) -> tuple[float, float, int]:
    """Return the eight-game pregame QB prior, preferring stable identity.

    ``player_key`` remains as a backwards-compatible fallback for older research
    fixtures, but production rows carrying Player Identity v3 never need to rely
    on spelling-sensitive name matching.
    """
    if logs is None or logs.empty:
        return np.nan, np.nan, 0

    identity = str(player_identity_key or "").strip()
    if identity and "_identity_key" in logs.columns:
        q = logs.loc[logs["_identity_key"].eq(identity)].copy()
    elif "_pkey" in logs.columns:
        q = logs.loc[logs["_pkey"].eq(_pkey(player_key))].copy()
    else:
        return np.nan, np.nan, 0

    if "season" in q.columns and "week" in q.columns:
        q = q.loc[
            (q["season"].lt(int(season)))
            | (q["season"].eq(int(season)) & q["week"].lt(int(week)))
        ].copy()
        q = q.sort_values(["season", "week"]).tail(HISTORY_GAMES)
    else:
        q = q.tail(HISTORY_GAMES)
    if q.empty:
        return np.nan, np.nan, 0
    att_col = next((c for c in ["pass_att", "attempts", "passing_attempts"] if c in q.columns), None)
    yd_col = next((c for c in ["pass_yards", "passing_yards"] if c in q.columns), None)
    ypa_col = next((c for c in ["ypa_game", "ypa"] if c in q.columns), None)
    att = pd.to_numeric(q[att_col], errors="coerce") if att_col else pd.Series(np.nan, index=q.index)
    mean_att = float(att.mean()) if att.notna().any() else np.nan
    ypa = pd.to_numeric(q[ypa_col], errors="coerce") if ypa_col else pd.Series(np.nan, index=q.index)
    mean_ypa = float(ypa.mean()) if ypa.notna().any() else np.nan
    if not np.isfinite(mean_ypa) and yd_col and att.notna().any():
        yards = pd.to_numeric(q[yd_col], errors="coerce")
        denom = float(att.sum(skipna=True))
        if denom > 0:
            mean_ypa = float(yards.sum(skipna=True) / denom)
    return mean_att, mean_ypa, int(len(q))


def _context_row(context: pd.DataFrame, team: str) -> pd.Series:
    q = context.loc[context["team"].eq(canon_team(team))]
    return q.iloc[0] if not q.empty else pd.Series(dtype="object")


def attempt_conversion(row: pd.Series, context: pd.DataFrame) -> float:
    off = _context_row(context, str(row.get("team", "")))
    value = _num(off.get("pass_attempts_per_dropback"))
    if not np.isfinite(value) or not (0.50 <= value <= 1.0):
        raise RuntimeError(f"invalid promoted pass-attempt conversion team={row.get('team')} value={value}")
    return float(value)


def controlled_environment(row: pd.Series, weather: pd.DataFrame | None = None) -> float:
    for name in ["controlled_environment", "controlled_venue", "is_dome", "indoor"]:
        value = _num(row.get(name))
        if np.isfinite(value):
            return float(int(value != 0))
    if weather is not None and not weather.empty:
        w = weather.copy()
        w.columns = [str(c).strip().lower() for c in w.columns]
        team = canon_team(row.get("team")); opp = canon_team(row.get("opponent"))
        if {"home", "away"}.issubset(w.columns):
            w["home"] = w["home"].map(canon_team); w["away"] = w["away"].map(canon_team)
            q = w.loc[
                ((w["home"].eq(team)) & (w["away"].eq(opp)))
                | ((w["home"].eq(opp)) & (w["away"].eq(team)))
            ]
            if not q.empty:
                for name in ["controlled_environment", "controlled_venue", "is_dome", "indoor"]:
                    value = _num(q.iloc[0].get(name))
                    if np.isfinite(value):
                        return float(int(value != 0))
                try:
                    from scripts.utils.stadium_locations import STADIUM_LOCATION
                    home = str(q.iloc[0]["home"])
                    info = STADIUM_LOCATION.get(home, {})
                    if "indoor" in info:
                        return float(int(bool(info["indoor"])))
                except Exception:
                    pass
    return np.nan


def build_feature_dict(
    row: pd.Series,
    *,
    base_proj: float,
    mc_proj: float,
    team_context: pd.DataFrame,
    player_logs: pd.DataFrame,
    weather: pd.DataFrame | None,
    season: int,
    week: int,
) -> dict:
    off = _context_row(team_context, str(row.get("team", "")))
    defense = _context_row(team_context, str(row.get("opponent", "")))
    ml = _num(row.get("ml_proj")); state = _num(row.get("state_proj"))
    comps = np.array([mc_proj, ml, state], dtype=float)
    finite = comps[np.isfinite(comps)]
    component_sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    component_range = float(np.max(finite) - np.min(finite)) if len(finite) else np.nan

    plays = _num(row.get("rules_plays_est"), _num(off.get("plays_est")))
    dropback_rate = _num(row.get("rules_pass_rate"), 0.57)
    conv = attempt_conversion(row, team_context)
    qb_share = _num(row.get("qb_pass_att_share"), 1.0)
    qb_share = float(np.clip(qb_share, 0.0, 1.0)) if np.isfinite(qb_share) else 1.0
    pred_attempts = (
        plays * dropback_rate * conv * qb_share
        if np.isfinite(plays) and np.isfinite(dropback_rate)
        else np.nan
    )
    pred_ypa = mc_proj / pred_attempts if np.isfinite(pred_attempts) and pred_attempts > 0 else np.nan

    player_key = row.get("player_clean_key", row.get("player", ""))
    identity_key = row.get("player_identity_key", "")
    qb_att, qb_ypa, _ = qb_prior(
        player_logs,
        str(player_key),
        int(season),
        int(week),
        player_identity_key=str(identity_key or ""),
    )

    return {
        "base_proj": float(base_proj),
        "mc_proj": float(mc_proj),
        "ml_proj": ml,
        "state_proj": state,
        "component_sd": component_sd,
        "component_range": component_range,
        "pred_attempts": pred_attempts,
        "pred_ypa": pred_ypa,
        "off_true_proe": _num(off.get("true_proe"), _num(off.get("proe"))),
        "off_neutral_pace": _num(off.get("neutral_pace_true"), _num(off.get("neutral_pace"))),
        "off_pass_rate": _num(off.get("pass_rate_off")),
        "off_plays": _num(off.get("plays_est")),
        "qb_prior_attempts": qb_att,
        "qb_prior_ypa": qb_ypa,
        "def_pass_epa_allowed": _num(defense.get("def_pass_epa_allowed"), _num(defense.get("def_pass_epa"))),
        "def_success_allowed": _num(defense.get("def_pass_success_allowed"), _num(defense.get("success_rate_def"))),
        "def_ypa_allowed": _num(defense.get("def_ypa_allowed")),
        "def_pass_rate_faced": _num(defense.get("pass_rate_faced")),
        "off_hit_sack_pressure": _num(
            off.get("hit_sack_pressure_rate_allowed"), _num(off.get("pressure_rate_allowed"))
        ),
        "def_hit_sack_pressure": _num(
            defense.get("hit_sack_pressure_rate_generated"),
            _num(defense.get("pressure_rate_generated")),
        ),
        "controlled_environment": controlled_environment(row, weather),
    }
