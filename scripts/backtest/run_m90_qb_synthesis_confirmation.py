from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHA = 20.0
RESIDUAL_CAP = 45.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 90


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _find_col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"missing required column among {names}")


def _metric_frame(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    ae = np.abs(err)
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(err.mean()),
        "correlation": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else np.nan,
        "median_ae": float(np.median(ae)),
        "tails100": int((ae >= 100).sum()),
        "under100": int((err <= -100).sum()),
        "over100": int((err >= 100).sum()),
    }


def _bootstrap_p_improve(y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y)
    wins = 0
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, n)
        b = np.abs(base[idx] - y[idx]).mean()
        c = np.abs(cand[idx] - y[idx]).mean()
        wins += int(c < b)
    return wins / BOOTSTRAP_DRAWS


def _prepare_trace(path: str) -> pd.DataFrame:
    df = _norm(pd.read_csv(path, low_memory=False))
    # M89 trace contract. Keep only columns that are genuinely pregame/model-state.
    actual_yards = _find_col(df, "actual_pass_yards", "actual")
    ensemble = _find_col(df, "ensemble_proj", "oos_ensemble")
    df["actual_pass_yards"] = pd.to_numeric(df[actual_yards], errors="coerce")
    df["ensemble_proj"] = pd.to_numeric(df[ensemble], errors="coerce")
    return df


def _feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    # Exact M89 football-only synthesis information families. The runner uses the
    # intersection so source-availability drift fails closed rather than creating
    # a different model between rotations.
    banned_tokens = (
        "actual", "result", "postgame", "casebook", "forensic", "sportsbook",
        "spread", "total_line", "moneyline", "implied_points", "market_",
        "closing", "final_score", "pass_yards_actual",
    )
    explicit = [
        "ensemble_proj", "mc_proj", "ml_proj", "state_proj",
        "pred_attempts", "mc_expected_pass_attempts", "implied_pred_ypa",
        "mc_ml_range", "mc_state_range", "ml_state_range",
        "component_range", "ml_minus_mc", "state_minus_mc",
        "off_pass_rate", "off_neutral_pass_rate", "off_true_proe",
        "off_seconds_per_play", "off_neutral_seconds_per_play",
        "off_pass_epa", "off_success_rate", "off_ypa", "off_explosive20_rate",
        "off_deep_attempt_rate", "off_sack_rate", "off_scramble_rate",
        "def_pass_rate_faced", "def_neutral_pass_rate_faced", "def_pass_epa_allowed",
        "def_success_rate_allowed", "def_ypa_allowed", "def_explosive20_allowed",
        "def_deep_rate_faced", "def_sack_rate_generated", "def_int_rate_generated",
        "opp_off_pass_epa", "opp_off_success_rate", "opp_off_plays_per_game",
        "opp_off_neutral_pass_rate", "opp_off_ypa",
        "home", "controlled_venue",
    ]
    common = set(train.columns) & set(test.columns)
    cols = [c for c in explicit if c in common]
    if "ensemble_proj" not in cols:
        cols.insert(0, "ensemble_proj")
    # Allow numeric M89 pregame fields not in the explicit list only when they are
    # clearly model-state/history variables and not market/postgame.
    for c in sorted(common):
        if c in cols or c in {"season", "week"}:
            continue
        if any(tok in c for tok in banned_tokens):
            continue
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        if c.startswith(("off_", "def_", "opp_off_", "pred_", "mc_", "ml_", "state_")):
            cols.append(c)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-trace", required=True)
    ap.add_argument("--test-trace", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _prepare_trace(args.train_trace)
    test = _prepare_trace(args.test_trace)

    features = _feature_columns(train, test)
    if len(features) < 5:
        raise RuntimeError(f"M90 feature intersection unexpectedly small: {features}")

    train = train.dropna(subset=["actual_pass_yards", "ensemble_proj"]).copy()
    test = test.dropna(subset=["actual_pass_yards", "ensemble_proj"]).copy()
    y_train = train["actual_pass_yards"].to_numpy(float)
    base_train = train["ensemble_proj"].to_numpy(float)
    target_resid = np.clip(y_train - base_train, -RESIDUAL_CAP, RESIDUAL_CAP)

    numeric = features
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
        ], remainder="drop")),
        ("ridge", Ridge(alpha=ALPHA)),
    ])
    pipe.fit(train[features], target_resid)

    y = test["actual_pass_yards"].to_numpy(float)
    base = test["ensemble_proj"].to_numpy(float)
    correction = np.clip(pipe.predict(test[features]), -RESIDUAL_CAP, RESIDUAL_CAP)
    cand = base + correction

    base_m = _metric_frame(y, base)
    cand_m = _metric_frame(y, cand)
    p_improve = _bootstrap_p_improve(y, base, cand)
    gates = {
        "mae_gain": base_m["mae"] - cand_m["mae"],
        "mae_gain_gate": (base_m["mae"] - cand_m["mae"]) >= 1.0,
        "rmse_nonworse": cand_m["rmse"] <= base_m["rmse"],
        "correlation_nonworse": cand_m["correlation"] >= base_m["correlation"],
        "tails_nonincrease": cand_m["tails100"] <= base_m["tails100"],
        "bootstrap_p_improve": p_improve,
        "bootstrap_gate": p_improve >= 0.90,
        "sportsbook_features_in_football_model": False,
        "postgame_casebook_features_used_for_prediction": False,
    }
    gates["all_gates_pass"] = bool(
        gates["mae_gain_gate"]
        and gates["rmse_nonworse"]
        and gates["correlation_nonworse"]
        and gates["tails_nonincrease"]
        and gates["bootstrap_gate"]
    )

    scoreboard = pd.DataFrame([
        {"model": "base", **base_m},
        {"model": "football_synthesis", **cand_m},
    ])
    scoreboard.to_csv(out_dir / "m90_rotated_confirmation_scoreboard.csv", index=False)

    trace_out = test[[c for c in ["season", "week", "team", "player_clean_key", "actual_pass_yards", "ensemble_proj"] if c in test.columns]].copy()
    trace_out["m90_correction"] = correction
    trace_out["m90_football_synthesis"] = cand
    trace_out["m90_base_error"] = base - y
    trace_out["m90_synthesis_error"] = cand - y
    trace_out.to_csv(out_dir / "m90_rotated_confirmation_trace.csv", index=False)

    decision = {
        "migration": "M90",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "ridge_alpha": ALPHA,
        "residual_cap": RESIDUAL_CAP,
        "features": features,
        "base": base_m,
        "football_synthesis": cand_m,
        "gates": gates,
        "disposition": "PROMOTE_M89_FOOTBALL_SYNTHESIS" if gates["all_gates_pass"] else "DO_NOT_PROMOTE_M89_SYNTHESIS",
        "catastrophic_casebook_reopened": False,
    }
    (out_dir / "m90_decision.json").write_text(json.dumps(decision, indent=2))
    print("=== M90 ROTATED CONFIRMATION ===")
    print(scoreboard.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
