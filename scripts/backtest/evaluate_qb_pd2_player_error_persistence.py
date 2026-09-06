#!/usr/bin/env python3
"""QB-PD2 frozen walk-forward individual player-error persistence diagnostic."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HIST = 8
MIN_PRIOR = 4
MIN_ROWS = 500


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, found {len(hits)}")
    return hits[0]


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"empty source {path}")
    return x


def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_walkforward(trace: pd.DataFrame) -> pd.DataFrame:
    need = {"season", "week", "player_clean_key", "actual_pass_yards", "base_proj", "football_synthesis"}
    missing = need - set(trace.columns)
    if missing:
        raise RuntimeError(f"M89 trace missing {sorted(missing)}")
    q = trace.loc[_num(trace["season"]).isin([2024, 2025])].copy()
    if len(q) != 884:
        raise RuntimeError(f"M89 row parity drift expected=884 got={len(q)}")
    q["season"] = _num(q["season"]).astype(int)
    q["week"] = _num(q["week"]).astype(int)
    q["player_key"] = q["player_clean_key"].map(_key)
    q["actual"] = _num(q["actual_pass_yards"])
    q["base"] = _num(q["base_proj"])
    q["synth"] = _num(q["football_synthesis"])
    if q[["actual", "base", "synth"]].isna().any().any():
        raise RuntimeError("non-finite M89 core values")
    q["base_err"] = q["base"] - q["actual"]
    q["synth_err"] = q["synth"] - q["actual"]
    q["base_abs"] = q["base_err"].abs()
    q["synth_abs"] = q["synth_err"].abs()
    q["synth_adv"] = q["base_abs"] - q["synth_abs"]
    q = q.sort_values(["season", "week", "player_key"], kind="stable").reset_index(drop=True)

    rows = []
    hist_by_player: dict[str, list[dict]] = {}
    for r in q.itertuples(index=False):
        hist = hist_by_player.get(r.player_key, [])[-HIST:]
        rec = {
            "season": r.season,
            "week": r.week,
            "player_key": r.player_key,
            "player": getattr(r, "player", getattr(r, "player_clean_key", r.player_key)),
            "actual": r.actual,
            "base": r.base,
            "synth": r.synth,
            "target_synth_error": r.synth_err,
            "target_synth_abs_error": r.synth_abs,
            "target_synth_advantage": r.synth_adv,
            "prior_games": len(hist),
        }
        if hist:
            h = pd.DataFrame(hist)
            rec["prior8_synth_bias"] = float(h["synth_err"].mean())
            rec["prior8_synth_mae"] = float(h["synth_abs"].mean())
            rec["prior8_base_mae"] = float(h["base_abs"].mean())
            rec["prior8_synth_advantage"] = float(h["synth_adv"].mean())
            rec["last_prior_season"] = int(h.iloc[-1]["season"])
            rec["last_prior_week"] = int(h.iloc[-1]["week"])
        else:
            for c in ["prior8_synth_bias", "prior8_synth_mae", "prior8_base_mae", "prior8_synth_advantage", "last_prior_season", "last_prior_week"]:
                rec[c] = np.nan
        rows.append(rec)
        hist_by_player.setdefault(r.player_key, []).append({
            "season": r.season, "week": r.week,
            "synth_err": r.synth_err, "synth_abs": r.synth_abs,
            "base_abs": r.base_abs, "synth_adv": r.synth_adv,
        })

    out = pd.DataFrame(rows)
    leak = out.loc[
        out["last_prior_season"].notna()
        & ((out["last_prior_season"] > out["season"]) | ((out["last_prior_season"] == out["season"]) & (out["last_prior_week"] >= out["week"])))
    ]
    if len(leak):
        raise RuntimeError(f"walk-forward leakage rows={len(leak)}")
    return out


def _quartile_gap(g: pd.DataFrame, feature: str, outcome: str) -> float:
    f = _num(g[feature]); y = _num(g[outcome])
    q25, q75 = float(f.quantile(.25)), float(f.quantile(.75))
    return float(y.loc[f.ge(q75)].mean() - y.loc[f.le(q25)].mean())


def _season_gaps(g: pd.DataFrame, feature: str, outcome: str) -> dict[int, float]:
    out = {}
    for s in [2024, 2025]:
        q = g.loc[g["season"].eq(s)]
        out[s] = _quartile_gap(q, feature, outcome) if len(q) >= 60 and _num(q[feature]).nunique() >= 4 else np.nan
    return out


def score(wf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = wf.loc[wf["prior_games"].ge(MIN_PRIOR)].copy()
    rows = []

    # A directional bias persistence
    spearman = float(_num(g["prior8_synth_bias"]).corr(_num(g["target_synth_error"]), method="spearman"))
    gap = _quartile_gap(g, "prior8_synth_bias", "target_synth_error")
    signq = g.loc[_num(g["prior8_synth_bias"]).abs().ge(5)].copy()
    sign_agree = float((np.sign(_num(signq["prior8_synth_bias"])) == np.sign(_num(signq["target_synth_error"]))).mean()) if len(signq) else np.nan
    sg = _season_gaps(g, "prior8_synth_bias", "target_synth_error")
    pass_a = bool(len(g) >= MIN_ROWS and spearman >= .08 and gap >= 15 and sign_agree >= .55 and sg.get(2024, -np.inf) > 0 and sg.get(2025, -np.inf) > 0)
    rows.append({"diagnostic":"DIRECTIONAL_BIAS_PERSISTENCE","rows":len(g),"spearman":spearman,"quartile_gap":gap,"sign_agreement":sign_agree,"gap_2024":sg.get(2024),"gap_2025":sg.get(2025),"passes":pass_a})

    # B difficulty persistence
    spearman_b = float(_num(g["prior8_synth_mae"]).corr(_num(g["target_synth_abs_error"]), method="spearman"))
    gap_b = _quartile_gap(g, "prior8_synth_mae", "target_synth_abs_error")
    sgb = _season_gaps(g, "prior8_synth_mae", "target_synth_abs_error")
    pass_b = bool(len(g) >= MIN_ROWS and spearman_b >= .08 and gap_b >= 8 and sgb.get(2024, -np.inf) > 0 and sgb.get(2025, -np.inf) > 0)
    rows.append({"diagnostic":"INDIVIDUAL_DIFFICULTY_PERSISTENCE","rows":len(g),"spearman":spearman_b,"quartile_gap":gap_b,"sign_agreement":np.nan,"gap_2024":sgb.get(2024),"gap_2025":sgb.get(2025),"passes":pass_b})

    # C synthesis reliability persistence
    spearman_c = float(_num(g["prior8_synth_advantage"]).corr(_num(g["target_synth_advantage"]), method="spearman"))
    gap_c = _quartile_gap(g, "prior8_synth_advantage", "target_synth_advantage")
    sgc = _season_gaps(g, "prior8_synth_advantage", "target_synth_advantage")
    pass_c = bool(len(g) >= MIN_ROWS and spearman_c >= .08 and gap_c >= 5 and sgc.get(2024, -np.inf) > 0 and sgc.get(2025, -np.inf) > 0)
    rows.append({"diagnostic":"SYNTHESIS_RELIABILITY_PERSISTENCE","rows":len(g),"spearman":spearman_c,"quartile_gap":gap_c,"sign_agreement":np.nan,"gap_2024":sgc.get(2024),"gap_2025":sgc.get(2025),"passes":pass_c})

    metrics = pd.DataFrame(rows)
    winners = metrics.loc[metrics["passes"], "diagnostic"].tolist()
    summary = {
        "migration":"QB_PD2_PLAYER_ERROR_PERSISTENCE",
        "source_rows":int(len(wf)),
        "scoreable_rows":int(len(g)),
        "players":int(wf["player_key"].nunique()),
        "history_window":HIST,
        "minimum_prior_games":MIN_PRIOR,
        "walk_forward_leakage_violations":0,
        "sportsbook_inputs_used":False,
        "model_fitting_used":False,
        "production_changed":False,
        "passing_diagnostics":winners,
        "disposition":"QB_PLAYER_ERROR_PERSISTENCE_DETECTED" if winners else "NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE",
    }
    return metrics, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_pd2_player_error_persistence"))
    a = ap.parse_args()
    trace = _read(_one(a.m89_root, "m89_2024_2025_synthesis_trace.csv"))
    wf = build_walkforward(trace)
    metrics, summary = score(wf)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    wf.to_csv(a.out_dir / "qb_pd2_walkforward_casebook.csv", index=False)
    metrics.to_csv(a.out_dir / "qb_pd2_metrics.csv", index=False)
    (a.out_dir / "qb_pd2_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
