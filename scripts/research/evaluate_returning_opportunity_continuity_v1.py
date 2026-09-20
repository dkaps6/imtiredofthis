#!/usr/bin/env python3
"""Evaluate frozen Returning Opportunity Continuity Experiment V1.

Plan: docs/research/RETURNING_OPPORTUNITY_CONTINUITY_EXPERIMENT_V1.md
2025 is inspected only for a family that passes the frozen 2024 primary gate.
No sportsbook fields are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.build_role_room_redundancy_audit import build_production_opportunity_state

KEY = ["season", "week", "team", "player_identity_key"]
FAMILIES = {
    "RB_RUSH_CONTINUITY": {"position": "RB", "domain": "rush", "feature": "prior_rush_share_game_returning_overlap", "tier": "PRIMARY"},
    "RB_TARGET_CONTINUITY": {"position": "RB", "domain": "tgt", "feature": "prior_tgt_share_game_returning_overlap", "tier": "PRIMARY"},
    "WR_TARGET_CONTINUITY": {"position": "WR", "domain": "tgt", "feature": "prior_tgt_share_game_returning_overlap", "tier": "PRIMARY"},
    "TE_TARGET_CONTINUITY": {"position": "TE", "domain": "tgt", "feature": "prior_tgt_share_game_returning_overlap", "tier": "PRIMARY"},
    "QB_RUSH_CONTINUITY": {"position": "QB", "domain": "rush", "feature": "prior_rush_share_game_returning_overlap", "tier": "DESCRIPTIVE_REPLICATION_ONLY"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _domain_spec(domain: str):
    if domain == "rush":
        num, den = "rushes", "team_rushes"
    elif domain == "tgt":
        num, den = "targets", "team_targets"
    else:
        raise ValueError(domain)
    features = [f"prod_{domain}_prior_share", f"prod_{domain}_prior_games", f"prod_{domain}_current_share", f"prod_{domain}_current_games", f"prod_{domain}_playerform_blend"]
    return num, den, features


def _fit(train, features):
    t = train[["outcome_share", *features]].dropna()
    x = np.column_stack([np.ones(len(t)), t[features].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(x, t["outcome_share"].to_numpy(float), rcond=None)
    return beta


def _predict(df, features, beta):
    x = np.column_stack([np.ones(len(df)), df[features].to_numpy(float)])
    return x @ beta


def _metrics(y, pred):
    resid = pred - y
    ae = np.abs(resid)
    return {"rows": int(len(y)), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(resid * resid))), "bias": float(resid.mean()), "p90_ae": float(np.quantile(ae, .90)), "p95_ae": float(np.quantile(ae, .95))}


def _delta(c, b, name):
    return (c[name] - b[name]) / b[name] if b[name] else np.nan


def _gate(base, cand, season):
    d = {n: _delta(cand, base, n) for n in ["mae", "rmse", "p90_ae", "p95_ae"]}
    bias_abs_delta = abs(cand["bias"]) - abs(base["bias"])
    if season == 2024:
        gates = {"rows_gate": cand["rows"] >= 500, "mae_gate": d["mae"] <= -0.01, "rmse_gate": d["rmse"] <= .005, "p90_gate": d["p90_ae"] <= .01, "bias_gate": bias_abs_delta <= .002}
    else:
        gates = {"rows_gate": cand["rows"] >= 500, "mae_gate": d["mae"] < 0, "rmse_gate": d["rmse"] <= .005, "p90_gate": d["p90_ae"] <= .01, "bias_gate": bias_abs_delta <= .002}
    return {"mae_relative_delta": d["mae"], "rmse_relative_delta": d["rmse"], "p90_relative_delta": d["p90_ae"], "p95_relative_delta": d["p95_ae"], "bias_abs_delta": bias_abs_delta, **gates, "season_gate_pass": bool(all(gates.values()))}


def evaluate(history, context):
    h, c = history.copy(), context.copy()
    h.columns = [str(x).strip().lower() for x in h.columns]
    c.columns = [str(x).strip().lower() for x in c.columns]
    features = sorted({s["feature"] for s in FAMILIES.values()})
    need = set(KEY + features)
    if need - set(c.columns):
        raise RuntimeError(f"missing context columns: {sorted(need-set(c.columns))}")
    if c.duplicated(KEY).any():
        raise RuntimeError("duplicate context player-game keys")
    state = build_production_opportunity_state(h)
    x = h[KEY + ["position", "rushes", "team_rushes", "targets", "team_targets"]].merge(c[KEY + features], on=KEY, how="inner", validate="one_to_one").merge(state, on=KEY, how="left", validate="one_to_one")
    metrics, gates = [], []
    for family, spec in FAMILIES.items():
        num, den, base_features = _domain_spec(spec["domain"])
        z = x[x.position.astype(str).str.upper().eq(spec["position"])].copy()
        denom, numer = pd.to_numeric(z[den], errors="coerce"), pd.to_numeric(z[num], errors="coerce")
        z["outcome_share"] = np.where(denom > 0, numer / denom, np.nan)
        z = z.dropna(subset=["outcome_share", *base_features, spec["feature"]])
        train = z[z.season.between(2019, 2023)]
        b0 = _fit(train, base_features)
        b1 = _fit(train, base_features + [spec["feature"]])
        for season in [2024]:
            ev = z[z.season.eq(season)]
            bm = _metrics(ev.outcome_share.to_numpy(float), _predict(ev, base_features, b0))
            cm = _metrics(ev.outcome_share.to_numpy(float), _predict(ev, base_features + [spec["feature"]], b1))
            for model, m in [("BASELINE", bm), ("CANDIDATE", cm)]: metrics.append({"family": family, "family_tier": spec["tier"], "position": spec["position"], "domain": spec["domain"], "feature": spec["feature"], "evaluation_season": season, "model": model, "train_seasons": "2019-2023", "train_rows": len(train), **m})
            g = _gate(bm, cm, season)
            primary_pass = g["season_gate_pass"] and spec["tier"] != "DESCRIPTIVE_REPLICATION_ONLY"
            gates.append({"family": family, "family_tier": spec["tier"], "evaluation_season": season, "gate_type": "PRIMARY", **g, "replication_inspected": bool(primary_pass), "family_disposition": "CONTINUITY_SIGNAL_PRIMARY_PASS" if primary_pass else ("DESCRIPTIVE_ONLY" if spec["tier"] == "DESCRIPTIVE_REPLICATION_ONLY" else "CONTINUITY_SIGNAL_PRIMARY_FAIL_CLOSED_V1")})
            if not primary_pass: continue
            rep = z[z.season.eq(2025)]
            rbm = _metrics(rep.outcome_share.to_numpy(float), _predict(rep, base_features, b0))
            rcm = _metrics(rep.outcome_share.to_numpy(float), _predict(rep, base_features + [spec["feature"]], b1))
            for model, m in [("BASELINE", rbm), ("CANDIDATE", rcm)]: metrics.append({"family": family, "family_tier": spec["tier"], "position": spec["position"], "domain": spec["domain"], "feature": spec["feature"], "evaluation_season": 2025, "model": model, "train_seasons": "2019-2023", "train_rows": len(train), **m})
            rg = _gate(rbm, rcm, 2025)
            disposition = "CONTINUITY_SIGNAL_REPLICATED" if rg["season_gate_pass"] else "CONTINUITY_SIGNAL_PRIMARY_PASS_REPLICATION_FAILED"
            gates[-1]["family_disposition"] = disposition
            gates.append({"family": family, "family_tier": spec["tier"], "evaluation_season": 2025, "gate_type": "REPLICATION", **rg, "replication_inspected": True, "family_disposition": disposition})
    return pd.DataFrame(metrics), pd.DataFrame(gates)


def main():
    p = argparse.ArgumentParser()
    for name in ["history", "context", "metrics-out", "gates-out", "manifest-out"]: p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    metrics, gates = evaluate(pd.read_csv(a.history), pd.read_csv(a.context))
    a.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(a.metrics_out, index=False); gates.to_csv(a.gates_out, index=False)
    dispositions = gates.groupby("family", sort=True)["family_disposition"].first().to_dict()
    manifest = {"experiment": "RETURNING_OPPORTUNITY_CONTINUITY_EXPERIMENT_V1", "history_sha256": sha256(a.history), "context_sha256": sha256(a.context), "sportsbook_read": False, "replication_policy": "2025_INSPECTED_ONLY_AFTER_2024_PRIMARY_PASS", "dispositions": dispositions}
    a.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(gates.to_string(index=False)); print(json.dumps(manifest, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
