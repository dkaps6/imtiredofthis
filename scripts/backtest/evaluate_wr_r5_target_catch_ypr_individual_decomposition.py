#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 2130


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def rate(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = num(numer)
    denom = num(denom)
    out = pd.Series(np.nan, index=numer.index, dtype=float)
    pos = denom.gt(0)
    out.loc[pos] = numer.loc[pos] / denom.loc[pos]
    zero = denom.eq(0) & numer.eq(0)
    out.loc[zero] = 0.0
    return out


def shapley_row(projected: tuple[float, float, float], actual: tuple[float, float, float]) -> tuple[float, float, float]:
    contrib = np.zeros(3, dtype=float)
    for perm in itertools.permutations(range(3)):
        state = list(projected)
        before = float(np.prod(state))
        for idx in perm:
            state[idx] = actual[idx]
            after = float(np.prod(state))
            contrib[idx] += after - before
            before = after
    contrib /= 6.0
    return float(contrib[0]), float(contrib[1]), float(contrib[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd5-root", type=Path, required=True)
    ap.add_argument("--wr-r1-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    nd = pd.read_csv(one(a.nd5_root, "wr_nd5_casebook.csv"), low_memory=False)
    nd.columns = [str(c).strip().lower() for c in nd.columns]
    if len(nd) != EXPECTED_ROWS:
        raise RuntimeError(f"ND5 row drift expected={EXPECTED_ROWS} got={len(nd)}")

    paired = pd.read_csv(one(a.wr_r1_root, "wr_r1_paired_wr_casebook.csv"), low_memory=False)
    paired.columns = [str(c).strip().lower() for c in paired.columns]
    keys = ["season", "week", "team", "player_clean_key"]

    base = paired.loc[
        num(paired["season"]).eq(2025)
        & paired["position"].astype(str).str.upper().eq("WR")
    ].copy()

    rec = base.loc[base["market"].astype(str).str.lower().eq("receptions"), keys + ["actual_m38", "mc_proj_m38"]].rename(
        columns={"actual_m38": "actual_rec", "mc_proj_m38": "proj_rec"}
    )
    yds = base.loc[base["market"].astype(str).str.lower().eq("rec_yards"), keys + ["actual_m38", "mc_proj_m38"]].rename(
        columns={"actual_m38": "actual_yards", "mc_proj_m38": "proj_yards"}
    )
    rr = rec.merge(yds, on=keys, how="inner", validate="one_to_one")

    keep = [
        "season", "week", "team", "player_clean_key", "player", "position", "m38_wr_rank", "m38_wr_role",
        "pred_targets", "actual_targets", "actual_rec_yards"
    ]
    keep = [c for c in keep if c in nd.columns]
    x = nd[keep].merge(rr, on=keys, how="inner", validate="one_to_one")
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"merged row drift expected={EXPECTED_ROWS} got={len(x)}")

    for c in ["pred_targets", "actual_targets", "actual_rec_yards", "actual_rec", "proj_rec", "actual_yards", "proj_yards"]:
        x[c] = num(x[c])

    actual_yard_truth_diff = float((x["actual_rec_yards"] - x["actual_yards"]).abs().max())
    if actual_yard_truth_diff > 1e-6:
        raise RuntimeError(f"actual yard source mismatch {actual_yard_truth_diff}")

    x["actual_catch_rate"] = rate(x["actual_rec"], x["actual_targets"])
    x["proj_catch_rate"] = rate(x["proj_rec"], x["pred_targets"])
    x["actual_ypr"] = rate(x["actual_yards"], x["actual_rec"])
    x["proj_ypr"] = rate(x["proj_yards"], x["proj_rec"])

    scoreable = x[["actual_catch_rate", "proj_catch_rate", "actual_ypr", "proj_ypr", "pred_targets", "actual_targets"]].notna().all(axis=1)
    z = x.loc[scoreable].copy()

    proj_recon = z["pred_targets"] * z["proj_catch_rate"] * z["proj_ypr"]
    actual_recon = z["actual_targets"] * z["actual_catch_rate"] * z["actual_ypr"]
    proj_recon_err = float((proj_recon - z["proj_yards"]).abs().max()) if len(z) else np.nan
    actual_recon_err = float((actual_recon - z["actual_yards"]).abs().max()) if len(z) else np.nan
    if proj_recon_err > 1e-6 or actual_recon_err > 1e-6:
        raise RuntimeError(f"factor reconciliation failed projected={proj_recon_err} actual={actual_recon_err}")

    comps = []
    for _, r in z.iterrows():
        comps.append(shapley_row(
            (float(r.pred_targets), float(r.proj_catch_rate), float(r.proj_ypr)),
            (float(r.actual_targets), float(r.actual_catch_rate), float(r.actual_ypr)),
        ))
    c = pd.DataFrame(comps, columns=["target_component", "catch_component", "ypr_component"], index=z.index)
    z = pd.concat([z, c], axis=1)
    z["yard_residual"] = z["actual_yards"] - z["proj_yards"]
    z["decomp_recon"] = z[["target_component", "catch_component", "ypr_component"]].sum(axis=1)
    decomp_err = float((z["yard_residual"] - z["decomp_recon"]).abs().max()) if len(z) else np.nan
    if decomp_err > 1e-6:
        raise RuntimeError(f"Shapley decomposition failed {decomp_err}")

    z["yard_ae"] = z["yard_residual"].abs()
    z["target_error"] = z["actual_targets"] - z["pred_targets"]
    z["rec_error"] = z["actual_rec"] - z["proj_rec"]

    profiles = []
    for pk, g in z.groupby("player_clean_key"):
        if len(g) < 8:
            continue
        ta = float(g["target_component"].abs().mean())
        ca = float(g["catch_component"].abs().mean())
        ya = float(g["ypr_component"].abs().mean())
        vals = {"TARGETS": ta, "CATCH": ca, "YPR": ya}
        order = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
        dom = order[0][0] if order[0][1] >= 1.25 * order[1][1] and order[0][1] >= 1.25 * order[2][1] else "MIXED"
        total = ta + ca + ya
        profiles.append({
            "player_key": pk,
            "player": str(g["player"].iloc[0]) if "player" in g.columns else pk,
            "games": int(len(g)),
            "yard_mae": float(g["yard_ae"].mean()),
            "yard_bias_actual_minus_proj": float(g["yard_residual"].mean()),
            "target_mae": float(g["target_error"].abs().mean()),
            "target_bias_actual_minus_proj": float(g["target_error"].mean()),
            "reception_mae": float(g["rec_error"].abs().mean()),
            "reception_bias_actual_minus_proj": float(g["rec_error"].mean()),
            "target_component_mean": float(g["target_component"].mean()),
            "target_component_abs": ta,
            "catch_component_mean": float(g["catch_component"].mean()),
            "catch_component_abs": ca,
            "ypr_component_mean": float(g["ypr_component"].mean()),
            "ypr_component_abs": ya,
            "target_component_share": ta / total if total else np.nan,
            "catch_component_share": ca / total if total else np.nan,
            "ypr_component_share": ya / total if total else np.nan,
            "yard_miss20": float(g["yard_ae"].ge(20).mean()),
            "yard_miss30": float(g["yard_ae"].ge(30).mean()),
            "yard_miss40": float(g["yard_ae"].ge(40).mean()),
            "dominant_mechanism": dom,
        })
    ps = pd.DataFrame(profiles)
    counts = ps["dominant_mechanism"].value_counts().to_dict() if len(ps) else {}

    result = {
        "migration": "WR_R5_TARGET_CATCH_YPR_INDIVIDUAL_DECOMPOSITION",
        "nd5_rows": int(len(nd)),
        "merged_rows": int(len(x)),
        "scoreable_rows": int(len(z)),
        "unscoreable_rows": int((~scoreable).sum()),
        "qualifying_players": int(len(ps)),
        "dominant_mechanism_counts": {str(k): int(v) for k, v in counts.items()},
        "actual_yard_truth_max_abs_diff": actual_yard_truth_diff,
        "projected_factor_reconciliation_max_abs_error": proj_recon_err,
        "actual_factor_reconciliation_max_abs_error": actual_recon_err,
        "shapley_decomposition_max_abs_error": decomp_err,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": "WR_TARGET_CATCH_YPR_INDIVIDUAL_MECHANISMS_MAPPED",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "wr_r5_mechanism_casebook.csv", index=False)
    ps.sort_values("yard_mae", ascending=False).to_csv(a.out_dir / "wr_r5_individual_mechanisms.csv", index=False)
    (a.out_dir / "wr_r5_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    if len(ps):
        print(ps.sort_values("yard_mae", ascending=False).head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
