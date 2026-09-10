#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

KEYS = ["season", "week", "team", "player_clean_key"]
STATES = ["D1", "D2_SHORT", "D2_MEDIUM", "D2_LONG", "D3_SHORT", "D3_MEDIUM", "D3_LONG", "D4"]
GROUPS = {
    "FIRST_DOWN": ["D1"],
    "SECOND_DOWN": ["D2_SHORT", "D2_MEDIUM", "D2_LONG"],
    "LATE_DOWN": ["D3_SHORT", "D3_MEDIUM", "D3_LONG", "D4"],
}
TOL = 1e-10


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(x):
    return pd.to_numeric(x, errors="coerce")


def canon(v):
    t = canon_team(v)
    return "WAS" if t == "WSH" else t


def canon_keys(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].fillna("").astype(str).map(canon)
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    return x


def corr_metrics(x, y) -> dict:
    z = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(z) < 3 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return {"n": int(len(z)), "pearson": np.nan, "spearman": np.nan, "same_sign": np.nan}
    return {
        "n": int(len(z)),
        "pearson": float(z.x.corr(z.y, method="pearson")),
        "spearman": float(z.x.corr(z.y, method="spearman")),
        "same_sign": float((np.sign(z.x) == np.sign(z.y)).mean()),
    }


def load_parent(root: Path) -> pd.DataFrame:
    d = pd.read_csv(one(root, "down_distance_decomposition_casebook.csv"), low_memory=False)
    d.columns = [str(c).strip() for c in d.columns]
    need = set(KEYS + ["within_state_rate_contrib", "fixed_057_residual"])
    for s in STATES:
        need.update({f"p_{s}", f"q_{s}", f"a_{s}", f"b_{s}"})
    missing = sorted(need - set(d.columns))
    if missing:
        raise RuntimeError(f"parent missing columns {missing}")
    d = canon_keys(d)
    if len(d) != 884 or d.duplicated(KEYS).any():
        raise RuntimeError(f"parent integrity failure rows={len(d)}")
    counts = {int(k): int(v) for k, v in d.season.value_counts().to_dict().items()}
    if counts != {2024: 444, 2025: 440}:
        raise RuntimeError(f"parent season drift {counts}")
    for c in ["within_state_rate_contrib", "fixed_057_residual"] + [f"{p}_{s}" for s in STATES for p in ["p", "q", "a", "b"]]:
        d[c] = num(d[c])
    return d


def add_contributions(d: pd.DataFrame) -> pd.DataFrame:
    z = d.copy()
    for s in STATES:
        z[f"c_{s}"] = 0.5 * (z[f"p_{s}"] + z[f"a_{s}"]) * (z[f"b_{s}"] - z[f"q_{s}"])
    for g, states in GROUPS.items():
        z[f"g_{g}"] = sum(z[f"c_{s}"] for s in states)
    z["state_sum"] = sum(z[f"c_{s}"] for s in STATES)
    z["group_sum"] = sum(z[f"g_{g}"] for g in GROUPS)
    return z


def summaries(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    state_rows = []
    season_views = [
        ("2024", z.season.eq(2024)),
        ("2025", z.season.eq(2025)),
        ("POOLED_2024_2025", pd.Series(True, index=z.index)),
    ]
    cohorts = [
        ("ALL", pd.Series(True, index=z.index)),
        ("ABS_RATE_MISS_0_08_PLUS", z.fixed_057_residual.abs().ge(0.08)),
        ("ABS_RATE_MISS_0_12_PLUS", z.fixed_057_residual.abs().ge(0.12)),
    ]
    for sl, sm in season_views:
        for cl, cm in cohorts:
            g = z.loc[sm & cm].copy()
            if g.empty:
                continue
            group_mass = {k: float(g[f"g_{k}"].abs().mean()) for k in GROUPS}
            denom = sum(group_mass.values())
            arr = np.column_stack([g[f"g_{k}"].abs().to_numpy(float) for k in GROUPS])
            labels = np.array(list(GROUPS), object)
            dominant = labels[arr.argmax(axis=1)]
            for k in GROUPS:
                v = g[f"g_{k}"]
                group_rows.append({
                    "season": sl, "cohort": cl, "group": k, "n": int(len(g)),
                    "mean_contribution": float(v.mean()),
                    "mean_abs_contribution": group_mass[k],
                    "abs_group_mass_share": float(group_mass[k] / denom) if denom else np.nan,
                    "sign_agreement_with_within_state": float((np.sign(v) == np.sign(g.within_state_rate_contrib)).mean()),
                    "sign_agreement_with_fixed_057_residual": float((np.sign(v) == np.sign(g.fixed_057_residual)).mean()),
                    "dominant_group_row_rate": float((dominant == k).mean()),
                    "p50_abs": float(v.abs().quantile(.50)),
                    "p75_abs": float(v.abs().quantile(.75)),
                    "p90_abs": float(v.abs().quantile(.90)),
                })
            for s in STATES:
                v = g[f"c_{s}"]
                state_rows.append({
                    "season": sl, "cohort": cl, "state": s, "n": int(len(g)),
                    "mean_contribution": float(v.mean()),
                    "mean_abs_contribution": float(v.abs().mean()),
                    "sign_agreement_with_within_state": float((np.sign(v) == np.sign(g.within_state_rate_contrib)).mean()),
                    "sign_agreement_with_fixed_057_residual": float((np.sign(v) == np.sign(g.fixed_057_residual)).mean()),
                    "p50_abs": float(v.abs().quantile(.50)),
                    "p75_abs": float(v.abs().quantile(.75)),
                    "p90_abs": float(v.abs().quantile(.90)),
                })
    return pd.DataFrame(group_rows), pd.DataFrame(state_rows)


def load_shared(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(one(root, "qb_wr_shared_pass_volume_primary_2025.csv"), low_memory=False)
    s = pd.read_csv(one(root, "qb_wr_shared_pass_volume_secondary_2024_2025.csv"), low_memory=False)
    for d in (p, s):
        d.columns = [str(c).strip().lower() for c in d.columns]
    p = canon_keys(p)
    s = canon_keys(s)
    if len(p) != 440 or len(s) != 884 or p.duplicated(KEYS).any() or s.duplicated(KEYS).any():
        raise RuntimeError("shared cohort integrity failure")
    return p, s


def shared_attribution(z: pd.DataFrame, root: Path) -> pd.DataFrame:
    p, s = load_shared(root)
    cols = KEYS + ["within_state_rate_contrib"] + [f"c_{x}" for x in STATES] + [f"g_{x}" for x in GROUPS]
    p = p.merge(z[cols], on=KEYS, how="left", validate="one_to_one")
    s = s.merge(z[cols], on=KEYS, how="left", validate="one_to_one")
    sigs = {"WITHIN_STATE_TOTAL": "within_state_rate_contrib"}
    sigs.update({f"STATE_{x}": f"c_{x}" for x in STATES})
    sigs.update({f"GROUP_{x}": f"g_{x}" for x in GROUPS})
    rows = []
    views = [
        ("PRIMARY_WR_TARGET_MASS", "2025", p, "wr_target_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "POOLED_2024_2025", s, "wr_reception_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "2024", s.loc[s.season.eq(2024)], "wr_reception_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "2025", s.loc[s.season.eq(2025)], "wr_reception_mass_residual"),
    ]
    for view, sl, d, target in views:
        base = corr_metrics(d["within_state_rate_contrib"], d[target])
        for label, col in sigs.items():
            m = corr_metrics(d[col], d[target])
            rec = {"view": view, "season": sl, "signal": label, **m,
                   "parent_within_state_abs_spearman": abs(float(base["spearman"])) if np.isfinite(base["spearman"]) else np.nan,
                   "leave_one_group_out_abs_spearman": np.nan,
                   "leave_one_group_out_spearman_drop": np.nan}
            if label.startswith("GROUP_"):
                group = label.replace("GROUP_", "")
                without = d["within_state_rate_contrib"] - d[f"g_{group}"]
                wm = corr_metrics(without, d[target])
                rec["leave_one_group_out_abs_spearman"] = abs(float(wm["spearman"])) if np.isfinite(wm["spearman"]) else np.nan
                rec["leave_one_group_out_spearman_drop"] = rec["parent_within_state_abs_spearman"] - rec["leave_one_group_out_abs_spearman"]
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-root", type=Path, required=True)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    z = add_contributions(load_parent(a.parent_root))
    state_err = float((z.state_sum - z.within_state_rate_contrib).abs().max())
    group_err = float((z.group_sum - z.within_state_rate_contrib).abs().max())
    finite = bool(np.isfinite(z[[f"c_{s}" for s in STATES]].to_numpy(float)).all())
    gs, ss = summaries(z)
    sh = shared_attribution(z, a.shared_root)

    integ = {
        "exact_884_parent_rows": len(z) == 884,
        "exact_444_440_season_split": {int(k): int(v) for k, v in z.season.value_counts().to_dict().items()} == {2024: 444, 2025: 440},
        "exact_440_884_shared_cohorts": True,
        "no_duplicate_canonical_keys": not z.duplicated(KEYS).any(),
        "all_eight_state_contributions_finite": finite,
        "state_sum_identity": state_err <= TOL,
        "group_sum_identity": group_err <= TOL,
        "shared_receiver_joins_exact_unique": True,
        "zero_sportsbook_inputs": True,
        "zero_model_fitting": True,
        "zero_new_target_game_pbp_loading": True,
        "zero_production_changes": True,
    }

    pool = gs.loc[(gs.season.eq("POOLED_2024_2025")) & (gs.cohort.eq("ALL"))].set_index("group")
    y24 = gs.loc[(gs.season.eq("2024")) & (gs.cohort.eq("ALL"))].set_index("group")
    y25 = gs.loc[(gs.season.eq("2025")) & (gs.cohort.eq("ALL"))].set_index("group")
    wrt = sh.loc[(sh.view.eq("PRIMARY_WR_TARGET_MASS")) & (sh.season.eq("2025"))].set_index("signal")
    wrr = sh.loc[(sh.view.eq("SECONDARY_WR_RECEPTION_MASS")) & (sh.season.eq("POOLED_2024_2025"))].set_index("signal")

    routing = {}
    for g in GROUPS:
        others = [x for x in GROUPS if x != g]
        vp = float(pool.loc[g, "mean_abs_contribution"])
        second = max(float(pool.loc[o, "mean_abs_contribution"]) for o in others)
        v24 = float(y24.loc[g, "mean_abs_contribution"])
        v25 = float(y25.loc[g, "mean_abs_contribution"])
        m24 = max(float(y24.loc[x, "mean_abs_contribution"]) for x in GROUPS)
        m25 = max(float(y25.loc[x, "mean_abs_contribution"]) for x in GROUPS)
        stable = (v24 >= m24 and v25 >= .9 * m25) or (v25 >= m25 and v24 >= .9 * m24)
        st = float(wrt.loc[f"GROUP_{g}", "spearman"])
        sr = float(wrr.loc[f"GROUP_{g}", "spearman"])
        dt = float(wrt.loc[f"GROUP_{g}", "leave_one_group_out_spearman_drop"])
        dr = float(wrr.loc[f"GROUP_{g}", "leave_one_group_out_spearman_drop"])
        routing[g] = {
            "largest_pooled": vp >= second,
            "season_stability": bool(stable),
            "pooled_lead_ge_20pct": vp >= 1.20 * second,
            "wr_target_abs_spearman_ge_0_30": abs(st) >= .30,
            "wr_reception_abs_spearman_ge_0_20": abs(sr) >= .20,
            "leave_one_group_out_drop_ge_0_05_either_view": max(dt, dr) >= .05,
            "pooled_mean_abs": vp,
            "second_pooled_mean_abs": second,
            "mean_abs_2024": v24,
            "mean_abs_2025": v25,
            "wr_target_spearman_2025": st,
            "wr_reception_spearman_pooled": sr,
            "wr_target_leaveout_drop": dt,
            "wr_reception_leaveout_drop": dr,
        }

    qualifying = [g for g, r in routing.items() if all([
        r["largest_pooled"], r["season_stability"], r["pooled_lead_ge_20pct"],
        r["wr_target_abs_spearman_ge_0_30"], r["wr_reception_abs_spearman_ge_0_20"],
        r["leave_one_group_out_drop_ge_0_05_either_view"],
    ])]
    mapping = {
        "FIRST_DOWN": "FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC",
        "SECOND_DOWN": "SECOND_DOWN_SHARED_PRIMARY_DIAGNOSTIC",
        "LATE_DOWN": "LATE_DOWN_SHARED_PRIMARY_DIAGNOSTIC",
    }
    if not all(integ.values()):
        disposition = "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE"
    elif len(qualifying) == 1:
        disposition = mapping[qualifying[0]]
    else:
        disposition = "DISTRIBUTED_WITHIN_STATE_SHARED_MECHANISM"

    result = {
        "migration": "QB_PASS_RATE_STATE_SHARED_ATTRIBUTION_V1",
        "disposition": disposition,
        "production_actionable": False,
        "qualifying_primary": qualifying,
        "identity_max_abs_errors": {"state_sum": state_err, "group_sum": group_err},
        "integrity_gates": integ,
        "routing": routing,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "state_shared_attribution_casebook.csv", index=False)
    gs.to_csv(a.out_dir / "state_group_summary.csv", index=False)
    ss.to_csv(a.out_dir / "state_individual_summary.csv", index=False)
    sh.to_csv(a.out_dir / "state_shared_receiver_attribution.csv", index=False)
    (a.out_dir / "state_shared_attribution_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(integ.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
