#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

KEYS = ["season", "week", "team", "player_clean_key"]
ZONES = ["BACKED_UP", "OWN_OPEN_FIELD", "PLUS_TERRITORY", "RED_ZONE"]
COMPONENTS = {
    "ZONE_REFERENCE_LEVEL": "c_level",
    "FIELD_POSITION_OCCUPANCY": "c_field_position_occupancy",
    "WITHIN_ZONE_PASS_PROPENSITY": "c_within_zone_pass_propensity",
}
TOL = 1e-10
HISTORY_GAMES = 8
SHRINK_GAMES = 4.0


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


def zone_of(y: float) -> str | None:
    if not np.isfinite(y):
        return None
    if y >= 80:
        return "BACKED_UP"
    if y > 50:
        return "OWN_OPEN_FIELD"
    if y > 20:
        return "PLUS_TERRITORY"
    return "RED_ZONE"


def load_parent(root: Path) -> pd.DataFrame:
    d = pd.read_csv(one(root, "state_shared_attribution_casebook.csv"), low_memory=False)
    d.columns = [str(c).strip() for c in d.columns]
    need = KEYS + ["p_D1", "a_D1", "q_D1", "b_D1", "c_D1", "fixed_057_residual"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise RuntimeError(f"parent missing {missing}")
    d = canon_keys(d[need].copy())
    for c in ["p_D1", "a_D1", "q_D1", "b_D1", "c_D1", "fixed_057_residual"]:
        d[c] = num(d[c])
    counts = {int(k): int(v) for k, v in d.season.value_counts().to_dict().items()}
    if len(d) != 884 or counts != {2024: 444, 2025: 440} or d.duplicated(KEYS).any() or d.duplicated(["season", "week", "team"]).any():
        raise RuntimeError(f"parent integrity drift rows={len(d)} seasons={counts}")
    return d


def regular_only(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    c = "season_type" if "season_type" in x.columns else ("game_type" if "game_type" in x.columns else None)
    if c:
        s = x[c].fillna("").astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            x = x.loc[keep].copy()
    return x


def to_pd(o):
    if isinstance(o, pd.DataFrame):
        return o.copy()
    if hasattr(o, "to_pandas"):
        return o.to_pandas()
    return pd.DataFrame(o)


def load_pbp() -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl
    frames = []
    audit = {}
    for season in (2023, 2024, 2025):
        raw = nfl.load_pbp(seasons=[season])
        p = regular_only(to_pd(raw))
        p.columns = [str(c).strip().lower() for c in p.columns]
        required = {"week", "posteam", "defteam", "down", "ydstogo", "yardline_100", "qb_dropback", "rush_attempt"}
        missing = sorted(required - set(p.columns))
        if missing:
            raise RuntimeError(f"PBP {season} missing {missing}")
        p["season"] = season
        for c in ["week", "down", "ydstogo", "yardline_100", "qb_dropback", "rush_attempt"]:
            p[c] = num(p[c])
        p["posteam"] = p.posteam.fillna("").astype(str).map(canon)
        p["defteam"] = p.defteam.fillna("").astype(str).map(canon)
        two = num(p["two_point_attempt"]).fillna(0).eq(1) if "two_point_attempt" in p else pd.Series(False, index=p.index)
        nop = num(p["no_play"]).fillna(0).eq(1) if "no_play" in p else pd.Series(False, index=p.index)
        eligible = p.week.between(1, 18) & p.down.eq(1) & p.posteam.ne("") & p.defteam.ne("") & (p.qb_dropback.fillna(0).eq(1) | p.rush_attempt.fillna(0).eq(1)) & ~two & ~nop
        q = p.loc[eligible, ["season", "week", "posteam", "defteam", "yardline_100", "qb_dropback"]].copy()
        q["dropback"] = q.qb_dropback.fillna(0).eq(1).astype(int)
        q["zone"] = q.yardline_100.map(lambda v: zone_of(float(v)) if pd.notna(v) else None)
        q["decomp"] = q.zone.notna()
        audit[str(season)] = {
            "eligible_first_down_plays": int(len(q)),
            "decomposable_first_down_plays": int(q.decomp.sum()),
            "yardline_coverage": float(q.decomp.mean()) if len(q) else np.nan,
        }
        frames.append(q)
    return pd.concat(frames, ignore_index=True), audit


def game_table(p: pd.DataFrame) -> pd.DataFrame:
    x = p.copy()
    x["decomp_db"] = x.dropback * x.decomp.astype(int)
    base = x.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        eligible_d1_plays=("dropback", "size"),
        decomposable_d1_plays=("decomp", "sum"),
        decomposable_d1_dropbacks=("decomp_db", "sum"),
    ).rename(columns={"posteam": "team", "defteam": "opponent"})
    q = x.loc[x.decomp].copy()
    g = q.groupby(["season", "week", "posteam", "defteam", "zone"], as_index=False).agg(n=("dropback", "size"), db=("dropback", "sum"))
    npiv = g.pivot_table(index=["season", "week", "posteam", "defteam"], columns="zone", values="n", fill_value=0, aggfunc="sum")
    dpiv = g.pivot_table(index=["season", "week", "posteam", "defteam"], columns="zone", values="db", fill_value=0, aggfunc="sum")
    for z in ZONES:
        if z not in npiv:
            npiv[z] = 0
        if z not in dpiv:
            dpiv[z] = 0
    npiv = npiv[ZONES].reset_index().rename(columns={**{z: f"n_{z}" for z in ZONES}, "posteam": "team", "defteam": "opponent"})
    dpiv = dpiv[ZONES].reset_index().rename(columns={**{z: f"db_{z}" for z in ZONES}, "posteam": "team", "defteam": "opponent"})
    out = base.merge(npiv, on=["season", "week", "team", "opponent"], how="left", validate="one_to_one").merge(dpiv, on=["season", "week", "team", "opponent"], how="left", validate="one_to_one")
    for c in [f"n_{z}" for z in ZONES] + [f"db_{z}" for z in ZONES]:
        out[c] = num(out[c]).fillna(0)
    out["team"] = out.team.map(canon)
    out["opponent"] = out.opponent.map(canon)
    out["ord"] = out.season.astype(int) * 100 + out.week.astype(int)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate field-position team-week")
    return out


def hist_occ(g: pd.DataFrame, zone: str) -> float:
    den = float(num(g.decomposable_d1_plays).sum())
    return float(num(g[f"n_{zone}"]).sum() / den) if den > 0 else np.nan


def hist_rate(g: pd.DataFrame, zone: str) -> float:
    den = float(num(g[f"n_{zone}"]).sum())
    return float(num(g[f"db_{zone}"]).sum() / den) if den > 0 else np.nan


def shr(v: float, n_games: int, league: float) -> float:
    if n_games > 0 and np.isfinite(v):
        return float((n_games * v + SHRINK_GAMES * league) / (n_games + SHRINK_GAMES))
    return np.nan


def ref_for(r, games: pd.DataFrame) -> dict:
    target_ord = int(r.season) * 100 + int(r.week)
    prior = games.loc[games.ord < target_ord]
    if prior.empty:
        raise RuntimeError("missing strict-prior league history")
    off = prior.loc[prior.team.eq(r.team)].sort_values("ord").tail(HISTORY_GAMES)
    deff = prior.loc[prior.opponent.eq(r.opponent)].sort_values("ord").tail(HISTORY_GAMES)
    rec = {
        "max_prior_ord": int(prior.ord.max()),
        "off_prior_games": int(len(off)),
        "def_prior_games": int(len(deff)),
    }
    po = {}
    qr = {}
    for z in ZONES:
        lo = hist_occ(prior, z)
        lr = hist_rate(prior, z)
        oo = shr(hist_occ(off, z), len(off), lo)
        od = shr(hist_occ(deff, z), len(deff), lo)
        qo = shr(hist_rate(off, z), len(off), lr)
        qd = shr(hist_rate(deff, z), len(deff), lr)
        ov = [v for v in (oo, od) if np.isfinite(v)]
        rv = [v for v in (qo, qd) if np.isfinite(v)]
        po[z] = float(np.mean(ov)) if ov else float(lo)
        qr[z] = float(np.clip(np.mean(rv) if rv else lr, .05, .95))
    den = sum(po.values())
    if not np.isfinite(den) or den <= 0:
        raise RuntimeError("invalid reference zone occupancy")
    po = {z: float(v / den) for z, v in po.items()}
    for z in ZONES:
        rec[f"p_{z}"] = po[z]
        rec[f"q_{z}"] = qr[z]
    return rec


def corr_metrics(x, y) -> dict:
    d = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(d) < 3 or d.x.nunique() < 2 or d.y.nunique() < 2:
        return {"n": int(len(d)), "pearson": np.nan, "spearman": np.nan, "same_sign": np.nan}
    return {
        "n": int(len(d)),
        "pearson": float(d.x.corr(d.y, method="pearson")),
        "spearman": float(d.x.corr(d.y, method="spearman")),
        "same_sign": float((np.sign(d.x) == np.sign(d.y)).mean()),
    }


def summarize(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    views = [("2024", z.season.eq(2024)), ("2025", z.season.eq(2025)), ("POOLED_2024_2025", pd.Series(True, index=z.index))]
    cohorts = [("ALL", pd.Series(True, index=z.index)), ("ABS_RATE_MISS_0_08_PLUS", z.fixed_057_residual.abs().ge(.08)), ("ABS_RATE_MISS_0_12_PLUS", z.fixed_057_residual.abs().ge(.12))]
    for sl, sm in views:
        for cl, cm in cohorts:
            g = z.loc[sm & cm].copy()
            if g.empty:
                continue
            mass = {k: float(g[v].abs().mean()) for k, v in COMPONENTS.items()}
            den = sum(mass.values())
            arr = np.column_stack([g[v].abs().to_numpy(float) for v in COMPONENTS.values()])
            labels = np.array(list(COMPONENTS), object)
            dom = labels[arr.argmax(axis=1)]
            for k, c in COMPONENTS.items():
                v = g[c]
                rows.append({
                    "season": sl, "cohort": cl, "component": k, "n": int(len(g)),
                    "mean_contribution": float(v.mean()), "mean_abs_contribution": mass[k],
                    "abs_mass_share": float(mass[k] / den) if den else np.nan,
                    "sign_agreement_with_parent_c_d1": float((np.sign(v) == np.sign(g.c_D1)).mean()),
                    "dominant_row_rate": float((dom == k).mean()),
                    "p50_abs": float(v.abs().quantile(.50)), "p75_abs": float(v.abs().quantile(.75)), "p90_abs": float(v.abs().quantile(.90)),
                })
    return pd.DataFrame(rows)


def zone_summary(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sl, g in [("2024", z.loc[z.season.eq(2024)]), ("2025", z.loc[z.season.eq(2025)]), ("POOLED_2024_2025", z)]:
        for x in ZONES:
            oc = .5 * (g[f"a_{x}"] - g[f"p_{x}"]) * (g[f"q_{x}"] + g[f"b_{x}"])
            rc = .5 * (g[f"p_{x}"] + g[f"a_{x}"]) * (g[f"b_{x}"] - g[f"q_{x}"])
            rows.append({
                "season": sl, "zone": x, "n": int(len(g)),
                "actual_occupancy": float(g[f"a_{x}"].mean()), "reference_occupancy": float(g[f"p_{x}"].mean()),
                "actual_within_zone_dbr": float(g[f"b_{x}"].mean()), "reference_within_zone_dbr": float(g[f"q_{x}"].mean()),
                "mean_occupancy_delta": float((g[f"a_{x}"] - g[f"p_{x}"]).mean()),
                "mean_within_zone_dbr_delta": float((g[f"b_{x}"] - g[f"q_{x}"]).mean()),
                "mean_occupancy_contribution": float(oc.mean()), "mean_abs_occupancy_contribution": float(oc.abs().mean()),
                "mean_rate_contribution": float(rc.mean()), "mean_abs_rate_contribution": float(rc.abs().mean()),
            })
    return pd.DataFrame(rows)


def load_shared(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(one(root, "qb_wr_shared_pass_volume_primary_2025.csv"), low_memory=False)
    s = pd.read_csv(one(root, "qb_wr_shared_pass_volume_secondary_2024_2025.csv"), low_memory=False)
    for d in (p, s):
        d.columns = [str(c).strip().lower() for c in d.columns]
    p, s = canon_keys(p), canon_keys(s)
    if len(p) != 440 or len(s) != 884 or p.duplicated(KEYS).any() or s.duplicated(KEYS).any():
        raise RuntimeError("shared receiver cohort drift")
    return p, s


def shared_attribution(z: pd.DataFrame, root: Path) -> pd.DataFrame:
    p, s = load_shared(root)
    keep = KEYS + ["c_D1", *COMPONENTS.values()]
    p = p.merge(z[keep], on=KEYS, how="left", validate="one_to_one")
    s = s.merge(z[keep], on=KEYS, how="left", validate="one_to_one")
    if p[keep[4:]].isna().any().any() or s[keep[4:]].isna().any().any():
        raise RuntimeError("shared component join missing")
    signals = {"PARENT_C_D1": "c_D1", **COMPONENTS}
    rows = []
    views = [
        ("PRIMARY_WR_TARGET_MASS", "2025", p, "wr_target_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "POOLED_2024_2025", s, "wr_reception_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "2024", s.loc[s.season.eq(2024)], "wr_reception_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "2025", s.loc[s.season.eq(2025)], "wr_reception_mass_residual"),
    ]
    for view, sl, d, target in views:
        for label, c in signals.items():
            rows.append({"view": view, "season": sl, "signal": label, **corr_metrics(d[c], d[target])})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-root", type=Path, required=True)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    parent = load_parent(a.parent_root)
    pbp, pbp_audit = load_pbp()
    games = game_table(pbp)
    z = parent.merge(games, on=["season", "week", "team"], how="left", validate="one_to_one")
    if len(z) != 884 or z.opponent.isna().any():
        raise RuntimeError("target PBP alignment failure")
    refs = pd.DataFrame([ref_for(r, games) for _, r in z.iterrows()], index=z.index)
    z = pd.concat([z, refs], axis=1)
    for x in ZONES:
        z[f"a_{x}"] = num(z[f"n_{x}"]) / num(z.decomposable_d1_plays)
        raw = num(z[f"db_{x}"]) / num(z[f"n_{x}"]).replace(0, np.nan)
        z[f"b_{x}"] = raw.where(num(z[f"n_{x}"]).gt(0), z[f"q_{x}"])
    z["r_zone"] = sum(z[f"p_{x}"] * z[f"q_{x}"] for x in ZONES)
    z["b_field"] = sum(z[f"a_{x}"] * z[f"b_{x}"] for x in ZONES)
    z["level_unscaled"] = z.r_zone - z.q_D1
    z["occupancy_unscaled"] = .5 * (
        sum((z[f"a_{x}"] - z[f"p_{x}"]) * z[f"q_{x}"] for x in ZONES)
        + sum((z[f"a_{x}"] - z[f"p_{x}"]) * z[f"b_{x}"] for x in ZONES)
    )
    z["rate_unscaled"] = .5 * (
        sum(z[f"p_{x}"] * (z[f"b_{x}"] - z[f"q_{x}"]) for x in ZONES)
        + sum(z[f"a_{x}"] * (z[f"b_{x}"] - z[f"q_{x}"]) for x in ZONES)
    )
    z["parent_weight"] = .5 * (z.p_D1 + z.a_D1)
    z["c_level"] = z.parent_weight * z.level_unscaled
    z["c_field_position_occupancy"] = z.parent_weight * z.occupancy_unscaled
    z["c_within_zone_pass_propensity"] = z.parent_weight * z.rate_unscaled

    ref_occ_err = float((sum(z[f"p_{x}"] for x in ZONES) - 1).abs().max())
    act_occ_err = float((sum(z[f"a_{x}"] for x in ZONES) - 1).abs().max())
    target_rate_err = float((z.b_field - z.b_D1).abs().max())
    unscaled_err = float(((z.level_unscaled + z.occupancy_unscaled + z.rate_unscaled) - (z.b_D1 - z.q_D1)).abs().max())
    scaled_err = float(((z.c_level + z.c_field_position_occupancy + z.c_within_zone_pass_propensity) - z.c_D1).abs().max())
    coverage = {
        "pooled": float(z.decomposable_d1_plays.sum() / z.eligible_d1_plays.sum()),
        "2024": float(z.loc[z.season.eq(2024), "decomposable_d1_plays"].sum() / z.loc[z.season.eq(2024), "eligible_d1_plays"].sum()),
        "2025": float(z.loc[z.season.eq(2025), "decomposable_d1_plays"].sum() / z.loc[z.season.eq(2025), "eligible_d1_plays"].sum()),
    }

    cs = summarize(z)
    zs = zone_summary(z)
    sr = shared_attribution(z, a.shared_root)

    integrity = {
        "exact_884_parent_rows": len(z) == 884,
        "exact_444_440_season_split": {int(k): int(v) for k, v in z.season.value_counts().to_dict().items()} == {2024: 444, 2025: 440},
        "exact_440_884_shared_receiver_cohorts": True,
        "no_duplicate_canonical_keys": not z.duplicated(KEYS).any(),
        "yardline_coverage_pooled_ge_0_99": coverage["pooled"] >= .99,
        "yardline_coverage_2024_ge_0_985": coverage["2024"] >= .985,
        "yardline_coverage_2025_ge_0_985": coverage["2025"] >= .985,
        "four_zones_mutually_exclusive_exhaustive": True,
        "reference_zone_occupancy_identity": ref_occ_err <= TOL,
        "actual_zone_occupancy_identity": act_occ_err <= TOL,
        "target_field_rate_reconciles_parent_b_d1": target_rate_err <= TOL,
        "unscaled_three_part_identity": unscaled_err <= TOL,
        "scaled_three_part_reconciles_parent_c_d1": scaled_err <= TOL,
        "all_reference_inputs_strictly_prior": bool((z.max_prior_ord < (z.season.astype(int) * 100 + z.week.astype(int))).all()),
        "zero_sportsbook_game_market_inputs": True,
        "zero_model_fitting": True,
        "zero_production_changes": True,
        "target_game_pbp_diagnostic_only": True,
        "shared_receiver_joins_exact_unique": True,
    }

    pool = cs.loc[(cs.season.eq("POOLED_2024_2025")) & (cs.cohort.eq("ALL"))].set_index("component")
    y24 = cs.loc[(cs.season.eq("2024")) & (cs.cohort.eq("ALL"))].set_index("component")
    y25 = cs.loc[(cs.season.eq("2025")) & (cs.cohort.eq("ALL"))].set_index("component")
    wrt = sr.loc[(sr.view.eq("PRIMARY_WR_TARGET_MASS")) & (sr.season.eq("2025"))].set_index("signal")
    wrr = sr.loc[(sr.view.eq("SECONDARY_WR_RECEPTION_MASS")) & (sr.season.eq("POOLED_2024_2025"))].set_index("signal")
    routing = {}
    for k in COMPONENTS:
        others = [o for o in COMPONENTS if o != k]
        vp = float(pool.loc[k, "mean_abs_contribution"])
        second = max(float(pool.loc[o, "mean_abs_contribution"]) for o in others)
        v24 = float(y24.loc[k, "mean_abs_contribution"])
        v25 = float(y25.loc[k, "mean_abs_contribution"])
        m24 = max(float(y24.loc[o, "mean_abs_contribution"]) for o in COMPONENTS)
        m25 = max(float(y25.loc[o, "mean_abs_contribution"]) for o in COMPONENTS)
        stable = (v24 >= m24 and v25 >= .9 * m25) or (v25 >= m25 and v24 >= .9 * m24)
        st = float(wrt.loc[k, "spearman"])
        srp = float(wrr.loc[k, "spearman"])
        routing[k] = {
            "largest_pooled": vp >= second,
            "season_stability": bool(stable),
            "pooled_lead_ge_20pct": vp >= 1.20 * second,
            "wr_target_abs_spearman_ge_0_25": abs(st) >= .25,
            "wr_reception_abs_spearman_ge_0_15": abs(srp) >= .15,
            "pooled_mean_abs": vp,
            "second_pooled_mean_abs": second,
            "mean_abs_2024": v24,
            "mean_abs_2025": v25,
            "wr_target_spearman_2025": st,
            "wr_reception_spearman_pooled": srp,
        }
    qualifying = [k for k, r in routing.items() if all([
        r["largest_pooled"], r["season_stability"], r["pooled_lead_ge_20pct"],
        r["wr_target_abs_spearman_ge_0_25"], r["wr_reception_abs_spearman_ge_0_15"],
    ])]
    mapping = {
        "ZONE_REFERENCE_LEVEL": "FIRST_DOWN_ZONE_REFERENCE_LEVEL_PRIMARY_DIAGNOSTIC",
        "FIELD_POSITION_OCCUPANCY": "FIRST_DOWN_FIELD_POSITION_OCCUPANCY_PRIMARY_DIAGNOSTIC",
        "WITHIN_ZONE_PASS_PROPENSITY": "FIRST_DOWN_WITHIN_FIELD_POSITION_PROPENSITY_PRIMARY_DIAGNOSTIC",
    }
    if not all(integrity.values()):
        disposition = "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE"
    elif len(qualifying) == 1:
        disposition = mapping[qualifying[0]]
    else:
        disposition = "MIXED_FIRST_DOWN_FIELD_POSITION_MECHANISM"

    result = {
        "migration": "QB_FIRST_DOWN_FIELD_POSITION_DECOMP_V1",
        "disposition": disposition,
        "production_actionable": False,
        "qualifying_primary": qualifying,
        "coverage": coverage,
        "identity_max_abs_errors": {
            "reference_zone_occupancy": ref_occ_err,
            "actual_zone_occupancy": act_occ_err,
            "target_field_rate_vs_parent_b_d1": target_rate_err,
            "unscaled_three_part": unscaled_err,
            "scaled_parent_c_d1": scaled_err,
        },
        "integrity_gates": integrity,
        "routing": routing,
        "pbp_audit": pbp_audit,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "first_down_field_position_casebook.csv", index=False)
    cs.to_csv(a.out_dir / "first_down_field_position_component_summary.csv", index=False)
    zs.to_csv(a.out_dir / "first_down_field_position_zone_summary.csv", index=False)
    sr.to_csv(a.out_dir / "first_down_field_position_shared_receiver_attribution.csv", index=False)
    (a.out_dir / "first_down_field_position_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(integrity.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
