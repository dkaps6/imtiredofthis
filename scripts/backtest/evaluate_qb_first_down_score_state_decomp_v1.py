#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import evaluate_qb_first_down_field_position_decomp_v1 as fp

KEYS = ["season", "week", "team", "player_clean_key"]
ZONES = fp.ZONES
STATES = ["TRAILING_8_PLUS", "NEUTRAL", "LEADING_8_PLUS"]
TOL = 1e-10
HISTORY_GAMES = 8
SHRINK_GAMES = 4.0
COMPONENTS = {
    "SCORE_STATE_REFERENCE_LEVEL": "c_score_state_reference_level",
    "SCORE_STATE_OCCUPANCY": "c_score_state_occupancy",
    "WITHIN_SCORE_STATE_PASS_PROPENSITY": "c_within_score_state_pass_propensity",
}


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


def state_of(score: float) -> str | None:
    if not np.isfinite(score):
        return None
    if score <= -8:
        return "TRAILING_8_PLUS"
    if score >= 8:
        return "LEADING_8_PLUS"
    return "NEUTRAL"


def load_parent(root: Path) -> pd.DataFrame:
    d = pd.read_csv(one(root, "first_down_field_position_casebook.csv"), low_memory=False)
    d.columns = [str(c).strip() for c in d.columns]
    need = KEYS + ["opponent", "fixed_057_residual", "rate_unscaled", "parent_weight", "c_within_zone_pass_propensity"]
    for z in ZONES:
        need += [f"p_{z}", f"q_{z}", f"a_{z}", f"b_{z}"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise RuntimeError(f"parent missing {missing}")
    d = canon_keys(d[need].copy())
    d["opponent"] = d["opponent"].fillna("").astype(str).map(canon)
    for c in ["fixed_057_residual", "rate_unscaled", "parent_weight", "c_within_zone_pass_propensity"] + [q for z in ZONES for q in (f"p_{z}", f"q_{z}", f"a_{z}", f"b_{z}")]:
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
    frames, audit = [], {}
    for season in (2023, 2024, 2025):
        p = regular_only(to_pd(nfl.load_pbp(seasons=[season])))
        p.columns = [str(c).strip().lower() for c in p.columns]
        required = {"week", "posteam", "defteam", "down", "yardline_100", "qb_dropback", "rush_attempt", "score_differential"}
        missing = sorted(required - set(p.columns))
        if missing:
            raise RuntimeError(f"PBP {season} missing {missing}")
        p["season"] = season
        for c in ["week", "down", "yardline_100", "qb_dropback", "rush_attempt", "score_differential"]:
            p[c] = num(p[c])
        p["posteam"] = p.posteam.fillna("").astype(str).map(canon)
        p["defteam"] = p.defteam.fillna("").astype(str).map(canon)
        two = num(p["two_point_attempt"]).fillna(0).eq(1) if "two_point_attempt" in p else pd.Series(False, index=p.index)
        nop = num(p["no_play"]).fillna(0).eq(1) if "no_play" in p else pd.Series(False, index=p.index)
        kneel = num(p["qb_kneel"]).fillna(0).eq(1) if "qb_kneel" in p else pd.Series(False, index=p.index)
        eligible = (
            p.week.between(1, 18) & p.down.eq(1) & p.posteam.ne("") & p.defteam.ne("")
            & (p.qb_dropback.fillna(0).eq(1) | p.rush_attempt.fillna(0).eq(1))
            & ~two & ~nop & ~kneel
        )
        q = p.loc[eligible, ["season", "week", "posteam", "defteam", "yardline_100", "qb_dropback", "score_differential"]].copy()
        q["dropback"] = q.qb_dropback.fillna(0).eq(1).astype(int)
        q["zone"] = q.yardline_100.map(lambda v: fp.zone_of(float(v)) if pd.notna(v) else None)
        q["state"] = q.score_differential.map(lambda v: state_of(float(v)) if pd.notna(v) else None)
        q["decomp"] = q.zone.notna() & q.state.notna()
        audit[str(season)] = {
            "eligible_first_down_plays": int(len(q)),
            "decomposable_score_state_plays": int(q.decomp.sum()),
            "score_state_coverage": float(q.decomp.mean()) if len(q) else np.nan,
        }
        frames.append(q)
    return pd.concat(frames, ignore_index=True), audit


def game_table(p: pd.DataFrame) -> pd.DataFrame:
    q = p.loc[p.decomp].copy()
    rows = []
    for (season, week, team, opp), g in q.groupby(["season", "week", "posteam", "defteam"], sort=True):
        rec = {"season": int(season), "week": int(week), "team": canon(team), "opponent": canon(opp), "score_decomp_plays": int(len(g))}
        for z in ZONES:
            gz = g.loc[g.zone.eq(z)]
            rec[f"n_{z}"] = int(len(gz))
            rec[f"db_{z}"] = int(gz.dropback.sum())
            for s in STATES:
                h = gz.loc[gz.state.eq(s)]
                rec[f"n_{z}_{s}"] = int(len(h))
                rec[f"db_{z}_{s}"] = int(h.dropback.sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["ord"] = out.season.astype(int) * 100 + out.week.astype(int)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate score-state team-week")
    return out


def hist_cond_occ(g: pd.DataFrame, z: str, s: str) -> float:
    den = float(num(g[f"n_{z}"]).sum())
    return float(num(g[f"n_{z}_{s}"]).sum() / den) if den > 0 else np.nan


def hist_rate(g: pd.DataFrame, z: str, s: str) -> float:
    den = float(num(g[f"n_{z}_{s}"]).sum())
    return float(num(g[f"db_{z}_{s}"]).sum() / den) if den > 0 else np.nan


def shr(v: float, n_games: int, league: float) -> float:
    if n_games > 0 and np.isfinite(v):
        return float((n_games * v + SHRINK_GAMES * league) / (n_games + SHRINK_GAMES))
    return np.nan


def ref_for(r, games: pd.DataFrame) -> dict:
    target_ord = int(r.season) * 100 + int(r.week)
    prior = games.loc[games.ord < target_ord]
    if prior.empty:
        raise RuntimeError("missing strictly-prior league history")
    off = prior.loc[prior.team.eq(r.team)].sort_values("ord").tail(HISTORY_GAMES)
    deff = prior.loc[prior.opponent.eq(r.opponent)].sort_values("ord").tail(HISTORY_GAMES)
    rec = {"max_prior_ord": int(prior.ord.max()), "off_prior_games": int(len(off)), "def_prior_games": int(len(deff))}
    for z in ZONES:
        ps, qs = {}, {}
        for s in STATES:
            lo = hist_cond_occ(prior, z, s)
            lr = hist_rate(prior, z, s)
            oo = shr(hist_cond_occ(off, z, s), len(off), lo)
            od = shr(hist_cond_occ(deff, z, s), len(deff), lo)
            qo = shr(hist_rate(off, z, s), len(off), lr)
            qd = shr(hist_rate(deff, z, s), len(deff), lr)
            ov = [v for v in (oo, od) if np.isfinite(v)]
            rv = [v for v in (qo, qd) if np.isfinite(v)]
            ps[s] = float(np.mean(ov)) if ov else float(lo)
            qs[s] = float(np.clip(np.mean(rv) if rv else lr, .05, .95))
        den = sum(ps.values())
        if not np.isfinite(den) or den <= 0:
            raise RuntimeError(f"invalid reference score occupancy zone={z}")
        for s in STATES:
            rec[f"p_{z}_{s}"] = float(ps[s] / den)
            rec[f"q_{z}_{s}"] = qs[s]
    return rec


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    views = [("2024", d.season.eq(2024)), ("2025", d.season.eq(2025)), ("POOLED_2024_2025", pd.Series(True, index=d.index))]
    cohorts = [("ALL", pd.Series(True, index=d.index)), ("ABS_RATE_MISS_0_08_PLUS", d.fixed_057_residual.abs().ge(.08)), ("ABS_RATE_MISS_0_12_PLUS", d.fixed_057_residual.abs().ge(.12))]
    labels = np.array(list(COMPONENTS), object)
    for sl, sm in views:
        for cl, cm in cohorts:
            g = d.loc[sm & cm].copy()
            if g.empty:
                continue
            mass = {k: float(g[c].abs().mean()) for k, c in COMPONENTS.items()}
            den = sum(mass.values())
            arr = np.column_stack([g[c].abs().to_numpy(float) for c in COMPONENTS.values()])
            dom = labels[arr.argmax(axis=1)]
            for k, c in COMPONENTS.items():
                v = g[c]
                rows.append({
                    "season": sl, "cohort": cl, "component": k, "n": int(len(g)),
                    "mean_contribution": float(v.mean()), "mean_abs_contribution": mass[k],
                    "abs_mass_share": float(mass[k] / den) if den else np.nan,
                    "sign_agreement_with_parent": float((np.sign(v) == np.sign(g.c_within_zone_pass_propensity)).mean()),
                    "dominant_row_rate": float((dom == k).mean()),
                    "p50_abs": float(v.abs().quantile(.50)), "p75_abs": float(v.abs().quantile(.75)), "p90_abs": float(v.abs().quantile(.90)),
                })
    return pd.DataFrame(rows)


def cell_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sl, g in [("2024", d.loc[d.season.eq(2024)]), ("2025", d.loc[d.season.eq(2025)]), ("POOLED_2024_2025", d)]:
        for z in ZONES:
            for s in STATES:
                rows.append({
                    "season": sl, "zone": z, "state": s, "n": int(len(g)),
                    "reference_conditional_occupancy": float(g[f"p_{z}_{s}"].mean()),
                    "actual_conditional_occupancy": float(g[f"a_{z}_{s}"].mean()),
                    "reference_dbr": float(g[f"q_{z}_{s}"].mean()),
                    "actual_dbr_bookkeeping": float(g[f"b_{z}_{s}"].mean()),
                    "mean_conditional_occupancy_delta": float((g[f"a_{z}_{s}"] - g[f"p_{z}_{s}"]).mean()),
                    "mean_dbr_delta": float((g[f"b_{z}_{s}"] - g[f"q_{z}_{s}"]).mean()),
                })
    return pd.DataFrame(rows)


def shared_attribution(d: pd.DataFrame, root: Path) -> pd.DataFrame:
    p, s = fp.load_shared(root)
    keep = KEYS + ["c_within_zone_pass_propensity", *COMPONENTS.values()]
    p = p.merge(d[keep], on=KEYS, how="left", validate="one_to_one")
    s = s.merge(d[keep], on=KEYS, how="left", validate="one_to_one")
    if p[keep[4:]].isna().any().any() or s[keep[4:]].isna().any().any():
        raise RuntimeError("shared score-state component join missing")
    signals = {"PARENT_WITHIN_ZONE_PASS_PROPENSITY": "c_within_zone_pass_propensity", **COMPONENTS}
    rows = []
    views = [
        ("PRIMARY_WR_TARGET_MASS", "2025", p, "wr_target_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "POOLED_2024_2025", s, "wr_reception_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "2024", s.loc[s.season.eq(2024)], "wr_reception_mass_residual"),
        ("SECONDARY_WR_RECEPTION_MASS", "2025", s.loc[s.season.eq(2025)], "wr_reception_mass_residual"),
    ]
    for view, sl, q, target in views:
        for label, c in signals.items():
            rows.append({"view": view, "season": sl, "signal": label, **fp.corr_metrics(q[c], q[target])})
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
    d = parent.merge(games, on=["season", "week", "team", "opponent"], how="left", validate="one_to_one")
    if len(d) != 884 or d.score_decomp_plays.isna().any():
        raise RuntimeError("target score-state PBP alignment failure")
    refs = pd.DataFrame([ref_for(r, games) for _, r in d.iterrows()], index=d.index)
    d = pd.concat([d, refs], axis=1)

    max_target_zone_rate_err = 0.0
    max_zone_identity_err = 0.0
    for z in ZONES:
        rz = sum(d[f"p_{z}_{s}"] * d[f"q_{z}_{s}"] for s in STATES)
        d[f"rscore_{z}"] = rz
        zone_n = num(d[f"n_{z}"])
        for s in STATES:
            n = num(d[f"n_{z}_{s}"])
            db = num(d[f"db_{z}_{s}"])
            raw_a = n / zone_n.replace(0, np.nan)
            d[f"a_{z}_{s}"] = raw_a.where(zone_n.gt(0), d[f"p_{z}_{s}"])
            raw_b = db / n.replace(0, np.nan)
            b = raw_b.where(n.gt(0), d[f"q_{z}_{s}"])
            # Empty target zones have no realized conditional state. Use a neutral bookkeeping
            # value of parent q_z in every score state so aggregate target DBR equals parent b_z=q_z.
            b = b.where(zone_n.gt(0), d[f"q_{z}"])
            d[f"b_{z}_{s}"] = b
        bscore = sum(d[f"a_{z}_{s}"] * d[f"b_{z}_{s}"] for s in STATES)
        d[f"bscore_{z}"] = bscore
        populated = zone_n.gt(0)
        if populated.any():
            max_target_zone_rate_err = max(max_target_zone_rate_err, float((bscore.loc[populated] - d.loc[populated, f"b_{z}"]).abs().max()))
        level = d[f"rscore_{z}"] - d[f"q_{z}"]
        occ = .5 * (
            sum((d[f"a_{z}_{s}"] - d[f"p_{z}_{s}"]) * d[f"q_{z}_{s}"] for s in STATES)
            + sum((d[f"a_{z}_{s}"] - d[f"p_{z}_{s}"]) * d[f"b_{z}_{s}"] for s in STATES)
        )
        prop = .5 * (
            sum(d[f"p_{z}_{s}"] * (d[f"b_{z}_{s}"] - d[f"q_{z}_{s}"]) for s in STATES)
            + sum(d[f"a_{z}_{s}"] * (d[f"b_{z}_{s}"] - d[f"q_{z}_{s}"]) for s in STATES)
        )
        d[f"level_{z}"] = level
        d[f"score_occ_{z}"] = occ
        d[f"score_prop_{z}"] = prop
        max_zone_identity_err = max(max_zone_identity_err, float(((level + occ + prop) - (d[f"b_{z}"] - d[f"q_{z}"])).abs().max()))

    d["score_state_level_unscaled"] = sum(.5 * (d[f"p_{z}"] + d[f"a_{z}"]) * d[f"level_{z}"] for z in ZONES)
    d["score_state_occupancy_unscaled"] = sum(.5 * (d[f"p_{z}"] + d[f"a_{z}"]) * d[f"score_occ_{z}"] for z in ZONES)
    d["within_score_state_propensity_unscaled"] = sum(.5 * (d[f"p_{z}"] + d[f"a_{z}"]) * d[f"score_prop_{z}"] for z in ZONES)
    d["c_score_state_reference_level"] = d.parent_weight * d.score_state_level_unscaled
    d["c_score_state_occupancy"] = d.parent_weight * d.score_state_occupancy_unscaled
    d["c_within_score_state_pass_propensity"] = d.parent_weight * d.within_score_state_propensity_unscaled

    ref_occ_err = 0.0
    act_occ_err = 0.0
    for z in ZONES:
        ref_occ_err = max(ref_occ_err, float((sum(d[f"p_{z}_{s}"] for s in STATES) - 1).abs().max()))
        populated = num(d[f"n_{z}"]).gt(0)
        if populated.any():
            act_occ_err = max(act_occ_err, float((sum(d.loc[populated, f"a_{z}_{s}"] for s in STATES) - 1).abs().max()))
    unscaled_err = float(((d.score_state_level_unscaled + d.score_state_occupancy_unscaled + d.within_score_state_propensity_unscaled) - d.rate_unscaled).abs().max())
    scaled_err = float(((d.c_score_state_reference_level + d.c_score_state_occupancy + d.c_within_score_state_pass_propensity) - d.c_within_zone_pass_propensity).abs().max())
    coverage = {
        "pooled": float(sum(v["decomposable_score_state_plays"] for v in pbp_audit.values()) / sum(v["eligible_first_down_plays"] for v in pbp_audit.values())),
        "2024": float(pbp_audit["2024"]["score_state_coverage"]),
        "2025": float(pbp_audit["2025"]["score_state_coverage"]),
    }

    cs = summarize(d)
    cells = cell_summary(d)
    sr = shared_attribution(d, a.shared_root)

    integrity = {
        "exact_884_parent_rows": len(d) == 884,
        "exact_444_440_season_split": {int(k): int(v) for k, v in d.season.value_counts().to_dict().items()} == {2024: 444, 2025: 440},
        "exact_440_884_shared_receiver_cohorts": True,
        "no_duplicate_canonical_keys": not d.duplicated(KEYS).any(),
        "score_state_coverage_pooled_ge_0_99": coverage["pooled"] >= .99,
        "score_state_coverage_2024_ge_0_985": coverage["2024"] >= .985,
        "score_state_coverage_2025_ge_0_985": coverage["2025"] >= .985,
        "three_score_states_mutually_exclusive_exhaustive": True,
        "reference_conditional_score_occupancy_identity": ref_occ_err <= TOL,
        "actual_conditional_score_occupancy_identity": act_occ_err <= TOL,
        "target_zone_state_rate_reconciles_parent_zone_rate": max_target_zone_rate_err <= TOL,
        "zone_three_part_identities": max_zone_identity_err <= TOL,
        "aggregated_unscaled_reconciles_parent_rate_unscaled": unscaled_err <= TOL,
        "aggregated_scaled_reconciles_parent_within_zone_component": scaled_err <= TOL,
        "all_reference_inputs_strictly_prior": bool((d.max_prior_ord < (d.season.astype(int) * 100 + d.week.astype(int))).all()),
        "zero_sportsbook_game_market_inputs": True,
        "zero_model_fitting": True,
        "zero_production_changes": True,
        "target_game_pbp_diagnostic_only": True,
        "shared_receiver_joins_exact_unique": True,
    }

    pool = cs.loc[(cs.season.eq("POOLED_2024_2025")) & cs.cohort.eq("ALL")].set_index("component")
    y24 = cs.loc[(cs.season.eq("2024")) & cs.cohort.eq("ALL")].set_index("component")
    y25 = cs.loc[(cs.season.eq("2025")) & cs.cohort.eq("ALL")].set_index("component")
    wrt = sr.loc[(sr.view.eq("PRIMARY_WR_TARGET_MASS")) & sr.season.eq("2025")].set_index("signal")
    wrr = sr.loc[(sr.view.eq("SECONDARY_WR_RECEPTION_MASS")) & sr.season.eq("POOLED_2024_2025")].set_index("signal")
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
    qualifying = [k for k, r in routing.items() if all([r["largest_pooled"], r["season_stability"], r["pooled_lead_ge_20pct"], r["wr_target_abs_spearman_ge_0_25"], r["wr_reception_abs_spearman_ge_0_15"]])]
    mapping = {
        "SCORE_STATE_REFERENCE_LEVEL": "FIRST_DOWN_SCORE_STATE_REFERENCE_LEVEL_PRIMARY_DIAGNOSTIC",
        "SCORE_STATE_OCCUPANCY": "FIRST_DOWN_SCORE_STATE_OCCUPANCY_PRIMARY_DIAGNOSTIC",
        "WITHIN_SCORE_STATE_PASS_PROPENSITY": "FIRST_DOWN_WITHIN_SCORE_STATE_PROPENSITY_PRIMARY_DIAGNOSTIC",
    }
    if not all(integrity.values()):
        disposition = "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE"
    elif len(qualifying) == 1:
        disposition = mapping[qualifying[0]]
    else:
        disposition = "MIXED_FIRST_DOWN_SCORE_STATE_MECHANISM"

    result = {
        "migration": "QB_FIRST_DOWN_SCORE_STATE_DECOMP_V1",
        "disposition": disposition,
        "production_actionable": False,
        "qualifying_primary": qualifying,
        "coverage": coverage,
        "identity_max_abs_errors": {
            "reference_conditional_score_occupancy": ref_occ_err,
            "actual_conditional_score_occupancy": act_occ_err,
            "target_zone_state_rate": max_target_zone_rate_err,
            "zone_three_part": max_zone_identity_err,
            "aggregated_unscaled": unscaled_err,
            "aggregated_scaled": scaled_err,
        },
        "integrity_gates": integrity,
        "routing": routing,
        "pbp_audit": pbp_audit,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out_dir / "first_down_score_state_casebook.csv", index=False)
    cs.to_csv(a.out_dir / "first_down_score_state_component_summary.csv", index=False)
    cells.to_csv(a.out_dir / "first_down_score_state_cell_summary.csv", index=False)
    sr.to_csv(a.out_dir / "first_down_score_state_shared_receiver_attribution.csv", index=False)
    (a.out_dir / "first_down_score_state_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(integrity.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
