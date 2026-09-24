#!/usr/bin/env python3
"""Blind-season TE-R5P receiving-yard distribution-width validation.

Research only. Reconstructs the exact fold-safe production-order TE-R5P
specialist distributions, estimates one football-only width factor on one
historical season, and applies it unchanged to the other season. The football
mean is invariant by construction. Sportsbook lines are secondary evaluation
only and never enter the fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import ev_roi, implied_prob, no_vig, signal
from scripts.backtest.grade_historical_market_vegas_benchmark_v1 import select_one_book_row
from scripts.operations.grade_market_track_record_v1 import american_profit, num, outcome_side
from scripts.research.grade_empirical_fair_prob_v1 import KEYS, _canon_keys, empirical_over_probability, rescale_outcomes

MARKET = "rec_yards"
POSITION = "TE"
FIT_TEST_DIRECTIONS = ((2024, 2025), (2025, 2024))


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _load_meta(distribution_dir: Path) -> pd.DataFrame:
    paths = sorted(distribution_dir.glob("*_metadata.csv"))
    if not paths:
        raise RuntimeError(f"no metadata files in {distribution_dir}")
    meta = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    meta = _canon_keys(meta)
    meta = meta.loc[meta["market"].eq(MARKET)].copy()
    if meta.duplicated(KEYS).any():
        raise RuntimeError("duplicate TE rec_yards distribution metadata identity")
    return meta


def _row_arrays(meta: pd.DataFrame, distribution_dir: Path) -> dict:
    cache: dict[str, object] = {}
    out: dict[tuple, np.ndarray] = {}
    for _, row in meta.iterrows():
        fn = str(row["npz_file"])
        if Path(fn).name != fn:
            raise RuntimeError(f"invalid shard path: {fn}")
        if fn not in cache:
            path = distribution_dir / fn
            if not path.exists():
                raise RuntimeError(f"missing distribution shard: {path}")
            cache[fn] = np.load(path, allow_pickle=False)
        arr = np.asarray(cache[fn][str(row["array_key"])], dtype=float)
        if len(arr) != int(row["draws"]):
            raise RuntimeError("distribution draw-count drift")
        if len(arr) != 2000:
            raise RuntimeError(f"unexpected TE replay draw count: {len(arr)}")
        if not np.isfinite(arr).all():
            raise RuntimeError("non-finite distribution draw")
        out[tuple(row[c] for c in KEYS)] = arr
    return out


def empirical_crps(draws: np.ndarray, actual: float) -> float:
    x = np.sort(np.asarray(draws, dtype=float))
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(actual):
        return np.nan
    n = len(x)
    first = float(np.mean(np.abs(x - float(actual))))
    i = np.arange(1, n + 1, dtype=float)
    second = float(np.sum((2.0 * i - n - 1.0) * x) / (n * n))
    return first - second


def _aligned(arr: np.ndarray, mean: float) -> np.ndarray:
    out = rescale_outcomes(arr, float(mean))
    delta = abs(float(np.mean(out)) - float(mean))
    if not np.isfinite(delta) or delta > 1e-8:
        raise RuntimeError(f"mean alignment failed: delta={delta}")
    return out


def _widen(arr: np.ndarray, mean: float, k: float) -> np.ndarray:
    out = float(mean) + (arr - float(mean)) * float(k)
    out = out + (float(mean) - float(np.mean(out)))
    if abs(float(np.mean(out)) - float(mean)) > 1e-8:
        raise RuntimeError("widened mean drift")
    return out


def _prepare_projection(projection_file: Path) -> pd.DataFrame:
    p = _read(projection_file, "specialist projection")
    if "ensemble_proj" not in p.columns:
        raise RuntimeError("specialist projection missing ensemble_proj")
    p["proj"] = num(p["ensemble_proj"]); p["actual"] = num(p["actual"])
    p["season"] = pd.to_numeric(p["season"], errors="coerce"); p["week"] = pd.to_numeric(p["week"], errors="coerce")
    p["position"] = p["position"].astype(str).str.upper().str.strip()
    p = p.loc[p["position"].eq(POSITION) & p["market"].astype(str).eq(MARKET) & p["season"].isin([2024, 2025]) & p["proj"].notna() & p["actual"].notna()].copy()
    if p.empty or p.duplicated(KEYS).any():
        raise RuntimeError("invalid TE rec_yards projection identity")
    return _canon_keys(p)


def fit_k(proj, meta, arrays, fit_season):
    q = proj.loc[proj["season"].eq(fit_season)].merge(meta[KEYS+["array_key","npz_file","draws"]], on=KEYS, how="inner", validate="one_to_one")
    expected = int(proj["season"].eq(fit_season).sum())
    if len(q) != expected or not len(q): raise RuntimeError(f"fit distribution coverage mismatch: {len(q)} != {expected}")
    sds=[]; residuals=[]
    for _,r in q.iterrows():
        a=_aligned(arrays[tuple(r[c] for c in KEYS)], float(r["proj"])); sds.append(float(np.std(a,ddof=1))); residuals.append(float(r["actual"])-float(r["proj"]))
    mean_sd=float(np.mean(sds)); residual_sd=float(np.std(residuals,ddof=1)); k=residual_sd/mean_sd
    return {"fit_season":int(fit_season),"n":int(len(q)),"mean_row_mc_sd":mean_sd,"residual_sd":residual_sd,"k":float(k)}


def evaluate_season(proj, meta, arrays, *, test_season, k):
    q=proj.loc[proj["season"].eq(test_season)].merge(meta[KEYS+["array_key","npz_file","draws"]],on=KEYS,how="inner",validate="one_to_one")
    expected=int(proj["season"].eq(test_season).sum())
    if len(q)!=expected or not len(q): raise RuntimeError(f"test distribution coverage mismatch: {len(q)} != {expected}")
    rows=[]
    for _,r in q.iterrows():
        key=tuple(r[c] for c in KEYS); base=_aligned(arrays[key],float(r["proj"])); wide=_widen(base,float(r["proj"]),float(k)); actual=float(r["actual"])
        b05,b10,b90,b95=np.quantile(base,[.05,.10,.90,.95]); w05,w10,w90,w95=np.quantile(wide,[.05,.10,.90,.95])
        rows.append({**{c:r[c] for c in KEYS},"position":POSITION,"proj":float(r["proj"]),"actual":actual,"k":float(k),"base_sd":float(np.std(base,ddof=1)),"wide_sd":float(np.std(wide,ddof=1)),"base_crps":empirical_crps(base,actual),"wide_crps":empirical_crps(wide,actual),"base_cover80":bool(b10<=actual<=b90),"wide_cover80":bool(w10<=actual<=w90),"base_cover90":bool(b05<=actual<=b95),"wide_cover90":bool(w05<=actual<=w95),"base_width80":float(b90-b10),"wide_width80":float(w90-w10),"base_width90":float(b95-b05),"wide_width90":float(w95-w05),"base_mean":float(np.mean(base)),"wide_mean":float(np.mean(wide))})
    d=pd.DataFrame(rows); b80=float(d.base_cover80.mean()); w80=float(d.wide_cover80.mean()); b90=float(d.base_cover90.mean()); w90=float(d.wide_cover90.mean())
    s={"test_season":int(test_season),"n":int(len(d)),"k":float(k),"point_mae_base":float((d.proj-d.actual).abs().mean()),"point_mae_wide":float((d.proj-d.actual).abs().mean()),"max_abs_mean_shift":float((d.wide_mean-d.base_mean).abs().max()),"crps_base":float(d.base_crps.mean()),"crps_wide":float(d.wide_crps.mean()),"crps_improvement_pct":float((d.base_crps.mean()-d.wide_crps.mean())/d.base_crps.mean()),"coverage80_base":b80,"coverage80_wide":w80,"coverage80_gap_base":abs(b80-.8),"coverage80_gap_wide":abs(w80-.8),"coverage90_base":b90,"coverage90_wide":w90,"coverage90_gap_base":abs(b90-.9),"coverage90_gap_wide":abs(w90-.9),"width80_base":float(d.base_width80.mean()),"width80_wide":float(d.wide_width80.mean()),"width90_base":float(d.base_width90.mean()),"width90_wide":float(d.wide_width90.mean())}
    return d,s


def _secondary_market_eval(detail, meta, arrays, props, *, test_season, k):
    p=props.copy(); p["season"]=pd.to_numeric(p["season"],errors="coerce"); p=p.loc[p.season.eq(test_season)&p.market.astype(str).eq(MARKET)].copy(); selected=select_one_book_row(p)
    keep=["game_id","player_clean_key","market","book","line","over_odds","under_odds","player"]
    q=detail.merge(selected[keep],on=["game_id","player_clean_key","market"],how="inner")
    if q.empty: raise RuntimeError(f"no historical market matches for TE rec_yards season {test_season}")
    q["line"]=num(q.line); pb=[]; pw=[]
    for _,r in q.iterrows():
        key=tuple(r[c] for c in KEYS); base=_aligned(arrays[key],float(r.proj)); wide=_widen(base,float(r.proj),float(k)); pb.append(empirical_over_probability(base,float(r.line))); pw.append(empirical_over_probability(wide,float(r.line)))
    outputs=[]; summaries={}
    for label,p_over in (("base",pb),("wide",pw)):
        z=q.copy(); z["p_over"]=p_over; z["p_under"]=1-z.p_over; y=(num(z.actual)>num(z.line)).astype(float); pp=np.clip(num(z.p_over),1e-6,1-1e-6)
        z["over_implied"]=z.over_odds.map(implied_prob); z["under_implied"]=z.under_odds.map(implied_prob); z["over_novig"]=[no_vig(a,b) for a,b in zip(z.over_implied,z.under_implied)]; z["under_novig"]=[no_vig(a,b) for a,b in zip(z.under_implied,z.over_implied)]; z["ev_over"]=[ev_roi(p,o) for p,o in zip(z.p_over,z.over_odds)]; z["ev_under"]=[ev_roi(p,o) for p,o in zip(z.p_under,z.under_odds)]
        bo=z.ev_under.isna()|(z.ev_over.fillna(-np.inf)>=z.ev_under.fillna(-np.inf)); z["side"]=np.where(bo,"OVER","UNDER"); z["best_ev"]=np.where(bo,z.ev_over,z.ev_under); z["best_model_p"]=np.where(bo,z.p_over,z.p_under); z["best_market_p"]=np.where(bo,z.over_novig,z.under_novig); z["prob_edge"]=z.best_model_p-z.best_market_p; z["chosen_odds"]=np.where(bo,z.over_odds,z.under_odds); z["signal"]=[signal(e,pe) for e,pe in zip(z.best_ev,z.prob_edge)]; z["actual_side"]=[outcome_side(a,l) for a,l in zip(z.actual,z.line)]; z["bet_result"]=np.select([z.actual_side.eq("PUSH"),z.side.eq(z.actual_side)],["PUSH","WIN"],default="LOSS"); z["unit_result"]=np.where(z.bet_result.eq("WIN"),[american_profit(o) for o in z.chosen_odds],np.where(z.bet_result.eq("LOSS"),-1.,0.))
        strong=z.loc[z.signal.eq("STRONG_EDGE")&z.bet_result.isin(["WIN","LOSS"])]
        summaries[label]={"rows":int(len(z)),"brier":float(np.mean((pp-y)**2)),"log_loss":float(-np.mean(y*np.log(pp)+(1-y)*np.log(1-pp))),"strong_rows":int(len(strong)),"strong_win_rate":float(strong.bet_result.eq("WIN").mean()) if len(strong) else np.nan,"strong_roi":float(strong.unit_result.mean()) if len(strong) else np.nan}; z["variant"]=label; outputs.append(z)
    return pd.concat(outputs,ignore_index=True),summaries


def run(projection_file, distribution_dir, props_file, out_dir):
    proj=_prepare_projection(projection_file)
    meta=_load_meta(distribution_dir); meta=meta.loc[meta.season.isin([2024,2025])].copy()
    # Mechanical scope fix: distribution metadata contains WR and TE rec_yards rows.
    # Width V2 is TE-only, so semi-join to the frozen TE projection cohort BEFORE
    # opening NPZ shards. This changes no row, fit, gate, or hypothesis; it only
    # prevents loading irrelevant WR arrays into memory.
    meta=meta.merge(proj[KEYS].drop_duplicates(),on=KEYS,how="inner",validate="one_to_one")
    if len(meta)!=len(proj): raise RuntimeError(f"TE metadata scope mismatch: {len(meta)} != {len(proj)}")
    arrays=_row_arrays(meta,distribution_dir); props=_read(props_file,"historical props")
    fits={s:fit_k(proj,meta,arrays,s) for s in (2024,2025)}; evaluations={}; market_eval={}; details=[]; markets=[]
    for fs,ts in FIT_TEST_DIRECTIONS:
        d,s=evaluate_season(proj,meta,arrays,test_season=ts,k=fits[fs]["k"]); d["fit_season"]=fs; d["test_season"]=ts; details.append(d); m,ms=_secondary_market_eval(d,meta,arrays,props,test_season=ts,k=fits[fs]["k"]); m["fit_season"]=fs; m["test_season"]=ts; markets.append(m); evaluations[f"fit{fs}_test{ts}"]=s; market_eval[f"fit{fs}_test{ts}"]=ms
    bb=[]; ww=[]
    for v in market_eval.values():
        for variant,bucket in (("base",bb),("wide",ww)): bucket.append((v[variant]["rows"],v[variant]["brier"],v[variant]["log_loss"]))
    weighted=lambda b,i:sum(x[0]*x[i] for x in b)/sum(x[0] for x in b)
    pooled={"matched_rows":int(sum(x[0] for x in bb)),"brier_base":float(weighted(bb,1)),"brier_wide":float(weighted(ww,1)),"log_loss_base":float(weighted(bb,2)),"log_loss_wide":float(weighted(ww,2))}
    gates={"point_mae_invariant_both_directions":all(abs(v["point_mae_wide"]-v["point_mae_base"])<=1e-10 for v in evaluations.values()),"mean_shift_le_1e_8_both_directions":all(v["max_abs_mean_shift"]<=1e-8 for v in evaluations.values()),"crps_strict_improve_both_directions":all(v["crps_wide"]<v["crps_base"] for v in evaluations.values()),"coverage80_gap_improve_both_directions":all(v["coverage80_gap_wide"]<v["coverage80_gap_base"] for v in evaluations.values()),"coverage90_gap_improve_both_directions":all(v["coverage90_gap_wide"]<v["coverage90_gap_base"] for v in evaluations.values()),"pooled_brier_nonworse":pooled["brier_wide"]<=pooled["brier_base"]+1e-12,"pooled_log_loss_nonworse":pooled["log_loss_wide"]<=pooled["log_loss_base"]+1e-12,"sportsbook_inputs_used_to_fit_k":0}; gates["qualified"]=bool(all(v for k,v in gates.items() if k!="sportsbook_inputs_used_to_fit_k"))
    sds=[]; residuals=[]
    for _,r in proj.iterrows(): a=_aligned(arrays[tuple(r[c] for c in KEYS)],float(r.proj)); sds.append(float(np.std(a,ddof=1))); residuals.append(float(r.actual)-float(r.proj))
    future_k=float(np.std(residuals,ddof=1)/np.mean(sds)); result={"study":"TE_R5P_REC_YARDS_WIDTH_V2","status":"research_only","production_changed":False,"sportsbook_inputs_used_to_fit_k":0,"fit":fits,"blind_evaluations":evaluations,"secondary_market_calibration":market_eval,"pooled_secondary_market_calibration":pooled,"gates":gates,"future_k_if_qualified":future_k,"disposition":"TE_R5P_REC_YARDS_WIDTH_V2_QUALIFIED" if gates["qualified"] else "TE_R5P_REC_YARDS_WIDTH_V2_FAILED_CLOSED"}
    out_dir.mkdir(parents=True,exist_ok=True); pd.concat(details,ignore_index=True).to_csv(out_dir/"blind_row_detail.csv",index=False); pd.concat(markets,ignore_index=True).to_csv(out_dir/"blind_market_detail.csv",index=False); pd.DataFrame(fits.values()).to_csv(out_dir/"fit_factors.csv",index=False); (out_dir/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# TE-R5P Receiving-Yards Width V2 — Result","",f"**Disposition:** `{result['disposition']}`","","Research only. No production change.","","## Fit factors",""]
    for s in (2024,2025): x=fits[s]; lines.append(f"- {s}: n={x['n']}, mean MC SD={x['mean_row_mc_sd']:.3f}, residual SD={x['residual_sd']:.3f}, k=**{x['k']:.4f}**")
    lines += ["","## Blind directions",""]
    for key,v in evaluations.items(): lines += [f"### {key}",f"- n: **{v['n']}**",f"- CRPS: **{v['crps_base']:.4f} -> {v['crps_wide']:.4f}** ({v['crps_improvement_pct']*100:+.2f}%)",f"- 80% coverage: **{v['coverage80_base']:.3f} -> {v['coverage80_wide']:.3f}** (gap {v['coverage80_gap_base']:.3f} -> {v['coverage80_gap_wide']:.3f})",f"- 90% coverage: **{v['coverage90_base']:.3f} -> {v['coverage90_wide']:.3f}** (gap {v['coverage90_gap_base']:.3f} -> {v['coverage90_gap_wide']:.3f})",f"- point MAE invariant: **{v['point_mae_base']:.4f}**",f"- max mean shift: **{v['max_abs_mean_shift']:.3g}**",""]
    lines += ["## Secondary historical-line calibration","",f"- pooled Brier: **{pooled['brier_base']:.5f} -> {pooled['brier_wide']:.5f}**",f"- pooled log loss: **{pooled['log_loss_base']:.5f} -> {pooled['log_loss_wide']:.5f}**","","## Frozen gates",""]+[f"- {k}: **{v}**" for k,v in gates.items()]+["",f"Predeclared future-only pooled factor if qualified: **{future_k:.4f}**.","","No live 2026 outcome was used to fit k. Any production integration requires a separate validation."]
    (out_dir/"RESULT.md").write_text("\n".join(lines)+"\n"); print(json.dumps(result,indent=2,sort_keys=True)); return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--projection-file",type=Path,required=True); ap.add_argument("--distribution-dir",type=Path,required=True); ap.add_argument("--props",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); run(a.projection_file,a.distribution_dir,a.props,a.out_dir); return 0

if __name__=="__main__": raise SystemExit(main())
