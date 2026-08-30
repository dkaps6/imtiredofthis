#!/usr/bin/env python3
"""Migration 80: canonical-v3 QB research-frontier batch audit.

M80 is diagnostic/source-contract only. It fits ZERO predictive models and does
not select a winner on 2025 outcomes. The purpose is to (1) re-baseline current
canonical-v3 catastrophic QB misses, (2) expand the no-retest ledger to M1-M79,
and (3) audit genuinely new information sources independently before any M81
predictive development screen.

Jobs are intentionally isolated:
  tail     - canonical-v3 failure re-baseline + M1-M79 no-retest crosswalk
  ftn      - FTN charting source/field audit; novel-only fields vs duplicates
  route    - route x coverage-shell historical feasibility / live-2026 constraint
  blocker  - true blocker x rusher acquisition frontier audit
  master   - combine job summaries into one advance/defer/reject table

No sportsbook fields. No predictive model fitting. No threshold/model search.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

EXPECTED_CANONICAL_SHA256 = "c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
EXPECTED_ROWS = 884
EXPECTED_SEASONS = {2024: 444, 2025: 440}
MARKET_TOKENS = ("market", "spread", "moneyline", "sportsbook", "implied_total", "game_total", "prop_line")

FTN_DUPLICATE_OR_CLOSED = {
    "qb_location": "formation/shotgun-family already explored M67/M68",
    "n_offense_backfield": "generic formation/personnel tendency already explored M67",
    "is_no_huddle": "no-huddle/opening/tendency already explored M67/M68",
    "is_play_action": "generic play-action tendency already explored M67/M68/M70",
    "n_pass_rushers": "generic pass-rush/pressure-count family overlaps M45/M56/M69",
}
FTN_NOVEL_CANDIDATES = {
    "is_motion": "pre-snap motion response, not generic formation rate",
    "is_screen_pass": "screen-specific tactical response",
    "is_rpo": "RPO-specific tactical response",
    "is_trick_play": "rare tactical tag; audit only",
    "is_qb_out_of_pocket": "QB out-of-pocket response",
    "is_interception_worthy": "decision-quality mechanism",
    "is_throw_away": "pressure-response/drive-ending mechanism distinct from generic pressure rate",
    "read_thrown": "read-progression distribution",
    "is_catchable_ball": "throw quality/catchability mechanism",
    "is_contested_ball": "contested-target environment",
    "is_created_reception": "receiver-created completion mechanism",
    "is_drop": "receiver drop mechanism",
    "n_blitzers": "explicit blitz construction/count distinct from generic pressure rate",
    "is_qb_fault_sack": "QB-attributable sack mechanism",
}

NO_RETEST_ROWS = [
    ("QB historical YPA / shrinkage", "M4A,M39+", "USED/EXHAUSTED", "QB YPA already foundational; do not repackage rolling YPA"),
    ("generic supervised ML on historical performance", "M4B,M61,M66+", "PROHIBITED", "new algorithm on same information universe is not a new migration"),
    ("QB state/regime transition", "M4C", "USED/EXHAUSTED", "state model already implemented"),
    ("ensemble/model remix", "M4D,M61,M66", "PROHIBITED", "same information + new blend is not new information"),
    ("worst-miss/catastrophic classifier", "M7,M62,M69,M70", "DO_NOT_REPEAT", "M80 may only re-baseline on corrected canonical-v3, not refit old mechanisms"),
    ("primary QB / role / opportunity identity", "M8,M47", "USED/EXHAUSTED", "starter and attempt-share identity already repaired"),
    ("dropback to official-attempt conversion", "M9", "CANONICAL", "structural repair already in foundation"),
    ("pace to team play volume", "M10", "CANONICAL", "structural repair already in foundation"),
    ("generic injuries", "M11,M67,M77-M79", "DO_NOT_RETEST", "exact inactives also failed cleanly in M79"),
    ("weather", "M12", "DEFER", "pregame context exists but low-priority modifier, not frontier hypothesis"),
    ("box counts", "M13,M14", "DO_NOT_RETEST", "historical box context already recovered"),
    ("man/zone + coverage-shell frequencies", "M14,M72,M75", "DO_NOT_RETEST_AS_STANDALONE", "Cover 0/1/2/3/4/6/9 and 2-Man already recovered"),
    ("WR-CB exact responsibility from participation", "M15", "NO_GO_SOURCE", "on-field defenders != assignment; do not infer fake matchups"),
    ("PROE / score-state / lead-trail pass tendency", "M16-M21,M64-M65", "DO_NOT_RETEST", "fixed architecture beat dynamic script variants"),
    ("generic team pass-rate history", "M17-M21,M40-M42", "DO_NOT_RETEST", "attempt differentiation remained weak"),
    ("generic pressure mismatch / pressure rate", "M16,M22-M23,M45,M56,M69,M72", "DO_NOT_RETEST", "gentle rule already promoted; richer aggregate families failed"),
    ("receiving target-pool breadth", "M31-M35", "DO_NOT_RETEST_FOR_QB", "allocation work already completed"),
    ("WR hierarchy", "M36-M38", "CANONICAL", "validated hierarchy already promoted"),
    ("attempt/YPA oracle decomposition", "M39,M53+,M64+,M73+", "DIAGNOSTIC_ONLY", "headroom known; oracle is not a pregame predictor"),
    ("raw/decompression correction", "M57-M58", "REJECTED", "some MAE/corr gain but unacceptable catastrophic/collateral behavior"),
    ("same-feature catastrophic classifier", "M62", "REJECTED", "failed frozen prospective gates"),
    ("high/low attempt surprise classifier", "M63", "REJECTED", "low side partial only; high-volume surprise unexplained"),
    ("generative possessions/drives attempts", "M64-M65", "REJECTED", "DBR/state occupancy failed prospectively"),
    ("opening script / playcaller", "M67-M68", "DO_NOT_RETEST", "weak correlations did not translate to actionable model gain"),
    ("QB efficiency volatility/risk", "M70-M71", "DO_NOT_RETEST", "negative as predictive family"),
    ("receiver explosive profile x aggregate defense", "M72", "REJECTED", "no replicated matchup bridge"),
    ("NGS separation/cushion/aDOT/YACOE x secondary", "M75", "REJECTED", "negative on corrected canonical-v3"),
    ("exact depth-chart/personnel discontinuity", "M77", "REJECTED", "2025 and 2024 worsened under frozen test"),
    ("official inactive identity", "M78-M79", "REJECTED", "clean source qualified; predictive test worsened MAE/components"),
    ("FTN novel-only tactical fields", "none M1-M79", "NOVEL", "source itself not previously ingested; duplicate fields must be excluded"),
    ("route-concept x coverage-shell compatibility", "none M1-M79", "NOVEL_WITH_SOURCE_CONSTRAINT", "shells exist from M14; route interaction not modeled"),
    ("true blocker x true rusher assignment", "none M1-M79", "NOVEL_WITH_SOURCE_CONSTRAINT", "generic pressure/OL continuity are not individual assignments"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_url_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "m80-qb-frontier-audit"})
    with urlopen(req, timeout=120) as r:
        raw = r.read(); final = r.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {"url": final, "bytes": len(raw), "sha256": sha256_bytes(raw)}


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c).strip().lower() for c in x.columns]; return x


def require_canonical(path: Path) -> pd.DataFrame:
    digest = sha256_bytes(path.read_bytes())
    if digest != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical-v3 SHA drift: {digest}")
    x = lower(pd.read_csv(path, low_memory=False))
    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"canonical row drift: {len(x)}")
    counts = {int(k): int(v) for k, v in pd.to_numeric(x.season).value_counts().to_dict().items()}
    if counts != EXPECTED_SEASONS:
        raise RuntimeError(f"canonical season-count drift: {counts}")
    bad = [c for c in x.columns if any(t in c for t in MARKET_TOKENS)]
    if bad:
        raise RuntimeError(f"market boundary violated: {bad}")
    return x


def job_tail(canonical: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    x = require_canonical(canonical)
    for c in ["pred_attempts","actual_attempts","pred_pass_yards","actual_pass_yards"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["pred_ypa"] = x.pred_pass_yards / x.pred_attempts.replace(0, np.nan)
    x["actual_ypa"] = x.actual_pass_yards / x.actual_attempts.replace(0, np.nan)
    x["signed_error"] = x.pred_pass_yards - x.actual_pass_yards
    x["abs_error"] = x.signed_error.abs()
    x["attempt_error"] = x.pred_attempts - x.actual_attempts
    x["ypa_error"] = x.pred_ypa - x.actual_ypa
    x["oracle_attempts_pred"] = x.actual_attempts * x.pred_ypa
    x["oracle_ypa_pred"] = x.pred_attempts * x.actual_ypa
    x["gain_if_perfect_attempts"] = x.abs_error - (x.oracle_attempts_pred - x.actual_pass_yards).abs()
    x["gain_if_perfect_ypa"] = x.abs_error - (x.oracle_ypa_pred - x.actual_pass_yards).abs()
    diff = (x.gain_if_perfect_attempts - x.gain_if_perfect_ypa).abs()
    x["dominant_axis"] = np.where(diff < 10, "MIXED", np.where(x.gain_if_perfect_attempts > x.gain_if_perfect_ypa, "ATTEMPTS", "YPA"))
    x["direction"] = np.where(x.signed_error < 0, "UNDERPROJECTED", "OVERPROJECTED")
    tails = x.loc[x.abs_error >= 100].sort_values("abs_error", ascending=False).copy()
    worst41 = x.nlargest(41, "abs_error").copy()
    tails.to_csv(out / "m80_canonical_v3_tail100.csv", index=False)
    worst41.to_csv(out / "m80_canonical_v3_worst41.csv", index=False)
    summary = pd.DataFrame([
        {"metric":"canonical_rows","value":len(x)},
        {"metric":"tail100_games","value":len(tails)},
        {"metric":"worst41_share","value":41/len(x)},
        {"metric":"tail100_underprojected","value":int((tails.direction=="UNDERPROJECTED").sum())},
        {"metric":"tail100_overprojected","value":int((tails.direction=="OVERPROJECTED").sum())},
        {"metric":"tail100_attempt_dominant","value":int((tails.dominant_axis=="ATTEMPTS").sum())},
        {"metric":"tail100_ypa_dominant","value":int((tails.dominant_axis=="YPA").sum())},
        {"metric":"tail100_mixed","value":int((tails.dominant_axis=="MIXED").sum())},
        {"metric":"predictive_model_fit","value":False},
    ])
    summary.to_csv(out / "m80_tail_summary.csv", index=False)
    pd.DataFrame(NO_RETEST_ROWS, columns=["family","migrations","status","reason"]).to_csv(out / "m80_m1_m79_no_retest_crosswalk.csv", index=False)
    print(summary.to_string(index=False))


def field_coverage(df: pd.DataFrame, field: str) -> float:
    if field not in df.columns or len(df) == 0: return 0.0
    s = df[field]
    if pd.api.types.is_bool_dtype(s): return float(s.notna().mean())
    if pd.api.types.is_numeric_dtype(s): return float(pd.to_numeric(s, errors="coerce").notna().mean())
    z = s.astype("string").str.strip(); return float((z.notna() & z.ne("") & z.str.lower().ne("nan")).mean())


def job_ftn(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    inventory=[]; source=[]
    for season in [2022, 2023, 2024, 2025]:
        url=f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        try:
            df, meta = read_url_parquet(url); df=lower(df)
            weeks = sorted(pd.to_numeric(df.get("week"), errors="coerce").dropna().astype(int).unique().tolist()) if "week" in df else []
            source.append({"season":season,"rows":len(df),"weeks":len(weeks),"min_week":min(weeks) if weeks else np.nan,"max_week":max(weeks) if weeks else np.nan,**meta,"status":"OK"})
            for field, reason in {**FTN_DUPLICATE_OR_CLOSED, **FTN_NOVEL_CANDIDATES}.items():
                inventory.append({"season":season,"field":field,"coverage":field_coverage(df,field),"classification":"DUPLICATE_CLOSED" if field in FTN_DUPLICATE_OR_CLOSED else "NOVEL_CANDIDATE","reason":reason})
        except Exception as exc:
            source.append({"season":season,"rows":0,"weeks":0,"min_week":np.nan,"max_week":np.nan,"url":url,"bytes":0,"sha256":"","status":f"ERROR:{type(exc).__name__}:{exc}"})
    s=pd.DataFrame(source); inv=pd.DataFrame(inventory)
    s.to_csv(out/"m80_ftn_source_audit.csv",index=False); inv.to_csv(out/"m80_ftn_field_inventory.csv",index=False)
    ok_2425 = all(int(s.loc[s.season.eq(y),"rows"].max() or 0)>0 for y in [2024,2025]) if not s.empty else False
    novel_live = inv.loc[(inv.season==2025)&(inv.classification=="NOVEL_CANDIDATE")& (inv.coverage>=0.80),"field"].nunique() if not inv.empty else 0
    decision=pd.DataFrame([{"candidate":"FTN_NOVEL_ONLY","historical_2024_2025_available":bool(ok_2425),"novel_fields_2025_coverage_ge_80pct":int(novel_live),"in_season_update_contract":True,"predictive_model_fit":False,"advance_to_m81_development":bool(ok_2425 and novel_live>=5),"notes":"exclude all DUPLICATE_CLOSED fields; FTN charting is published after games and can only become strictly-prior history for later target games"}])
    decision.to_csv(out/"m80_ftn_decision.csv",index=False); print(decision.to_string(index=False))


def job_route(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for season in [2024,2025]:
        url=f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
        try:
            df,meta=read_url_parquet(url); df=lower(df)
            route=field_coverage(df,"route"); shell=field_coverage(df,"defense_coverage_type"); manzone=field_coverage(df,"defense_man_zone_type")
            both=float((df.get("route",pd.Series(index=df.index,dtype=object)).notna() & df.get("defense_coverage_type",pd.Series(index=df.index,dtype=object)).notna()).mean()) if len(df) else 0.0
            rows.append({"season":season,"rows":len(df),"route_coverage":route,"coverage_shell_coverage":shell,"man_zone_coverage":manzone,"route_and_shell_same_play":both,**meta,"status":"OK"})
        except Exception as exc:
            rows.append({"season":season,"rows":0,"route_coverage":0.0,"coverage_shell_coverage":0.0,"man_zone_coverage":0.0,"route_and_shell_same_play":0.0,"url":url,"bytes":0,"sha256":"","status":f"ERROR:{type(exc).__name__}:{exc}"})
    audit=pd.DataFrame(rows); audit.to_csv(out/"m80_route_shell_source_audit.csv",index=False)
    hist=bool(len(audit)==2 and (audit.rows>0).all() and (audit.route_and_shell_same_play>0.20).all())
    decision=pd.DataFrame([{"candidate":"ROUTE_X_COVERAGE_SHELL","historical_science_feasible":hist,"coverage_shell_itself_is_new":False,"route_interaction_is_new":True,"in_season_2026_source_available":False,"predictive_model_fit":False,"advance_to_m81_development":False,"status":"HOLD_FOR_DEPLOYABLE_LIVE_SOURCE" if hist else "NO_GO_SOURCE","notes":"M14 already recovered shells; nflverse 2023+ participation is postseason-only, so do not build an undeployable predictive winner unless an in-season route source is found"}])
    decision.to_csv(out/"m80_route_shell_decision.csv",index=False); print(decision.to_string(index=False))


def job_blocker(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Data-existence audit only. Big Data Bowl exposes exact assignment fields, but
    # it is a competition snapshot rather than a complete/live 2024-25 contract.
    fields=["blockedPlayerNFLId1","blockedPlayerNFLId2","blockedPlayerNFLId3","pressureAllowedAsBlocker","timeToPressureAllowedAsBlocker","pff_primaryDefensiveCoverageMatchupNflId"]
    inv=pd.DataFrame({"field":fields,"exists_in_public_big_data_bowl_2025":True,"repo_implementation_exists":False,"complete_2024_2025_weekly_contract":False,"live_2026_contract":False})
    inv.to_csv(out/"m80_blocker_rusher_field_inventory.csv",index=False)
    decision=pd.DataFrame([{"candidate":"TRUE_BLOCKER_X_RUSHER","information_is_genuinely_new":True,"public_example_data_exists":True,"complete_2024_2025_history_available":False,"live_2026_source_available":False,"predictive_model_fit":False,"advance_to_m81_development":False,"status":"HOLD_DATA_ACQUISITION","notes":"do not substitute aggregate pressure/OL continuity; require exact weekly blocker-rusher assignment contract before predictive testing"}])
    decision.to_csv(out/"m80_blocker_rusher_decision.csv",index=False); print(decision.to_string(index=False))


def job_master(inputs: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    candidates=[]
    specs=[
        ("FTN_NOVEL_ONLY","m80_ftn_decision.csv"),
        ("ROUTE_X_COVERAGE_SHELL","m80_route_shell_decision.csv"),
        ("TRUE_BLOCKER_X_RUSHER","m80_blocker_rusher_decision.csv"),
    ]
    for name, fname in specs:
        hits=list(inputs.rglob(fname))
        if not hits:
            candidates.append({"candidate":name,"status":"MISSING_AUDIT_OUTPUT","advance_to_m81_development":False}); continue
        row=pd.read_csv(hits[0]).iloc[0].to_dict(); candidates.append(row)
    master=pd.DataFrame(candidates)
    master.to_csv(out/"m80_master_frontier_decision.csv",index=False)
    interpretation={
        "migration":"M80",
        "status":"DIAGNOSTIC_SOURCE_FRONTIER",
        "predictive_model_fit":False,
        "canonical_sha256":EXPECTED_CANONICAL_SHA256,
        "selection_rule":"M81 may develop only candidates explicitly marked advance_to_m81_development=true. Historical-only/deployment-blocked sources remain HOLD. No dead M1-M79 family may be renamed as new.",
        "combination_rule":"If 2+ genuinely distinct families later pass M81 development independently, a preregistered survivor stack may be tested on development before one frozen untouched confirmation.",
    }
    (out/"m80_contract.json").write_text(json.dumps(interpretation,indent=2)+"\n")
    print(master.to_string(index=False))


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--job",required=True,choices=["tail","ftn","route","blocker","master"])
    p.add_argument("--canonical",type=Path,default=Path("data/backtests/qb_frontier_canonical_v3_football_only/qb_frontier_canonical_v3_football_only.csv"))
    p.add_argument("--out",type=Path,required=True); p.add_argument("--inputs",type=Path,default=Path("data/backtests/qb_m80_inputs"))
    a=p.parse_args()
    if a.job=="tail": job_tail(a.canonical,a.out)
    elif a.job=="ftn": job_ftn(a.out)
    elif a.job=="route": job_route(a.out)
    elif a.job=="blocker": job_blocker(a.out)
    else: job_master(a.inputs,a.out)
    return 0

if __name__=="__main__": raise SystemExit(main())
