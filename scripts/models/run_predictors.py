# scripts/models/run_predictors.py
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

# -------- Optional SciPy for SGP bivariate normal ----------
try:
    from scipy.stats import multivariate_normal, norm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -------- Import ensemble WITHOUT re-importing package symbols (no circulars) ----
try:
    from . import ensemble  # pkg-relative; __init__.py must stay light/empty
except Exception:
    import scripts.models.ensemble as ensemble  # fallback if needed

# -------- Safe Leg type: try shared_types, else define a local minimal dataclass ----
try:
    from .shared_types import Leg  # preferred if you already have it
except Exception:
    from dataclasses import dataclass
    @dataclass
    class Leg:
        player_id: str
        player: str
        team: str
        market: str
        line: float
        features: dict

# -------- Config pulled from your repo structure (already referenced by engine) ----
try:
    from scripts.config import LOG_DIR, RUN_ID, MONTE_CARLO_TRIALS
except Exception:
    import os
    from datetime import datetime, timezone
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    RUN_ID = os.getenv("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    # Prefer MC_TRIALS env (already set in your workflow), else MONTE_CARLO_TRIALS, else default
    MONTE_CARLO_TRIALS = int(os.getenv("MC_TRIALS", os.getenv("MONTE_CARLO_TRIALS", "20000")))

# --- NO-NUKES tweak: ensure LOG_DIR behaves like a Path even if provided as str ---
if not isinstance(LOG_DIR, Path):
    LOG_DIR = Path(LOG_DIR)

# ----------------------- Utils -----------------------
def _read_csv(path: str, cols: List[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=cols or [])
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=cols or [])

def american_to_prob(odds) -> float | None:
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0: return 100.0 / (o + 100.0)
    if o < 0: return (-o) / ((-o) + 100.0)
    return None

def prob_to_american(p: float) -> int | None:
    if p <= 0 or p >= 1: return None
    return -int(round(100*p/(1-p))) if p >= 0.5 else int(round(100*(1-p)/p))

def tier_from_edge(e: float | None) -> str:
    if e is None: return "RED"
    if e >= 0.06: return "ELITE"
    if e >= 0.04: return "GREEN"
    if e >= 0.01: return "AMBER"
    return "RED"

# ----------------------- Coverage & CB shadow -----------------------
def _load_coverage() -> Dict[str, set]:
    df = _read_csv("data/coverage.csv", ["defense_team","tag"])
    if df.empty: return {}
    df["defense_team"] = df["defense_team"].astype(str).str.upper()
    cov = {}
    for t, grp in df.groupby("defense_team"):
        cov[t] = set(x.strip().lower() for x in grp["tag"].dropna().astype(str))
    return cov

def _load_cb_assign() -> Dict[Tuple[str,str], float]:
    df = _read_csv("data/cb_assignments.csv", ["defense_team","receiver","cb","penalty","quality"])
    if df.empty: return {}
    df["defense_team"] = df["defense_team"].astype(str).str.upper()
    df["receiver"] = df["receiver"].astype(str).str.lower()
    penalty = {}
    for _,r in df.iterrows():
        p = r.get("penalty")
        if pd.isna(p):
            q = str(r.get("quality","")).lower()
            p = {"elite":0.20, "good":0.12, "avg":0.05}.get(q, 0.10)
        p = max(0.0, min(0.25, float(p)))
        penalty[(r["defense_team"], r["receiver"])] = p
    return penalty

def coverage_multiplier(position: str, opp_team: str, player: str,
                        cov_tags: Dict[str,set], cb_pen: Dict[Tuple[str,str], float]) -> Tuple[float,float,str]:
    note_parts = []
    tgt_mult = 1.0; ypt_mult = 1.0
    pos = (position or "").upper()
    opp = (opp_team or "").upper()

    tags = cov_tags.get(opp, set())
    if tags:
        if ("heavy_man" in tags or "top_shadow" in tags) and (pos.startswith("WR") or pos == "WR"):
            tgt_mult *= 0.92; ypt_mult *= 0.94; note_parts.append("cov:man/shadow")
        if "heavy_zone" in tags and (pos.startswith("TE") or pos == "TE"):
            tgt_mult *= 1.06; ypt_mult *= 1.04; note_parts.append("cov:zone+TE")

    pen = cb_pen.get((opp, (player or "").lower()))
    if pen and (pos.startswith("WR") or pos == "WR"):
        tgt_mult *= (1.0 - pen)
        ypt_mult *= (1.0 - min(pen*0.75, 0.20))
        note_parts.append(f"CB:-{int(pen*100)}%")

    return tgt_mult, ypt_mult, ",".join(note_parts)

# ----------------------- Team λ and rush/pass mix -----------------------
def _sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

def _team_expected_tds(team: str, odds_game: pd.DataFrame, merged_rows: pd.DataFrame) -> float:
    try:
        g = odds_game[(odds_game['home_team']==team) | (odds_game['away_team']==team)]
        total = float(g[g['market']=='totals']['point'].dropna().head(1).values[0]) if not g.empty else None
        spread_series = g[g['market']=='spreads']['point'].dropna().head(1)
        spread = float(spread_series.values[0]) if not spread_series.empty else None
        if total is not None:
            if spread is None: team_pts = total/2.0
            else:
                is_home = (not g.empty) and (g.iloc[0]['home_team']==team)
                team_pts = total/2.0 + ( -spread/2.0 if is_home else spread/2.0 )
            return max(0.1, team_pts/6.8)
    except Exception:
        pass
    try:
        trows = merged_rows[merged_rows['team']==team]
        plays = float(trows.get('pace').dropna().mean() or 60.0)
        return max(0.1, (plays*0.20)/6.8)
    except Exception:
        return 1.8

def _rush_mix_from_def(opp_row: pd.Series | None) -> float:
    if opp_row is None or opp_row.empty: return 0.50
    dr = float(opp_row.get('def_rush_epa', 0.0))
    dp = float(opp_row.get('def_pass_epa', 0.0))
    sack = float(opp_row.get('def_sack_rate', 0.0))
    light = float(opp_row.get('light_box_rate', 0.0))
    heavy = float(opp_row.get('heavy_box_rate', 0.0))
    base = (dr - dp) * 2.5
    box_adj = (light - heavy) * 1.5
    press_adj = sack * 1.2
    mix = _sigmoid(base + box_adj + press_adj)
    return min(0.85, max(0.15, mix))

# ----------------------- Player TD probabilities -----------------------
def p_anytime_td(position: str, rz_tgt_share: float, rz_carry_share: float,
                 team_lambda_td: float, rush_mix: float) -> Tuple[float,float]:
    pos = (position or '').upper()
    if   pos.startswith('RB'): tgt_w, car_w = 0.25, 0.75
    elif pos.startswith('TE'): tgt_w, car_w = 0.80, 0.20
    elif pos.startswith('QB'): tgt_w, car_w = 0.05, 0.20
    else:                      tgt_w, car_w = 0.85, 0.15

    rz_tgt_share  = max(0.0, min(1.0, float(rz_tgt_share or 0.0)))
    rz_carry_share= max(0.0, min(1.0, float(rz_carry_share or 0.0)))

    eff_share = tgt_w*rz_tgt_share*(1.0 - rush_mix) + car_w*rz_carry_share*rush_mix
    eff_share = max(0.0, min(1.0, eff_share))

    lam_player = max(0.01, float(team_lambda_td))*eff_share
    p_any = 1.0 - math.exp(-lam_player)
    return p_any, lam_player

def p_two_plus_td(lam_player: float) -> float:
    return 1.0 - math.exp(-lam_player)*(1.0 + lam_player)

# ----------------------- SGP helpers -----------------------
def _bvn_joint_prob(pA: float, pB: float, rho: float) -> float:
    pA = min(max(pA, 1e-6), 1-1e-6)
    pB = min(max(pB, 1e-6), 1-1e-6)
    if _HAS_SCIPY:
        zA = norm.ppf(pA); zB = norm.ppf(pB)
        cov = [[1.0, rho],[rho,1.0]]
        return float(multivariate_normal(mean=[0,0], cov=cov).cdf([zA, zB]))
    return pA*pB + rho*math.sqrt(pA*(1-pA)*pB*(1-pB))

# ----------------------- Main run -----------------------
def run(season: int):
    Path('outputs').mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)

    pf=_read_csv('data/player_form.csv', ['player','team','position'])
    tf=_read_csv('data/team_form.csv', ['team'])
    props=_read_csv('outputs/props_raw.csv', ['player','team','opp_team','market','line','over_odds','under_odds','book','commence_time','event_id','position'])
    odds_game=_read_csv('outputs/odds_game.csv', ['event_id','commence_time','sport_key','home_team','away_team','market','point','book'])

    for df,col in [(pf,'team'),(tf,'team'),(props,'team')]:
        if col in df.columns: df[col]=df[col].astype(str).str.upper()
    for col in ['opp_team','home_team','away_team']:
        if col in props.columns: props[col]=props[col].astype(str).str.upper()
        if col in odds_game.columns: odds_game[col]=odds_game[col].astype(str).str.upper()
    if 'player' in props.columns: props['player'] = props['player'].astype(str)

    # Market fair p (de-vig) just in case it hasn't been computed upstream
    p_market_fair=[]
    for _,r in props.iterrows():
        p_o=american_to_prob(r.get('over_odds')); p_u=american_to_prob(r.get('under_odds'))
        if p_o is None and p_u is None: p_market_fair.append(0.5)
        elif p_o is None:               p_market_fair.append(1-p_u)
        elif p_u is None:               p_market_fair.append(p_o)
        else:
            s=p_o+p_u; p_market_fair.append(p_o/s if s>0 else 0.5)
    props['p_market_fair']=p_market_fair

    merged=props.merge(pf, on=['player','team'], how='left', suffixes=('','_pf')).merge(tf, on='team', how='left', suffixes=('','_tf'))

    cov_tags = _load_coverage()
    cb_pen   = _load_cb_assign()

    out_rows=[]; leg_records=[]

    for _,r in merged.iterrows():
        market=str(r.get('market',''))
        player=str(r.get('player',''))
        team=str(r.get('team',''))
        opp=str(r.get('opp_team',''))
        position=str(r.get('position') or r.get('position_pf') or "")
        tgt_mult, ypt_mult, covnote = coverage_multiplier(position, opp, player, cov_tags, cb_pen)

        # ---------- Anytime TD ----------
        if market == 'player_anytime_td':
            lam_team = _team_expected_tds(team, odds_game, merged)
            opp_row = tf[tf['team']==opp].head(1).squeeze() if not tf.empty else pd.Series(dtype='float64')
            rush_mix = _rush_mix_from_def(opp_row)
            rz_tgt = (r.get('rz_tgt_share') or 0.0) * tgt_mult
            rz_car = (r.get('rz_carry_share') or 0.0)
            p_final, lam_player = p_anytime_td(position, rz_tgt, rz_car, lam_team, rush_mix)
            p_mkt = american_to_prob(r.get('over_odds')) or 0.5
            edge = p_final - p_mkt; fair = prob_to_american(p_final)
            note = f"AnyTD λ_team={lam_team:.2f}, rush_mix={rush_mix:.2f}" + (f", {covnote}" if covnote else "")
            out_rows.append({'event_id':r.get('event_id'),'player':player,'team':team,'opp_team':opp,'market':market,'line':1.0,
                             'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,'tier':tier_from_edge(edge),'notes':note})
            leg_records.append({'event_id':r.get('event_id'),'team':team,'player':player,'market':market,'p':p_final,'side':'yes','position':position})
            continue

        # ---------- 2+ TD ----------
        if market in ('player_2_or_more_tds','player_two_plus_tds'):
            lam_team = _team_expected_tds(team, odds_game, merged)
            opp_row = tf[tf['team']==opp].head(1).squeeze() if not tf.empty else pd.Series(dtype='float64')
            rush_mix = _rush_mix_from_def(opp_row)
            rz_tgt = (r.get('rz_tgt_share') or 0.0) * tgt_mult
            rz_car = (r.get('rz_carry_share') or 0.0)
            _, lam_player = p_anytime_td(position, rz_tgt, rz_car, lam_team, rush_mix)
            p_final = p_two_plus_td(lam_player)
            p_mkt = (american_to_prob(r.get('over_odds')) or american_to_prob(r.get('under_odds')) or 0.5)
            edge = p_final - p_mkt; fair = prob_to_american(p_final)
            note = f"2+TD λ_player={lam_player:.2f}" + (f", {covnote}" if covnote else "")
            out_rows.append({'event_id':r.get('event_id'),'player':player,'team':team,'opp_team':opp,'market':market,'line':2.0,
                             'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,'tier':tier_from_edge(edge),'notes':note})
            leg_records.append({'event_id':r.get('event_id'),'team':team,'player':player,'market':market,'p':p_final,'side':'yes','position':position})
            continue

        # ---------- Continuous markets ----------
        try: line=float(r.get('line'))
        except Exception: continue

        feats={'mu': r.get('mu'),'sd': r.get('sd'),'sd_widen': r.get('sd_widen',1.0),
               'eff_mu': r.get('eff_mu'),'eff_sd': r.get('eff_sd'),
               'p_market_fair': r.get('p_market_fair',0.5),
               'target_share': r.get('target_share',0.0),'rush_share': r.get('rush_share',0.0),
               'qb_ypa': r.get('qb_ypa',0.0),'light_box_rate': r.get('light_box_rate',0.0),
               'heavy_box_rate': r.get('heavy_box_rate',0.0),'def_sack_rate': r.get('def_sack_rate',0.0),
               'def_pass_epa': r.get('def_pass_epa',0.0),'pace': r.get('pace',0.0),'proe': r.get('proe',0.0)}

        # Apply coverage/CB impact to receiving markets
        if market in ('player_rec_yds','player_receptions','player_rush_rec_yds'):
            if feats.get('mu') is not None and not pd.isna(feats['mu']):
                if market in ('player_rec_yds','player_rush_rec_yds'): feats['mu']  = float(feats['mu'])  * ypt_mult
                if market == 'player_receptions':                   feats['mu']  = float(feats['mu'])  * tgt_mult
            if feats.get('eff_mu') is not None and not pd.isna(feats['eff_mu']):
                if market in ('player_rec_yds','player_rush_rec_yds'): feats['eff_mu'] = float(feats['eff_mu']) * ypt_mult
                if market == 'player_receptions':                   feats['eff_mu'] = float(feats['eff_mu']) * tgt_mult

        # Build leg and blend
        leg=Leg(player_id=f"{player}|{team}|{market}|{line}", player=player, team=team, market=market, line=line, features=feats)
        blended=ensemble.blend(leg, context={'w_mc':0.25,'w_bayes':0.25,'w_markov':0.25,'w_ml':0.25})

        p_mkt=blended.get('p_market',0.5); p_final=blended.get('p_final',0.5)
        edge=(p_final-p_mkt) if p_mkt is not None else None
        fair=prob_to_american(p_final) if p_final is not None else None

        note = blended.get('notes','')
        if covnote and market in ('player_rec_yds','player_receptions','player_rush_rec_yds'):
            note = (note + ("; " if note else "") + covnote)

        out_rows.append({'event_id':r.get('event_id'),'player':player,'team':team,'opp_team':opp,'market':market,'line':line,
                         'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,'tier':tier_from_edge(edge),'notes':note})

        leg_records.append({'event_id':r.get('event_id'),'team':team,'player':player,'market':market,'p':p_final,'side':'over','position':position})

    # ---------- Write singles ----------
    singles=pd.DataFrame(out_rows).sort_values(['tier','edge'], ascending=[True, False])
    singles_path=Path('outputs/master_model_predictions.csv'); singles.to_csv(singles_path, index=False)
    (LOG_DIR/'summary.json').write_text(json.dumps({'run_id':RUN_ID,'season':int(season),
        'rows':int(len(singles)),'mc_trials':int(MONTE_CARLO_TRIALS),'provider':os.getenv('PROVIDER_USED','unknown')}, indent=2))
    singles.to_csv(LOG_DIR/'master_model_predictions.csv', index=False)
    print(f"[predictors] ✅ wrote {len(singles)} rows → {singles_path}")
    print(f"[predictors] log → {LOG_DIR}")

    # ---------- Same-Game Parlay suggestions ----------
    if leg_records:
        legs_df = pd.DataFrame(leg_records)
        RHO = {('player_pass_yds','player_rec_yds'): 0.60, ('player_rec_yds','player_pass_yds'): 0.60,
               ('player_rush_yds','player_pass_yds'): -0.35, ('player_pass_yds','player_rush_yds'): -0.35}

        if not tf.empty:
            pass_z = np.nanpercentile(tf.get('def_pass_epa', pd.Series([0])), 40)
            rush_z = np.nanpercentile(tf.get('def_rush_epa', pd.Series([0])), 60)

        sgp_rows=[]
        for ev_id, g in legs_df.groupby('event_id'):
            if g.empty or ev_id is None: continue
            by_market = { (row['team'], row['market']): row for _,row in g.iterrows() }

            teams = g['team'].dropna().unique().tolist()
            for tm in teams:
                qb = by_market.get((tm, 'player_pass_yds'))
                wrs = [row for key,row in by_market.items() if key[0]==tm and row['market']=='player_rec_yds']
                if qb and wrs:
                    rho = RHO[('player_pass_yds','player_rec_yds')]
                    for wr in wrs:
                        jp = _bvn_joint_prob(qb['p'], wr['p'], rho); fair = prob_to_american(jp)
                        sgp_rows.append({'event_id': ev_id,'team': tm,
                                         'legs': f"QB pass o{qb.get('line','')}; WR rec o{wr.get('line','')}",
                                         'players': f"{qb['player']} + {wr['player']}",
                                         'rho': rho, 'joint_prob': jp, 'fair_odds': fair, 'note': 'QB↔WR +ρ'})

                rb = by_market.get((tm, 'player_rush_yds')); qb = by_market.get((tm, 'player_pass_yds'))
                if rb and qb:
                    opp = singles.loc[(singles['event_id']==ev_id) & (singles['team']==tm),'opp_team']
                    opp = opp.iloc[0] if not opp.empty else ''
                    rho = -0.35
                    if opp and not tf.empty:
                        opp_row = tf[tf['team']==opp].head(1)
                        if not opp_row.empty:
                            if float(opp_row['def_rush_epa'].iloc[0] or 0) >= rush_z and float(opp_row['def_pass_epa'].iloc[0] or 0) <= pass_z:
                                rho = -0.50
                    jp = _bvn_joint_prob(rb['p'], qb['p'], rho); fair = prob_to_american(jp)
                    sgp_rows.append({'event_id': ev_id,'team': tm,
                                     'legs': f"RB rush o{rb.get('line','')}; QB pass o{qb.get('line','')}",
                                     'players': f"{rb['player']} + {qb['player']}",
                                     'rho': rho, 'joint_prob': jp, 'fair_odds': fair, 'note': 'RB↔QB −ρ'})

        if sgp_rows:
            sgp = pd.DataFrame(sgp_rows).sort_values('joint_prob', ascending=False)
            sgp_path = Path('outputs/sgp_candidates.csv'); sgp.to_csv(sgp_path, index=False)
            print(f"[predictors] 🧩 SGP pairs → {sgp_path} rows={len(sgp)}")
        else:
            print("[predictors] (no SGP pairs found)")

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--season', type=int, required=True)
    a=ap.parse_args(); run(a.season)
