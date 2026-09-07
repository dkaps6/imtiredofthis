"""Joint Monte Carlo simulation for player-prop outcomes.

Each simulated game shares team plays, pass rate, and efficiency shocks across
players. Player target/carry opportunities are allocated with multinomial draws,
so same-team outcomes compete for finite volume instead of being simulated as
independent normal distributions.

Migration 3: canonical football rules alter pre-simulation assumptions.
Migration 4A: empirical-Bayesian baselines can feed those rule-adjusted inputs.
Migration 38: calibrated WR hierarchy sharpening redistributes, but does not add,
team WR target-share mass before canonical target allocation.

Research integration V1 adds an opt-in C2 pass/receiving conservation adapter.
The default production behavior of :func:`simulate` is unchanged.  The adapter
uses the exact shared pass-attempt/pass-efficiency state created by the canonical
simulation, replaces only receptions/receiving-yards/primary-QB passing-yards,
and leaves rushing/TD arrays byte-for-byte untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from scripts.config import MC

MARKET_MAP = {
    "player_pass_yds": "pass_yards", "player_passing_yards": "pass_yards", "pass_yards": "pass_yards",
    "player_rush_yds": "rush_yards", "player_rushing_yards": "rush_yards", "rush_yards": "rush_yards",
    "player_reception_yds": "rec_yards", "player_rec_yds": "rec_yards", "player_receiving_yards": "rec_yards", "rec_yards": "rec_yards",
    "player_receptions": "receptions", "receptions": "receptions",
    "player_rush_att": "rush_att", "rush_att": "rush_att",
    "player_rush_reception_yds": "rush_rec_yards", "player_rush_rec_yds": "rush_rec_yards", "rush_rec_yards": "rush_rec_yards",
    "player_anytime_td": "anytime_td", "anytime_td": "anytime_td", "atd": "anytime_td",
}

WR_POSITIONS = {"WR", "LWR", "RWR", "SWR"}
WR_TARGET_HIERARCHY_MULTIPLIERS = (1.40, 1.14, 0.91, 0.78)
PASS_CATCHER_POSITIONS = {"WR", "LWR", "RWR", "SWR", "TE", "RB", "FB"}
C2_RESIDUAL_CATCH_RATE = 0.64
C2_RESIDUAL_YPT = 7.5
C2_YPR_MIN = 3.0
C2_YPR_MAX = 35.0


@dataclass
class SimulationResult:
    values: Dict[tuple[str, str, str], np.ndarray]
    iterations: int
    # Internal shared team-game states.  Existing callers can ignore this field.
    # Capturing the arrays does not add random draws or alter default outputs.
    team_states: Dict[tuple[str, str, str], np.ndarray] | None = None


def _num(row: pd.Series, *names, default=np.nan) -> float:
    for name in names:
        if name in row.index:
            try:
                value = float(row.get(name))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return float(default)


def _clip_prob(value: float, default: float) -> float:
    if not np.isfinite(value): value = default
    return float(np.clip(value, 0.001, 0.999))


def _team_inputs(team_rows: pd.DataFrame) -> tuple[float, float]:
    row = team_rows.iloc[0]
    plays = _num(row, "rules_plays_est", "plays_est", "pbp_plays_offense")
    if not np.isfinite(plays):
        pace = _num(row, "pace", "neutral_pace")
        plays = 1800.0 / pace if np.isfinite(pace) and pace > 0 else 64.0
    plays = float(np.clip(plays, 50.0, 80.0))
    pass_rate = _num(row, "rules_pass_rate")
    if not np.isfinite(pass_rate):
        proe = _num(row, "proe", "pass_rate_over_expected", default=0.0)
        pass_rate = 0.58 + (proe if np.isfinite(proe) else 0.0)
        team_wp = _num(row, "team_wp")
        if np.isfinite(team_wp): pass_rate += -0.02 if team_wp >= 0.60 else 0.02 if team_wp <= 0.40 else 0.0
    return plays, float(np.clip(pass_rate, 0.35, 0.75))


def _allocate_counts(rng: np.random.Generator, totals: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Allocate integer opportunities across modeled players plus a residual bucket."""
    n_iter=len(totals); n_players=len(shares)
    if n_players == 0: return np.empty((n_iter,0),dtype=int)
    clean=np.nan_to_num(shares.astype(float),nan=0.0,posinf=0.0,neginf=0.0); clean=np.clip(clean,0.0,0.95)
    total_share=clean.sum()
    if total_share > 0.95: clean *= 0.95 / total_share
    residual=max(0.0,1.0-clean.sum()); probs=np.append(clean,residual); probs=probs/probs.sum()
    allocations=np.empty((n_iter,n_players),dtype=int)
    for i,total in enumerate(totals.astype(int)): allocations[i]=rng.multinomial(max(0,int(total)),probs)[:n_players]
    return allocations


def _top_n_shares(shares: np.ndarray, n: int = 5) -> np.ndarray:
    """Retain only the team's top-N rushing shares before canonical normalization."""
    clean=np.nan_to_num(shares.astype(float),nan=0.0,posinf=0.0,neginf=0.0); clean=np.clip(clean,0.0,0.95)
    if len(clean) <= n: return clean
    order=np.argsort(-clean, kind="stable"); keep=order[:n]; out=np.zeros_like(clean); out[keep]=clean[keep]
    return out


def _sharpen_wr_target_shares(team_df: pd.DataFrame, shares: np.ndarray) -> np.ndarray:
    """Apply the Migration 37 winner while preserving total team WR target mass.

    WRs are ranked by their leakage-safe pregame target shares. Multipliers are
    applied to WR1/WR2/WR3/WR4+ and then renormalized back to the original WR
    target-share total. RB/TE/FB and all other player shares are unchanged.
    """
    clean=np.nan_to_num(np.asarray(shares,dtype=float),nan=0.0,posinf=0.0,neginf=0.0); clean=np.clip(clean,0.0,0.95)
    positions=team_df.get("position",pd.Series("",index=team_df.index)).fillna("").astype(str).str.upper().to_numpy()
    wr_idx=np.flatnonzero(np.isin(positions,list(WR_POSITIONS)))
    if len(wr_idx) <= 1: return clean
    wr=clean[wr_idx].copy(); total=float(wr.sum())
    if total <= 0.0: return clean
    order=np.argsort(-wr,kind="stable"); mult=np.ones(len(wr),dtype=float)
    for rank,idx in enumerate(order): mult[idx]=WR_TARGET_HIERARCHY_MULTIPLIERS[min(rank,len(WR_TARGET_HIERARCHY_MULTIPLIERS)-1)]
    sharpened=wr*mult
    if sharpened.sum() <= 0.0: return clean
    sharpened*=total/float(sharpened.sum())
    out=clean.copy(); out[wr_idx]=sharpened
    return out


def _player_key(row: pd.Series) -> str:
    value=row.get("player_clean_key")
    if pd.notna(value) and str(value).strip(): return str(value).strip()
    return "".join(ch.lower() for ch in str(row.get("player","")) if ch.isalnum())


def simulate(metrics: pd.DataFrame, *, iterations: int | None=None, seed: int | None=None, allocation_trace: list[dict] | None=None) -> SimulationResult:
    iterations=int(iterations or MC.get("iterations",25000)); seed=int(MC.get("seed",42) if seed is None else seed); rng=np.random.default_rng(seed); values={}; team_states={}
    if metrics.empty: return SimulationResult(values,iterations,team_states)
    frame=metrics.copy(); frame["player_clean_key"]=frame.apply(_player_key,axis=1)
    game_key="event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"]=frame.apply(lambda r:"|".join(sorted([str(r.get("team","")),str(r.get("opponent",""))])),axis=1); game_key="_game_key"
    player_cols=[game_key,"team","player_clean_key"]; players=frame.sort_values(player_cols).drop_duplicates(player_cols,keep="last")
    for game,game_df in players.groupby(game_key,dropna=False):
        game_pace_shock=rng.normal(0.0,2.0,iterations)
        for team,team_df in game_df.groupby("team",dropna=False):
            if pd.isna(team) or not str(team).strip(): continue
            plays_mean,pass_rate_mean=_team_inputs(team_df); plays=np.rint(np.clip(rng.normal(plays_mean,3.5,iterations)+game_pace_shock,45,85)).astype(int); pass_rate=np.clip(rng.normal(pass_rate_mean,0.035,iterations),0.25,0.82); pass_att=rng.binomial(plays,pass_rate); rush_att=plays-pass_att
            pass_eff_shock=np.clip(rng.normal(1.0,0.09,iterations),0.65,1.35); rush_eff_shock=np.clip(rng.normal(1.0,0.10,iterations),0.60,1.40)
            # Capture the already-drawn shared game state without changing RNG order.
            team_states[(str(game),str(team),"plays")]=plays.copy(); team_states[(str(game),str(team),"pass_rate")]=pass_rate.copy(); team_states[(str(game),str(team),"pass_att")]=pass_att.copy(); team_states[(str(game),str(team),"rush_att")]=rush_att.copy(); team_states[(str(game),str(team),"pass_eff_shock")]=pass_eff_shock.copy(); team_states[(str(game),str(team),"rush_eff_shock")]=rush_eff_shock.copy()
            raw_target_shares=np.array([_num(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0) for _,r in team_df.iterrows()]); target_shares=_sharpen_wr_target_shares(team_df,raw_target_shares); raw_rush_shares=np.array([_num(r,"rules_rush_share","bayes_rush_share","rush_share",default=0.0) for _,r in team_df.iterrows()]); rush_shares=_top_n_shares(raw_rush_shares,5)
            targets=_allocate_counts(rng,pass_att,target_shares); carries=_allocate_counts(rng,rush_att,rush_shares)
            if allocation_trace is not None:
                clean=np.clip(np.nan_to_num(rush_shares.astype(float),nan=0.0,posinf=0.0,neginf=0.0),0.0,0.95); raw_sum=float(clean.sum()); used=clean.copy()
                if raw_sum>0.95: used*=0.95/raw_sum
                residual=max(0.0,1.0-float(used.sum())); probs=np.append(used,residual); probs=probs/probs.sum(); team_rush_mean=float(np.mean(rush_att)) if len(rush_att) else np.nan
                for j,(_,trace_row) in enumerate(team_df.iterrows()): allocation_trace.append({"event_id":str(game),"team":str(team),"player_clean_key":_player_key(trace_row),"sim_selected_market":str(trace_row.get("market","")),"raw_player_rush_share":float(clean[j]),"raw_team_rush_share_sum":raw_sum,"final_player_probability":float(probs[j]),"residual_probability":float(probs[-1]),"team_rush_total_mean":team_rush_mean,"expected_carries_from_final_probability":team_rush_mean*float(probs[j]),"realized_multinomial_mean_carries":float(carries[:,j].mean())})
            for j,(_,row) in enumerate(team_df.iterrows()):
                pkey=_player_key(row)
                if not pkey: continue
                role=str(row.get("model_role",row.get("role","")) or "").upper(); position=str(row.get("position","") or "").upper(); catch_rate=_clip_prob(_num(row,"rules_catch_rate","bayes_receptions_per_target","receptions_per_target","catch_rate",default=0.64),0.64); receptions=rng.binomial(targets[:,j],catch_rate); vol_mult=float(np.clip(_num(row,"rules_volatility_mult",default=1.0),0.75,1.50))
                ypt=_num(row,"rules_ypt","bayes_ypt","ypt"); ypt=7.5 if not np.isfinite(ypt) or ypt<=0 else ypt; rec_mu=targets[:,j]*ypt*pass_eff_shock; rec_sd=np.maximum(6.0,np.sqrt(np.maximum(targets[:,j],1))*ypt*0.55)*vol_mult; rec_yards=np.clip(rng.normal(rec_mu,rec_sd),0.0,None)
                ypc=_num(row,"rules_ypc","bayes_ypc","ypc"); ypc=4.2 if not np.isfinite(ypc) or ypc<=0 else ypc; rush_mu=carries[:,j]*ypc*rush_eff_shock; rush_sd=np.maximum(3.0,np.sqrt(np.maximum(carries[:,j],1))*ypc*0.65)*vol_mult; rush_yards=np.clip(rng.normal(rush_mu,rush_sd),0.0,None)
                values[(str(game),pkey,"receptions")]=receptions.astype(float); values[(str(game),pkey,"rec_yards")]=rec_yards; values[(str(game),pkey,"rush_att")]=carries[:,j].astype(float); values[(str(game),pkey,"rush_yards")]=rush_yards; values[(str(game),pkey,"rush_rec_yards")]=rush_yards+rec_yards
                if position=="QB" or role.startswith("QB"):
                    ypa=_num(row,"rules_ypa","bayes_ypa","ypa","ypa_prior"); ypa=7.0 if not np.isfinite(ypa) or ypa<=0 else ypa; qb_noise=np.clip(rng.normal(1.0,0.07*vol_mult,iterations),0.72,1.28); values[(str(game),pkey,"pass_yards")]=np.clip(pass_att*ypa*pass_eff_shock*qb_noise,0.0,None)
                td_rate=_num(row,"offensive_td_rate")
                if np.isfinite(td_rate) and td_rate>=0:
                    rz=_num(row,"rz_share",default=np.nan); rz_mult=float(np.clip(0.75+rz,0.75,1.35)) if np.isfinite(rz) else 1.0; wp=_num(row,"team_wp"); script_mult=1.0+(0.08*(wp-0.5) if np.isfinite(wp) else 0.0); lam=max(0.0,td_rate*rz_mult*script_mult); team_scoring_shock=np.clip(rng.normal(1.0,0.12,iterations),0.65,1.35); p_iter=np.clip(1.0-np.exp(-lam*team_scoring_shock),0.001,0.98); values[(str(game),pkey,"anytime_td")]=rng.binomial(1,p_iter).astype(float)
    return SimulationResult(values,iterations,team_states)


def _primary_qb_row(team_df: pd.DataFrame) -> pd.Series | None:
    candidates=[]
    for idx,row in team_df.iterrows():
        pos=str(row.get("position","") or "").upper().strip(); role=str(row.get("model_role",row.get("role","")) or "").upper().strip()
        if pos!="QB" and not role.startswith("QB"): continue
        candidates.append((_num(row,"qb_projection_eligible",default=0.0),_num(row,"qb_role_score",default=0.0),str(_player_key(row)),idx,row))
    if not candidates: return None
    candidates.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True)
    return candidates[0][4]


def apply_pass_receiving_conservation(
    base: SimulationResult,
    metrics: pd.DataFrame,
    *,
    anchor_map: dict[tuple[str,str],float],
    seed: int=5601,
    conservation_trace: list[dict] | None=None,
) -> SimulationResult:
    """Apply the frozen C2 receiving process to a completed canonical simulation.

    This is intentionally an opt-in research adapter.  It consumes the exact
    team-game pass-attempt and pass-efficiency arrays already drawn by
    :func:`simulate`, preserves the M38 target hierarchy, generates receiving
    yards from completed receptions (zero receptions => zero yards), adds an
    explicit residual receiver bucket, and scales the team receiving process to
    the supplied pregame QB mean anchor.  Only receptions/receiving yards,
    rush+receiving yards and the primary QB passing-yard array are replaced.
    All rushing and TD arrays are copied from ``base`` unchanged.
    """
    if base.team_states is None:
        raise RuntimeError("C2 integration requires canonical shared team_states from simulate()")
    rng=np.random.default_rng(int(seed))
    values={k:np.asarray(v).copy() for k,v in base.values.items()}
    frame=metrics.copy(); frame["player_clean_key"]=frame.apply(_player_key,axis=1)
    game_key="event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"]=frame.apply(lambda r:"|".join(sorted([str(r.get("team","")),str(r.get("opponent",""))])),axis=1); game_key="_game_key"
    players=frame.sort_values([game_key,"team","player_clean_key"]).drop_duplicates([game_key,"team","player_clean_key"],keep="last")
    for game,game_df in players.groupby(game_key,dropna=False):
        for team,team_df in game_df.groupby("team",dropna=False):
            if pd.isna(team) or not str(team).strip(): continue
            game_s=str(game); team_s=str(team); team_df=team_df.reset_index(drop=True)
            pass_att=base.team_states.get((game_s,team_s,"pass_att")); pass_eff_shock=base.team_states.get((game_s,team_s,"pass_eff_shock"))
            if pass_att is None or pass_eff_shock is None:
                raise RuntimeError(f"missing canonical pass state game={game_s} team={team_s}")
            pass_att=np.asarray(pass_att,dtype=int); pass_eff_shock=np.asarray(pass_eff_shock,dtype=float)
            if len(pass_att)!=base.iterations or len(pass_eff_shock)!=base.iterations:
                raise RuntimeError(f"C2 shared-state length mismatch game={game_s} team={team_s}")

            raw=np.array([_num(r,"rules_tgt_share","bayes_tgt_share","target_share","tgt_share",default=0.0) for _,r in team_df.iterrows()],dtype=float)
            shares=_sharpen_wr_target_shares(team_df,raw)
            positions=team_df.get("position",pd.Series("",index=team_df.index)).fillna("").astype(str).str.upper().str.strip().to_numpy()
            pass_mask=np.isin(positions,list(PASS_CATCHER_POSITIONS))
            shares=np.where(pass_mask,shares,0.0)
            targets=_allocate_counts(rng,pass_att,shares)
            residual_targets=np.maximum(0,pass_att-targets.sum(axis=1))

            rec_arrays={}; yard_arrays={}; zero_rec_positive=0
            for j,(_,row) in enumerate(team_df.iterrows()):
                if not pass_mask[j]: continue
                pkey=_player_key(row)
                if not pkey: continue
                catch_rate=_clip_prob(_num(row,"rules_catch_rate","bayes_receptions_per_target","receptions_per_target","catch_rate",default=C2_RESIDUAL_CATCH_RATE),C2_RESIDUAL_CATCH_RATE)
                receptions=rng.binomial(targets[:,j],catch_rate)
                ypt=_num(row,"rules_ypt","bayes_ypt","ypt"); ypt=C2_RESIDUAL_YPT if not np.isfinite(ypt) or ypt<=0 else float(ypt)
                ypr=float(np.clip(ypt/catch_rate,C2_YPR_MIN,C2_YPR_MAX))
                vol=float(np.clip(_num(row,"rules_volatility_mult",default=1.0),0.75,1.50))
                mu=receptions.astype(float)*ypr*pass_eff_shock
                sd=np.maximum(3.0,np.sqrt(np.maximum(receptions,1))*ypr*0.55)*vol
                yards=np.clip(rng.normal(mu,sd),0.0,None); yards=np.where(receptions>0,yards,0.0)
                zero_rec_positive+=int(((receptions==0)&(yards>0)).sum())
                rec_arrays[pkey]=receptions.astype(float); yard_arrays[pkey]=yards

            residual_rec=rng.binomial(residual_targets,C2_RESIDUAL_CATCH_RATE)
            residual_ypr=C2_RESIDUAL_YPT/C2_RESIDUAL_CATCH_RATE
            residual_mu=residual_rec.astype(float)*residual_ypr*pass_eff_shock
            residual_sd=np.maximum(3.0,np.sqrt(np.maximum(residual_rec,1))*residual_ypr*0.55)
            residual_yards=np.clip(rng.normal(residual_mu,residual_sd),0.0,None); residual_yards=np.where(residual_rec>0,residual_yards,0.0)

            raw_modeled=np.sum(np.vstack(list(yard_arrays.values())),axis=0) if yard_arrays else np.zeros(base.iterations,dtype=float)
            raw_total=raw_modeled+residual_yards; raw_mean=float(np.mean(raw_total))
            anchor=float(anchor_map.get((game_s,team_s),np.nan))
            if not np.isfinite(anchor) or anchor<=0: raise RuntimeError(f"missing/invalid C2 pass-yard anchor game={game_s} team={team_s} anchor={anchor}")
            if not np.isfinite(raw_mean) or raw_mean<=0: raise RuntimeError(f"invalid C2 raw receiving mean game={game_s} team={team_s} raw_mean={raw_mean}")
            scale=anchor/raw_mean

            scaled={pkey:arr*scale for pkey,arr in yard_arrays.items()}; scaled_residual=residual_yards*scale
            modeled_sum=np.sum(np.vstack(list(scaled.values())),axis=0) if scaled else np.zeros(base.iterations,dtype=float)
            qb_yards=modeled_sum+scaled_residual
            for pkey,receptions in rec_arrays.items():
                values[(game_s,pkey,"receptions")]=receptions
                values[(game_s,pkey,"rec_yards")]=scaled[pkey]
                rush=values.get((game_s,pkey,"rush_yards"))
                if rush is not None: values[(game_s,pkey,"rush_rec_yards")]=np.asarray(rush,dtype=float)+scaled[pkey]
            qb_row=_primary_qb_row(team_df)
            if qb_row is not None:
                qkey=_player_key(qb_row)
                if qkey: values[(game_s,qkey,"pass_yards")]=qb_yards
            gap=qb_yards-(modeled_sum+scaled_residual)
            if conservation_trace is not None:
                conservation_trace.append({
                    "event_id":game_s,"team":team_s,"anchor_mean":anchor,"candidate_qb_mean":float(np.mean(qb_yards)),
                    "raw_receiver_mean":raw_mean,"scale":float(scale),"residual_receiver_yards_mean":float(np.mean(scaled_residual)),
                    "zero_reception_positive_yards":int(zero_rec_positive),"max_abs_gap":float(np.max(np.abs(gap))),
                    "mean_abs_gap":float(np.mean(np.abs(gap))),"pass_att_mean":float(np.mean(pass_att)),
                })
    return SimulationResult(values,base.iterations,base.team_states)


def lookup(result: SimulationResult,row: pd.Series,market: str) -> np.ndarray | None:
    game=row.get("event_id")
    if pd.isna(game) or not str(game).strip(): game="|".join(sorted([str(row.get("team","")),str(row.get("opponent",""))]))
    return result.values.get((str(game),_player_key(row),MARKET_MAP.get(str(market).lower(),str(market).lower())))
