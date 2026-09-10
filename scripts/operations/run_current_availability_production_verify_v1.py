#!/usr/bin/env python3
"""Dynamic production verification for the certified current-availability stack.

Verification-only adapter. It generalizes only the immutable certification
runner's historical 30-team snapshot assumption to the current certified
eligible team set. No sportsbook data is read and no model parameters change.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scripts._opponent_map import canon_team
from scripts.utils.eligible_team_set_v1 import expected_current_teams, validate_current_team_set
import scripts.operations.run_current_availability_football_stack_cert_v1 as cert_runner
import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
import scripts.run_pricing_with_full_roster_universe_v5_production as v5

DATA=Path('data')
OUT=DATA/'current_player_availability_production_verification.json'

def read_csv(path:Path)->pd.DataFrame:
    if not path.is_file() or path.stat().st_size<=0: raise RuntimeError(f'required production verification input missing/empty: {path}')
    x=pd.read_csv(path,low_memory=False)
    if x.empty: raise RuntimeError(f'required production verification input has zero rows: {path}')
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def main()->int:
    active=read_csv(DATA/'roles_current_production_eligible_v1.csv')
    avail=read_csv(DATA/'current_player_availability.csv')
    game_cert=read_csv(DATA/'current_player_availability_game_certification.csv')
    form=read_csv(DATA/'player_form_consensus.csv')
    expected=expected_current_teams()
    if expected is None: raise RuntimeError('production verification requires explicit ACTIVE_ROLES_CSV')
    active_teams={canon_team(x) for x in active.team.dropna().astype(str)}; active_teams.discard('')
    if active_teams!=expected: raise RuntimeError(f'active roles != helper expected teams missing={sorted(expected-active_teams)} extra={sorted(active_teams-expected)}')
    eligible=pd.to_numeric(game_cert.production_eligible,errors='coerce').fillna(0).eq(1)
    eligible_games=int(eligible.sum())
    if len(expected)!=2*eligible_games: raise RuntimeError(f'eligible team/game arithmetic mismatch teams={len(expected)} games={eligible_games}')
    team_rows=pd.concat([game_cert[['away_team']].rename(columns={'away_team':'team'}),game_cert[['home_team']].rename(columns={'home_team':'team'})])
    if team_rows.team.astype(str).duplicated().any(): raise RuntimeError('certification contains duplicate weekly team assignment')
    validate_current_team_set(form.team.dropna().astype(str).unique(),label='production verification PlayerForm')
    unavailable=avail.loc[pd.to_numeric(avail.definitive_unavailable,errors='coerce').fillna(0).eq(1)].copy()
    bad=set(zip(unavailable.team.map(canon_team).astype(str),unavailable.player_clean_key.astype(str)))
    form_keys=set(zip(form.team.map(canon_team).astype(str),form.player_clean_key.astype(str)))
    if bad & form_keys: raise RuntimeError(f'definitive unavailable survived PlayerForm: {sorted(bad&form_keys)[:20]}')
    lookup=cert_runner.build_lookup_metrics(form)
    base._identity_frame=v2._canonical_identity_frame
    base._validate_priced_distribution_coverage=v2._install_provider_player_aliases_and_validate
    base._build_full_universe=v3._build_with_promoted_entitlement_specialists
    base.canonical_simulate=v5._simulate_v5
    result=base._full_roster_simulate(lookup,iterations=4000,seed=20260910)
    required={'r22':DATA/'rb_receiving_tail_production_audit.json','r26':DATA/'rb_r26_receptions_production_audit.json','ent':DATA/'target_entitlement_v1_audit.json','qb':DATA/'qb_c2_production_integration_audit.json','universe':DATA/'football_simulation_universe_audit.json','p3':DATA/'rb_rush_rec_conservation_input_audit.json'}
    payloads={}
    for name,path in required.items():
        if not path.is_file(): raise RuntimeError(f'promoted football stack did not emit required audit: {path}')
        payloads[name]=json.loads(path.read_text(encoding='utf-8'))
    r22,r26,ent,qb,universe,p3=(payloads[k] for k in ['r22','r26','ent','qb','universe','p3'])
    sim_players={str(k[1]) for k in result.values}; unavailable_in_sim=sorted({k for _,k in bad if k in sim_players})
    if unavailable_in_sim: raise RuntimeError(f'unavailable players received simulation arrays: {unavailable_in_sim}')
    football_teams=int(universe.get('football_teams',0)); canonical_games=int(universe.get('canonical_games',0))
    if football_teams!=len(expected): raise RuntimeError(f'football universe team mismatch expected={len(expected)} observed={football_teams}')
    if canonical_games!=eligible_games: raise RuntimeError(f'football universe game mismatch expected={eligible_games} observed={canonical_games}')
    checks={
      'te_r5p_pool_preserved':bool(ent.get('te_r5p_team_pool_preserved',False)),
      'wr_r15_wr1_anchor_preserved':bool(ent.get('wr_r15_m38_wr1_anchor_preserved',False)),
      'wr_r15_wr2plus_pool_preserved':bool(ent.get('wr_r15_wr2plus_pool_preserved',False)),
      'wr_r15_wr_room_mass_preserved':bool(ent.get('wr_r15_wr_room_mass_preserved',False)),
      'wr_r15_non_wr_entitlement_preserved':bool(ent.get('wr_r15_non_wr_entitlement_preserved',False)),
    }
    if not all(checks.values()): raise RuntimeError(f'entitlement conservation failed: {checks}')
    if qb.get('disposition')!='QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS': raise RuntimeError(f'QB C2 integration failure: {qb}')
    if int(qb.get('sportsbook_inputs_to_selector',0))!=0 or int(qb.get('sportsbook_inputs_to_starter_selection',0))!=0 or int(qb.get('sportsbook_inputs_to_c2_generation',0))!=0: raise RuntimeError('QB C2 sportsbook leakage')
    if r22.get('disposition') not in {'RB_R22_PRODUCTION_TAIL_INTEGRATION_PASS','R22_PRODUCTION_TAIL_INTEGRATION_PASS'}: raise RuntimeError(f'R22 disposition unexpected: {r22.get("disposition")}')
    if float(r22.get('max_mean_delta',0.0))>1e-10: raise RuntimeError(f'R22 mean drift: {r22.get("max_mean_delta")}')
    if int(r26.get('sportsbook_inputs_used',r26.get('sportsbook_inputs_to_r26_football',0)))!=0: raise RuntimeError('R26 sportsbook leakage')
    if p3.get('disposition')!='RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3': raise RuntimeError(f'P3 conservation failure: {p3}')
    if int(universe.get('priced_distribution_misses',-1))!=0: raise RuntimeError('football-only verification missed generated distributions')
    out={'disposition':'CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION','sportsbook_inputs_used':0,'eligible_teams':len(expected),'eligible_games':eligible_games,'withheld_games':int((~eligible).sum()),'football_teams':football_teams,'canonical_games':canonical_games,'unavailable_rows':len(unavailable),'unavailable_players_in_playerform':0,'unavailable_players_with_simulation_arrays':0,'qb_c2_disposition':qb.get('disposition'),'r22_disposition':r22.get('disposition'),'r22_max_mean_delta':r22.get('max_mean_delta'),'r26_disposition':r26.get('disposition'),'p3_conservation_disposition':p3.get('disposition'),**checks}
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
