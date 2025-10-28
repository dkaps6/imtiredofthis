import pandas as pd, numpy as np
from pathlib import Path
import re

def _read_csv(path):
    if not path: return pd.DataFrame()
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def normalize_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def pf_compact(name: str) -> str:
    tokens = re.sub(r'[^A-Za-z\s\-]', '', str(name)).strip().split()
    if not tokens: return ''
    return f"{tokens[0][0].upper()}{tokens[-1].capitalize()}"

def load_all(cfg):
    P = cfg['paths']

    # Core inputs
    props   = _read_csv(P['props'])
    pf      = _read_csv(P['players'])
    tf      = _read_csv(P['teams'])
    cov     = _read_csv(P['coverage'])
    lines   = _read_csv(P['lines'])

    # Sharp/team extras (optional)
    sharp_off       = _read_csv(P.get('sharp_off',''))
    sharp_def       = _read_csv(P.get('sharp_def',''))
    sharp_def_tend  = _read_csv(P.get('sharp_def_tend',''))
    sharp_team_form = _read_csv(P.get('sharp_team_form',''))

    # Advanced features (optional)
    wx          = _read_csv(P.get('weather',''))
    qb_mob      = _read_csv(P.get('qb_mobility',''))
    wr_exp      = _read_csv(P.get('wr_cb_exposure',''))
    proe        = _read_csv(P.get('proe_splits',''))

    # Normalize teams
    for df in [tf, sharp_off, sharp_def, sharp_def_tend, sharp_team_form, lines, proe]:
        if 'team' in df.columns: df['team'] = normalize_team(df['team'])
    if 'offense_team' in cov.columns: cov['offense_team'] = normalize_team(cov['offense_team'])
    if 'defense_team' in cov.columns: cov['defense_team'] = normalize_team(cov['defense_team'])
    if 'team' in pf.columns: pf['team'] = normalize_team(pf['team'])
    if 'offense_team' in lines.columns: lines['offense_team'] = normalize_team(lines['offense_team'])
    if 'defense_team' in lines.columns: lines['defense_team'] = normalize_team(lines['defense_team'])

    # Opponent schedule map from coverage (offense_team -> defense_team)
    schedule = {}
    if not cov.empty and {'offense_team','defense_team'}.issubset(cov.columns):
        schedule = dict(zip(cov['offense_team'], cov['defense_team']))

    # Merge props (FanDuel only) → player_form
    if not props.empty:
        props_fd = props[props['book'].astype(str).str.lower().eq('fanduel')].copy() if 'book' in props.columns else props.copy()
        pf['pf_key'] = pf['player']
        props_fd['pf_key'] = props_fd['player'].map(pf_compact)
        merged = props_fd.merge(pf, on='pf_key', how='left', suffixes=('','_pf'))

        team_series = merged.get('team', merged.get('team_pf', pd.Series('', index=merged.index)))
        merged['team'] = normalize_team(team_series.fillna(''))
        merged['opponent'] = merged['team'].map(schedule)

        # join lines (optional) by team/opponent if event_id missing
        if 'event_id' not in merged.columns:
            merged['event_id'] = pd.NA
        if not lines.empty and {'offense_team','defense_team'}.issubset(lines.columns):
            # reduce lines to unique key
            gl = lines.drop_duplicates(subset=['offense_team','defense_team'])
            merged = merged.merge(gl, left_on=['team','opponent'], right_on=['offense_team','defense_team'], how='left', suffixes=('','_gl'))
    else:
        merged = pd.DataFrame()

    # Build unified team frame
    team_frame = pd.DataFrame({'team': pd.unique(tf['team'])}) if 'team' in tf.columns else pd.DataFrame(columns=['team'])
    for add in [tf, sharp_team_form, sharp_off, sharp_def, sharp_def_tend, proe]:
        if not add.empty and 'team' in add.columns:
            cols = ['team'] + [c for c in add.columns if c != 'team']
            team_frame = team_frame.merge(add[cols], on='team', how='left')

    # Attach advanced feature frames for later stages
    # (We keep them separate; features.build will decide how to use them)
    merged._wx = wx
    merged._qb_mob = qb_mob
    merged._wr_exp = wr_exp

    return merged, team_frame, cov, lines
