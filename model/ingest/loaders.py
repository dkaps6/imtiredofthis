import pandas as pd, numpy as np
from pathlib import Path
import re

def _read_csv(path):
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def normalize_team(s):
    return s.astype(str).str.upper().str.strip()

def pf_compact(name):
    tokens = re.sub(r'[^A-Za-z\s\-]', '', str(name)).strip().split()
    if not tokens: return ''
    return f"{tokens[0][0].upper()}{tokens[-1].capitalize()}"

def load_all(cfg):
    paths = cfg['paths']
    props = _read_csv(paths['props']); pf = _read_csv(paths['players']); tf = _read_csv(paths['teams'])
    cov = _read_csv(paths['coverage']); lines = _read_csv(paths['lines'])
    sharp_off = _read_csv(paths.get('sharp_off','')); sharp_def = _read_csv(paths.get('sharp_def',''))
    sharp_def_tend = _read_csv(paths.get('sharp_def_tend','')); sharp_team_form = _read_csv(paths.get('sharp_team_form',''))

    if 'team' in tf.columns: tf['team'] = normalize_team(tf['team'])
    for df in [sharp_off, sharp_def, sharp_def_tend, sharp_team_form]:
        if 'team' in df.columns: df['team'] = normalize_team(df['team'])
    if 'offense_team' in cov.columns: cov['offense_team'] = normalize_team(cov['offense_team'])
    if 'defense_team' in cov.columns: cov['defense_team'] = normalize_team(cov['defense_team'])
    if 'team' in pf.columns: pf['team'] = normalize_team(pf['team'])

    schedule = {}
    if not cov.empty and 'offense_team' in cov.columns and 'defense_team' in cov.columns:
        schedule = dict(zip(cov['offense_team'], cov['defense_team']))

    if not props.empty:
        props_fd = props[props['book'].str.lower().eq('fanduel')].copy() if 'book' in props.columns else props.copy()
        pf['pf_key'] = pf['player']
        props_fd['pf_key'] = props_fd['player'].apply(pf_compact)
        merged = props_fd.merge(pf, on='pf_key', how='left', suffixes=('','_pf'))
        team_series = merged.get('team') if 'team' in merged.columns else merged.get('team_pf')
        if team_series is None: team_series = pd.Series('', index=merged.index)
        merged['team'] = normalize_team(team_series.fillna(''))
        merged['opponent'] = merged['team'].map(schedule)
    else:
        merged = pd.DataFrame()

    team_frame = pd.DataFrame({'team': pd.unique(tf['team'])}) if 'team' in tf.columns else pd.DataFrame(columns=['team'])
    for add in [tf, sharp_team_form, sharp_off, sharp_def, sharp_def_tend]:
        if not add.empty and 'team' in add.columns:
            cols = ['team'] + [c for c in add.columns if c != 'team']
            team_frame = team_frame.merge(add[cols], on='team', how='left')

    return merged, team_frame, cov, lines
