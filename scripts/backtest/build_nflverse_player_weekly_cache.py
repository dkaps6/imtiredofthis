#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd


def to_pandas(x):
    if isinstance(x, pd.DataFrame): return x
    if hasattr(x, 'to_pandas'): return x.to_pandas()
    return pd.DataFrame(x)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seasons', default='2018-2025')
    ap.add_argument('--out-dir', type=Path, default=Path('data/research_cache/nflverse_player_weekly'))
    args=ap.parse_args()
    a,b=[int(x) for x in args.seasons.split('-')]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import nflreadpy as nfl
    rows=[]
    for season in range(a,b+1):
        raw=nfl.load_player_stats(seasons=[season], summary_level='week')
        df=to_pandas(raw)
        if df.empty: raise RuntimeError(f'zero weekly player rows for {season}')
        df.columns=[str(c) for c in df.columns]
        sort=[c for c in ['season','week','recent_team','player_id','player_display_name'] if c in df.columns]
        if sort: df=df.sort_values(sort, kind='stable').reset_index(drop=True)
        p=args.out_dir/f'player_weekly_{season}.parquet'
        df.to_parquet(p, index=False)
        rows.append({'season':season,'rows':len(df),'columns':len(df.columns),'bytes':p.stat().st_size,'sha256':sha256(p),'file':str(p)})
        print(f'[player-cache] {season}: rows={len(df)} cols={len(df.columns)} bytes={p.stat().st_size}')
    manifest=pd.DataFrame(rows)
    manifest.to_csv(args.out_dir/'manifest.csv', index=False)
    print(manifest.to_string(index=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
