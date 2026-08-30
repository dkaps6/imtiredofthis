#!/usr/bin/env python3
"""Diagnostic only: record every candidate inactive bullet that M78 cannot parse.

No model fitting and no gate changes. This exists solely to repair the official
inactive source parser from observed NFL.com markup/token variants rather than
loosening the M78 completeness requirement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import audit_qb_official_inactive_availability_v3 as v3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    snapshots = []
    for season in v3.SEASONS:
        for url in m78.discover_article_urls(season, snapshots):
            try:
                r = m78.request(url)
            except Exception:
                continue
            soup = BeautifulSoup(r.text, 'html.parser')
            h1 = soup.find('h1')
            title = m78.norm_space(h1.get_text(' ', strip=True) if h1 else soup.title.get_text(' ', strip=True) if soup.title else '')
            week = m78.extract_week(title)
            if week is None:
                continue
            for team, _label, ul in v3._team_section_candidates(soup):
                bullets = v3._candidate_bullets(ul)
                for text in bullets:
                    parsed = m78.parse_player_bullet(text)
                    if parsed is not None:
                        continue
                    first_token = text.split()[0] if text.split() else ''
                    rows.append({
                        'season': season,
                        'week': week,
                        'team': team,
                        'first_token': first_token,
                        'raw_bullet': text,
                        'article_title': title,
                        'article_url': r.url,
                    })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(['season','week','team','raw_bullet','article_url']).sort_values(['season','week','team','raw_bullet'])
    df.to_csv(out / 'm78_unparsed_candidate_bullets.csv', index=False)
    print('unparsed_candidate_bullets=', len(df))
    if len(df):
        print(df[['season','week','team','first_token','raw_bullet']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
