#!/usr/bin/env python3
"""Regression for the frozen QB C2 availability-aware coverage seam."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from scripts.utils.eligible_team_set_v1 import validate_current_team_set

TEAMS32 = [
    'ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC',
    'LAC','LAR','LV','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SEA','SF','TB','TEN','WAS'
]
ELIGIBLE30 = [t for t in TEAMS32 if t not in {'NE','SEA'}]


def must_fail(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError('expected RuntimeError')


def main() -> int:
    old = os.environ.pop('ACTIVE_ROLES_CSV', None)
    try:
        a = validate_current_team_set(TEAMS32, label='qb regression legacy')
        assert a['mode'] == 'LEGACY_32_TEAM' and a['observed_teams'] == 32, a
        must_fail(lambda: validate_current_team_set(ELIGIBLE30, label='qb regression legacy-short'))

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'active.csv'
            pd.DataFrame({'team': ELIGIBLE30, 'player_clean_key': [f'p{i}' for i in range(30)]}).to_csv(p, index=False)
            os.environ['ACTIVE_ROLES_CSV'] = str(p)
            a = validate_current_team_set(ELIGIBLE30, label='qb regression explicit')
            assert a['mode'] == 'EXPLICIT_CURRENT_AVAILABILITY' and a['observed_teams'] == 30, a
            must_fail(lambda: validate_current_team_set(TEAMS32, label='qb regression explicit-extra'))
            must_fail(lambda: validate_current_team_set(ELIGIBLE30[:-2], label='qb regression explicit-missing'))

        print('CURRENT_AVAILABILITY_QB_C2_ELIGIBLE_TEAM_SEAM_REGRESSION_PASS')
        return 0
    finally:
        if old is not None:
            os.environ['ACTIVE_ROLES_CSV'] = old
        else:
            os.environ.pop('ACTIVE_ROLES_CSV', None)


if __name__ == '__main__':
    raise SystemExit(main())
