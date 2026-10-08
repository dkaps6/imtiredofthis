from pathlib import Path


def test_full_slate_uses_validated_2026_production_wiring():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    required = [
        'scripts/run_sharpfootball_v2.py',
        'scripts/run_team_form_context.py --season "${SEASON}" --box-backfill-prev',
        'scripts/run_qb_promoted_context.py',
        'scripts/run_live_odds_gate.py',
        'data/player_identity_validation.csv',
        'data/provider_readiness_v3.csv',
        'data/team_context_v3.csv',
        'scripts/audit_2026_production_readiness.py --strict',
    ]
    missing = [needle for needle in required if needle not in text]
    assert not missing, f'canonical Full Slate wiring regressed: {missing}'


def test_full_slate_live_pricing_requires_active_slate_odds_gate():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    live_guard = "steps.live_odds.outputs.available"
    sportsbook_guard = "steps.sportsbook.outputs.available == 'true'"
    assert live_guard in text
    assert text.count(sportsbook_guard) >= 6
    assert 'Resolve sportsbook availability' in text
    assert 'No current active-slate player prop markets are posted' in text


def test_full_slate_preserved_replay_is_no_credit_and_noncurrent():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    assert 'Restore pinned previously-paid sportsbook snapshot' in text
    assert 'Restore acquisition-time pregame football locks for preserved replay' in text
    assert 'scripts/operations/restore_preserved_pregame_game_lock_v1.py' in text
    assert text.index('Restore acquisition-time pregame football locks for preserved replay') < text.index('Build PlayerForm from production-eligible current roles')
    assert 'REUSE_PAID_ODDS_RUN_ID' in text
    assert 'odds_api_refetched' in text
    assert 'PRESERVED_REPLAY' in text
    assert 'fetch_live_odds and reuse_paid_odds_run_id are mutually exclusive' in text


def test_full_slate_main_push_defaults_to_no_credit_mode():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    assert 'push:\n    branches: [main]' in text
    assert "FETCH_LIVE_ODDS: ${{ github.event.inputs.fetch_live_odds || 'false' }}" in text
