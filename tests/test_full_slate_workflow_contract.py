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
    guard = "steps.live_odds.outputs.available == 'true'"
    assert text.count(guard) >= 4
    assert 'No current active-slate player prop markets are posted' in text


def test_full_slate_main_push_defaults_to_no_credit_mode():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    assert 'push:\n    branches: [main]' in text
    assert "FETCH_LIVE_ODDS: ${{ github.event.inputs.fetch_live_odds || 'false' }}" in text
