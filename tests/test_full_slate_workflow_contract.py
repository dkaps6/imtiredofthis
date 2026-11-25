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
    assert 'REUSE_PAID_ODDS_RUN_ID' in text
    assert 'odds_api_refetched' in text
    assert 'PRESERVED_REPLAY' in text
    assert 'fetch_live_odds and reuse_paid_odds_run_id are mutually exclusive' in text


def test_full_slate_main_push_defaults_to_no_credit_mode():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    assert 'push:\n    branches: [main]' in text
    # Since d3047ae a push may spend credits only when its head commit carries the
    # explicit authorization tag; any other push (and any dispatch left at its
    # default) still resolves to 'false'.
    expected = (
        "FETCH_LIVE_ODDS: ${{ github.event.inputs.fetch_live_odds || "
        "(github.event_name == 'push' && contains(github.event.head_commit.message, "
        "'[LIVE_ODDS_AUTHORIZED]') && 'true') || 'false' }}"
    )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith('FETCH_LIVE_ODDS: ${{')]
    assert lines == [expected], lines


def test_full_slate_regate_reuses_a_failed_gate_capture_without_refetch():
    text = Path('.github/workflows/full-slate.yml').read_text(encoding='utf-8')
    assert 'reuse_paid_odds_regate:' in text
    step = text[text.index('- name: Restore pinned previously-paid sportsbook snapshot'):]
    step = step[:step.index('- name: Resolve sportsbook availability')]
    # only inside the no-fetch replay step, pinned to the verified artifact
    assert "if: ${{ env.FETCH_LIVE_ODDS != 'true' && env.REUSE_PAID_ODDS_RUN_ID != '' }}" in step
    assert 'PINNED_PAID_ODDS_ARTIFACT_VERIFIED' in step
    assert 'LIVE_ODDS_PRESERVED_RAW_DIR=/tmp/preserved_paid_odds' in step
    assert 'reuse_paid_odds_regate is only for a run whose gate failed' in step
    assert "s['odds_api_refetched']=False" in step
    assert step.index('PINNED_PAID_ODDS_ARTIFACT_VERIFIED') < step.index('LIVE_ODDS_PRESERVED_RAW_DIR=')
