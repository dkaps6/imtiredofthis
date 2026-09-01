#!/usr/bin/env python3
from pathlib import Path

p = Path('docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md')
s = p.read_text()

s = s.replace('Current research branch: `research-rb-m96e-workload-risk-guard`',
              'Current research branch: `research-rb-market-benchmark`')
s = s.replace('No M91-M96C RB research has been promoted to production.',
              'No M91-M96E RB research or downstream RB market benchmark has been promoted to production.')

section = r'''
# Latest downstream benchmark: RB Market Benchmark — M94C vs 2025 archived DK/FD rushing-yard lines

Full results: `docs/migrations/RB_MARKET_BENCHMARK_RESULTS.md`.

Authoritative:

- workflow `RB Market Benchmark`
- run **`33499129109`**
- job **`99828098063`**
- tested SHA **`a26ad1a9991c2f9303d30e4f5b4cff25c3e9d30c`**
- artifact **`9796956965`**
- artifact SHA256 **`6759e7d8157ade3d4f9237e21a30feacb2507f77f03904ca85740683b7f96475`**
- execution success
- sportsbook inputs into football model `0`; football model change `0`; feature/weight/threshold search `0`

This was a downstream benchmark only. The source is the public Action Network-derived 2025 archive previously audited in M60B. Only exact full-game `rushing_yards` straight props from DraftKings/FanDuel were eligible. The archive does not preserve a trustworthy fixed pre-kick timestamp, so these rows are **archived latest / closing-like**, not a 30-minute-before-kickoff snapshot.

The first benchmark run `33498879907` was mechanically green but scientifically unusable because a broad source filter admitted combo/milestone markets and the archive's abbreviated names did not match the first full-name join. Run #2 repaired only exact-market filtering and identity mechanics; no model or benchmark metric logic changed.

Exact common market-covered RB player-games: **`899`**.

- M94C MAE **`25.515051`**, RMSE `34.364907`, bias `-0.579911`, corr `.453546`.
- Vegas DK/FD consensus MAE **`23.701891`**, RMSE `32.493543`, bias `-4.327030`, corr `.529751`.
- Vegas consensus therefore beat M94C by **`1.813160` MAE yards** on the exact common sample.
- Head-to-head: M94C closer `403`, market closer `496`; model closer rate `44.83%`.

The market edge is not uniform. When M94C and market were within 5 yards (`n=277`), M94C had a tiny MAE edge: `24.5481` vs `24.6390`. As disagreement widened, the market advantage grew: at `15+` yards disagreement (`n=211`), M94C MAE `31.9875` vs market `26.4716`, a **`5.5159`-yard market advantage**.

The most damaging regime is M94C materially **above** market. At M94C >=15 yards above consensus (`n=144`), M94C was closer only `36.11%`; its MAE was `33.0303` vs market `25.3924`. Large model-high disagreement is therefore a strong forensic sign of stale/overconfident workload/role state, but the market line itself must not become an upstream football feature.

Postgame actual-carry diagnosis shows Vegas did **not** solve the high-workload tail:

- actual 0-5 carries (`n=188`): M94C MAE `19.8329`, market `15.6596` — market better by `4.1733`.
- actual 6-10 (`n=232`): M94C `22.1260`, market `19.9655` — market better by `2.1605`.
- actual 20+ (`n=94`): M94C **`38.9831`**, market `39.8032` — M94C better by `0.8201`.
- actual 25+ (`n=23`): M94C **`51.0291`**, market `54.7826` — M94C better by `3.7535`.

For actual 25+ games both systems were drastically low: actual mean `123.09`, M94C mean `73.28`, market mean `69.52`. The extreme workload tail is still unsolved by both.

The benchmark identifies a **new, separately justified football-data research path** rather than permission to retune exposed M96 thresholds:

1. false-high workload suppression / pregame workload-collapse detection;
2. Week-1, rookie, new-team and new-role initialization;
3. current depth chart / transaction / practice-injury / availability timing available before kickoff;
4. coaching/backfield usage priors and potentially offensive-line availability;
5. rookie/draft/college workload priors where leakage-safe.

The benchmark is external evidence about *where* the model is missing football information. Sportsbook lines remain downstream only and may not be used as a feature, training target, ensemble input, or pregame router in the independent football model.

# NEXT LEGITIMATE RB RESEARCH PATH — RB-ND1

Name: **RB-ND1 — Pregame Role Initialization / Availability Data Audit**

Primary question:

> Does M94C lack timely football-only role/availability information—especially Week 1, rookies, new teams/roles, and abrupt workload collapses—that can explain the market benchmark's false-high/false-low regimes without using sportsbook information upstream?

Required first step is a **source/coverage audit**, not a tuned model. Inventory leakage-safe pregame historical availability for 2024-2025 (and earlier if comparable): official/current depth charts or archived depth snapshots, roster transactions/team changes, rookie/draft status, practice/injury/game-status timing, inactive/active information available before kickoff, coaching/backfield usage history, and offensive-line availability where reliable. Explicitly record timestamp semantics and coverage. Do not use postgame inactive knowledge as if it were known pregame.

Only after a source passes timing/coverage integrity should a small precommitted football-only experiment be opened. The sportsbook benchmark may define broad problem regimes for forensic analysis but **may not be a model feature/target or threshold selector**.

The historical M96 router stop remains in force: do not reopen M96F or tune M96C/D/E thresholds/features against exposed 2025 labels. RB-ND1 is a separately justified new-data-source path.
'''

if '# Latest downstream benchmark: RB Market Benchmark' not in s:
    marker = '# AUTONOMOUS_RB_RESEARCH_STOP'
    if marker in s:
        s = s.replace(marker, section + '\n' + marker, 1)
    else:
        s += '\n' + section

# Clarify the existing stop without deleting it.
old = ('Retrospective RB efficiency refinement is now frozen. The final precommitted router solved most of the high-workload safety issue but missed the predeclared global materiality gate. Further threshold/feature variants on exposed 2025 outcomes would be overfitting rather than independent evidence.')
new = (old + '\n\nThis stop applies to additional M96 carry/efficiency/router retuning on exposed historical labels. The downstream RB market benchmark subsequently established a separately justified **new-data-source audit path (RB-ND1)** focused on timely football-only role/availability information. The autonomous loop remains stopped from blindly opening more M96 variants; any RB-ND1 work must preserve the market-as-downstream-only contract and begin with source/timestamp integrity rather than model tuning.')
if old in s and 'This stop applies to additional M96 carry/efficiency/router retuning' not in s:
    s = s.replace(old, new, 1)

p.write_text(s)
print('handoff updated')
