# M95B — RB Offensive × Defensive Matchup Engine

## Purpose
M95A showed a strong descriptive/pregame truth: established RBs, especially workhorses, perform materially better against demonstrably weak run defenses than against strong run defenses. M95B converts that truth into a compact, prospective A-vs-B model rather than a raw defensive feature dump.

The football equation under test is:

**RB/player profile + rushing offense + expected rushing structure** × **opponent run-defense profile** → **carries, efficiency, rushing yards and upside distribution**.

Workhorse status is treated as usage evidence, not talent.

## Frozen scientific rules
- Research only. No production coefficient or projection code is changed.
- No sportsbook/market inputs.
- Every target row is leakage-safe: player, team and defense features use only games completed before that season/week.
- 2023 → 2024 is the first forward test; 2023+2024 → 2025 is the out-of-year validation.
- The four model families are specified before looking at 2025 results; 2025 does not select features, weights or hyperparameters.
- Compact family scores are used to avoid the correlated-feature failure observed in M95A.

## Four progressive model families
1. **Role baseline** — recent carries, RB-pool share, workload history, team RB-pool/concentration and QB-rush share.
2. **Role + offense** — adds the RB's own rushing-quality/explosive profile and his team's rushing strength, rushing structure, pace/rush tendency and short-yardage/red-zone profile.
3. **Role + offense + defense** — adds compact opponent run-efficiency, explosive-vulnerability, resistance, RB-specific damage and red-zone vulnerability families.
4. **Full matchup interactions** — adds explicit A×B interactions: player efficiency × defensive efficiency weakness; player explosiveness × explosive vulnerability; role × RB-specific vulnerability; team rush strength × run-defense weakness; team structure × run-resistance weakness; short/red-zone offense × short/red-zone defense; plus directional and shotgun matchup features where available.

## Offensive/player profile
M95B uses pregame information tied to the individual RB and his team, including recent carries/share, rushing production, targets, workload tail history, player PBP rushing efficiency/success/first-down/stuff/explosive rates, team rushing efficiency/success/explosive rates, rushing tendency, neutral/early-down rushing tendency, RB-vs-QB share structure, backfield concentration, and short-yardage/red-zone rushing performance.

## Defense profile
Compact opponent families are built from pregame rolling nflverse PBP and M95A RB-results-against-defense history:
- fundamental run efficiency;
- explosive-run vulnerability;
- run resistance/short-yardage;
- RB-specific historical damage;
- red-zone/goal-line vulnerability;
- directional/shotgun context for explicit matchup interactions.

## Missing/limited advanced data
M95B exports an audit instead of pretending unavailable data exist. Current frozen/free sources do not provide reliable true yards before contact, true yards after contact, missed tackles forced/allowed, run-block win rate, adjusted line yards/second-level yards, detailed run-concept charting, or a complete historical OL/DL personnel/injury strength model. Box-rate/legacy contextual fields remain research-only until provider provenance is fully certified.

## Targets and diagnostics
Regression: carries, rushing yards, rush+receiving yards, and YPC (with minimum-carry filtering for stability).

Classification: 15+/20+/25+ carries, 75+/100+ rushing yards, and 20+ yard rushing-explosive occurrence where player-PBP matching is available.

Report all-RB metrics plus 0–5, 6–10, 11–14, 15+, 20+, 25+, workhorse/strong-starter/committee/light roles, weak/middle/strong defenses, and explicit good-RB/weak-defense and strong-rush-offense/weak-defense interaction slices.

## Promotion policy
M95B cannot promote directly to production. It can only advance the compact matchup engine for later integration with the opportunity architecture if the forward tests show stable incremental signal, especially in rushing yards/efficiency, without materially damaging carry calibration or the middle workload slices.
