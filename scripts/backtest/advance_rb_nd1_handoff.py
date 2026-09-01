#!/usr/bin/env python3
from pathlib import Path

p = Path('docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md')
s = p.read_text()
s = s.replace('- Current research branch: `research-rb-market-benchmark`', '- Current research branch: `research-rb-nd1-forensic-atlas`')
start = s.index('# NEXT LEGITIMATE RB RESEARCH PATH — RB-ND1')
end = s.index('## Fresh-chat startup procedure', start)
new = r'''# Latest completed diagnostic: RB-ND1 — Forensic Failure Atlas

Full results: `docs/migrations/RB_ND1_FORENSIC_FAILURE_ATLAS_RESULTS.md`.

Authoritative:

- workflow `RB-ND1 Forensic Failure Atlas`
- run **`33503240202`**
- job **`99841197836`**
- tested SHA **`1b689322b48e7de52530bca5a9e2d7039a7a1a9b`**
- artifact **`9798563163`**
- artifact SHA256 **`d92ccd3fdc36edf523ed6da3af3c76e227e2b5a5ae8c9a49680b6661a53c7d7c`**
- execution success; fit/search `0`; sportsbook input to football model `0`; production change `0`
- disposition **`ADVANCE_RB_ND2_PREGAME_ROLE_STATE_PLAYER_SHARE_RECONSTRUCTION`**

RB-ND1 reverse-engineered all 1,393 frozen M94C 2025 RB/FB player-games with exact two-factor Shapley decompositions.

Carry decomposition (`team rush attempts × player team-rush share`):

- team-volume absolute contribution share **`39.87%`**
- **player-share/backfield-allocation share `60.13%`**

Rushing-yard decomposition (`carries × YPC`):

- opportunity **`51.51%`**
- efficiency **`48.49%`**

This confirms the overall opportunity/efficiency problem is joint, while identifying **player allocation/share as the larger opportunity subproblem**.

Largest compound error classes by share of total absolute rushing-yard error:

- `OPPORTUNITY__PLAYER_SHARE`: **`31.63%`**
- `EFFICIENCY__TEAM_VOLUME`: `18.06%`
- `EFFICIENCY__PLAYER_SHARE`: `17.54%`
- `OPPORTUNITY__TEAM_VOLUME`: `10.37%`

The three player-share-primary classes together account for roughly **55.5% of total absolute rushing-yard error**.

Special overlapping forensic flags:

- explosive shock: n `157`, MAE `47.31`, involved `25.35%` of total abs error;
- non-RB/QB rushing competition: n `329`, involved `24.75%`;
- game-script miss: n `325`, involved `23.60%`;
- role collapse: n `59`, MAE `32.59`;
- new-role initialization: n `8`, MAE `41.81`.

On the exact market-covered M96E evaluation window (W6-18, n `633`):

- M94C MAE **`25.595447`**
- M96E MAE **`25.527995`**
- archived DK/FD consensus MAE **`24.139810`**

So M96E does move M94C toward the market, but only `0.06745` yards/game on this listed-RB universe. It is too small to solve the dominant role/allocation gap.

### Critical new information/data finding

The authoritative M94C 2025 RB trace has **zero populated `role` rows and zero populated `rules_role` rows across all 1,393 RB/FB games**. Historical inputs use nflverse weekly rosters, but depth information is merged only when its source is explicitly week-tagged; unsafe date/full-season depth snapshots are intentionally skipped to prevent leakage.

Therefore the backtest has been asking historical usage/context to infer current backfield role **without a true historical pregame depth-role state**. This is a new missing-information family and strongly matches the false-high/false-low market diagnosis, Week-1 rookie/new-team failures, role-collapse cases and the 60.13% player-share carry decomposition.

Do not retrospectively use today's Ourlads depth chart. Any role reconstruction must prove pregame/timestamp integrity.

# NEXT MIGRATION — RB-ND2

Name: **RB-ND2 — Pregame Role-State / Player-Share Reconstruction**

Primary question:

> Can leakage-safe current-role information materially improve player share/backfield allocation while keeping M94C's team rush-volume layer fixed, thereby reducing ordinary/listed-RB carry and rushing-yard MAE without damaging 20+/25+ workload outcomes?

Required source families to audit/build before fitting:

1. lagged nflverse participation/snap involvement and any stable role fields;
2. lagged carries/touches, team RB share, red-zone/short-yardage usage and backfield concentration;
3. current-week weekly roster membership/status;
4. same-week historical injury/practice/game-status fields with explicit week/timing semantics and no future fill;
5. competing-RB availability/vacancy/backfield member count;
6. team-change/new-team and no-history/rookie indicators;
7. Week-1 prior-role construction from prior-season continuity plus new-player priors;
8. timestamp-safe historical depth chart only if source provenance proves pregame availability;
9. QB/non-RB rushing competition as a separate allocation input.

Initial ND2 architecture must **freeze M94C team rush attempts** and predict player share/allocation, not raw rushing yards. Evaluate resulting carries first, then translate through the frozen yard layer and benchmark actual outcomes. Sportsbook remains downstream only.

Precommitted goals must include materially improving all/listed-RB carry and rush-yard MAE, reducing false-high 0-10 carry cases, and preserving 20+/25+ workload performance. Do not reopen old M96 threshold tuning.

# RETROSPECTIVE FAMILY STOP BOUNDARY

The old `AUTONOMOUS_RB_RESEARCH_STOP` applies specifically to further exposed-sample M96C/D/E router/threshold retuning. It **does not** close RB research and it does not prohibit RB-ND2, which is a separately justified missing-information/data family discovered through the downstream market benchmark and RB-ND1 forensic audit.

A specific family should still be stopped when further variants would amount to answer-key fitting. When that happens, move only to a genuinely different evidence-backed football mechanism/data family.

'''
s = s[:start] + new + s[end:]
s = s.replace('> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first and verify the latest authoritative GitHub Actions run/artifact. Respect `AUTONOMOUS_RB_RESEARCH_STOP`: RB retrospective refinement is frozen pending genuinely prospective 2026 evidence. Preserve all modeling/validation rules and do not restart old research.', '> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first and verify the latest authoritative GitHub Actions run/artifact. Continue from `NEXT MIGRATION — RB-ND2`. The old autonomous stop applies only to exposed M96 router retuning, not the new role-state/player-share data family. Preserve all modeling/validation rules and keep sportsbook data downstream only.')
p.write_text(s)
print('updated canonical handoff to RB-ND2')
