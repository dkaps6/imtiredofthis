# Active-Roster Receiver Room State V1 — Diagnostic Result

Date: 2026-09-25

Disposition: **ACTIVE_ROSTER_RECEIVER_ROOM_STATE_V1_NOT_SUPPORTED**

This was discovery-only. No candidate was scored or qualified.

## Authority

- branch: `research-active-roster-receiver-room-state-v1`
- authoritative run: `36172146432`
- job: `108194031342`
- tested head: `7ea29fec84555465d90c0a470aaf98fd559b031b`
- artifact: `10880536886`
- digest: `sha256:10aca2c157a6269ada4487787b1c5e671a5df2a51aec34d3854161b4a1ee54de`
- discovery seasons: 2022-2023
- confirmation seasons inspected: none
- parameters fit: `0`
- candidate variants scored: `0`
- sportsbook inputs: `0`

## Pooled discovery

Macro room-composition MAE:
`0.081330 -> 0.080858`

Macro oracle room-target MAE:
`2.450679 -> 2.442002`

By room:
- WR composition MAE: `0.092917 -> 0.093338` — worse
- WR oracle target MAE: `2.818050 -> 2.836944` — worse
- TE composition MAE: `0.079148 -> 0.077627`
- TE oracle target MAE: `2.368981 -> 2.329353`
- RB_FB composition MAE: `0.071924 -> 0.071610`
- RB_FB oracle target MAE: `2.165004 -> 2.159709`

Frozen WR protection gates failed in both discovery seasons.

## Coverage

State coverage was substantial but incomplete:
- WR row coverage ~65.6%; entitlement-weighted ~70.5%
- TE row coverage ~66.6%; entitlement-weighted ~68.1%
- RB_FB row coverage ~67.0%; entitlement-weighted ~70.0%

Complete room-state coverage was particularly sparse for WR rooms:
- WR team-games full state: 62
- WR team-games with fallback: 960

## WR1 interaction

- positive WR1 state gap rate: ~94.4%
- mean WR1 state gap: ~0.0481
- correlation between WR1 state gap and needed WR-room mass correction: ~0.0666

So the prior WR1 absolute-share state signal does not become a reliable whole-room composition rule.

## Frozen gate result

PASS:
- pooled macro composition improves
- pooled TE composition nonworse
- pooled RB_FB composition nonworse
- pooled macro oracle targets improves
- sportsbook inputs zero
- parameters fit zero
- candidate variants zero

FAIL:
- pooled WR composition improves
- WR composition improves in 2022
- WR composition improves in 2023
- pooled WR oracle targets improves

## Interpretation

This closes the exact active-roster receiver room-state formulation.

Do not rescue it with:
- WR-only routing;
- coverage thresholds;
- player-state thresholds;
- current-games thresholds;
- position-specific weighting;
- 2024-2025 outcome inspection.

The result reinforces:
- WR1 current-state information is real at an absolute/team-share level;
- it is not sufficient as a room-composition correction;
- room opportunity should continue through independently justified room-volume research.

The current room-volume lead is `RECEIVER_ROOM_TARGETS_PER_PLAY_V1`, not this state formulation.
