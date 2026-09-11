# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
2. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

The Week-1 live Full Slate mechanical incident is now **repaired and merged**. PR #523 was merged to `main` at `f84c6242da02b1804b4b9675c3a3a3e679838e10`. Post-merge Repo CI run `34656484043` and no-live Full Slate run `34656483980` both passed. The exact preserved paid artifact from run `34650067599` previously replayed through repaired Steps 29-31 with zero certification blockers and zero additional OddsAPI acquisition.

Do **not** restart the roster/event-scope repair, preserved-artifact replay, player-identity audit, rematch investigation, Knight alias investigation, downstream current-availability certification repair, or broad historical M107/M108 search. Do not spend another paid live-odds call merely to rediscover mechanical bugs.

Production model science remains frozen and unchanged by the repair. The active merged handoff contains the exact integrity verdict, paid-run lineage, replay evidence, merge SHA, and next authorized step.

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover an authoritative M108 workflow/script/run/PR proving this was a canonical repository gate. Do not invent or require an M108 test by name unless concrete GitHub lineage is later recovered.

---

## PARKED SCIENCE CHECKPOINT

The QB/WR shared-opportunity / first-down pass-propensity / public pregame-intent V1B lane remains preserved at:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Resume that lane only after confirming no newer production incident supersedes the merged checkpoint.

No production-science change is authorized by the live-repair work itself.