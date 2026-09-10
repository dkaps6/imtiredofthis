# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority before availability promotion:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack before availability promotion:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability production promotion and clean Full Slate verification.  
**Production availability is NOT yet merged to main.**

GitHub is canonical; chat memory is secondary. Preserve first valid scientific/integration results and every mechanical failure. Do not lower frozen gates, retune post-result, allow sportsbook to define football, or mutate R26/R22/scientific model parameters without separately frozen authority.

## Prior complete handoff authority

The complete detailed handoff immediately before this implementation checkpoint is main commit `93b974acf9563c5b875d7d08476b339fcd5549c6`, blob `0ef4a1f1443ee0327344b5d6ae7002a577dd9d0e`. It contains full R27/R27D closure, availability source/timing/candidate lineage, preserved mechanical failures, eligible-team/QB seams, 35-gate Run3 evidence, Run4A finalization, and the remaining roadmap. That history remains canonical.

## Immutable promotion authority — 35/35 PASS

Source certification evidence:
- certification branch `ops-current-player-availability-35gate-cert-v1`
- source head `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- source run/job `34461561636` / `102820358570`
- gates 1-34 artifact `10145975346`
- gates 1-34 digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- Run4A wrapper run `34463888613`
- finalized artifact `10146675272`
- finalized digest `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- passed gates `35/35`
- disposition `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- production promoted in certification result: `false`.

## Frozen production promotion

Dedicated branch: `ops-current-player-availability-production-promotion-v1`.

Frozen plan commit: `9b3241996c50f1a05d124acd370fa09f0bc07e35`.

Frozen plan file: `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_PROMOTION_V1_FROZEN_PLAN.md`, blob `2d350379377ae65b2fb094504624da0591f53f7e`.

The plan authorizes only certified current-roster/current-availability plumbing, current-role seams, Full Slate build ordering, and availability-aware current-team coverage validation. It prohibits changes to M89/M90/C2 science, M38, WR-R15, TE-R5P, P3, R26 science, R22, historical research, T-75 timing semantics, QUESTIONABLE/DOUBTFUL semantics, and sportsbook-to-football boundaries.

## CURRENT LIVE STATE — promotion implementation PARTIALLY COMPLETE, NOT LOCKED/VERIFIED

Promotion branch latest implementation head at this checkpoint: `02424422bdf7eb0d671fd14f97765da9d657aeb2`.

Certified files now copied/promoted onto the dedicated branch:
- `scripts/providers/ourlads_depth_status_v1.py` — certified source blob `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- `scripts/providers/nfl_official_inactives_v1.py` — certified source blob `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- `scripts/build/build_current_player_availability_v1.py` — certified resolver source blob `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- `scripts/build/build_reconciled_active_roles_v1.py` — certified source blob `b2d05882cada048337f1f5f8b8db8ec7f9eef001`
- `scripts/build/build_production_eligible_active_roles_v1.py` — certified source blob `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/validate_current_player_availability_timing_v1.py` — certified T-75 source blob `d67ae30ed6e837f62098671c499d05462fe9d837`
- `scripts/run_current_player_availability_candidate_prep_v1.py` — certified orchestrator source blob `1b16c8052f079ba94423b21a755ce4107f60a311`
- `scripts/run_player_form_current_roles_v1.py` — certified strict-prior/current-role wrapper source blob `0a82649801794c560ccfd0ef368b7a340ec6f38f`
- `scripts/run_rb_week1_current_roles_v1.py` — certified RB current-role wrapper source blob `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/run_team_form_context_week1_prior_v1.py` — certified Week1 prior wrapper source blob `240f62d018fa419567663cae3c96647483305361`
- `scripts/utils/current_roles_v1.py` — certified current-role seam source blob `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/utils/eligible_team_set_v1.py` — certified eligible-team helper source blob `77b591e431378ec984c51e8a032262e673d4c843`
- `scripts/operations/apply_current_availability_eligible_team_seam_v1.py` — certified full-universe/R26 transformer source blob `b64ec5ccd59728121a250433e40e77e3e1013a05`.

Implementation commits made during this checkpoint include:
- `7031875e97a2f26b0dd04438b2966409ac07685f` Ourlads status sidecar
- `659068cdc6899839b29c0950bef45d68712a8939` official inactives provider
- `a31bb7b6f31500a4dff282933d806a2f44b98b3f` availability resolver
- `cc7d39854ed7a3c970efa93d092c236400905d43` production-eligible role filter
- `3e4000679e9aeffc9ebb5d9fe64ee76b8e7ffc20` reconciled active roles
- `f7267f30098406786ada37f50bc5e74f9cfa81e5` availability prep orchestrator
- `9eb62ac836e901264f16dfcb36a9b3b9149560a7` PlayerForm current-role wrapper
- `605b0ccce533163b64f287cb58204fb4d99e391d` current-role utility
- `507c54daf4addd97d05619e365db056645785c78` RB current-role wrapper
- `1983e3258c5972232b3ad870a642cfc535f87560` Week1 TeamForm prior wrapper
- `eaaaa59533e8f6a594c7cda44f399c1f7bd153ae` eligible-team helper
- `02424422bdf7eb0d671fd14f97765da9d657aeb2` eligible-team transformer.

**Important:** this is a partial production implementation only. It is NOT yet an implementation lock, NOT yet a branch verification PASS, NOT merged to `main`, and NOT a production-complete state.

## Exact next action

1. Continue only on `ops-current-player-availability-production-promotion-v1` from head `02424422bdf7eb0d671fd14f97765da9d657aeb2` (or verify newer legitimate commits first).
2. Copy the remaining certified QB C2 current-output seam transformers from the immutable 35-gate lineage, preserving the separate 32-team QB state-context source-integrity guard.
3. Verify all promoted files against their certified source blobs; if any promoted blob differs, resolve before locking. Do not silently accept hand-copied drift.
4. Wire `.github/workflows/full-slate.yml` to the frozen order: schedule -> current roster/status -> injuries -> official inactives -> T-75 certification -> availability/current eligible roles -> TeamForm/QB context -> PlayerForm -> model stack -> RB P3 -> QB C2 -> M38/TE-R5P/WR-R15/R22/R26/outer P3 -> sportsbook downstream only.
5. Set `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` for current-role consumers; unavailable/withheld players/games may never be resurrected by odds matching.
6. Apply the exact certified eligible-team/QB current-output seams before pricing, or implement identical semantics directly; legacy mode must retain 32 teams, explicit availability mode must require exactly the certified eligible team set.
7. Freeze an exact production implementation lock containing source blobs, workflow blob, scientific protected hashes, 35/35 authority, and prohibited-change assertions **before the first branch verification run**.
8. Execute dedicated no-odds clean-checkout Full Slate branch verification. Preserve any mechanical failure separately; semantic/scientific invariant failure means no promotion.
9. Only after complete branch verification PASS may the implementation be promoted to `main`.
10. After main promotion, execute a clean-main no-odds Full Slate verification and preserve run/job/artifact/digest before closing availability and moving to the next roadmap item.

## Remaining roadmap after availability

1. Finish availability production promotion + branch verification + clean-main verification — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles, building on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package and prospective grading.
