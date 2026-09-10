# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority before availability promotion:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack before availability promotion:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability production promotion and clean Full Slate verification.  
**Production availability is NOT yet merged to main.**

GitHub is canonical; chat memory is secondary. Preserve first valid scientific/integration results and every mechanical failure. Do not lower frozen gates, retune post-result, allow sportsbook to define football, or mutate R26/R22/scientific model parameters without separately frozen authority.

## Prior detailed authority

The complete pre-promotion handoff is main commit `93b974acf9563c5b875d7d08476b339fcd5549c6`, blob `0ef4a1f1443ee0327344b5d6ae7002a577dd9d0e`. It contains full R27/R27D closure, availability source/timing/candidate lineage, preserved mechanical failures, eligible-team/QB seam history, 35-gate Run3 evidence, Run4A finalization, and the remaining roadmap.

## Immutable 35/35 promotion authority

- certification branch: `ops-current-player-availability-35gate-cert-v1`
- source head: `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- first valid gates 1-34 run/job: `34461561636` / `102820358570`
- gates 1-34 artifact: `10145975346`
- gates 1-34 digest: `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- gate-35 evidence-only wrapper run: `34463888613`
- finalized artifact: `10146675272`
- finalized digest: `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- passed gates: `35/35`
- disposition: `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- certification result production_promoted: `false`.

## Frozen production promotion

Branch: `ops-current-player-availability-production-promotion-v1`.

Frozen promotion plan:
- commit `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- file `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_PROMOTION_V1_FROZEN_PLAN.md`
- blob `2d350379377ae65b2fb094504624da0591f53f7e`.

Frozen dynamic production-verification plan:
- commit `3a560ad4c73adfd992035d4c650e406c050ed309`
- file `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_VERIFY_V1_FROZEN_PLAN.md`
- blob `a304f4bcf977c421aef7b7eb1a960579f0c2150c`.

The separate verification plan exists because the immutable 35-gate football-stack runner legitimately hard-coded the historical certification snapshot's 30 eligible teams. Production may legitimately have 32, 30, or another even team count under the same T-75 rule. The historical runner remains immutable; the new verification-only runner may generalize only that count to the exact same-run certified eligible team set.

## Exact promoted implementation blobs

Canonical production workflow candidate:
- `.github/workflows/full-slate.yml`: `41bacd4b756c32279169c80e0ff603a4780b7bee`
- workflow-wiring commit: `753a301881361a71c25a9b3850eea8adf65fee25`.

Availability/current-role plumbing is restored byte-for-byte to certified source blobs:
- Ourlads status provider `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- NFL official inactives provider `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- availability resolver `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- T-75 timing validator `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active roles builder `b2d05882cada048337f1f5f8b8db8ec7f9eef001`
- production-eligible roles filter `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- availability prep orchestrator `1b16c8052f079ba94423b21a755ce4107f60a311`
- PlayerForm current-role/strict-prior wrapper `0a82649801794c560ccfd0ef368b7a340ec6f38f`
- RB P3 current-role wrapper `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- Week1 TeamForm strict-prior wrapper `240f62d018fa419567663cae3c96647483305361`
- current-role seam `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- eligible-team helper `77b591e431378ec984c51e8a032262e673d4c843`
- full-universe/R26 coverage transformer `b64ec5ccd59728121a250433e40e77e3e1013a05`
- QB C2 starter coverage transformer `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- QB C2 primary coverage transformer `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`
- immutable historical no-odds football-stack runner `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`.

Integrity note: initial hand-copies of several files were semantically equivalent but formatting-compacted, producing different Git blobs. This was caught **before implementation lock and before any production verification run**. The affected files were restored to exact certified blobs; no such drift is accepted by the lock.

New verification-only implementation:
- `scripts/operations/run_current_availability_production_verify_v1.py`
- blob `0a84913d850b44504271fba1985f053b2838b970`
- implementation commit `7fb0349587a6ad95c3486321edf6b8f751c2c803`.

Production verification launcher:
- `.github/workflows/current-player-availability-production-verify-v1.yml`
- blob `8f1bd9d931cf588574cd45c0879708092487fa39`
- final launcher commit `cfb2cb1d7e917d82affecfb0bcf2da50b4cf60a9`.

## Canonical Full Slate wiring now on promotion branch

The candidate `full-slate.yml` now implements the frozen order:
1. raw Ourlads roles;
2. authoritative team-week map;
3. weather/injury acquisition;
4. Ourlads status + official NFL inactives + T-75 game certification + availability reconciliation + production-eligible roles **before opportunity**;
5. TeamForm (Week1 strict-prior wrapper when applicable) and complete 32-team QB promoted context;
6. Coverage/PBP enrichment;
7. PlayerForm from explicit production-eligible current roles with strict-prior publication;
8. canonical football model stack;
9. RB P3 from explicit current roles;
10. complete 32-team QB C2 state-context source build;
11. certified current-output coverage seams for full-universe/R26 and QB C2;
12. only then, when requested, sportsbook acquisition/matching/pricing.

`ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` is the current-role authority. Sportsbook matching uses the eligible roles and priced output is explicitly forbidden from resurrecting withheld teams. The complete QB state-context source remains exactly 32 teams; only current-output coverage follows the certified eligible team set.

## FORMAL IMPLEMENTATION LOCK + FIRST PRODUCTION VERIFICATION — LIVE

Implementation lock:
- file `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_PROMOTION_V1_IMPLEMENTATION_LOCK.md`
- lock/head commit `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`.

The lock pins the 35/35 authority, both frozen plans, the canonical Full Slate blob, all exact certified provider/helper/wrapper/transformer blobs, the immutable historical runner, and the dynamic production-verification blob. It prohibits model-science/history/artifact changes.

First locked production branch verification:
- workflow `Current Player Availability Production Verify V1`
- Run `34470549704`
- Job `102849180219`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- current state at handoff write: **in progress / installing dependencies**
- no PASS/FAIL yet
- no production promotion yet.

The launcher is designed to verify all pinned blobs and the exact 35/35 digest, dispatch the promotion branch's canonical `full-slate.yml` with `FETCH_LIVE_ODDS=false`, wait for that clean Full Slate run, download its exact artifact, verify the complete 32-team QB state-context source, apply only the certified current-output seams in the verifier checkout, execute the dynamic no-odds football verification, run strict repo/2026 readiness audits, and upload immutable result lineage.

## Exact next action

1. Inspect Run `34470549704` / Job `102849180219` first.
2. If the launcher fails before semantic verification, preserve it as `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`; diagnose the exact step and freeze only the minimum mechanical repair before retrying.
3. If it dispatches a canonical Full Slate child run, record that child run/job/artifact/digest and inspect its failure/success independently.
4. A semantic/scientific invariant failure means NO promotion. Do not lower or reinterpret the frozen boundary.
5. Only a final `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION` result may authorize promotion to `main`.
6. After branch PASS, preserve exact promotion commit lineage, promote to `main`, and execute a clean-main no-odds Full Slate verification before declaring availability production-complete.
7. Once availability closes, move automatically to the next documented roadmap item.

## Remaining roadmap after availability

1. Finish availability branch verification + main promotion + clean-main verification — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles, building on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package and prospective grading.
