# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority before availability promotion:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Pre-promotion production stack:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier.  
**Active lane:** current roster / late-week availability production promotion.  
**IMPORTANT: availability is NOT yet promoted to `main`.**

GitHub is canonical; chat memory is secondary. Preserve first valid science/integration results and all mechanical failures. No post-result threshold changes, no sportsbook-defined football, no target/future leakage, and no R26/R22/model-science mutation without separately frozen authority.

## Historical detailed handoff authority

For full R27/R27D history, source audits, candidate failures, 35-gate fixture history, QB/current-team seam history, and all older run artifacts, read main commit `93b974acf9563c5b875d7d08476b339fcd5549c6`, blob `0ef4a1f1443ee0327344b5d6ae7002a577dd9d0e`.

## Immutable availability promotion authority

- certification branch: `ops-current-player-availability-35gate-cert-v1`
- first valid gates 1-34 run/job: `34461561636` / `102820358570`
- gates 1-34 artifact/digest: `10145975346` / `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- gate-35 evidence-only wrapper run: `34463888613`
- final 35/35 artifact/digest: `10146675272` / `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- disposition: `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- production promoted in certification result: `false`.

## Frozen production promotion

Branch: `ops-current-player-availability-production-promotion-v1`.

Promotion plan:
- commit `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- blob `2d350379377ae65b2fb094504624da0591f53f7e`.

Dynamic production-verification plan:
- commit `3a560ad4c73adfd992035d4c650e406c050ed309`
- blob `a304f4bcf977c421aef7b7eb1a960579f0c2150c`.

The dynamic verification plan exists because the immutable 35-gate runner hard-coded the historical certification snapshot's 30 eligible teams. Production eligibility can legitimately be 32, 30, or another even count under T-75. The immutable runner remains unchanged; only a separately frozen verification-only adapter may derive the same-run certified eligible-team count dynamically.

## Exact production candidate implementation

Canonical candidate Full Slate:
- `.github/workflows/full-slate.yml`
- wiring commit `753a301881361a71c25a9b3850eea8adf65fee25`
- blob `41bacd4b756c32279169c80e0ff603a4780b7bee`.

Exact certified blobs promoted:
- Ourlads status `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- official NFL inactives `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- availability resolver `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- T-75 validator `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active roles `b2d05882cada048337f1f5f8b8db8ec7f9eef001`
- production-eligible roles filter `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- availability prep orchestrator `1b16c8052f079ba94423b21a755ce4107f60a311`
- PlayerForm current-role/strict-prior wrapper `0a82649801794c560ccfd0ef368b7a340ec6f38f`
- RB P3 current-role wrapper `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- Week1 TeamForm prior wrapper `240f62d018fa419567663cae3c96647483305361`
- current-role seam `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- eligible-team helper `77b591e431378ec984c51e8a032262e673d4c843`
- full-universe/R26 current-team transformer `b64ec5ccd59728121a250433e40e77e3e1013a05`
- QB C2 starter transformer `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- QB C2 primary transformer `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`
- immutable historical football-stack verifier `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`.

Initial hand-copies of several files produced formatting-only blob drift. This was caught BEFORE lock/run and every affected certified file was restored byte-for-byte. Do not accept semantically equivalent but blob-different certified files.

Dynamic production verifier:
- `scripts/operations/run_current_availability_production_verify_v1.py`
- commit `7fb0349587a6ad95c3486321edf6b8f751c2c803`
- blob `0a84913d850b44504271fba1985f053b2838b970`.

Verification launcher:
- `.github/workflows/current-player-availability-production-verify-v1.yml`
- final pre-lock commit `cfb2cb1d7e917d82affecfb0bcf2da50b4cf60a9`
- blob `8f1bd9d931cf588574cd45c0879708092487fa39`.

## Locked production semantics

Canonical Full Slate now resolves availability before opportunity, uses `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` for current-role consumers, uses the strict-prior Week1 wrappers, keeps QB C2 state-context source integrity at exactly 32 teams, changes only current-output coverage to the exact certified eligible-team set, and moves sportsbook acquisition/matching after football eligibility. Odds cannot resurrect a withheld team/player.

No M89/M90/C2 science, M38, WR-R15, TE-R5P, P3, R26 science, R22, model artifact, or historical research result is authorized to change.

## FORMAL IMPLEMENTATION LOCK

- lock file: `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_PROMOTION_V1_IMPLEMENTATION_LOCK.md`
- lock/head: `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`.

This lock pins the exact 35/35 authority, both frozen plans, canonical Full Slate blob, every certified provider/helper/wrapper/transformer blob, and the dynamic verifier before execution.

## FIRST LOCKED PRODUCTION VERIFICATION — LIVE

Parent verification launcher:
- Run `34470549704`
- Job `102849180219`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- frozen implementation verification step: **PASS**
- canonical Full Slate dispatch step: **PASS**
- parent currently waiting for child Full Slate.

Canonical child Full Slate:
- Run `34470613780`
- Job `102849391150`
- run number `560`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- `FETCH_LIVE_ODDS=false`
- current state at this handoff: **IN PROGRESS**
- already passed: setup, dependency installation, season context, no-odds boundary, raw Ourlads roles, authoritative team-week map, weather/injuries, **availability before opportunity**, strict-prior Week1 TeamForm, promoted QB M89/M90 context.
- currently executing: Coverage v2 matchup intelligence.
- no production PASS/FAIL has been declared yet.

If child Full Slate succeeds, parent Run `34470549704` will download its exact `run_34470613780` artifact, independently verify 32-team QB state-context source integrity, apply only the certified current-output seam transforms in the verifier checkout, execute `run_current_availability_production_verify_v1.py`, run strict repo/2026 readiness audits, and upload immutable production-verification lineage.

## Exact next action

1. Inspect child Run `34470613780` / Job `102849391150` first, then parent `34470549704` / `102849180219`.
2. Preserve any failure exactly. If the failure is plumbing/mechanical, label it `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`, diagnose it, freeze the minimum repair before retrying. Do not change frozen semantic gates.
3. A semantic/scientific invariant failure means NO promotion.
4. Only `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION` may authorize promotion to `main`.
5. After a branch PASS, preserve run/job/artifact/digest and promotion lineage, promote the exact verified implementation to `main`, then run a clean-main no-odds Full Slate verification before closing availability.
6. After availability production closes, move automatically to the next roadmap item.

## Remaining roadmap

1. Availability branch verification + main promotion + clean-main verification — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles, building on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package and prospective grading.
