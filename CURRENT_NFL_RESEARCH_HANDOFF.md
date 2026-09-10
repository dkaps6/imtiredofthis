# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Current operational production authority before this documentation-only handoff commit:** `3079d8ab0512c5a1304662609e3e880d6846292f`  
**Protected scientific/model authority remains:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Availability/current-role production:** **COMPLETE / PROMOTED / CLEAN-MAIN VERIFIED**  
**Master betting workbook publisher:** **COMPLETE / PROMOTED / FULL-SLATE ARTIFACT VERIFIED**  
**RB receiving-yard mean research lane:** **CLOSED / NO NEW MEAN INTEGRATION AUTHORIZED**  
**Next active model-development lane:** **QB opportunity / efficiency audit before any new frozen experiment**

GitHub is canonical; chat memory is secondary. Preserve first valid scientific/integration results and every mechanical failure. Do not post-hoc alter frozen gates, do not use sportsbook information to define football projections or player roles, do not leak target/future outcomes into pregame features, and do not silently mutate protected production science.

## Historical detailed handoff authority

The immediately prior detailed handoff is preserved at main commit `ba75716a33ff378f15508e8157213b1d146113f5`, handoff blob `c5a560615a38b33e82e0cadc355205e2b96223e7`.

That prior handoff contains the full detailed lineage for:
- R27/R27B/R27C/R27C2/R27D RB receiving-yard work;
- current-player availability source/timing research;
- 35/35 availability integration certification;
- availability production-verification Run1/Run2 mechanical failures and Run3 PASS;
- PR #513 availability promotion;
- clean-main verification;
- post-promotion P3 test-contract repair via PR #514;
- final green Repo CI and no-odds Full Slate before the workbook publisher.

For deeper pre-promotion history, preserve earlier referenced handoff commits/blobs from that file. Do not delete or rewrite historical result records.

## Current production stack

Football projection stack remains:
- QB mean: M89/M90
- QB distribution: mean-neutral C2
- WR: M38 WR1 + WR-R15 WR2+
- TE: TE-R5P
- RB rushing: P3
- RB receptions: R26
- RB receiving-yard mean: existing production YPT/mean path
- RB receiving-yard distribution/tails: R22 using frozen R19 assets, exactly mean-preserving
- current player availability/current roles: promoted availability-first production plumbing
- sportsbook: downstream only after football eligibility/projection generation

Operational production authority now additionally includes the automatic downstream workbook publisher, but the scientific/model authority remains unchanged.

## Availability/current-role production — CLOSED / COMPLETE

Availability is now canonical production behavior. Core locked semantics:
- resolve current availability before opportunity;
- `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` is current eligible role authority;
- definite unavailable players cannot survive into eligible roles, PlayerForm, or simulation arrays;
- official NFL inactive authority applies inside the frozen T-75 window;
- kicked-off games become `KICKED_OFF_LOCKED` and are withheld from current betting/output eligibility;
- sportsbook acquisition occurs only after football eligibility and cannot resurrect withheld games or unavailable players;
- sportsbook inputs cannot define carries, targets, receptions, passing opportunity, current roster eligibility, or player hierarchy.

Availability production verification and promotion were completed before the workbook work. See prior handoff blob `c5a560615a38b33e82e0cadc355205e2b96223e7` for exact Run1/Run2/Run3 lineage and PR #513/#514 details.

## RB receiving-yard mean lane — CLOSED / NO INTEGRATION

Latest valid R27D strict-prior YACOE residual result remains:
- branch `research-rb-r27d-yacoe-residual-v1`
- first valid run/job `34436178615` / `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity `18/18 PASS`
- scientific gates `4/13 PASS`

Conclusion remains: strict-prior relative xYAC/YACOE persistence did not provide a reliable vacancy-RB1 receiving-yard mean correction. No R27D integration, retuning, or post-hoc router is authorized. R26 receptions and R22 tails remain protected. Future RB receiving-mean work requires genuinely new pregame information/mechanism.

## MASTER BETTING WORKBOOK — PROMOTED AUTOMATIC FULL-SLATE OUTPUT

### Goal

Every Full Slate run must automatically publish one human-facing downstream betting-model workbook while preserving the football-first architecture. The workbook is a reporting/pricing product only; it must never feed Vegas lines, prices, probabilities, or PLAY/LEAN decisions back into football projections.

### Production files

The production integration adds:
- `scripts/build_master_betting_workbook_v1.py` — stable Full Slate entrypoint
- `scripts/master_betting_workbook_core_v2.py` — hardened workbook/identity/position publisher
- `.github/workflows/full-slate.yml` — additive workbook-build step

Workbook output path:
- `outputs/NFL_BETTING_MODEL_MASTER.xlsx`

Existing Full Slate artifact upload already captures `outputs/**`, therefore the workbook is included automatically inside the normal `run_<github.run_id>` artifact package.

### Integration lineage

Integration branch:
- `ops-master-betting-workbook-artifact-v1`

Relevant branch commits:
- initial builder commit `50f142261651858eba0070f4ede06794201904a3`
- Full Slate wiring commit `487d4c4608705aa46338dfbeff97f3cb5f7e952b`
- hardened position-resolution core commit `33d1bd7cd5d518d753be5f72b0f627a04c9bc0cd`
- final branch head / stable-entrypoint routing commit `f9d1d333aaca1a4223e592dd1505eb689ca015f1`

PR:
- PR `#515`
- title `Publish master betting workbook with every Full Slate run`
- changed files: exactly `.github/workflows/full-slate.yml`, `scripts/build_master_betting_workbook_v1.py`, and `scripts/master_betting_workbook_core_v2.py`
- reporting/integration only; no QB/RB/WR/TE projection science, availability science, R22/R26/P3 logic, or sportsbook ordering changed.

PR Repo CI:
- Run `34505333837`
- head `f9d1d333aaca1a4223e592dd1505eb689ca015f1`
- conclusion `SUCCESS`.

Merged production commit:
- `3079d8ab0512c5a1304662609e3e880d6846292f`
- merge message: `Publish master betting workbook with every Full Slate run`

### First production Full Slate proof of automatic workbook publication

Production main Full Slate:
- Run `34505435434`
- Job `102966347569`
- Full Slate run number `569`
- head `3079d8ab0512c5a1304662609e3e880d6846292f`
- conclusion `SUCCESS`
- artifact `10163711046`
- artifact name `run_34505435434`
- artifact digest `sha256:e272ea4c1398bae9c9fa617566cfd17565fbd19efeacff01c08c7f193b16214b`

The exact artifact was downloaded and physically verified to contain:
- `outputs/NFL_BETTING_MODEL_MASTER.xlsx`

This was a no-live-odds run. The generated workbook correctly reported:
- workbook disposition `MASTER_BETTING_WORKBOOK_PUBLISHED`
- current availability rows `468`
- game certification rows `16`
- pricing status `NO_LIVE_ODDS_REQUESTED`
- sportsbook downstream only `true`
- priced offers `0`
- unresolved position rows `0`
- betting recommendations suppressed because live odds were intentionally not requested.

Therefore automatic publication is **PRODUCTION COMPLETE**.

## Workbook operating contract

The workbook is intended to become the one operational human-facing betting board for each Full Slate run.

Expected workbook content includes:
- Dashboard
- Master Betting Board
- Best Snapshot/Current Edges
- Game Certification
- Availability & Roles
- Market Science/lineage
- run/commit metadata

When `FETCH_LIVE_ODDS=false`:
- the workbook still publishes;
- football outputs/availability remain current;
- no sportsbook rows are invented;
- betting decisions are suppressed.

When `FETCH_LIVE_ODDS=true` and active eligible markets exist:
- lines/offers enter only after football projections/eligibility are established;
- workbook can show model projection, Vegas line, odds, model probability, no-vig market probability, line difference, probability edge, EV, best side, availability/game-state gates, and PLAY/LEAN/PASS/BLOCKED display status;
- those display labels remain downstream and are not themselves scientific model inputs.

Anytime-TD remains blocked/research-only unless/until a separately dedicated TD model is scientifically certified; generic ATD rows must not be promoted as authoritative wager recommendations merely because pricing exists.

## Position / identity audit from first workbook prototype

The manually generated prototype exposed avoidable `Position=UNKNOWN` rows. Audit result:
- 26 unique players / 90 sportsbook-offer rows initially had `Position=UNKNOWN`;
- all were in the Anytime TD market;
- 23/26 were not truly unknown — they were generational-suffix identity mismatches such as `Jr.`, `II`, or `III` between sportsbook keys and current football sources;
- examples include cases where a paid snapshot key retained a suffix while Ourlads/PlayerForm identity normalization dropped it;
- three remaining snapshot-only names were absent from the current active-roster join and required football-position fallback evidence rather than guessing.

Permanent publisher fix:
- exact current player/team identity first;
- suffix-insensitive team-matched identity alias next;
- unique safe suffix alias where appropriate;
- historical football-position authority / deterministic model-market position evidence as fallback;
- otherwise explicit unresolved state and fail-closed row behavior;
- workbook records `Position Source` so position provenance is auditable.

First production-generated no-odds workbook reported `unresolved position rows = 0`.

Important WR-role distinction remains:
- Ourlads `depth_role` is not a trustworthy team-wide WR1/WR2/WR3 hierarchy because LWR/RWR/SWR lane/depth structures can create duplicate labels;
- downstream model hierarchy is separately derived by PlayerForm/current model role and must remain distinct from Ourlads depth-role display;
- sportsbook lines must never determine which player is a team's WR1/WR2/etc.;
- lack of a sportsbook prop must not define football roster eligibility.

## Week 1 operational note

Week 1 has started. NE-SEA has already kicked off and should be treated as `KICKED_OFF_LOCKED` by the current availability/game-certification system. Missing current live props for that game are expected and are not an Odds API failure.

A preserved pre-Week-1 Vegas/paid replay exists and contains earlier NE-SEA lines, but those are historical downstream evidence only. Do not substitute them for current live prices unless a separately explicit historical comparison/replay task calls for them.

Preserved paid replay reference used for workbook prototyping:
- Run `34228468564`
- Artifact `10056788145`
- artifact name `paid-full-slate-replay-v1`
- digest `sha256:eb30732faf61aaaec9cb532308dd9573712d8f025cb532cefa091b9fd777a4ac`

## R26 Week 1 prospective seal — preserve for later grading

R26Q immutable Week 1 prospective seal remains preserved:
- Run `34400524030`
- Job `102630996205`
- Artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`.

R26S frozen evaluator remains the required post-Week-1 grading path:
- plan `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock `3a706fa52f91f6584f6fd6594563239dd6ea3b53`
- canonical dry run `34411889262`
- Job `102667966181`
- Artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`

Do not recompute the seal. Rerun the exact frozen evaluator only when complete authoritative Week 1 outcomes are available.

## NEXT ACTIVE ACTION — QB OPPORTUNITY / EFFICIENCY

The next development lane is QB opportunity/efficiency, not another RB receiving-mean retry.

Before freezing any new QB candidate:
1. audit all prior QB migrations/research/result records and current production code;
2. inventory what has already been tested for attempts, dropbacks, pass rate, YPA/efficiency, sacks/pressure, scrambles, explosive passing, receiver correlations, game environment, and distribution/tails;
3. identify the exact remaining production error decomposition using strict-prior football information;
4. explicitly avoid reinventing failed or already-promoted QB work;
5. conduct source/schema availability audit for any genuinely new proposed information before freezing a candidate;
6. freeze question, mechanism, data boundary, walk-forward protocol, baseline/candidate, and gates before first scientific result;
7. sportsbook remains benchmark/downstream only.

Current conceptual QB roadmap includes:
- attempts / dropbacks / pass-rate opportunity
- yards per attempt / efficiency
- sacks and scrambles
- explosive passing / receiver interaction only where independently football-supported
- distribution/tail calibration after mean/opportunity decomposition

Do not change production simply because Week 1 has begun. Speed matters, but frozen scientific lineage and pregame integrity remain mandatory.

## Resume rule for next chat/session

Read this file first, then verify live GitHub `main`, recent Actions runs/artifacts, and the active research branch before acting. GitHub is canonical.

Current stop point after workbook completion:
- availability: production complete
- automatic master betting workbook: production complete
- RB receiving-yard mean lane: closed/no integration
- R26Q/R26S prospective grading: sealed/pending complete Week 1 outcomes
- next work: QB opportunity/efficiency anti-reinvention audit and error decomposition
