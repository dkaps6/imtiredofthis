# Current Player Availability — Production Promotion V1 Implementation Lock

Status: `RELOCKED_AFTER_FROZEN_RUN2_R22_DISPOSITION_REPAIR_BEFORE_RUN3`

## Promotion authority

- frozen production promotion plan commit: `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- promotion plan blob: `2d350379377ae65b2fb094504624da0591f53f7e`
- frozen production verification plan commit: `3a560ad4c73adfd992035d4c650e406c050ed309`
- verification plan blob: `a304f4bcf977c421aef7b7eb1a960579f0c2150c`
- 35/35 finalized artifact: `10146675272`
- 35/35 finalized digest: `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- first-valid gates 1-34 run/job: `34461561636` / `102820358570`
- first-valid gates 1-34 artifact/digest: `10145975346` / `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`.

## Preserved production-branch Run1 mechanical failure

Parent verification:
- run/job `34470549704` / `102849180219`
- evidence artifact `10149440197`
- digest `sha256:355a8eb8340bdc3281461457cef39a205ffaeedddbe0543b9a291cf1d3db38ae`.

Child Full Slate:
- run/job `34470613780` / `102849391150`
- artifact `10149436841`
- digest `sha256:62fbde8ff296d8df8fbe2c40b5fa3d04a281640281c53dea84be8d14a751f946`
- all football stages through certified availability-aware current-output seams passed
- strict repo audit failed only because the workflow no longer directly invoked `scripts/run_player_form_v2_loader.py`
- main promotion did not occur.

Run1 disposition: `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`.

## Frozen Run1 minimum repair authority

- repair plan commit: `90a297ab94b75ffbe5da7d050928d01f9da60e1d`
- repair plan blob: `9b7ffb85d6bfb0895abaca97d4d2642ae3a07288`
- scope: static audit compatibility only; no Full Slate/model/availability/T-75/sportsbook semantic change.

Repaired audit blobs:
- `scripts/utils/audit_repo.py`: `c5a465892c7f9386a1c567e26622f6835e190aa6`
- `scripts/audit_2026_production_readiness.py`: `d9c47ab0d784510722b8d8f879d53067c4de3aa0`.

The repaired audits recognize the certified current-role PlayerForm wrapper only while also proving that it delegates to `scripts.run_player_form_v2_loader`, executes `loader.main()`, resolves explicit certified current roles, republishes strict-prior history, and rejects target/future-week active-season rows. The protected legacy loader remains a material audited dependency.

## Preserved production-branch Run2 mechanical failure

Parent verification:
- run/job `34477957076` / `102873279264`
- head `944426bae1a6d9292fded5dcf41f8951e7357467`
- evidence artifact `10152500791`
- digest `sha256:0f24bbf30094aaf8e22133eb70669d7f4e29275faa0d9b29ae48c051e66d6c14`
- final conclusion `FAILURE`.

Canonical child Full Slate:
- run `34478009290`
- head `944426bae1a6d9292fded5dcf41f8951e7357467`
- artifact `10152451377`
- digest `sha256:dd7a2cace7a5ec014ad020d9959238b7896e372ad3b9c748c14e58ec077fdeb3`
- conclusion `SUCCESS`.

The Run1 audit-compatibility repair therefore succeeded. Run2 then failed only in the dynamic verifier because its accepted R22 disposition literals omitted the canonical production adapter disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`. The child R22 audit itself reported `integration_valid=true`, zero sportsbook inputs, zero current/future outcomes, all R22 gates true, and `max_mean_delta=3.552713678800501e-15`.

Run2 disposition: `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`.

## Frozen Run2 minimum repair authority

- repair plan commit: `538c984eb5c06a72e8b4423a65b564ff6af8d7f2`
- repair plan blob: `e5a0eb9495b2d154320fb448d5f25b33fed20ae2`
- repair implementation commit: `12dcd9a0c33b1c85f60c7a19e98de8d527c90927`
- scope: add only the canonical current R22 production-adapter PASS disposition to the existing accepted-disposition check; retain the exact mean-neutrality threshold and all other invariants.

## Exact implementation blobs for Run3

Canonical production workflow candidate:
- `.github/workflows/full-slate.yml`: `41bacd4b756c32279169c80e0ff603a4780b7bee`.

Production verification launcher:
- `.github/workflows/current-player-availability-production-verify-v1.yml`: `27e53abdab162f8bb6f66064a3bdceb0f086136a`
- launcher update commit: `488e682432db407400f01eaf15e9ee8647a1a82c`
- pins both mechanical repair plans and the repaired dynamic verifier.

Availability and current-role plumbing:
- `scripts/providers/ourlads_depth_status_v1.py`: `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- `scripts/providers/nfl_official_inactives_v1.py`: `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- `scripts/build/build_current_player_availability_v1.py`: `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- `scripts/validate_current_player_availability_timing_v1.py`: `d67ae30ed6e837f62098671c499d05462fe9d837`
- `scripts/build/build_reconciled_active_roles_v1.py`: `b2d05882cada048337f1f5f8b8db8ec7f9eef001`
- `scripts/build/build_production_eligible_active_roles_v1.py`: `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/run_current_player_availability_candidate_prep_v1.py`: `1b16c8052f079ba94423b21a755ce4107f60a311`
- `scripts/run_player_form_current_roles_v1.py`: `0a82649801794c560ccfd0ef368b7a340ec6f38f`
- `scripts/run_rb_week1_current_roles_v1.py`: `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/run_team_form_context_week1_prior_v1.py`: `240f62d018fa419567663cae3c96647483305361`
- `scripts/utils/current_roles_v1.py`: `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/utils/eligible_team_set_v1.py`: `77b591e431378ec984c51e8a032262e673d4c843`.

Certified current-output seam transformers:
- full-universe/R26 seam: `b64ec5ccd59728121a250433e40e77e3e1013a05`
- QB C2 starter seam: `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- QB C2 primary seam: `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`.

Verification-only runners:
- immutable historical 35-gate football-stack runner: `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`
- repaired dynamic production verification runner: `1d4a1be7c1bed0d2ae5d8bf8fd1f70f9ae3221fe`.

Strict audit authorities remain:
- `scripts/utils/audit_repo.py`: `c5a465892c7f9386a1c567e26622f6835e190aa6`
- `scripts/audit_2026_production_readiness.py`: `d9c47ab0d784510722b8d8f879d53067c4de3aa0`.

## Locked production semantics

- availability resolves before any current player opportunity;
- raw Ourlads provider artifact remains audit evidence and is not overwritten;
- `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` is the explicit current-role authority for PlayerForm/RB/current-output consumers;
- TeamForm Week 1 uses only the certified strict-prior prior-season wrapper;
- complete QB C2 state-context source integrity remains exactly 32 teams;
- only current-output coverage guards become availability-aware and require the exact certified eligible-team set;
- sportsbook acquisition/matching is downstream of football eligibility and may not resurrect withheld games or definitive-unavailable players;
- no sportsbook input defines carries, targets, receptions, passing opportunity, roster eligibility, or player roles;
- no M89/M90/C2 science, M38, WR-R15, TE-R5P, P3, R26 science, R22 tail science, model artifact, or historical research result changes are authorized.

## Run3 authorization

Creation of this relock authorizes exactly the repaired dedicated production-branch verification launcher. It must dispatch the branch's exact canonical Full Slate with live odds disabled, require that clean child run to succeed, stage its immutable outputs, execute the separately frozen dynamic production verifier, run both repaired strict audits, and upload immutable verification evidence.

The already-played NE-SEA Week 1 game is expected to remain `KICKED_OFF_LOCKED` and withheld from current production eligibility. Sportsbook remains disabled for this verification.

Only `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION` authorizes a subsequent promotion to `main`. Any new mechanical failure must be preserved and repaired under a separately frozen minimum repair. Any semantic/scientific invariant failure means no promotion.