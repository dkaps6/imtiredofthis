# Week 5 (Oct 8, 2026) Full Slate — Production Recovery Result

Status: **FULL SLATE PRICED / WORKFLOW SUCCESS / DATA QUALITY NOT CERTIFIED**

This is an operations handoff, not a new scientific model fit or promotion.

## Canonical football execution

- Repository `dkaps6/imtiredofthis`.
- Canonical production entry point `.github/workflows/full-slate.yml`.
- 2026 Week 5: 15 games / 30 scheduled teams; bye: CAR, KC.
- PR #674 fixed QB C2 source context and original protected availability seams.
- Source head for successful no-odds validation `66c22026880178e60d8011ea2400127249be83ac`.
- `37852305029`: canonical no-live-odds Full Slate **SUCCESS**.
- One-shot paid acquisition dispatcher `37852477816` **SUCCESS**.
- Single paid live-source Full Slate `37852811339`: football, C2, availability and sportsbook acquisition passed; failed later on a bye-week source-quality hard-coded 32-team guard.
- Preserved exact paid-source artifact `11582178322` name `run_37852811339`, digest `sha256:4948e08003fcab0f519b35bd4b23d8c895554e4faced993238575acdc5d2e765`.
- PR #675: limit late-game C2 active-team context to certified **complete game** subset; validate full schedule separately; accept raw Ourlads league roster on bye weeks only if extra teams are authoritative bye clubs. Merged.
- PR #676: replace 32-team coverage quality count with exact scheduled-team identities, unique nonblank keys and coverage flags, plus negative tests. Merged.
- No QB/WR/TE/RB model science, efficiency, weights, or C2 selector parameters were changed by these production repairs.

## Final offline recovery

**SUCCESS**: [GitHub Actions `37854558239`](https://github.com/dkaps6/imtiredofthis/actions/runs/37854558239)

- Source branch: `ops/week5-20261008-paid-artifact-offline-recovery`.
- Execution SHA: `a74976ee4775d1d833e3f428a566b13dc2f31593`.
- Archive `11582934259`, name `week5_offline_paid_recovery_37854558239`.
- Digest `sha256:b6c60b1eaddbb8e8e862c571be452d93ab40e44a468a9e82c3ccbe7fe9f51b56`.
- Artifact expiry 2027-01-06.
- Exact original paid source artifact digest verified before extraction.
- `FETCH_LIVE_ODDS=false`; **zero additional OddsAPI acquisitions** during recovery.
- Restore earlier football artifacts rather than recompute fresh current-game inputs.
- Reapply original production eligibility guards and exact QB C2 model.
- Data quality classifier executed, then frozen pricing offers and deterministic metrics produced.
- Canonical `scripts/run_pricing_with_full_roster_universe_v3.py` completed.
- Football universe: 434 modeled players, 30 teams, 15 games; pricing produced 2,954 side rows before final-board quarantine.
- `QB_C2_PRICING_LINEAGE_STAMP_CERTIFIED`; 30 certified C2 football quarterbacks, 28 selected C2, no QB mean/model value changes. C2 audit found 29 priced QB identities, but one non-starter-market identity was explicitly quarantined before stamping.
- `data/manual_final_board_quarantine.csv` has one **Week-5-only** BAL/Tyler Huntley record, prompted by Lamar Jackson's documented ankle injury and ongoing official Week-5 starter uncertainty. Baltimore's team depth lists Lamar QB1, while the sportsbook posted Huntley markets. The final-board-only quarantine withholds Huntley; it does NOT adjust team passes/receivers/RB or install a sportsbook-chosen starter upstream. Retain the exact documented source. Reassess after official Week-5 starter designation.
- Final-board quarantine removed **6** Huntley side rows (four pass yards among six), leaving **2,948** priced offer-side rows.
- `outputs/props_priced_clean.csv`: final priced offers, quarantined.
- `outputs/NFL_BETTING_MODEL_MASTER.xlsx`: generated master workbook.
- Repo audit and production readiness script run successfully, archival upload successful.
- Workflow *green* **does not mean** the data-quality audit is fully certified.

## Preserved data-quality concerns

`data/full_slate_data_quality_audit.json`:

- disposition: `FULL_SLATE_DATA_QUALITY_NOT_CERTIFIED`;
- certification blockers: **2**;
- historical identity: `REVIEW_REQUIRED_POSSIBLE_VETERAN_ALIAS`. One CIN WR `Mitch Tinsley` has a potential `Mitchell Tinsley` historic GSIS alias (he has no directly priced sportsbook markets in the snapshot). Do not automatically fuse those identities without source-proven evidence;
- injuries: `PARTIAL_SCOPE_NOT_PROVEN`: official current-week positive injury rows only cover 21/30 scheduled teams; missing positive rows cannot be taken to mean no injuries. Do not quietly declare full injury coverage;
- direct WR/CB matchup feeds are unavailable and explicitly gated off; team-scheme coverage is 30/30;
- raw leaguewide Ourlads roster properly includes CAR/KC bye teams; downstream football and live odds scope excludes both;
- live sportsbook artifact contained 757 compact player prop identities/markets from 360 unique players; original input had 134 quarantined invalid/unsupported rows, reconciled by source status.

**Do not treat this as a fully certified/high-confidence betting card.** Relative model edge alone is insufficient to bet everything. No probabilities, lines, or picks have been manually substituted.

## Future action

1. A Week-5 rerun must continue to use the canonical Full Slate and honor a fresh T-75 window, strict roster/schedule/source guards, and only source-certified starter data. Do not mechanically accept Huntley as BAL primary because a sportsbook posted him; do not treat him as permanently quarantined after an official starter confirmation. The quarantine is week-specific and should be updated if the starter authority is genuinely resolved.
2. For original Oct-8 snapshot reuse artifact `11582178322` and successful offline recovery `11582934259`; no need for another paid OddsAPI call.
3. Before wagering, resolve 21/30 partial injury report scope and CIN possible veteran-alias concern, or clearly mark remaining outputs uncertified; avoid betting based on unqualified outputs.
4. Continue separate research branch PR #672 without mixing model science into operations.
