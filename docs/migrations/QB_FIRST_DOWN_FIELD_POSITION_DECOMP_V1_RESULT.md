# QB First-Down Field-Position Decomposition V1 — Authoritative Result

## Disposition

`FIRST_DOWN_WITHIN_FIELD_POSITION_PROPENSITY_PRIMARY_DIAGNOSTIC`

The frozen diagnostic completed successfully. Field-position occupancy does not explain the established first-down shared QB/receiver pass-opportunity miss. The dominant mechanism remains the team's realized pass-origin propensity **within the same first-down field-position zone**.

## Authoritative run

- Workflow run: `34545401969`
- Job: `103096807810`
- Branch: `research-qb-first-down-field-position-decomp-v1`
- Tested head: `1527ed2c2935175b222b9a428b2c2113a6eceb58`
- Artifact: `qb-first-down-field-position-decomp-v1`
- Artifact ID: `10178825973`
- Artifact digest: `sha256:734b1603b69f45a39f33e6d4770690bcc81b42f8a3b3dae33b74dce6c2cbeb37`
- Parent rows: `884` = 444 in 2024 + 440 in 2025
- Shared receiver cohorts: exact 440 2025 WR-target rows and 884 pooled WR-reception rows
- Sportsbook inputs: zero
- Model fitting: zero
- Production changes: zero

## Frozen routing result

### WITHIN_ZONE_PASS_PROPENSITY — PRIMARY

- pooled mean absolute contribution: `0.04474731937078716`
- 2024 mean absolute contribution: `0.04415409734554893`
- 2025 mean absolute contribution: `0.04534593432352756`
- largest pooled component: yes
- pooled lead >=20% versus second-largest: yes
- season stability gate: yes
- 2025 WR-target Spearman: `0.416549531388451`
- pooled WR-reception Spearman: `0.31835559399817626`

All five frozen PRIMARY routing gates passed.

### FIELD_POSITION_OCCUPANCY — NOT PRIMARY

- pooled mean absolute contribution: `0.006228897430577961`
- 2024: `0.0059446340240805395`
- 2025: `0.006515745049861725`
- 2025 WR-target Spearman: `0.0050027963142183765`
- pooled WR-reception Spearman: `-0.016112244229219697`

This component is much smaller and has essentially no shared receiver relationship.

### ZONE_REFERENCE_LEVEL — NOT PRIMARY

- pooled mean absolute contribution: `0.0005191748236608816`
- 2025 WR-target Spearman: `0.004404511856513158`
- pooled WR-reception Spearman: `-0.030669418425460664`

This component is negligible.

## Integrity

All frozen integrity gates passed. First-down `yardline_100` coverage was 100% in 2023, 2024, and 2025. The four frozen zones were mutually exclusive and exhaustive. Every reconciliation identity held to floating-point tolerance (~1e-16), all reference inputs were strictly prior, and the receiver joins preserved exact cohort size and uniqueness.

## Scientific interpretation

The first-down shared opportunity miss is **not** primarily caused by where first downs occur on the field. It persists after holding first-down field-position zone constant.

The strongest surviving description is now:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION -> WITHIN-FIELD-POSITION PASS PROPENSITY`

This mechanism remains strongly shared with receiver opportunity error.

## Production consequence

None. This is a diagnostic result, not a qualified correction. Do not add field position as a generic QB pass-rate feature and do not move this diagnostic into production.

## Anti-loop / next boundary

- Do not retest field-position occupancy as the explanation; it failed decisively under the frozen decomposition.
- Do not rescue the failed first-down choice-economics D1 family.
- Do not repackage generic pass-rate history, pass-funnel, playcaller/opening-script, FTN tactical-call history, or M83 adaptive-defense information.
- Any next predictive source must plausibly explain **week-specific first-down pass choice after down and field-position are held constant**, or the project should explicitly classify this remaining portion as largely unobserved game-plan intent rather than indefinitely re-mining the same historical PBP.
