# PRISM WR/TE Opportunity-Bias Conservation Bridge V1 — Source Recovery Amendment

Date: 2026-10-18

Status: **FROZEN BEFORE RECOVERY AUDIT / NO BRIDGE OUTCOME SCORED**

This amendment supersedes only the **materialization path** in
`PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_PLAN.md`.

It does **not** change:
- the qualified signed-bias signal;
- the bridge topology;
- the support gates;
- the proposed bias factor;
- any scientific threshold;
- any outcome gate.

## 1. Why an amendment is required

The original immutable full-authority artifacts have expired from GitHub
artifact storage:
- WR-R15 run `34238301577` / artifact `10061328722`;
- TE-R5P run `34152797603` / artifact `10029942404`.

A source audit must not fail open by substituting a partial player panel or a
different model.

The exact authorities are therefore recovered only from later immutable
evidence that either:
1. reproduces the original WR authority byte-for-byte from its original source
   commit; or
2. preserves the exact TE-R5P production-order state and explicitly certifies
   that TE-R5P was applied.

## 2. WR authority recovery — exact deterministic rerun

Original WR-R15 source commit:
`02c3dd1a681d4ab2953683039e39830554f9ec9f`.

Original run:
`34238301577`.

Original evaluator:
`scripts/backtest/evaluate_wr_r15_wr1_anchor_participation_v1.py`.

Original fold contract:
- build leakage-safe 2022 inputs from 2021;
- build leakage-safe 2023 inputs from 2022;
- build leakage-safe 2024 inputs from 2023;
- run exactly:
  `--data-2022 ... --logs-2022 ... --data-2023 ... --logs-2023 ... --data-2024 ... --logs-2024 ... --iterations 2000`.

Historical Actions run `36203881629`, while the original authority artifact
was still alive, recorded the exact SHA-256 of:

`wr_r15_confirmation_predictions.csv`

as:

`91d108ee7981935d78179257b204a0204e9a4f8061e6eba4d6810845ed057d05`.

The recovered WR authority is accepted **only** if the rebuilt predictions CSV
matches that SHA-256 exactly.

No tolerance-based parity can replace the byte hash.

The rebuilt file supplies the original full-room:
- `pred_targets`;
- `mc_rec_yards`;
- player/team/week identity;
- 2023 and 2024 fold-safe OOS rows.

## 3. Independent preserved WR-2024 cross-check

Later run `36332720457` / artifact `10935999653` remains live.

Artifact:
`opportunity-authority-priority-v1`

Digest:
`sha256:4fde0a72c3aef7ae2412580333a7fefcd3078e53a31f0e9ebb945fd6a6ed1bb5`.

Its integrity record explicitly states:
- `wr_r15_2024_applied_all_weeks = true`;
- specialist assets invariant;
- team-volume inputs invariant;
- zero sportsbook inputs.

For 2024 WR signal rows, recovered WR-R15 `pred_targets` must equal the
preserved direct-detail `baseline_pred` on every exact overlapping
`season, week, team, player_clean_key` identity within `1e-9`.

Do **not** require raw WR-R15 `mc_rec_yards` to equal the later full-stack
`baseline_proj`; the latter is a downstream production-order receiving-yard
mean and is not the same accounting object. The original WR predictions CSV
byte hash is the authority for WR-R15 `mc_rec_yards`.

This target-count check is an independent semantic cross-check in addition to
the exact file hash.

## 4. TE authority recovery — preserved exact production-order state

Use the same still-live run `36332720457` / artifact `10935999653`.

The artifact integrity record explicitly certifies:
- `te_r5p_2024_2025_applied = true`;
- specialist assets invariant;
- source-priority boundary exact;
- team-volume inputs invariant;
- zero sportsbook inputs;
- zero target/future feature rows.

TE target opportunity source:
`opportunity_authority_priority_direct_detail.csv`
with family `TE_TARGET_OPPORTUNITY`.

Frozen parent target projection:
`baseline_pred`.

TE receiving-yard parent source:
`opportunity_authority_priority_market_detail.csv`
with:
- position `TE`;
- market `rec_yards`.

Frozen parent receiving-yard projection:
`baseline_proj`.

The source is used only for seasons 2024 and 2025.

Every qualified TE opportunity signal row must exact-join both preserved files.

## 5. Frozen signal source remains unchanged

Run:
`39109423266`.

Artifact:
`12128983909`.

Digest:
`sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490`.

Only:
- WR opportunity;
- TE opportunity

are eligible.

WR efficiency and TE efficiency remain CLOSED.

## 6. Frozen room topology and gates remain unchanged

Room:
`position + season + week + team`.

Bridge-topology eligibility remains:
- exact authority join;
- predicted targets >0;
- predicted receiving yards >0;
- finite positive implied predicted YPT;
- room size >=2.

Per position/season support remains:
- >=250 bridge-eligible rows;
- >=50 distinct supported players;
- 100% signal-to-authority join;
- zero duplicate authority identities;
- zero duplicate signal identities;
- zero nonpositive/nonfinite matched predicted target rows;
- zero nonpositive/nonfinite matched predicted YPT rows.

No gate may be changed after recovery results are seen.

## 7. Frozen mechanical factor inventory remains unchanged

For source/topology inspection only:

`bias_factor = exp(-hierarchical_signed_bias / predicted_receiving_yards)`.

Do not apply it to targets in this recovery audit.
Do not renormalize rooms.
Do not score actual outcomes.

## 8. Fail-closed rules

Recovery is blocked if:
- WR rebuilt prediction SHA differs by even one byte;
- WR-2024 independent preserved parity fails;
- preserved opportunity artifact digest/run identity fails;
- its integrity flags do not certify exact WR/TE specialist use;
- any qualified signal row lacks an exact parent join;
- any original source/topology gate fails.

Do not fall back to:
- generic historical ensemble;
- current all-history production coefficients;
- partial supported-player residual panels;
- reconstructed/fuzzy identity;
- sportsbook data;
- 2026 outcomes.

## 9. Integrity boundary

At this amendment:
- bridge outcome rows scored: **0**;
- target corrections applied: **0**;
- room renormalizations applied: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 / Week-6 outcomes used: **0**;
- sportsbook / OddsAPI used: **0**;
- production changed: **false**.

Disposition:
`PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_SOURCE_RECOVERY_FROZEN`.


## 10. WR conservation hierarchy clarification

WR-R15 is not a flat whole-room target-share model.

The exact authority source:
- freezes the M38 WR anchor entitlement;
- preserves the total WR-room mass;
- preserves the secondary-WR pool mass;
- redistributes only the secondary-WR pool through the R15 softmax;
- then passes the resulting entitlement state through the existing explicit
  simulator.

Therefore a later bridge, if source-ready, must intervene inside the same
hierarchy:
- anchor remains protected unless a separately frozen mechanism explicitly
  earns anchor eligibility;
- secondary supported players may receive the frozen signed-bias modifier;
- unsupported secondary players retain neutral factor 1;
- the exact secondary pool is renormalized back to its original mass;
- full WR-room and team-player mass must remain exact;
- downstream targets / receiving yards must be regenerated by the existing
  authority simulator, not by a hand-built flat target-count patch.

The current source audit may use the preserved full-room R15 entitlement state
from run `36203881629` / artifact `10893281689` only as topology and
conservation evidence. It must not infer a single flat team target pool from
those shares.

No bridge action is authorized by this clarification.
