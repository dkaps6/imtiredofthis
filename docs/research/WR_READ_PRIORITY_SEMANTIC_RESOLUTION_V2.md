# WR Read Priority Semantic Resolution V2

**STATUS: SOURCE-ONLY CONTRACT FROZEN BEFORE ANY WR OUTCOME TEST. NO PRODUCTION CHANGE.**

## Why V2 exists

The first source-only audit used the current nflreadr FTN dictionary literally and treated `read_thrown == 0` as `PRIMARY_READ`. That audit found the feature strongly nonredundant with WR-R15 opportunity, but the raw receiver-target frequency of `0` (~1-2% in 2023-2024) was too unusual to proceed without resolving semantics.

Claude independently re-audited nflverse/nflreadr issue #216, PR #319, and the nflverse-ftn ingestion code. The best available source evidence supports the current maintainer-approved mapping:

- `0` = first / primary read (2023+ only)
- `1` = second read
- `2` = third read or later
- `CHK` = checkdown
- `DES` = designed read / no progression
- `SD` = scramble drill
- in 2022, primary reads were not coded and appear as `NA`

The current dictionary clarification was approved by `tanho63`, the same maintainer who authored the earlier ambiguous issue text. nflverse-ftn passes FTN's raw `read` value through without recoding. Therefore this project will **not** relabel `1` as first read.

## Frozen source-only feature concept

The next source screen will **not** use literal first-read share. Instead it will test whether a broader, semantically defensible receiver process variable is measurable and nonredundant:

`EARLY_NO_EXTENDED_SHARE8`

Within the receiver's last eight strictly-prior target-bearing games:

- **EARLY / NO EXTENDED PROGRESSION**
  - 2022: `read_thrown` is `NA` (documented uncoded primary read) OR `DES`
  - 2023+: `read_thrown` is `0` OR `DES`
- **EXTENDED PROGRESSION**
  - all seasons: `read_thrown` is `1` OR `2`
- **EXCLUDED FROM BINARY DENOMINATOR**
  - `CHK`
  - `SD`
  - unknown/unrecognized codes

The feature is:

`EARLY_NO_EXTENDED_SHARE8 = early_or_no_extended / (early_or_no_extended + extended_progression)`

This is intentionally **not** called first-read share. It measures whether targets to the receiver tend to arrive before an extended QB progression versus after the progression advances.

## Support / chronology contract

- exact FTN game/play -> nflverse PBP receiver-ID join
- GSIS-first receiver identity using the same strictly-prior roster resolver already validated in WR-R17/R18/R19
- target-bearing games are selected first, strictly before the target game
- last 8 prior target-bearing games
- minimum 4 prior target-bearing games
- minimum 16 **classifiable progression targets** (`early/no-extended` + `extended`)
- `CHK`, `SD`, and unknown codes do not count toward the 16-target progression support floor
- no fuzzy identity
- no sportsbook
- no WR receiving-yard outcome fields
- no 2024 WR-R15 projection/outcome fields may be parsed or materialized

## Frozen source/redundancy eligibility screen

This V2 audit is not a football-result experiment. It asks only whether an R20 plan is source- and novelty-eligible.

Required:

1. FTN->PBP exact-play join >= 95% in every 2022-2024 season.
2. No unrecognized `read_thrown` values beyond the frozen key.
3. Supported 2023 WR-R15 authority coverage >= 60% under the 4-game / 16-classifiable-target floor.
4. `abs(Spearman(EARLY_NO_EXTENDED_SHARE8, entitlement_tgt_share)) < 0.75`.
5. Linear R² explaining `EARLY_NO_EXTENDED_SHARE8` from `entitlement_tgt_share + pred_targets + WR1 indicator` < 0.60.
6. WR outcomes loaded = false; sportsbook inputs = 0; 2024 WR-R15 projection/outcome fields parsed = false.

Pass disposition:
`READ_PRIORITY_R20_PLAN_ELIGIBLE_V2`

Fail disposition:
`READ_PRIORITY_SOURCE_REDUNDANCY_BLOCKED_V2`

## Important boundary

A V2 source-screen pass does **not** authorize football scoring by itself. It only permits a separately frozen R20 outcome plan, which must receive adversarial review before any 2023 WR receiving-yard outcome is opened.

No 2024 WR holdout outcome will be inspected in this source audit.