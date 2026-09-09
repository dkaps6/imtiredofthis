# RB R26R — 2026 Week 1 Prospective Observation Snapshot V1 — Frozen Plan

## Purpose

Capture a genuinely prospective, pregame sportsbook observation of the already-sealed R26Q/R26O RB receptions candidate without changing, rerunning, recalibrating, or otherwise allowing sportsbook information to influence any football projection.

This is a downstream benchmark snapshot only.

## Candidate

`RB_R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_V1`

## Immutable parent authority

R26Q prospective seal:

- run: `34400524030`
- artifact: `10123251043`
- artifact name: `rb-r26q-2026-week1-receptions-prospective-seal-v1`
- artifact digest: `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- head: `68661da94f03cab2f96182d47636cf55e088b5de`
- required disposition: `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`

The R26Q artifact itself seals R26O run `34399750746`, artifact `10123070453`, digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`, head `e7014a6e365cbb776e48085dcef12dfece744ca4`.

## Protected production authority

Protected production code remains:

`f8417f55b04ce0e19baf260e9d532765034c47f1`

R26R may use the protected live-odds/provider utilities only to capture and identify sportsbook offers. No sportsbook artifact may be passed into a football model or alter a sealed projection.

## Frozen scope

R26R must:

1. verify the exact R26Q artifact digest and head before reading it;
2. verify the exact R26Q PASS disposition and sealed-parent lineage;
3. verify all 28 R26Q gates passed;
4. verify all 17 sealed R26O evidence files remain byte-exact inside R26Q;
5. verify the sealed R26O receptions NPZ contains exactly 107 RB/FB arrays and that all per-array SHA256 values match the sealed manifest;
6. preserve the 104 changed vacancy-active RB/FB arrays and 3 unchanged CIN RB/FB arrays exactly;
7. build the current Week-1 schedule and current Ourlads roster only for live market scoping/player identity;
8. fetch a current live OddsAPI snapshot using the existing hardened live-odds boundary;
9. materialize exact bookmaker/player/market/line offers without inventing a consensus line;
10. filter the observation study to `player_receptions` only;
11. match sportsbook players to the sealed R26Q/R26O player universe deterministically by current team plus normalized player identity;
12. preserve every exact sportsbook book-line and American over/under price used in the snapshot;
13. compare the sealed production baseline reception mean and sealed R26O shadow/candidate mean with each exact book-line;
14. compute the sealed candidate distribution's `P(< line)`, `P(= line)`, and `P(> line)` directly from the immutable 25,000-draw array;
15. compute no-vig sportsbook over probability from exact paired over/under American odds when both sides are valid;
16. record market comparison only as downstream evidence. Market disagreement is not an upstream correction.

## Explicit non-scope / leakage barriers

R26R must not:

- use any 2026 Week-1 outcomes;
- use any sportsbook line, price, consensus, or implied probability as a football-model feature;
- rerun or regenerate R26N/R26O football values;
- refit R9;
- change entitlement shares, target shares, targets, catch rates, receptions means, or any distribution array;
- use current same-week depth/roster information to change the sealed candidate;
- promote anything to production;
- activate the shadow in production;
- tune a threshold from this sportsbook snapshot.

Current Ourlads roster/depth information is allowed only for sportsbook identity resolution and active-slate scoping. It is not a football input to the sealed R26Q candidate.

## Frozen observation outputs

Required outputs:

- `r26r_disposition.json`
- `r26r_gate_matrix.csv`
- `r26r_sealed_array_verification.csv`
- `r26r_receptions_book_line_snapshot.csv`
- `r26r_receptions_player_market_summary.csv`
- `r26r_market_capture_status.json`
- exact copies or SHA256 records for the live-odds status, pricing-offer audit, and exact pricing-offer source used by R26R.

The book-line snapshot must retain at minimum:

- event identity;
- team/opponent;
- player and sealed `player_clean_key`;
- book/book title;
- exact line;
- exact over/under American prices;
- sealed production baseline reception mean;
- sealed candidate/shadow reception mean;
- candidate-minus-baseline mean shift;
- baseline-minus-line gap;
- candidate-minus-line gap;
- candidate `P(< line)`, `P(= line)`, `P(> line)`;
- no-vig sportsbook over probability when computable;
- candidate-vs-market probability gap when computable;
- source/fetch timestamp when provided by the protected market capture.

A player-level summary may report descriptive min/median/max book line and counts, but no synthetic consensus line may replace or overwrite exact book-line evidence.

## Frozen gates

1. exact R26Q artifact digest verified by workflow;
2. exact R26Q head SHA;
3. exact R26Q PASS disposition;
4. R26Q gate matrix is 28/28;
5. R26Q parent lineage equals authoritative R26O run/artifact/digest/head;
6. R26Q reports 107 sealed arrays;
7. R26Q reports 104 changed reception arrays;
8. R26Q reports 17 sealed evidence files;
9. every R26Q sealed-file byte-exact flag is true;
10. sealed R26O manifest has exactly 107 RB/FB rows;
11. sealed R26O NPZ has exactly 107 members;
12. every NPZ member has exactly 25,000 draws;
13. every NPZ per-array SHA256 matches the sealed manifest;
14. manifest scope is exactly 104 vacancy-active + 3 nonvacancy CIN rows;
15. no football values regenerated;
16. no R9 refit;
17. no production parameters changed;
18. no production promotion or live-shadow activation authorized;
19. no 2026 Week-1 outcomes used;
20. sportsbook information used only after immutable candidate verification;
21. current roster/depth used only for market identity/scoping, not candidate generation;
22. live-odds provider boundary returns either a legitimate market-available state or a legitimate no-market-yet state;
23. if market is available, live identity disposition is ready and unresolved core player rows are zero;
24. if market is available, exact pricing offers materialize without a fabricated consensus line;
25. if `player_receptions` offers exist, every observed sealed-player match is deterministic and one-to-one within team;
26. if `player_receptions` offers exist, every retained book-line has finite line and valid over/under prices;
27. if `player_receptions` offers exist, candidate distribution probabilities are finite and sum to 1 within numerical tolerance;
28. if `player_receptions` offers exist, no candidate array/hash/value differs from the R26Q seal;
29. sportsbook lines/prices/implied probabilities are absent from all football-input construction paths in this run;
30. observation output contains no authority to change production or tune the sealed candidate.

## Frozen dispositions

If all parent/seal integrity gates pass and one or more valid `player_receptions` sportsbook book-lines are captured and matched:

`R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`

If all parent/seal integrity gates pass but legitimate current sportsbook state has no usable `player_receptions` market yet:

`R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_NO_RECEPTIONS_MARKET_YET`

This is not a scientific failure and authorizes only a later snapshot against the exact same sealed R26Q parent.

If any immutable-parent, identity, exactness, leakage, or observation-contract gate fails:

`R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_FAIL_NO_OBSERVATION`

## Authority ceiling

Even a PASS authorizes only prospective market observation and later postgame evaluation of the already-sealed candidate. It does not authorize production promotion, live-shadow production activation, model tuning, market correction, or any change to the sealed candidate.

## Side research note preserved for later study

An acute same-week injury/role-transfer question (for example, an established RB1 becoming inactive and an RB2 inheriting the start) is scientifically important but is not part of R26R and must not modify the sealed Week-1 candidate. The current RB receiving identity layer already carries player-specific historical identity features; a separately frozen study may later test whether sudden role-transfer magnitude is calibrated correctly for the replacement player's own profile rather than assuming the injured starter's profile.
