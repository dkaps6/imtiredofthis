# Market Snapshot History / CLV Capture Architecture V1 — Frozen Plan

Date frozen: 2026-09-29  
Status: **FROZEN BEFORE FUTURE CLOSING SNAPSHOTS — DOWNSTREAM RESEARCH INFRASTRUCTURE ONLY**  
Branch: `research-market-snapshot-history-v1`

## Authority

The frozen RB-PD2 forward plan explicitly authorized, after the first valid
prospective lock:
1. route-volume source readiness;
2. opponent-injury propagation;
3. CLV capture architecture as a downstream diagnostic only.

This V1 implements item 3 without buying any additional sportsbook data and
without feeding market information into football generation.

## Existing architecture gap

Current `scripts/operations/archive_priced_board_v1.py` intentionally keeps
only the latest row per natural market key. A later capture replaces the earlier
capture.

That is useful for a "latest known line" ledger but destroys the sequence of
market snapshots required for real closing-line-value / line-movement analysis.

This V1 does **not** change or remove the existing latest-row ledger.

It adds an independent append-only snapshot history.

## Zero-marginal-cost boundary

A snapshot may be archived only from a Full Slate artifact that already exists.

This research infrastructure:
- never calls OddsAPI;
- never triggers Full Slate;
- never requests a paid refresh;
- never creates a reason to buy a close snapshot;
- simply retains the market data from runs acquired for independent production /
  operational reasons.

No paid OddsAPI pull is authorized solely for CLV research.

## Exact timestamp authority

Use the source artifact's own `data/props_raw.csv`:
- `fetched_at` = odds acquisition timestamp;
- `commence_time` = event kickoff timestamp;
- `event_id` = event identity.

Do not substitute GitHub workflow completion time for odds-fetch time when the
provider artifact supplies `fetched_at`.

Every archived priced row must be joined to:
- exact `odds_fetched_at_utc`;
- exact `commence_time_utc`;
- `minutes_to_kickoff`.

A row captured at or after kickoff is not a valid pregame market snapshot for
that event and must be flagged / excluded from closing selection.

## Immutable snapshot unit

One snapshot file per source Full Slate run.

Suggested path:

`data/market_track_record/snapshots/<season>_wk<WW>/<timestamp>__run_<RUN_ID>.csv`

Snapshot contents preserve the priced-board row plus:
- source run ID;
- source git SHA;
- odds fetched timestamp;
- kickoff timestamp;
- minutes to kickoff;
- immutable snapshot ID.

A snapshot path may never be overwritten with different bytes.

## Market identity

Primary matching key across snapshots:

`(event_id, player_clean_key, market, book, side)`

Do **not** include `vegas_line` in the cross-snapshot identity key because line
movement is the object being measured.

Within one snapshot that primary key must be unique. Conflicts fail closed.

No cross-book substitution in the primary CLV calculation.

## Entry snapshot

A future CLV observation requires an explicitly identified canonical/published
entry board.

The entry snapshot is not automatically "earliest available" if an earlier run
was debugging, invalid, or unpublished.

Entry authority must retain:
- source run ID;
- source git SHA;
- board certification/disposition if available.

## Closing snapshot

For each entry market row independently, closing quote = the latest available
snapshot satisfying all:
1. same primary market identity;
2. source capture later than or equal to entry capture;
3. source capture strictly before that event's kickoff;
4. quote finite / valid;
5. snapshot was obtained for independent operational reasons, not a research-only
   paid pull.

If no such later quote exists:
`NO_VALID_CLOSE_CAPTURE`.

Do not borrow another book's closing quote for the primary metric.

## Frozen CLV outputs

### 1. Side-aligned line CLV

For OVER:
`line_clv = close_line - entry_line`

For UNDER:
`line_clv = entry_line - close_line`

Positive means the market moved in the selected/model side's direction and the
entry captured the more favorable line.

### 2. Same-line price CLV

Only when `close_line == entry_line` within 1e-12:

Convert entry and close selected-side American odds to decimal.

`price_clv_pct = (decimal_entry / decimal_close - 1) * 100`

Positive means the entry price was better than the later closing price.

If the line changed, price CLV is `NOT_COMPARABLE_LINE_CHANGED`; no synthetic
single score combining line and price is allowed in V1.

### 3. Frozen-model gap at close

Using the **entry** model projection only:

For OVER:
`model_gap_to_close = entry_model_proj - close_line`

For UNDER:
`model_gap_to_close = close_line - entry_model_proj`

This is diagnostic only. The closing quote never enters the football model.

## No-outcome boundary

CLV can be calculated before outcomes and must initially be treated separately
from ROI/hit rate.

No future game result is needed to define:
- entry snapshot;
- closing snapshot;
- line CLV;
- price CLV;
- close-model gap.

## Future evaluation population

Primary future cohort:
- exact canonical published betting-board rows;
- one selected offer per player/market decision lineage;
- same-book close available.

The append-only archive should retain the full priced market so future selected
boards can be joined without having to reacquire history.

## Interpretation

Positive CLV is market-confirmation evidence, not proof the football model is
correct.

Negative CLV is market-disagreement evidence, not permission to feed market
movement upstream.

No football coefficient, mean, width, selection threshold, or staking rule may
be fitted from CLV in this architecture phase.

## Required integrity checks

- source priced board contract valid;
- raw source has exact `fetched_at`;
- each event has one kickoff time;
- priced event IDs all resolve into raw event metadata;
- no post-kickoff row enters valid-close population;
- snapshot primary keys unique;
- source run / git SHA present;
- immutable file collision fails closed unless bytes are identical;
- no existing latest-row ledger semantics change.

## Authorized implementation

Build:
1. append-only market snapshot archiver;
2. pure snapshot comparison / CLV calculator;
3. tests proving no-overwrite, exact kickoff filtering, same-book matching,
   line direction, and same-line price math;
4. bounded workflow integration that runs only after an already-completed Full
   Slate artifact is downloaded by the existing archive workflow.

No model / pricing / selection mutation.
