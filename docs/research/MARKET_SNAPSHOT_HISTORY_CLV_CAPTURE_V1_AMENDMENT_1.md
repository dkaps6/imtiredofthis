# Market Snapshot History / CLV Capture V1 — Amendment 1

Date: 2026-09-29  
Status: **FROZEN BEFORE IMPLEMENTATION OUTPUT / BEFORE FUTURE MARKET SNAPSHOTS**

Parent:
`docs/research/MARKET_SNAPSHOT_HISTORY_CLV_CAPTURE_V1_PLAN.md`

## Why this amendment is required

The parent plan allowed the "latest available" pregame snapshot to act as the
closing snapshot. That is too loose: a quote captured many hours before kickoff
is market movement evidence, but it is not honestly a closing line.

No implementation result or future snapshot comparison has been inspected.
This is a pre-result contract correction.

## Corrected later-snapshot rule

A comparison quote must be:
- same primary market identity;
- strictly **later** than the entry quote's `odds_fetched_at_utc`;
- strictly before kickoff.

If no later quote exists:
`NO_LATER_PREGAME_CAPTURE`.

## Timing labels

For the latest valid later quote, compute exact minutes to kickoff.

### `VALID_T30_CLOSE`
`0 < minutes_to_kickoff <= 30`

Only this class may be called **closing-line value (CLV)** in V1.

### `VALID_T60_LATE_MARKET`
`30 < minutes_to_kickoff <= 60`

May be reported as late-market movement, not exact CLV.

### `PREGAME_MOVEMENT_ONLY`
`minutes_to_kickoff > 60`

May be reported only as earlier market movement.

### Invalid
At/after kickoff or missing timestamp/kickoff:
`INVALID_FOR_PREGAME_MOVEMENT`.

## Scientific boundary

Do not collapse the three timing classes into one "CLV" metric.

No paid OddsAPI refresh is authorized merely to obtain a T30 snapshot.
