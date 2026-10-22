# GSIS Point-in-Time Archive V1

## Purpose

This research lane preserves authenticated NFLGSIS tables as immutable weekly
source snapshots. It does not feed production or Full Slate. A snapshot taken
through Week N may be considered for Week N+1 research only after a separate
semantic and predictive validation.

The source reports are season/phase aggregates. A completed-season table is
not a historical weekly state. Do not use a full-season historical aggregate as
a pregame weekly feature, and do not reconstruct prior weeks by subtracting
season totals unless that method is separately proven valid for the report.

## Authorized capture surface

The collector uses the authenticated Work browser and the rendered table DOM.
It does not store or export credentials, cookies, tokens, authorization headers,
browser storage, request metadata, or session state. Backend transport remains
unidentified.

After each selector change, collection must wait until all of these are true:

1. season, phase, team, and report-mode selectors match the requested state;
2. the rendered table is present, or GSIS explicitly returns an empty/alert state;
3. two consecutive full rendered-table snapshots are identical.

Every raw cell retains displayed text, HTML cell type, colspan, and rowspan.
Metadata retains the capture timestamp, GSIS last-updated text, source version,
selected filters, report route, notes, table count, and row count.

## Weekly directory contract

```text
data/research/gsis/
  validation/
    2025_REG_lineup_detail_pilot/
      manifest.json
      raw_snapshot.json.gz
  snapshots/
    season_2026/
      REG/
        through_week_03/
          captured_20260928T194504Z/
            manifest.json
            raw_snapshot.json.gz
        through_week_04/
          captured_<UTC timestamp>/
            manifest.json
            raw_snapshot.json.gz
```

Each `captured_*` directory is immutable. The materializer fails when its target
already exists. A later capture always receives a new directory, even if it is
for the same through-week boundary.

## Current snapshot scope

The complete 2026 REG capture contains 320 team/view records:

- Lineup Detail: Offense and Defense;
- Down Analysis: Offense/Defense × Detailed/Grouped;
- Play Propensity: Field Position, Quarter, and Score Differential;
- Formation Usage: default view.

The team set is the 32 current franchises. Historical selector aliases such as
Oakland, San Diego, and St. Louis are excluded from current-season iteration.
Explicit empty or alert responses are retained in the manifest and must not be
filled from another team, mode, or source.

## Repeating the archive after each week

1. Complete normal interactive NFL/Okta authentication in the Work browser.
2. Select the current season and REG phase.
3. Collect all 320 requested team/view states with the stability guard above.
4. Normalize the browser capture to one JSON document with archive metadata and
   one record per team/view.
5. Materialize it into a new immutable directory:

```bash
python scripts/research/materialize_gsis_snapshot_v1.py \
  --kind current \
  --input /path/to/normalized_browser_capture.json \
  --output data/research/gsis/snapshots/season_2026/REG/through_week_<NN>/captured_<UTC>
```

6. Review the manifest for 320 unique records, source version, hashes, capture
   window, explicit empty responses, and report schemas.
7. Commit the new directory without editing prior captures.

The normalized input is transient. The committed authority is the deterministic
gzip payload plus manifest hashes. Decompress with standard gzip tooling.

## Week-3 boundary

The first current snapshot contains 2026 Week-3 information. It may be archived
now, but it must not be used to tune, diagnose, grade, or modify the production
model until the existing frozen Week-3 postmortem protocol permits that work.
