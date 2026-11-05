# WR-CB Prospective Public-Article Pregame Capture V1

**Prepared, NOT executed** on the nonexistent/unverified 2026 Week-4 article as of September 30, 2026. Source/production gate remains CLOSED.

Purpose: future free-source point-in-time capture so the current public HTML cannot later overwrite our pregame source evidence. This is not actual defensive responsibility tracking. No OddsAPI usage, paid feed, fitted parameters or football-model mutation.

- Triggerable research-only manual GitHub workflow: `.github/workflows/wr-cb-prospective-capture-v1.yml` with a specifically verified LIVE article URL, season and week. **A newly added workflow may not be dispatchable until GitHub recognizes it on the default branch; do not assume it is already running.** The script itself can be run from the research branch with the exact same arguments.
- `scripts/research/capture_fantasyalarm_wr_cb_pregame_v1.py` captures ONE article per invocation. Restricts exact HTTPS host and WR article path; verifies article title explicitly matches requested season/week. Preserves exact returned HTML bytes, SHA-256, network-fetch start and finish UTC, site publication/edit timestamps, exact factual pair identities, and per-WR real kickoff/schedule.
- Distinguishes early from late matchups row by row; a Thursday row cannot be made pregame by a Friday snapshot of a Sunday matchup. Empty output/URL mismatch fails closed. Ambiguous site modification metadata also quarantines.
- Output `snapshot_rows_audit.csv` and compact **editorial-grade-free** `pregame_factual_lock.json`. These are observational source facts, not roster/identity-certified or model-eligible. Missing teams and slots remain MISSING.
- GitHub Actions artifact stores raw HTML with **90-day retention**, not permanent. To protect source authority, commit the **sanitized** `pregame_factual_lock.json`, `capture_summary.json` and their hashes to GitHub on a new immutable evidence branch **BEFORE each game's kickoff**; never commit the vendor's raw HTML into this repository. Only then can a later strict-prior audit verify exactly what we observed in time.
- Independently verify declared kickoff, stable player identity, completeness/sampling mechanism, editor-projected-vs-actual matchup semantics and eventual row lineage before any scientific test or use of future 2026 data.
- The existing 2021-26 retrospective pages collected after their games are not retroactively pregame snapshots. Historical `dateModified` timestamp alone is not sufficient to recover the original pairings.

No scheduled scraping/recurring GitHub Actions job is installed. No prospectively locked Week-4 data exists at this checkpoint.

## Usable while PR #665 remains draft

The preexisting `research-wr-cb-free-archive-v1.yml` PR validation workflow now also checks `data/research/wr_cb_pregame_capture_request_v1.csv`. Its default state contains **only the CSV header**, so no prospective fetch occurs and no extra paid resource is used. Once a new, externally verified current-week public article is actually live, put **exactly one** `season,week,url` row in that request file and push the research branch before the applicable game's kickoff. The PR source-validation workflow will execute the one-article time-stamped capture and upload a separate `wr-cb-prospective-exact-capture` artifact. The unrelated historical source audit remains unchanged.

This pull-request-triggered mechanism avoids assuming GitHub can dispatch a brand-new workflow before that workflow exists on `main`. It does NOT autonomously watch article release or automatically make the factual lock durable; after capture, inspect the result and commit the sanitized factual lock/hash to GitHub before kickoff. Remove the request row after the specific capture to avoid repeat fetches on unrelated PR updates. Never enter a guessed URL, merge the model, or label absent slots as zero.
