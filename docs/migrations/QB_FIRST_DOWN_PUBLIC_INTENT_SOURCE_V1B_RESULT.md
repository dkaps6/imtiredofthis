# QB First-Down Public Intent Source V1B — Gold Retrieval Result

Date: 2026-09-24

Status: `RETRIEVAL_AUTOMATION_NOT_QUALIFIED`

## Canonical runs

Initial generic transport:
- run `36003845429`
- head `9cc72add8e1fc3fab4f7d375f41dee2d1ca49680`
- artifact `10809261974`
- digest `sha256:eb435e645f47cfca4f75c0ff31d6cd3887b82bd6108da7ca6445d5f91149fa57`

One allowed generic repair pass:
- head `847c00387a6a3ba1fa5eaf7af7a1a222069aee65`
- run `36004240741`
- artifact `10809622180`
- digest `sha256:82e8dfba390461c2bcdefc693d1c177fab9e62efe899b0f75693943efd58c92c`

## Integrity

- football outcomes read: 0
- model residuals read: 0
- sportsbook fields used: 0
- predictive models fit: 0
- production changes: 0

## Result

Gold rows: 13.

Initial DuckDuckGo HTML transport:
- eligible-candidate recall: 0%
- canonical-source retrieval: 0%
- transport returned HTTP 202 empty pages, then 403 rate limiting.

Repair added generic Bing HTML fallback:
- eligible-candidate recall: 0%
- canonical-source retrieval: 0%
- Bing returned redirect-wrapper/junk results rather than honoring the frozen site-restricted queries.

This clears the V1B frozen early-stop condition for an unstable/unsuitable generic public-search transport after one implementation + repair pass.

## Interpretation

This is an **operational retrieval failure**, not evidence that public pregame intent lacks football value.

The parent V1 manual gold prefix already proves that timestamp-safe eligible sources exist for the sampled team-weeks. What failed is scalable retrieval from unauthenticated search-engine HTML inside GitHub Actions.

Do not:
- lower the retrieval gates;
- add team-specific query hacks;
- continue cycling through public search engines;
- return to a 500+ row manual historical crawl.

A future restart requires a materially different retrieval source/connector with stable search access, not another HTML-scraping variant.

Current-week prospective shadow collection remains scientifically distinct and may continue because it freezes information before kickoff rather than reconstructing historical sources after outcomes.
