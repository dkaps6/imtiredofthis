# NFL GSIS 2026 Source / Access Audit V1 — Frozen Plan

**STATUS: FROZEN BEFORE FIRST CREDENTIALED RUN. RESEARCH-ONLY. NO PRODUCTION INTEGRATION.**

## Purpose

The user now has authorized NFL GSIS access and has stored credentials in GitHub Actions secrets:

- `NFLGSIS_USER`
- `NFLGSIS_PASS`

This study answers only:

1. Can a GitHub Actions runner authenticate to the current NFL GSIS GameStatsLive site using those authorized credentials?
2. Does the account reach the authenticated schedule/statistics surface?
3. What same-site report/navigation surfaces and network endpoints are visible after login?
4. What source families appear materially new versus the current 2026 model stack?

No GSIS field enters football modeling in V1.

## Current official login context

Public NFLGSIS currently states that its login process changed on **2026-07-07** and eligible NFL / club / media / related-site users should click Login and sign in with email and password.

Primary public entry:
`https://www.nflgsis.com/GameStatsLive/Auth/`

Expected authenticated schedule route:
`https://www.nflgsis.com/GameStatsLive/Schedule`

## Existing repository boundary

Legacy `scripts/providers/gsis_pull.py` was removed from production on 2026-08-31 during provider cleanup.

It was not a true credentialed GSIS client; its final implementation consumed nflverse / nflreadpy GSIS-derived public data.

Therefore this study must **not** resurrect that provider.

`AGENTS.md` explicitly requires a new 2026 source/semantic audit before any legacy GSIS-style source can return to production.

## Security contract

The workflow:
- receives credentials only through GitHub Actions secrets;
- never prints either secret;
- never writes credentials to artifacts;
- never stores browser cookies/storage-state in artifacts;
- never uploads screenshots of authenticated pages;
- strips query strings/fragments from captured internal URLs;
- records only route/path, HTTP status, content type, page title, link text and non-sensitive structural metadata;
- performs no mutation, submission, registration, account changes or downloads beyond ordinary authenticated GET/navigation;
- makes at most one login attempt per run;
- does not brute-force or retry bad credentials.

If the account presents MFA / OTP / CAPTCHA / device approval / other interactive authentication that cannot be completed non-interactively, classify it and stop without attempting bypass.

## V1 browser behavior

Use Playwright/Chromium because the 2026 login can redirect through the NFL identity surface and may require JavaScript.

The client must support:
1. direct legacy username/password fields if the current page exposes them;
2. email-first then password NFL identity flow;
3. one-step email/password flow;
4. detection of MFA / OTP / CAPTCHA / passkey / security challenge.

After apparent authentication, navigate to the expected Schedule route.

Authentication is considered successful only if:
- credentials were present;
- browser is no longer on a login/auth/challenge page;
- Schedule returns a usable page;
- page content is not a bad-credential / unauthorized / sign-in prompt.

## Authenticated inventory

On successful auth, collect:

### Page metadata
- final authenticated URL path only;
- title;
- response status when available;
- visible text character count;
- same-host anchor count;
- unique same-host route paths.

### Same-host links
For up to 100 unique links:
- link text trimmed to 120 characters;
- path only;
- no query string;
- no fragment.

### Network inventory
Observe same-host requests/responses during auth + Schedule load.

For up to 200 unique route/method/content-type tuples record:
- HTTP method;
- path only;
- status;
- content type.

Never record:
- Authorization headers;
- cookies;
- request bodies;
- query strings;
- response bodies;
- tokens.

## Classification

### `GSIS_2026_AUTHENTICATED_INVENTORY_READY`
Credentials authenticate successfully and Schedule/inventory are reachable.

### `GSIS_2026_INTERACTIVE_AUTH_REQUIRED`
Credentials are accepted into an MFA/OTP/CAPTCHA/passkey/device challenge that requires the user interactively.

### `GSIS_2026_AUTH_FAILED`
A single authorized login attempt returns bad credentials / unauthorized / login loop.

### `GSIS_2026_ACCESS_SURFACE_UNRESOLVED`
Credentials appear accepted but authenticated Schedule/report access cannot be proven.

### `GSIS_2026_SECRET_CONFIGURATION_MISSING`
One or both required GitHub Actions secrets are absent/empty.

## Promotion boundary

A successful V1 authorizes only a second frozen **data-semantic inventory**:
- inspect available report schemas/endpoints;
- capture small non-sensitive sample metadata;
- compare exact fields/freshness to current nflverse, participation, Ourlads, Sharp, injury and identity sources;
- identify genuinely new pregame information.

It does **not** authorize:
- Full Slate integration;
- replacing nflverse/Ourlads/Sharp;
- fitting model coefficients;
- Week-3 outcome use;
- sportsbook integration;
- automated bulk downloading.

Any future ingestion must freeze source lineage, freshness, leakage, licensing/terms, failure semantics, and historical availability first.
