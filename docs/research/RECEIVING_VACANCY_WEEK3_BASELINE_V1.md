# Receiving Vacancy Week-3 Baseline V1 — Frozen Pregame Snapshot

Date frozen: 2026-09-26  
Status: **PREGAME FROZEN — NO OUTCOMES / NO MODEL CHANGE**

This snapshot implements `RECEIVING_VACANCY_PROSPECTIVE_AUDIT_V1_PLAN.md`. Exact row-level survivor entitlements are preserved in the output-only diagnostic artifact described below.

## Authority

- Full Slate no-live-odds run: `36204768034`
- Full Slate artifact: `10892728623`
- source main: `f7d2011b73950488ea209124ba895b92c401b2b1`
- target-entitlement output replay run: `36276121522`
- replay artifact: `10917193224`
- replay digest: `sha256:e329c3a167043cb329dc3c638379d147a63df76d70a1ec1a4e536db9c039a700`
- exact full survivor file in artifact: `target_entitlement_rows.csv`
- candidate variants scored: **0**
- parameters fit: **0**
- sportsbook inputs: **0**
- target-game outcomes read: **0**
- production mutations: **0**

## Frozen unavailable receivers

| Team | Player | Pos | Latest strict-prior target share | Prior week |
|---|---|---:|---:|---:|
| HOU | Nico Collins | WR | 0.270270 | 1 |
| IND | Ashton Dulin | WR | 0.034483 | 1 |
| LAC | Brenen Thompson | WR | 0.115385 | 2 |
| LAC | Charlie Kolar | TE | 0.115385 | 2 |
| MIA | Caleb Douglas | WR | 0.130435 | 2 |
| NO | Barion Brown | WR | 0.000000 | 1 |
| NYJ | Mason Taylor | TE | 0.130435 | 1 |
| SF | Demarcus Robinson | WR | 0.142857 | 2 |
| WAS | Chig Okonkwo | TE | 0.103448 | 1 |

## Frozen team baseline

The baseline is production's current behavior after unavailable players are removed: survivors are passed through the existing rules, M38 hierarchy, and `TEAM_TARGET_ENTITLEMENT_V1` finite allocator. `Skill sum` excludes tiny QB receiving probability; residual remains the production 0.05 bucket.

| Team | Vacated prior share | Skill sum | Residual | Top-1 | Top-2 | Skill HHI |
|---|---:|---:|---:|---:|---:|---:|
| HOU | 0.270270 | 0.949591 | 0.050000 | 0.186384 | 0.350870 | 0.123926 |
| IND | 0.034483 | 0.945101 | 0.050000 | 0.173920 | 0.316692 | 0.114180 |
| LAC | 0.230769 | 0.949640 | 0.050000 | 0.187077 | 0.338452 | 0.117176 |
| MIA | 0.130435 | 0.949213 | 0.050000 | 0.150416 | 0.276088 | 0.101226 |
| NO | 0.000000 | 0.946368 | 0.050000 | 0.224561 | 0.365900 | 0.134845 |
| NYJ | 0.130435 | 0.949359 | 0.050000 | 0.249975 | 0.379473 | 0.137132 |
| SF | 0.142857 | 0.949379 | 0.050000 | 0.164203 | 0.290632 | 0.106780 |
| WAS | 0.103448 | 0.949403 | 0.050000 | 0.196041 | 0.351375 | 0.111856 |

## Frozen top-5 survivor entitlements by affected team

### HOU

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Dalton Schultz | TE | 0.186384 |
| 2 | Xavier Hutchinson | WR | 0.164486 |
| 3 | Kayshon Boutte | WR | 0.120060 |
| 4 | Cade Stover | TE | 0.095511 |
| 5 | Jared Wayne | WR | 0.095130 |

### IND

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Keenan Allen | WR | 0.173920 |
| 2 | Tyler Warren | TE | 0.142772 |
| 3 | Josh Downs | WR | 0.127728 |
| 4 | Drew Ogletree | TE | 0.094003 |
| 5 | Darius Slayton | WR | 0.087851 |

### LAC

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Ladd McConkey | WR | 0.187077 |
| 2 | Quentin Johnston | WR | 0.151375 |
| 3 | Oronde Gadsden II | TE | 0.110874 |
| 4 | Hayden Rucci | TE | 0.108402 |
| 5 | Omarion Hampton | RB | 0.079476 |

### MIA

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Malik Washington | WR | 0.150416 |
| 2 | De'Von Achane | RB | 0.125672 |
| 3 | Greg Dulcich | TE | 0.106271 |
| 4 | Kevin Coleman | WR | 0.103622 |
| 5 | Seydou Traore | TE | 0.097390 |

### NO

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Chris Olave | WR | 0.224561 |
| 2 | Juwan Johnson | TE | 0.141338 |
| 3 | Devaughn Vele | WR | 0.106412 |
| 4 | Noah Fant | TE | 0.093917 |
| 5 | Oscar Delp | TE | 0.090331 |

### NYJ

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Garrett Wilson | WR | 0.249975 |
| 2 | Adonai Mitchell | WR | 0.129498 |
| 3 | Kenyon Sadiq | TE | 0.099310 |
| 4 | Malik McClain | WR | 0.099241 |
| 5 | Breece Hall | RB | 0.082731 |

### SF

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Deebo Samuel Sr. | WR | 0.164203 |
| 2 | Mike Evans | WR | 0.126429 |
| 3 | George Kittle | TE | 0.123719 |
| 4 | Christian McCaffrey | RB | 0.121831 |
| 5 | Jacob Cowing | WR | 0.082063 |

### WAS

| Rank | Player | Pos | Entitlement share |
|---:|---|---:|---:|
| 1 | Terry McLaurin | WR | 0.196041 |
| 2 | Stefon Diggs | WR | 0.155334 |
| 3 | Antonio Williams | WR | 0.087652 |
| 4 | John Bates | TE | 0.070940 |
| 5 | Rachaad White | RB | 0.066930 |

## Postgame lock

After games are final, grade actual survivor target share and concentration against this snapshot without changing the cohort, baseline ranks, or entitlement values. The complete per-player baseline is the immutable artifact `10917193224`; this document is only the compact human-readable checkpoint.
