# WR-CB Verified Snapshot Source Lock V1 — 2026-09-30

**Purpose:** durably freeze sanitized pregame source inputs **before any WR-CB model science or protected 2025 outcome access**.

## Discovery source lock — 2024 Week 1

- exact Wayback capture: `2024-09-07T00:57:14Z`;
- source publication: `2024-09-06T23:59:46Z`;
- CDX digest: `JE62ZZZYJ4TELVSMS7LVWRPWWTXE53HA`;
- exact replay body SHA-256: `0854732b5af783b1d67fd64bcf6598a86c03a0f3669e70e4399b903c441dc33f`;
- archived JSON-LD articleBody SHA-256: `85998a169935efd81dece50d0d162d41d4f88c58bfdd18d1239b2afbe497680d`;
- verification run `36768720380`, artifact `11122412408`;
- 66 explicit factual pairings recovered; 57 still pregame at capture; **46 strict exact-week two-sided roster rows locked**;
- lock file: `data/research/wr_cb_verified_snapshot_lock_2024w01_v1.csv`;
- lock SHA-256: `99f37f47e9f9b3ccdbbb76175ee86c944fdcd17ed6e75fce48e7bb71dede6dd9`.

## Protected confirmation source lock — 2025 Week 14

- exact Wayback capture: `2025-12-06T13:14:09Z`;
- source publication: `2025-12-04T21:00:27Z`;
- exact CDX digest: `2VNA2WFCUG3X347ZKOY5MGW7Z4BEEFQN`;
- exact replay body SHA-256: `e6f0bc1bd5a676f542ab086ed4aecdc5dba2dd30579a9ec9f59c5223ff134816`;
- archived JSON-LD articleBody SHA-256: `03487ac0eb907191e1d68b2cda5cc125aa6e1e4ec4b99705f511c6be0f62402d`;
- verification run `36770684037` **SUCCESS**, artifact `11123566159`;
- 60 explicit factual pairings recovered; 56 still pregame at capture; **47 strict exact-week two-sided roster rows locked**;
- lock file: `data/research/wr_cb_verified_snapshot_lock_2025w14_v1.csv`;
- lock SHA-256: `c5b4875aa1d78fefeffb616b9a0684320d29c64babaaa4db004bcf3867509ed2`;
- verifier explicitly recorded `confirmation_outcomes_accessed=false`, `target_game_outcomes=false`, `parameters_fit=0`.

## Lock semantics

Rows contain only `season, week, WR team, WR GSIS ID, opponent, CB GSIS ID, alignment bucket`. No FantasyAlarm editorial grade, prose, salary, sportsbook field, game result or realized player outcome is retained.

These source locks are **immutable scientific inputs**, not model features. Never edit them in place. Any future correction requires a new version with a new filename/hash and explicit reason. 2025 Week 14 remains protected from outcome/model use until a scientific design is frozen. Missing rows remain missing, never zero exposure. Pairings remain editor-projected pregame alignments, not route-by-route observed coverage.

Source/model gate remains CLOSED.
