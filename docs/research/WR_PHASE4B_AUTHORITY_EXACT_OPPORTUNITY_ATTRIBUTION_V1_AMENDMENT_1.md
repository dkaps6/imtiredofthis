# WR Phase 4B Authority-Exact Opportunity Attribution V1 — Amendment 1

## Status

Frozen source-contract amendment only. The Phase 4B science, accounting formulas, materiality threshold, interpretation tree, and no-challenger/no-production-change boundaries remain unchanged.

This amendment is supported by:
- V2 strict-prior GSIS audit run `34922511782`, artifact `10378757263`, digest `sha256:70772b5c358f921e31681f0c2cfbf064556542f68dd9f4706fa0d09b9520c7fb`;
- V3 identity-temporal ablation run `34971134534`, artifact `10397407493`, digest `sha256:ea175f8b86cf4faba88cfd22559a0826e61480d3ecd84d688048e6815f806b44`;
- V3 result doc `docs/research/WR_PHASE4B_IDENTITY_TEMPORAL_ABLATION_V3_RESULT.md`;
- Claude conditional-PASS closure comment `5682161002`;
- GPT-5.6 overlap confirmation comment `5682192889`.

## Frozen actual-target identity contract

For this retrospective Phase 4B attribution diagnostic only:

1. Alias-to-GSIS resolution may use all audited 2022-2024 weekly-roster alias evidence because this is static identity metadata, not a predictive football feature.
2. Resolution remains deterministic and fail-closed:
   - same-team exact full alias;
   - same-team suffix-insensitive alias;
   - globally unique exact full alias;
   - globally unique suffix-insensitive alias;
   - no fuzzy matching;
   - ambiguity remains unresolved.
3. Actual target counts come only from nflverse PBP by stable `receiver_player_id`, using the unchanged rule:
   `REG week1-18 AND pass_attempt==1 AND two_point_attempt!=1 AND no_play!=1 AND receiver_player_id nonnull`.
4. A resolved GSIS identity with a present PBP team-game and no target events receives a verified actual target count of zero.
5. Unresolved or ambiguous identities are never imputed to zero.
6. This relaxation is forbidden for predictive feature construction and does not change any predictive temporal rule.

## V3 validation facts

- Layer 4: `5,320 / 5,321` resolved; one ambiguous row remains.
- Layers 2/3 canonical identity domain: `6,011 / 6,012` resolved; one ambiguous row remains.
- R15 graded authority: `4,193 / 4,193` resolved with `4,193 / 4,193` exact target parity and max absolute target delta `0.0`.
- The sole broader-domain ambiguity is `2024 W15 NYJ Brandon Smith`, where multiple same-team exact-name GSIS candidates exist. It remains fail-closed.

## Non-reassignment confirmation

The V3 static-identity rule is pure additive recovery relative to V2 strict-prior resolution:
- Layer 4: all `5,244 / 5,244` rows resolved by both V2 and V3 retain the identical GSIS ID;
- Layers 2/3: all `5,933 / 5,933` overlap rows retain the identical GSIS ID;
- R15 authority: all `4,154 / 4,154` overlap rows retain the identical GSIS ID;
- overlap target counts are unchanged.

Therefore the relaxed rule adds identities but does not reassign any previously resolved identity.

## Layer-specific fail-closed consequence

- Layer 4 metrics use the `5,320` resolved WR2+ rows and explicitly report/exclude the one ambiguous row.
- Layers 2/3 require a complete resolved canonical modeled room for team-game attribution. Any anchor-observable team-game containing an unresolved/ambiguous canonical identity is excluded from Layer 2/3 accounting and reported explicitly. This prevents a partial room from being mislabeled as complete.
- Layer 1 is unchanged and remains all `4,193` graded authority rows.

## Authorization state

Authorized next step: patch implementation, rerun synthetic/mechanical tests and source preflight, then obtain Claude code-level implementation review.

Receiving-yard attribution outcomes remain sealed until that implementation review passes.

No production change. No paid Full Slate. No RB work.
