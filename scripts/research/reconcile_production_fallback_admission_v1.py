#!/usr/bin/env python3
"""Reconcile card population fallback against what production actually priced.

Required by re-audit `6427742545`. The decomposition in
`decompose_position_prior_fallback_materiality_v1.py` is card-level and
DEFAULTS-based, so its mass figures are proxies. This script answers the
question that was actually asked -- current-production severity -- from the
durable priced board rather than from constants.

The board is production's own output and carries production's own
`bayes_evidence_state` per priced row, so "was this row admitted as a
zero-evidence player" needs no inference at all.

The audit asked for a three-way distinction. Two of the three are established
here from durable evidence:

  1. card has population fallback            -> the card's evidence_authority
  2. row is actually admitted to production   -> the board's bayes_evidence_state
  3. fallback POSTERIOR is nonzero            -> NOT ESTABLISHED, see below

(3) cannot be computed from this repository. Production's posterior for a
zero-evidence player is `_weighted_group_mean()` over the position group's own
`{metric}_prior` and `prior_games`, and those inputs live in
`data/player_form_consensus.csv` / `data/metrics_ready.csv`, which are
run-time artifacts that no run commits. Only the priced boards are durable.
That is a §5a durability gap, reported rather than papered over with the
DEFAULTS constant -- substituting the constant is exactly the error the
re-audit caught.

Outcome-blind: reads `bayes_evidence_state`, `player`, `team`, `market`,
`position`. It does not read any line, odds, price, edge, probability or
result column, and asserts as much.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FALLBACK = "position_prior_only"

# Columns this reconciliation is permitted to read off the priced board. The
# board necessarily contains lines, odds and edges; naming the allowlist makes
# it checkable that none of them were touched.
_BOARD_COLS_ALLOWED = (
    "player", "player_clean_key", "team", "opponent", "market",
    "source_market", "bayes_evidence_state", "bayes_applied", "rules_role",
    "season", "week",
)

# Markets the auditor has ruled uncertified and separate; a fallback row here
# is a different kind of problem from one in a graded yardage market.
_UNCERTIFIED_MARKETS = frozenset({"anytime_td"})


def load_board(path: Path) -> pd.DataFrame:
    board = pd.read_csv(path, low_memory=False)
    keep = [c for c in _BOARD_COLS_ALLOWED if c in board.columns]
    missing = {"player", "market", "bayes_evidence_state"} - set(keep)
    if missing:
        raise RuntimeError(f"board lacks required columns {missing}")
    return board[keep].copy()


def reconcile(card: pd.DataFrame, board: pd.DataFrame,
              week: int) -> tuple[dict, pd.DataFrame]:
    c = card.loc[card["target_week"].eq(week)
                 & card["can_receive_opportunity"].astype(bool)]
    card_fb = c.loc[c["evidence_authority"].eq(FALLBACK)]

    b = board.loc[pd.to_numeric(board["week"], errors="coerce").eq(week)] \
        if "week" in board.columns else board
    admitted = b.loc[b["bayes_evidence_state"].eq(FALLBACK)]

    unc = admitted["market"].isin(_UNCERTIFIED_MARKETS)

    def _names(df, col):
        return set(df[col].astype(str).str.strip()) - {"", "nan"}

    prod_names = _names(admitted, "player")
    card_names = _names(card_fb, "player")

    detail = (admitted.groupby(["player", "team", "market"], dropna=False)
              .size().reset_index(name="priced_rows")
              .sort_values(["player", "market"]))
    detail["market_uncertified"] = detail["market"].isin(_UNCERTIFIED_MARKETS)
    detail["also_fallback_in_card"] = detail["player"].astype(str).str.strip() \
        .isin(card_names)

    status = {
        "version": "PRODUCTION_FALLBACK_ADMISSION_RECONCILIATION_V1",
        "authorized_by": "GPT_RE_AUDIT_6427742545",
        "scientific_boundary": "DESCRIPTIVE_RECONCILIATION_ONLY_NO_FITTING",
        "outcomes_inspected": False,
        "board_columns_read": sorted(board.columns.tolist()),
        "week": week,
        # (1) card-level
        "card_eligible_cards": int(len(c)),
        "card_eligible_players": int(c["gsis_id"].nunique()),
        "card_fallback_cards": int(len(card_fb)),
        "card_fallback_players": int(card_fb["gsis_id"].nunique()),
        # (2) production admission, from production's own label
        "production_priced_rows": int(len(b)),
        "production_priced_players": int(b["player"].nunique()),
        "production_fallback_rows": int(len(admitted)),
        "production_fallback_players": int(len(prod_names)),
        "production_fallback_rows_by_market": {
            k: int(v) for k, v in admitted["market"].value_counts().items()},
        "production_fallback_rows_uncertified_market": int(unc.sum()),
        "production_fallback_rows_certified_market": int((~unc).sum()),
        "production_fallback_players_sorted": sorted(prod_names),
        # agreement
        "production_fallback_also_fallback_in_card": len(
            prod_names & card_names),
        "production_fallback_not_fallback_in_card": sorted(
            prod_names - card_names),
        # (3) explicitly not established
        "fallback_posterior_established": False,
        "fallback_posterior_blocker": (
            "production posterior for a zero-evidence player is "
            "_weighted_group_mean over the position group's own "
            "{metric}_prior and prior_games; those inputs live in "
            "data/player_form_consensus.csv / data/metrics_ready.csv, which "
            "are run-time artifacts that no run commits. Only priced boards "
            "are durable. Substituting the DEFAULTS constant is the error "
            "re-audit 6427742545 identified, so it is not substituted."),
        "durability_gap": (
            "to make fallback posterior auditable, a live run must archive the "
            "player_form/metrics_ready prior inputs alongside the priced board"),
    }
    return status, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, default=Path(
        "data/research/player_identity/player_identity_card_v1_2026.csv"))
    ap.add_argument("--board", type=Path, default=Path(
        "data/market_track_record/boards/2026_wk07.csv"))
    ap.add_argument("--week", type=int, default=7)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/research/player_identity"))
    a = ap.parse_args()

    card = pd.read_csv(a.cards, low_memory=False)
    board = load_board(a.board)

    forbidden = [c for c in board.columns
                 if any(k in c.lower() for k in
                        ("line", "odds", "price", "edge", "prob", "proj"))]
    if forbidden:
        raise RuntimeError(
            f"board slice carries decision columns {forbidden}; this "
            "reconciliation is outcome-blind")

    status, detail = reconcile(card, board, a.week)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"production_fallback_admission_v1_week{a.week}"
    detail.to_csv(a.out_dir / f"{stem}.csv", index=False)
    (a.out_dir / f"{stem}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"=== PRODUCTION FALLBACK ADMISSION — WEEK {a.week} ===")
    print(f"card eligible      : {status['card_eligible_cards']} cards, "
          f"{status['card_eligible_players']} players")
    print(f"card fallback      : {status['card_fallback_cards']} cards, "
          f"{status['card_fallback_players']} players")
    print(f"production priced  : {status['production_priced_rows']} rows, "
          f"{status['production_priced_players']} players")
    print(f"production fallback: {status['production_fallback_rows']} rows, "
          f"{status['production_fallback_players']} players")
    print(f"  by market        : "
          f"{json.dumps(status['production_fallback_rows_by_market'], sort_keys=True)}")
    print(f"  uncertified mkt  : {status['production_fallback_rows_uncertified_market']}"
          f" of {status['production_fallback_rows']} rows")
    print(f"\nagreement: {status['production_fallback_also_fallback_in_card']} "
          f"of {status['production_fallback_players']} production fallback "
          f"players are also fallback in the card")
    if status["production_fallback_not_fallback_in_card"]:
        print("  in production but not card-fallback: "
              f"{status['production_fallback_not_fallback_in_card']}")
    print("\nfallback POSTERIOR is explicitly NOT established here; see "
          "fallback_posterior_blocker in the status sidecar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
