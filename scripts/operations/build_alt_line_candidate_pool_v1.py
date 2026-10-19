#!/usr/bin/env python3
"""Screen a priced Full Slate board down to the alt-line candidate pool.

Implements the hard gates of
``docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`` so candidate
discovery is reproducible instead of eyeballed. This reads an already-priced
board and never re-prices, never fetches odds and never touches production
artifacts -- it is a read-only screen over outputs a run already produced.

The ranking quantity is cushion measured in model standard deviations. That is
the question alt-line construction actually asks: not "how big is the edge"
but "how much room does the football thesis have before the leg dies".
"""
from __future__ import annotations

import argparse
import csv
import sys

YARD_MARKETS = {"pass_yards", "rush_yards", "rec_yards", "receptions"}
SYNTH_TOL = 1e-6

# Hierarchy ranks that actually see meaningful offensive usage. A TE3 or RB3
# with a positive edge on a tiny line is the sparse-player fallback talking,
# not role evidence, and a QB2 priced for half a rushing yard is a book pricing
# someone who is not expected to play.
#
# This is matched against player_form's ``model_role``, the usage-derived
# hierarchy -- NOT ``depth_role``, which carries the Ourlads alignment slot
# (WR1=LWR, WR2=SWR, WR3=RWR) and says nothing about rank. Every rostered
# receiver has an alignment label of WR1-WR3, so gating on depth_role admits
# the entire room: TEN's Calvin Ridley reads depth_role=WR1 while being
# model_role=WR4 at an 11.7% target share, because he is the second body at
# the LWR slot.
PLAYABLE_ROLES = {"QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "TE2", "FB1"}

# Below these the market stops being a yardage distribution and becomes a
# near-binary event, where cushion measured in sd is meaningless.
MIN_LINE = {"pass_yards": 150.0, "rush_yards": 10.0, "rec_yards": 10.0,
            "receptions": 2.5}

# A rushing prop on a receiver, or a receiving prop on a quarterback, prices a
# gadget call rather than a role. "They might run a reverse" is not a thesis.
GADGET = {("rush_yards", "WR"), ("rush_yards", "TE"),
          ("rec_yards", "QB"), ("receptions", "QB")}


def _f(row, key):
    try:
        return float(row.get(key) or "")
    except (TypeError, ValueError):
        return None


def _load(path):
    if not path:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _quarantined(board_row, prop_q, board_q, season, week):
    player = (board_row.get("player") or "").strip().lower()
    for r in prop_q:
        if (r.get("player") or "").strip().lower() == player:
            return "prop_quarantine"
    for r in board_q:
        if (r.get("player") or "").strip().lower() != player:
            continue
        # A final-board quarantine is scoped to the season/week it was verified
        # for; an unrelated week must not inherit it.
        if str(r.get("season") or "").strip() == str(season) and \
           str(r.get("week") or "").strip() == str(week):
            return "final_board_quarantine"
    return None


def screen(board, form_by_key, prop_q, board_q, min_games=10):
    """Return (kept, rejected). Each kept row carries its cushion metrics."""
    kept, rejected = [], []
    for row in board:
        side = (row.get("side") or "").upper()
        market = row.get("market") or ""
        week = (row.get("week") or "").strip()
        season = (row.get("season") or "").strip()
        key = (row.get("player_clean_key") or "").strip().lower()
        form = form_by_key.get(key, {})

        def reject(reason):
            rejected.append({**row, "reject_reason": reason})

        if market not in YARD_MARKETS:
            reject(f"market_not_eligible:{market}")
            continue
        if (row.get("ensemble_status") or "") != "calibrated":
            reject(f"ensemble_not_calibrated:{row.get('ensemble_status')}")
            continue

        proj, line, sd = _f(row, "model_proj"), _f(row, "vegas_line"), _f(row, "model_sd")
        if proj is None or line is None or sd is None or sd <= 0:
            reject("missing_projection_or_sd")
            continue

        # A priced QB pass-yards row must not bypass the promoted synthesis.
        if market == "pass_yards":
            if (row.get("qb_synthesis_applied") or "") not in ("1", "True", "true"):
                reject("pass_yards_without_qb_synthesis")
                continue
            synth = _f(row, "qb_synthesis_proj")
            if synth is None or abs(synth - proj) > SYNTH_TOL:
                reject("qb_synthesis_proj_mismatch")
                continue

        # RB P3 is promoted for Week 1 only; any other week routing through it
        # is a contract violation, not a candidate.
        if (row.get("rb_synthesis_applied") or "") in ("1", "True", "true") and week != "1":
            reject(f"rb_synthesis_outside_week1:week={week}")
            continue

        q = _quarantined(row, prop_q, board_q, season, week)
        if q:
            reject(q)
            continue

        edge = _f(row, "edge_pct")
        if edge is None or edge <= 0:
            reject("non_positive_edge")
            continue

        # The documented systematic RB rushing/rush+rec under cluster is an
        # excluded family however low the model goes.
        pos_group = (form.get("position_group") or "").upper()
        if market == "rush_yards" and side == "UNDER" and pos_group == "RB":
            reject("rb_rush_under_cluster")
            continue

        # A share built on almost no games is noise, not a player baseline.
        games = (_f(form, "prior_games") or 0) + (_f(form, "current_games") or 0)
        if games < min_games:
            reject(f"insufficient_player_evidence:games={games:g}")
            continue

        # The player has to be someone the offense actually uses. Rank comes
        # from model_role; a missing rank fails closed rather than passing.
        model_role = (form.get("model_role") or "").strip().upper()
        if model_role not in PLAYABLE_ROLES:
            reject(f"not_a_playable_role:{model_role or 'unknown'}")
            continue

        if (market, pos_group) in GADGET:
            reject(f"gadget_market:{market}/{pos_group}")
            continue

        floor = MIN_LINE.get(market)
        if floor is not None and line < floor:
            reject(f"degenerate_low_line:{market}@{line:g}")
            continue

        cushion = (proj - line) if side == "OVER" else (line - proj)
        kept.append({
            **row,
            "position_group": pos_group,
            "model_role": model_role,
            "alignment_slot": form.get("depth_role") or "",
            "tgt_share": form.get("tgt_share") or "",
            "rush_share": form.get("rush_share") or "",
            "player_games": f"{games:g}",
            "cushion": f"{cushion:.3f}",
            "cushion_sd": f"{cushion / sd:.4f}",
        })
    return kept, rejected


def dedupe_by_book(rows, prefer="fanduel"):
    """One row per player+market+side, preferring the book that posts alts."""
    best = {}
    for r in rows:
        k = (r.get("player"), r.get("market"), r.get("side"))
        cur = best.get(k)
        if cur is None:
            best[k] = r
            continue
        r_pref = (r.get("book") or "") == prefer
        c_pref = (cur.get("book") or "") == prefer
        if r_pref and not c_pref:
            best[k] = r
        elif r_pref == c_pref and (_f(r, "edge_pct") or 0) > (_f(cur, "edge_pct") or 0):
            best[k] = r
    return list(best.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--player-form", required=True)
    ap.add_argument("--prop-quarantine", default="")
    ap.add_argument("--final-board-quarantine", default="")
    ap.add_argument("--min-games", type=float, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--reject-summary", action="store_true")
    args = ap.parse_args(argv)

    board = _load(args.board)
    form = _load(args.player_form)
    form_by_key = {(r.get("player_clean_key") or "").strip().lower(): r for r in form}

    kept, rejected = screen(
        board, form_by_key,
        _load(args.prop_quarantine), _load(args.final_board_quarantine),
        min_games=args.min_games,
    )
    pool = dedupe_by_book(kept)
    pool.sort(key=lambda r: -float(r["cushion_sd"]))

    print(f"board_rows={len(board)} kept={len(kept)} pool_after_book_dedupe={len(pool)}")

    if args.reject_summary:
        counts = {}
        for r in rejected:
            counts[r["reject_reason"].split(":")[0]] = \
                counts.get(r["reject_reason"].split(":")[0], 0) + 1
        print("\n=== REJECTIONS BY GATE ===")
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5}  {reason}")

    cols = ["market", "player", "team", "opponent", "side", "vegas_line",
            "model_proj", "model_sd", "cushion", "cushion_sd", "fair_prob",
            "vegas_odds", "edge_pct", "book", "position_group", "model_role",
            "alignment_slot", "tgt_share", "rush_share", "player_games"]
    print("\n=== CANDIDATE POOL (ranked by cushion in sd) ===")
    print("\t".join(cols))
    for r in pool:
        print("\t".join(str(r.get(c) or "") for c in cols))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(pool)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
