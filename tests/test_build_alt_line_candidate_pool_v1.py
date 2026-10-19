"""Gates of the alt-line candidate screen.

Each test pins one rule from
``docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md``. The screen
decides what is allowed to reach a ticket, so a silently loosened gate is the
failure mode worth testing for.
"""
import importlib.util
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "operations" / \
    "build_alt_line_candidate_pool_v1.py"
_spec = importlib.util.spec_from_file_location("alt_pool", _SRC)
alt_pool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alt_pool)


def board_row(**kw):
    base = dict(
        market="rec_yards", player="P", player_clean_key="p", team="A",
        opponent="B", side="OVER", vegas_line="50", model_proj="60",
        model_sd="20", ensemble_status="calibrated", edge_pct="0.05",
        season="2026", week="6", book="fanduel", qb_synthesis_applied="",
        qb_synthesis_proj="", rb_synthesis_applied="0", fair_prob="0.6",
        vegas_odds="-110",
    )
    base.update(kw)
    return base


WR_FORM = {"p": {"position_group": "WR", "prior_games": "10", "current_games": "5",
                 "depth_role": "WR1", "model_role": "WR1",
                 "tgt_share": "0.25", "rush_share": "0"}}
RB_FORM = {"p": {**WR_FORM["p"], "position_group": "RB",
                 "depth_role": "RB1", "model_role": "RB1"}}


def screen_one(row, form=None, prop_q=(), board_q=(), min_games=None):
    """Screen one row. min_games is left to the module default unless given,
    so a change to that default is actually exercised here."""
    kwargs = {} if min_games is None else {"min_games": min_games}
    kept, rejected = alt_pool.screen(
        [row], form if form is not None else WR_FORM,
        list(prop_q), list(board_q), **kwargs)
    return (kept[0] if kept else None), (rejected[0]["reject_reason"] if rejected else None)


def test_eligible_row_survives_and_cushion_is_in_sd_units():
    kept, reason = screen_one(board_row())
    assert reason is None
    assert kept["cushion"] == "10.000"
    assert float(kept["cushion_sd"]) == pytest.approx(0.5)


def test_under_cushion_measures_distance_below_the_line():
    kept, _ = screen_one(board_row(side="UNDER", vegas_line="80"))
    assert float(kept["cushion"]) == pytest.approx(20.0)


@pytest.mark.parametrize("market", ["anytime_td", "rush_rec_yards"])
def test_markets_outside_the_eligible_set_are_rejected(market):
    _, reason = screen_one(board_row(market=market))
    assert reason.startswith("market_not_eligible")


def test_uncalibrated_ensemble_is_rejected():
    _, reason = screen_one(board_row(ensemble_status="uncalibrated_mc_only"))
    assert reason.startswith("ensemble_not_calibrated")


def test_non_positive_edge_is_rejected():
    _, reason = screen_one(board_row(edge_pct="-0.01"))
    assert reason == "non_positive_edge"


def test_pass_yards_must_carry_the_promoted_synthesis():
    _, reason = screen_one(board_row(market="pass_yards"))
    assert reason == "pass_yards_without_qb_synthesis"


QB_FORM = {"p": {"position_group": "QB", "prior_games": "17", "current_games": "5",
                 "depth_role": "QB1", "model_role": "QB1",
                 "tgt_share": "0", "rush_share": "0.1"}}


def test_pass_yards_projection_must_equal_the_synthesis_mean():
    qb = dict(market="pass_yards", vegas_line="200", model_proj="220",
              qb_synthesis_applied="1")
    _, reason = screen_one(board_row(**qb, qb_synthesis_proj="219"), form=QB_FORM)
    assert reason == "qb_synthesis_proj_mismatch"
    kept, _ = screen_one(board_row(**qb, qb_synthesis_proj="220"), form=QB_FORM)
    assert kept is not None


def test_rb_synthesis_is_week_one_only():
    _, reason = screen_one(board_row(rb_synthesis_applied="1"))
    assert reason.startswith("rb_synthesis_outside_week1")
    kept, _ = screen_one(board_row(rb_synthesis_applied="1", week="1"))
    assert kept is not None


def test_rb_rushing_under_cluster_is_an_excluded_family():
    _, reason = screen_one(
        board_row(market="rush_yards", side="UNDER", vegas_line="80"), form=RB_FORM)
    assert reason == "rb_rush_under_cluster"


def test_rb_rushing_over_is_not_in_that_cluster():
    """The exclusion is the under cluster specifically, not RB rushing."""
    kept, _ = screen_one(board_row(market="rush_yards", side="OVER"), form=RB_FORM)
    assert kept is not None


def test_sparse_player_evidence_is_rejected():
    thin = {"p": {**WR_FORM["p"], "prior_games": "0", "current_games": "2"}}
    _, reason = screen_one(board_row(), form=thin)
    assert reason.startswith("insufficient_player_evidence")


def test_final_board_quarantine_applies_only_to_its_own_season_and_week():
    _, reason = screen_one(
        board_row(), board_q=[{"player": "P", "season": "2026", "week": "6"}])
    assert reason == "final_board_quarantine"
    kept, _ = screen_one(
        board_row(), board_q=[{"player": "P", "season": "2026", "week": "4"}])
    assert kept is not None, "a week-4 quarantine must not suppress a week-6 row"


def test_prop_quarantine_applies_to_every_week():
    _, reason = screen_one(board_row(), prop_q=[{"player": "P"}])
    assert reason == "prop_quarantine"


def test_book_dedupe_prefers_the_book_that_posts_alt_lines():
    deduped = alt_pool.dedupe_by_book([
        board_row(book="draftkings", edge_pct="0.09"),
        board_row(book="fanduel", edge_pct="0.01"),
    ])
    assert len(deduped) == 1
    assert deduped[0]["book"] == "fanduel", \
        "alt construction needs the book that actually posts alts, not the best main price"


def test_backup_quarterback_is_not_a_playable_role():
    qb2 = {"p": {"position_group": "QB", "prior_games": "8", "current_games": "3",
                 "depth_role": "QB2", "model_role": "QB2",
                 "tgt_share": "0", "rush_share": "0.12"}}
    _, reason = screen_one(
        board_row(market="rush_yards", vegas_line="12", model_proj="20"), form=qb2)
    assert reason == "not_a_playable_role:QB2"


@pytest.mark.parametrize("role", ["TE3", "RB3", "WR4", "WR5"])
def test_deep_backups_are_rejected(role):
    form = {"p": {"position_group": "TE", "prior_games": "15", "current_games": "5",
                  "depth_role": role, "model_role": role,
                  "tgt_share": "0.05", "rush_share": "0"}}
    _, reason = screen_one(board_row(vegas_line="12", model_proj="20"), form=form)
    assert reason == f"not_a_playable_role:{role}"


def test_receiver_rushing_prop_is_a_gadget_market():
    _, reason = screen_one(board_row(market="rush_yards", vegas_line="12", model_proj="20"))
    assert reason == "gadget_market:rush_yards/WR"


@pytest.mark.parametrize("market,line", [
    ("rec_yards", "5.5"), ("rush_yards", "0.5"), ("receptions", "0.5"),
    ("pass_yards", "120"),
])
def test_lines_below_the_floor_are_degenerate(market, line):
    form = {"p": {"position_group": "RB", "prior_games": "15", "current_games": "5",
                  "depth_role": "RB1", "model_role": "RB1",
                  "tgt_share": "0.1", "rush_share": "0.5"}}
    kw = dict(market=market, vegas_line=line, model_proj="999")
    if market == "pass_yards":
        kw.update(qb_synthesis_applied="1", qb_synthesis_proj="999")
        form["p"]["position_group"] = "QB"
        form["p"]["model_role"] = "QB1"
        form["p"]["depth_role"] = "QB1"
    _, reason = screen_one(board_row(**kw), form=form)
    assert reason.startswith("degenerate_low_line")


def test_evidence_floor_defaults_to_ten_games():
    form = {"p": {**WR_FORM["p"], "prior_games": "4", "current_games": "4"}}
    _, reason = screen_one(board_row(vegas_line="12", model_proj="20"), form=form)
    assert reason.startswith("insufficient_player_evidence")


def test_rank_comes_from_model_role_not_the_alignment_slot():
    """The bug this pins: Ourlads' WR1/WR2/WR3 name the LWR/SWR/RWR alignment,
    not the target hierarchy, so the second body at a slot still reads WR1.

    TEN's Calvin Ridley was depth_role=WR1 (Player 2 at LWR) while being
    model_role=WR4 on an 11.7% target share. Gating on depth_role admitted the
    whole receiver room, because every rostered receiver has an alignment label
    in WR1-WR3.
    """
    ridley = {"p": {"position_group": "WR", "prior_games": "6", "current_games": "5",
                    "depth_role": "WR1", "model_role": "WR4",
                    "tgt_share": "0.1166", "rush_share": "0"}}
    _, reason = screen_one(board_row(vegas_line="12.5", model_proj="21.8"), form=ridley)
    assert reason == "not_a_playable_role:WR4"

    # The reverse case has to pass: TEN's actual WR1 is Carnell Tate, whose
    # alignment slot is RWR and so reads depth_role=WR3.
    tate = {"p": {"position_group": "WR", "prior_games": "12", "current_games": "5",
                  "depth_role": "WR3", "model_role": "WR1",
                  "tgt_share": "0.2587", "rush_share": "0"}}
    kept, _ = screen_one(board_row(vegas_line="12.5", model_proj="21.8"), form=tate)
    assert kept is not None, "the room's real WR1 must not be filtered by its alignment slot"
    assert kept["rank_used"] == "WR1"
    assert kept["model_role"] == "WR1"
    assert kept["depth_role"] == "WR3"


def test_missing_rank_fails_closed():
    form = {"p": {**WR_FORM["p"], "model_role": ""}}
    _, reason = screen_one(board_row(vegas_line="12", model_proj="20"), form=form)
    assert reason == "not_a_playable_role:unknown"


def test_quarterback_rank_comes_from_depth_role_not_model_role():
    """model_role ranks quarterbacks by descending ypa, so a backup with a tiny
    sample and a fluky average outranks the starter. For a single-alignment
    position depth_role is the real depth order, so that is what gates here.

    TEN: Trubisky reads model_role=QB1 on ypa 8.94 over four games, while Cam
    Ward -- the actual starter, 22 games, ypa 5.82 -- reads model_role=QB2.
    """
    ward = {"p": {"position_group": "QB", "prior_games": "17", "current_games": "5",
                  "depth_role": "QB1", "model_role": "QB2",
                  "tgt_share": "0", "rush_share": "0.118"}}
    kept, _ = screen_one(board_row(
        market="pass_yards", vegas_line="200", model_proj="225",
        qb_synthesis_applied="1", qb_synthesis_proj="225"), form=ward)
    assert kept is not None, "the certified starter must not be filtered by an ypa rank"
    assert kept["rank_used"] == "QB1"

    # Enough games that the evidence floor is not what rejects him, so the
    # rank gate is the binding one.
    trubisky = {"p": {"position_group": "QB", "prior_games": "12", "current_games": "2",
                      "depth_role": "QB2", "model_role": "QB1",
                      "tgt_share": "0", "rush_share": "0.063"}}
    _, reason = screen_one(board_row(
        market="pass_yards", vegas_line="200", model_proj="225",
        qb_synthesis_applied="1", qb_synthesis_proj="225"), form=trubisky)
    assert reason == "not_a_playable_role:QB2", \
        "an ypa-promoted backup must not reach a ticket as the starter"
