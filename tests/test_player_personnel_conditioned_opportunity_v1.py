import math
import pandas as pd

from scripts.research.evaluate_player_personnel_conditioned_opportunity_v1 import (
    PERSONNEL_BUCKETS,
    canon_player_key,
    personnel_code,
    split_list,
    summarize,
)


def test_personnel_code_contract():
    assert personnel_code("1 RB, 1 TE, 3 WR") == "11"
    assert personnel_code("1 RB; 2 TE; 2 WR") == "12"
    assert personnel_code("2 RB, 1 TE, 2 WR") == "OTHER"
    assert personnel_code("") == "OTHER"


def test_parallel_list_parser():
    assert split_list("00-1;00-2;00-3") == ["00-1", "00-2", "00-3"]
    assert split_list("WR,TE,RB") == ["WR", "TE", "RB"]
    assert split_list(None) == []


def test_player_key_is_stable_nonempty():
    assert canon_player_key("Amon-Ra St. Brown")
    assert canon_player_key("Travis Kelce")
    assert canon_player_key("") == ""


def test_summarize_expected_sign_and_quartile_gap():
    q = pd.DataFrame({
        "player_clean_key": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "personnel_lift": [-.04, -.03, -.02, -.01, .01, .02, .03, .04],
        "realized_opportunity_gain": [-8, -7, -4, -2, 1, 3, 6, 9],
    })
    out = summarize(q)
    assert out["rho"] > 0.9
    assert out["q4_q1_gap"] > 0


def test_frozen_personnel_buckets():
    assert PERSONNEL_BUCKETS == ("11", "12", "OTHER")
