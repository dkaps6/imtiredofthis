# Frozen canonical high-value opportunity source safeguards.
from scripts.research.audit_player_high_value_scoring_opportunity_identity_v1 import family

def test_family_mapping():
    assert family("RB")=="RB"
    assert family("FB")=="RB"
    assert family("WR")=="WR"
    assert family("SWR")=="WR"
    assert family("TE")=="TE"
    assert family("QB")=="QB"
    assert family("CB")==""
