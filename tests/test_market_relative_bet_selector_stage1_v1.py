import numpy as np
import pandas as pd
from scripts.research.market_relative_bet_selector_stage1_v1 import fit_beta, consensus

def test_beta_no_intercept_and_clip():
    raw,b=fit_beta(np.array([1.,2.,3.]),np.array([.5,1.,1.5]))
    assert abs(raw-.5)<1e-12 and abs(b-.5)<1e-12
    raw,b=fit_beta(np.array([1.,2.]),np.array([-1.,-2.]))
    assert raw<0 and b==0.0
    raw,b=fit_beta(np.array([1.,2.]),np.array([2.,4.]))
    assert raw>1 and b==1.0

def test_consensus_line_median_and_book_count():
    p=pd.DataFrame([
      {"game_id":"g","player_clean_key":"p","market":"pass_yards","book":"A","line":250.5},
      {"game_id":"g","player_clean_key":"p","market":"pass_yards","book":"B","line":252.5},
    ])
    c=consensus(p)
    assert len(c)==1
    assert c.iloc[0].consensus_line==251.5
    assert c.iloc[0].consensus_book_count==2
