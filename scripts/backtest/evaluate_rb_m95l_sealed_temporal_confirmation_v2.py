"""Mechanical M95L reconstruction fix.

Run #1 failed before any confirmation metrics were produced because the temporal
M94C expanding-margin frame contains only state/margin columns while the frozen
offensive-play model requires the original full pregame feature matrix. This
wrapper reattaches the already-generated pregame margin predictions to the exact
same full target-week feature rows before applying the frozen state/plays models.

No model family, coefficient, feature set, cutoff, probability rule, gate, or
confirmation outcome is changed or inspected by this patch.
"""
import pandas as pd

import scripts.backtest.evaluate_rb_m95l_sealed_temporal_confirmation as m


def _fixed_m94c_week_prediction(x: pd.DataFrame, pred: pd.DataFrame, oof_margin: pd.DataFrame,
                                features: list[str], week: int):
    train = x.loc[m.num(x["week"]).lt(week)].copy()
    margin = oof_margin.loc[m.num(oof_margin["week"]).eq(week)].copy()
    mapper_train = oof_margin.loc[m.num(oof_margin["week"]).lt(week)].copy()
    if len(train) < 90 or len(mapper_train) < 90 or margin.empty:
        raise RuntimeError(f"M95L insufficient M94C temporal training for week {week}")

    test = x.loc[m.num(x["week"]).eq(week)].copy()
    add = m.TEAM_KEYS + ["pred_mean_margin", "pred_final_margin"]
    test = test.merge(
        margin[add].drop_duplicates(m.TEAM_KEYS),
        on=m.TEAM_KEYS, how="left", validate="one_to_one",
    )
    if test[["pred_mean_margin", "pred_final_margin"]].isna().any().any():
        raise RuntimeError(f"M95L temporal margin join incomplete for week {week}")
    test = m.c._margin_state_features(test)
    test = m.c._fit_predict_state_mapper(mapper_train, test, m.M94C_STATE_FAMILY)
    test = m.c._fit_predict_plays(train, test, features, m.M94C_PLAY_FAMILY)
    test["structured_team_rush_att"] = m.c._structured_team_rush(test)
    test["candidate_team_rush_att"] = (
        (1.0 - m.M94C_ALPHA) * m.num(test["baseline_team_rush_att"])
        + m.M94C_ALPHA * m.num(test["structured_team_rush_att"])
    ).clip(8, 50)
    rb = m.m94._player_candidate(pred, test, "candidate_team_rush_att")
    rb = rb.loc[m.num(rb["week"]).eq(week)].copy()
    return test, rb


m._m94c_week_prediction = _fixed_m94c_week_prediction

if __name__ == "__main__":
    raise SystemExit(m.main())
