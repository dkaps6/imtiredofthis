"""Mechanical M95L reconstruction fixes.

Run #1 failed before any confirmation metrics were produced because the temporal
M94C expanding-margin frame contains only state/margin columns while the frozen
offensive-play model requires the original full pregame feature matrix. This
wrapper reattaches the already-generated pregame margin predictions to the exact
same full target-week feature rows before applying the frozen state/plays models.

Run #2 then failed the 97% M94C player-join source guard. A source-only identity
diagnostic proved all 20 missing rows belong to five stable GSIS identities whose
2023 weekly-roster aliases differ from the frozen M95B keys. The workflow now
reconciles the historical walk-forward inputs by GSIS before scoring. This wrapper
also applies the same verified alias bridge to M95G's direct weekly roster,
injury, and depth source normalizer so the frozen role architecture sees the same
players under the same M95B keys.

No model family, coefficient, feature set, cutoff, probability rule, gate, carry
mean, sportsbook input, or confirmation outcome is changed or inspected here.
"""
import pandas as pd

import scripts.backtest.evaluate_rb_m95l_sealed_temporal_confirmation as m


# Source-only diagnostic run 33426776411 verified these aliases by stable GSIS ID.
# M95G strips suffixes before keying, so both provider and suffix-stripped forms
# are deliberately mapped onto the frozen M95B player_clean_key.
_M95L_ROLE_ALIAS = {
    "jefferywilson": "jeffwilson",
    "jeffwilson": "jeffwilson",
    "kennethgainwell": "kennygainwell",
    "kennygainwell": "kennygainwell",
    "chrisrodriguez": "chrisrodriguezjr",
    "chrisrodriguezjr": "chrisrodriguezjr",
    "kennethwalker": "kennethwalkeriii",
    "kennethwalkeriii": "kennethwalkeriii",
    "christopherbrooks": "chrisbrooks",
    "chrisbrooks": "chrisbrooks",
}
_ORIGINAL_ROLE_NORM = m.g.norm_name


def _m95l_role_norm(value: object) -> str:
    key = _ORIGINAL_ROLE_NORM(value)
    return _M95L_ROLE_ALIAS.get(key, key)


# This changes identity spelling only inside the sealed M95L process. The
# previously validated M95G/H/I/K source files and production code stay untouched.
m.g.norm_name = _m95l_role_norm


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
