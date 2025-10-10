# scripts/utils/odds.py
"""
Odds conversion utilities:
- American <-> Decimal <-> Implied Probability
- De-vig (remove bookmaker margin)
"""

def american_to_decimal(odds):
    """Convert American odds to decimal."""
    try:
        odds = float(odds)
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    except Exception:
        return None


def decimal_to_american(odds):
    """Convert decimal odds to American odds."""
    try:
        odds = float(odds)
        if odds >= 2:
            return round((odds - 1) * 100)
        else:
            return round(-100 / (odds - 1))
    except Exception:
        return None


def implied_prob_from_american(odds):
    """Return implied probability (0–1) from American odds."""
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except Exception:
        return None


def implied_prob_from_decimal(odds):
    """Return implied probability (0–1) from decimal odds."""
    try:
        return 1 / float(odds)
    except Exception:
        return None


def devig_two_way(prob1, prob2):
    """
    Remove the vig (juice) from a two-way market using normalized probabilities.
    """
    total = prob1 + prob2
    if total == 0:
        return (0.5, 0.5)
    return (prob1 / total, prob2 / total)


def expected_value(prob_model, odds_american):
    """Compute EV% given model probability and American odds."""
    p = prob_model
    o = american_to_decimal(odds_american)
    return (p * (o - 1)) - (1 - p)
