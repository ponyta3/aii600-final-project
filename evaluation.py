"""Scores in the units of the decision, and uncertainty by resampling whole days.

Trips on one day share its traffic, so the day is the unit that is resampled (decisions.md,
2026-11-10 scoring entry). Nothing here loads data or reads March; pass the trips in.
"""
import numpy as np
import pandas as pd

from scoring import _paired, coverage, quantile_loss


def trip_loss(actual, allowance, tau=0.90):
    """Quantile loss of each trip, in weighted minutes."""
    y, a = _paired(actual, allowance)
    return np.maximum(tau * (y - a), (tau - 1) * (y - a))


def summary(actual, allowance, tau=0.90):
    """One method on one set of trips: the loss and the minutes behind it."""
    y, a = _paired(actual, allowance)
    late = y > a
    return {
        "trips": int(y.size),
        "mean loss (weighted min)": quantile_loss(y, a, tau),
        "coverage": coverage(y, a),
        "late trips per 100": 100 * float(late.mean()),
        "minutes late, when late": float((y - a)[late].mean()) if late.any() else 0.0,
        "minutes unused, when on time": float((a - y)[~late].mean()) if (~late).any() else 0.0,
        "mean allowance (min)": float(a.mean()),
    }


def score_table(actual, allowances, tau=0.90):
    """summary() for each named method, one row per method."""
    table = pd.DataFrame({name: summary(actual, a, tau) for name, a in allowances.items()}).T
    return table.astype({"trips": int})


def resample_days(trips, rng, by_weekday=False, date_col="tpep_pickup_datetime"):
    """All trips of days drawn with replacement; by_weekday=True draws within each day of the week."""
    date = pd.to_datetime(trips[date_col]).dt.normalize()
    days = pd.Series(date.unique()).sort_values()
    groups = [g.to_numpy() for _, g in days.groupby(days.dt.day_of_week)] if by_weekday else [days.to_numpy()]
    drawn = np.concatenate([rng.choice(g, len(g)) for g in groups])
    rows = trips.groupby(date.to_numpy()).indices
    return trips.iloc[np.concatenate([rows[d] for d in drawn])].reset_index(drop=True)


def day_bootstrap_difference(dates, loss_a, loss_b, draws=10_000, seed=600):
    """Mean loss of a minus mean loss of b over trips, on resampled days (the same days for both)."""
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("the two losses must be nonempty 1-D arrays of the same length")
    by_day = pd.DataFrame({"d": a - b, "n": 1.0}).groupby(pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()).sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(by_day), size=(draws, len(by_day)))
    return by_day["d"].to_numpy()[pick].sum(axis=1) / by_day["n"].to_numpy()[pick].sum(axis=1)
