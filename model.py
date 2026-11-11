"""Quantile regression on pickup hour, day of week and pickup zone, with February additive calibration.

Fit on cleaned ordinary-day January trips, calibrate on February, then freeze (decisions.md,
2026-10-26 and 2026-11-10 entries). March is held out.

    Q_0.9(T | h, d, z) = a[w(d), b(h)] + g_d + c_z,   g_Mon = g_Sat = 0,  c_161 = 0
    allowance          = Q_0.9(T | h, d, z) + February offset

w(d) is weekday or weekend. b(h) is the pickup hour, with 8 pm to 4 am pooled into one night cell
(each of those hours has fewer than 100 January trips). Weekdays and weekends each get their own
hourly profile a; g_d moves the profile up or down for the day, c_z for the zone.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from evaluation import resample_days
from scoring import february_additive_offset, apply_offset, quantile_loss

TAU = 0.90
ZONES = (161, 162, 163, 230)
NIGHT = (20, 21, 22, 23, 0, 1, 2, 3)
CELLS = ("night",) + tuple(f"{h:02d}" for h in range(4, 20))
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _durations(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all() or not (y > 0).all():
        raise ValueError("durations must be a nonempty 1-D array of positive finite minutes")
    return y


def design(trips, by_day_type=True):
    """Indicator columns, the same for any input.

    by_day_type=True: one column per (weekday or weekend, hour cell), plus Tue-Fri, Sun and zone shifts.
    by_day_type=False: main effects only (hour cell, day, zone), kept for the structure check.
    """
    pickup = pd.to_datetime(trips["tpep_pickup_datetime"])
    hour = pickup.dt.hour.to_numpy()
    day = pickup.dt.day_of_week.to_numpy()  # Monday = 0
    zone = trips["PULocationID"].to_numpy()
    if not np.isin(zone, ZONES).all():
        raise ValueError(f"pickup zones must be in {ZONES}")
    cell = np.where(np.isin(hour, NIGHT), "night", np.char.zfill(hour.astype(str), 2))
    X = pd.DataFrame(index=trips.index)
    if by_day_type:
        day_type = np.where(day >= 5, "weekend", "weekday")
        for w in ("weekday", "weekend"):
            for c in CELLS:
                X[f"{w}_{c}"] = ((day_type == w) & (cell == c)).astype(float)
        shifted = (1, 2, 3, 4, 6)  # Monday and Saturday are the two profiles' reference days
    else:
        for c in CELLS:
            X[f"hour_{c}"] = (cell == c).astype(float)
        shifted = (1, 2, 3, 4, 5, 6)
    for d in shifted:
        X[DAY_NAMES[d]] = (day == d).astype(float)
    for z in ZONES[1:]:
        X[f"zone_{z}"] = (zone == z).astype(float)
    return X


def fit(trips, tau=TAU, by_day_type=True):
    """Coefficients of the tau-quantile regression in minutes, as a Series indexed by design column."""
    y = _durations(trips["duration_minutes"])
    X = design(trips, by_day_type)
    empty = X.columns[X.sum() == 0]
    if len(empty):
        raise ValueError(f"no fit trips for: {', '.join(empty)}")
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("coefficients are not identified: a reference day (Monday, Saturday) has no fit trips")
    return sm.QuantReg(y, X).fit(q=tau, max_iter=10000).params


def predict(params, trips):
    """Uncalibrated tau-quantile in minutes for each trip."""
    X = design(trips, by_day_type="weekday_night" in params.index)
    return X[params.index].to_numpy() @ params.to_numpy()


def calibrate(params, february_trips, tau=TAU):
    """Return (February additive offset, calibrated February allowances), in minutes."""
    feb = _durations(february_trips["duration_minutes"])
    predicted = predict(params, february_trips)
    offset = february_additive_offset(feb, predicted, tau=tau)
    return offset, apply_offset(predicted, offset)


def leave_one_day_out(trips, tau=TAU, by_day_type=True):
    """Quantile loss on each pickup date when the model is fit on all the other dates."""
    date = pd.to_datetime(trips["tpep_pickup_datetime"]).dt.date
    loss = {}
    for d in sorted(date.unique()):
        held = (date == d).to_numpy()
        params = fit(trips.loc[~held], tau, by_day_type)
        loss[d] = quantile_loss(trips.loc[held, "duration_minutes"], predict(params, trips.loc[held]), tau)
    return pd.Series(loss, name="quantile_loss")


def bootstrap_allowances(fit_trips, calibrate_trips, draws=300, seed=600, tau=TAU):
    """Refit on resampled fit days and recalibrate on resampled calibration days.

    Fit days are resampled within each day of the week, so every draw has all seven. Returns the calibrated
    allowance of every calibration trip in every draw (draws x trips), the offsets, and the draws skipped
    because a cell was empty.
    """
    rng = np.random.default_rng(seed)
    allowances, offsets, skipped = [], [], 0
    for _ in range(draws):
        try:
            params = fit(resample_days(fit_trips, rng, by_weekday=True), tau)
        except ValueError:
            skipped += 1
            continue
        offset, _ = calibrate(params, resample_days(calibrate_trips, rng), tau)
        offsets.append(offset)
        allowances.append(predict(params, calibrate_trips) + offset)
    return np.array(allowances), np.array(offsets), skipped
