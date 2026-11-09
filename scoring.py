"""Shared travel-time scoring and February additive calibration for AII 600.

Inputs are trip durations / allowances in minutes. This module does not read
March, choose exclusions, or load data. Pass identically filtered samples to
both baseline and conditional model.
"""

import numpy as np


def _paired(actual, allowance):
    y = np.asarray(actual, dtype=float)
    a = np.asarray(allowance, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("actual must be a nonempty 1-D array")
    if a.ndim == 0:
        a = np.full_like(y, float(a))
    if a.shape != y.shape:
        raise ValueError("allowance must be a scalar or match actual shape")
    if not np.isfinite(y).all() or not np.isfinite(a).all():
        raise ValueError("actual and allowance must contain only finite numbers")
    return y, a


def quantile_loss(actual, allowance, tau=0.90):
    """Mean pinball/asymmetric loss in weighted minutes; shortfall costs tau.

    At tau=.90, a minute under-allowed costs .9 and a minute over-allowed .1.
    """
    if not 0 < tau < 1:
        raise ValueError("tau must be between 0 and 1")
    y, a = _paired(actual, allowance)
    error = y - a
    return float(np.maximum(tau * error, (tau - 1) * error).mean())


def coverage(actual, allowance):
    """Fraction of trips arriving within allowance (actual <= allowance)."""
    y, a = _paired(actual, allowance)
    return float(np.mean(y <= a))


def february_additive_offset(actual_feb, predicted_feb, tau=0.90):
    """Residual tau-quantile for additive calibration; use FEBRUARY ONLY.

    Prediction after calibration: predicted + offset. Negative offsets allowed.
    NumPy's default linear quantile interpolation is used consistently.
    """
    if not 0 < tau < 1:
        raise ValueError("tau must be between 0 and 1")
    y, pred = _paired(actual_feb, predicted_feb)
    return float(np.quantile(y - pred, tau, method="linear"))


def apply_offset(predicted, offset):
    """Add the frozen February offset to predictions for another split."""
    pred = np.asarray(predicted, dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(offset):
        raise ValueError("predicted and offset must be finite")
    return pred + offset
