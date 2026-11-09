"""Fixed January P90 travel-time allowance and February additive calibration.

Fit and calibrate on cleaned ordinary-day trips only. March is held out.
"""
import numpy as np
from scoring import february_additive_offset, apply_offset


def _durations(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all() or not (y > 0).all():
        raise ValueError("durations must be a nonempty 1-D array of positive finite minutes")
    return y


def january_p90(january_durations):
    """Fit the constant allowance with linear quantile interpolation."""
    return float(np.quantile(_durations(january_durations), 0.90, method="linear"))


def calibrate_fixed_p90(january_allowance, february_durations):
    """Return (February additive offset, frozen calibrated allowance), in minutes."""
    feb = _durations(february_durations)
    if not np.isfinite(january_allowance) or january_allowance <= 0:
        raise ValueError("January allowance must be positive and finite")
    offset = february_additive_offset(feb, float(january_allowance), tau=0.90)
    final = float(apply_offset(float(january_allowance), offset))
    if not np.isfinite(final) or final <= 0:
        raise ValueError("calibrated allowance must be positive and finite")
    return offset, final
