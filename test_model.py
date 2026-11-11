"""Synthetic-data tests; no TLC data or March outcomes involved."""
import unittest

import numpy as np
import pandas as pd

from model import NIGHT, ZONES, bootstrap_allowances, calibrate, design, fit, leave_one_day_out, predict

TRUE_ZONE = {161: 0.0, 162: 5.0, 163: 10.0, 230: -5.0}
Q90_NOISE = 10 * np.log(10)  # 90th percentile of Exp(mean 10)


def synthetic(n, seed, shift=0.0):
    """A weekday rush the weekend does not have, a Friday effect, zone effects and Exp(10) noise."""
    rng = np.random.default_rng(seed)
    pickup = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n), unit="min")
    hour, day = pickup.hour.to_numpy(), pickup.day_of_week.to_numpy()
    zone = rng.choice(ZONES, n)
    night = np.isin(hour, NIGHT)
    centre = (
        np.where(night, 30.0, 35.0 + 0.5 * hour)
        + np.where((day < 5) & (hour >= 15) & (hour <= 18), 20.0, 0.0)
        + np.where(day == 4, 3.0, 0.0)
        + pd.Series(zone).map(TRUE_ZONE).to_numpy()
    )
    return pd.DataFrame({
        "tpep_pickup_datetime": pickup,
        "PULocationID": zone,
        "duration_minutes": centre + rng.exponential(10, n) + shift,
        "true_q90": centre + Q90_NOISE + shift,
    })


class TestQuantileModel(unittest.TestCase):
    def test_recovers_known_quantile(self):
        jan = synthetic(60_000, seed=1)
        pred = predict(fit(jan), jan)
        self.assertLess(np.abs(pred - jan["true_q90"]).mean(), 1.0)
        self.assertAlmostEqual(np.mean(jan["duration_minutes"] <= pred), 0.90, delta=0.01)

    def test_recovers_known_parameters(self):
        params = fit(synthetic(60_000, seed=2))
        self.assertAlmostEqual(params["weekday_17"] - params["weekend_17"], 20.0, delta=3.0)
        self.assertAlmostEqual(params["Fri"], 3.0, delta=1.5)
        self.assertAlmostEqual(params["zone_163"], 10.0, delta=1.5)

    def test_main_effects_miss_the_weekday_rush(self):
        jan = synthetic(60_000, seed=3)
        main = np.abs(predict(fit(jan, by_day_type=False), jan) - jan["true_q90"]).mean()
        both = np.abs(predict(fit(jan), jan) - jan["true_q90"]).mean()
        self.assertLess(both, main)

    def test_offset_absorbs_a_level_shift(self):
        params = fit(synthetic(60_000, seed=4))
        feb = synthetic(20_000, seed=5, shift=6.0)
        offset, allowance = calibrate(params, feb)
        self.assertAlmostEqual(offset, 6.0, delta=1.0)
        self.assertAlmostEqual(np.mean(feb["duration_minutes"] <= allowance), 0.90, delta=0.001)

    def test_design_has_fixed_columns(self):
        one = synthetic(1, seed=6)
        self.assertEqual(design(one).shape, (1, 2 * 17 + 5 + 3))
        self.assertEqual(design(one, by_day_type=False).shape, (1, 17 + 6 + 3))
        self.assertEqual(float(design(one).iloc[0, :34].sum()), 1.0)

    def test_predict_matches_columns_by_name(self):
        jan = synthetic(5_000, seed=7)
        params = fit(jan)
        np.testing.assert_allclose(predict(params.iloc[::-1], jan), predict(params, jan))

    def test_leave_one_day_out_scores_every_date(self):
        jan = synthetic(8_000, seed=8)
        loss = leave_one_day_out(jan)
        self.assertEqual(len(loss), 28)
        self.assertTrue((loss > 0).all())

    def test_invalid_input_rejected(self):
        jan = synthetic(5_000, seed=9)
        pickup = pd.to_datetime(jan["tpep_pickup_datetime"])
        bad_duration = jan.assign(duration_minutes=jan["duration_minutes"].where(jan.index > 0, 0.0))
        bad_zone = jan.assign(PULocationID=jan["PULocationID"].where(jan.index > 0, 132))
        empty_cell = jan[~((pickup.dt.day_of_week >= 5) & (pickup.dt.hour == 5))]
        no_monday = jan[pickup.dt.day_of_week != 0]
        no_saturday = jan[pickup.dt.day_of_week != 5]
        for bad in (bad_duration, bad_zone, empty_cell, no_monday, no_saturday):
            with self.assertRaises(ValueError):
                fit(bad)

    def test_bootstrap_allowances_shapes_and_seed(self):
        jan, feb = synthetic(8_000, seed=10), synthetic(500, seed=11)
        allowances, offsets, skipped = bootstrap_allowances(jan, feb, draws=5, seed=1)
        self.assertEqual(allowances.shape, (5 - skipped, 500))
        self.assertEqual(len(offsets), 5 - skipped)
        self.assertGreater(allowances.std(axis=0).mean(), 0)
        np.testing.assert_array_equal(allowances, bootstrap_allowances(jan, feb, draws=5, seed=1)[0])


if __name__ == "__main__":
    unittest.main()
