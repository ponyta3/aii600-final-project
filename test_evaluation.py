"""Synthetic-data tests; no TLC data or March outcomes involved."""
import unittest

import numpy as np
import pandas as pd

from evaluation import day_bootstrap_difference, resample_days, score_table, summary, trip_loss
from scoring import quantile_loss


def days_of_trips(n_days=28, per_day=50, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    pickup = [start + pd.Timedelta(days=d, minutes=int(m)) for d in range(n_days) for m in rng.integers(0, 1440, per_day)]
    return pd.DataFrame({"tpep_pickup_datetime": pickup, "duration_minutes": rng.exponential(10, n_days * per_day) + 40})


class TestEvaluation(unittest.TestCase):
    def test_trip_loss_averages_to_quantile_loss(self):
        y, a = [50, 60, 70, 80], [65, 65, 65, 65]
        np.testing.assert_allclose(trip_loss(y, a), [1.5, 0.5, 4.5, 13.5])
        self.assertAlmostEqual(trip_loss(y, a).mean(), quantile_loss(y, a))

    def test_summary_in_minutes(self):
        s = summary([50, 60, 70, 80], 65)
        self.assertEqual(s["trips"], 4)
        self.assertAlmostEqual(s["coverage"], 0.5)
        self.assertAlmostEqual(s["late trips per 100"], 50.0)
        self.assertAlmostEqual(s["minutes late, when late"], 10.0)
        self.assertAlmostEqual(s["minutes unused, when on time"], 10.0)
        self.assertAlmostEqual(s["mean loss (weighted min)"], 5.0)
        self.assertEqual(list(score_table([50, 60], {"a": 55, "b": [70, 70]}).index), ["a", "b"])

    def test_resample_days_keeps_whole_days(self):
        trips = days_of_trips()
        drawn = resample_days(trips, np.random.default_rng(1))
        self.assertEqual(len(drawn), len(trips))
        self.assertLess(drawn["tpep_pickup_datetime"].dt.date.nunique(), 28)
        self.assertTrue((drawn.groupby(drawn["tpep_pickup_datetime"].dt.date).size() % 50 == 0).all())

    def test_resample_within_weekday_keeps_every_weekday(self):
        trips = days_of_trips()
        for seed in range(20):
            drawn = resample_days(trips, np.random.default_rng(seed), by_weekday=True)
            self.assertEqual(drawn["tpep_pickup_datetime"].dt.day_of_week.value_counts().to_dict(), {d: 200 for d in range(7)})

    def test_day_bootstrap_sees_day_level_noise(self):
        trips = days_of_trips(per_day=200)
        rng = np.random.default_rng(2)
        day = trips["tpep_pickup_datetime"].dt.normalize()
        shock = day.map(dict(zip(day.unique(), rng.normal(0, 2, day.nunique())))).to_numpy()
        a, b = 3.0 + shock + rng.normal(0, 1, len(trips)), np.full(len(trips), 2.0)
        diff = day_bootstrap_difference(day, a, b, draws=2000, seed=3)
        self.assertAlmostEqual(diff.mean(), (a - b).mean(), delta=0.05)
        self.assertAlmostEqual(diff.std(), 2 / np.sqrt(28), delta=0.15)  # far above the trip-level 0.03
        np.testing.assert_array_equal(diff, day_bootstrap_difference(day, a, b, draws=2000, seed=3))

    def test_mismatched_losses_rejected(self):
        with self.assertRaises(ValueError):
            day_bootstrap_difference(["2024-01-01"], [1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
