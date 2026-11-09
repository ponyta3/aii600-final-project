"""Run with: python -m unittest -v test_scoring.py (synthetic values only)."""
import unittest
import numpy as np
from scoring import quantile_loss, coverage, february_additive_offset, apply_offset


class TestScoring(unittest.TestCase):
    def test_asymmetric_costs(self):
        self.assertAlmostEqual(quantile_loss([70], [60]), 9.0)
        self.assertAlmostEqual(quantile_loss([50], [60]), 1.0)

    def test_coverage_including_ties(self):
        self.assertAlmostEqual(coverage([40, 60, 80], 60), 2 / 3)

    def test_february_offset_and_application(self):
        # Five SYNTHETIC February residuals: 0, 2, 4, 6, 8.
        y_feb = np.array([50, 52, 54, 56, 58])
        pred_feb = np.repeat(50.0, 5)
        offset = february_additive_offset(y_feb, pred_feb)
        self.assertAlmostEqual(offset, 7.2)
        self.assertAlmostEqual(float(apply_offset([60], offset)[0]), 67.2)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            coverage([40, 50], [60])
        with self.assertRaises(ValueError):
            quantile_loss([], [])
        with self.assertRaises(ValueError):
            february_additive_offset([50], [50], tau=1)


if __name__ == "__main__":
    unittest.main()
