"""Synthetic-data tests; no TLC data or March outcomes involved."""
import unittest
import numpy as np
from baseline import january_p90, calibrate_fixed_p90


class TestFixedP90(unittest.TestCase):
    def test_january_quantile_linear(self):
        self.assertAlmostEqual(january_p90([10, 20, 30, 40, 50]), 46.0)

    def test_february_additive_calibration(self):
        offset, final = calibrate_fixed_p90(46.0, [15, 25, 35, 45, 55])
        self.assertAlmostEqual(offset, 5.0)  # February P90 is 51; 51 - 46 = 5
        self.assertAlmostEqual(final, 51.0)

    def test_negative_offset_is_allowed(self):
        offset, final = calibrate_fixed_p90(60.0, [10, 20, 30, 40, 50])
        self.assertAlmostEqual(offset, -14.0)
        self.assertAlmostEqual(final, 46.0)

    def test_invalid_durations_rejected(self):
        for bad in ([], [10, 0], [10, np.nan], [[10, 20]]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    january_p90(bad)


if __name__ == "__main__":
    unittest.main()
