import unittest

try:
    import ase  # noqa: F401
except ImportError:
    HAS_ASE = False
else:
    HAS_ASE = True

from orbit.periodic import minimum_image_distance, wrap_fractional


@unittest.skipUnless(HAS_ASE, "ASE is not installed")
class PeriodicTests(unittest.TestCase):
    def test_wrap_uses_half_open_cell(self):
        wrapped = wrap_fractional([-0.1, 1.0, 2.25])
        self.assertAlmostEqual(float(wrapped[0]), 0.9)
        self.assertAlmostEqual(float(wrapped[1]), 0.0)
        self.assertAlmostEqual(float(wrapped[2]), 0.25)

    def test_minimum_image_distance_supports_skew_cell(self):
        cell = [[3.0, 0.0, 0.0], [1.0, 2.8, 0.0], [0.2, 0.3, 4.0]]
        distance = minimum_image_distance([0.95, 0.0, 0.0], [0.05, 0.0, 0.0], cell)
        self.assertAlmostEqual(distance, 0.3, places=12)


if __name__ == "__main__":
    unittest.main()

