import unittest
import numpy as np

from orbit.pathfinding import straight_path_images


class StraightPathImagesTests(unittest.TestCase):
    def test_plus_c_path_is_linear_and_unwrapped(self):
        anchor = np.array([0.5, 0.5, 0.5])
        cell = np.diag([4.0, 5.0, 6.0])
        frac, distance = straight_path_images(anchor, (0, 0, 1), cell, 5)
        np.testing.assert_allclose(frac[0], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(frac[-1], [0.5, 0.5, 1.5])
        np.testing.assert_allclose(frac[:, 2], [0.5, 0.75, 1.0, 1.25, 1.5])
        np.testing.assert_allclose(distance, [0.0, 1.5, 3.0, 4.5, 6.0])

    def test_requires_two_images(self):
        with self.assertRaises(ValueError):
            straight_path_images(np.zeros(3), (0, 0, 1), np.eye(3), 1)


if __name__ == "__main__":
    unittest.main()
