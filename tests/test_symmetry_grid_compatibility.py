import unittest

import numpy as np

from orbit.sampling import symmetry_compatible_grid_divisions
from orbit.symmetry import SymmetryOperations


class SymmetryCompatibleGridTests(unittest.TestCase):
    def setUp(self):
        self.stabilizer = SymmetryOperations(
            rotations=np.asarray(
                [
                    [
                        [-1, 0, 1],
                        [0, -1, 1],
                        [0, 0, 1],
                    ],
                    [
                        [0, -1, 0],
                        [1, 0, -1],
                        [0, 0, -1],
                    ],
                    [
                        [0, 1, -1],
                        [-1, 0, 0],
                        [0, 0, -1],
                    ],
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ],
                dtype=int,
            ),
            translations=np.zeros((4, 3), dtype=float),
        )

    def test_refines_liinse2_grid(self):
        result = symmetry_compatible_grid_divisions(
            np.asarray([16, 16, 20], dtype=int),
            self.stabilizer,
        )
        self.assertEqual(tuple(result), (20, 20, 20))

    def test_preserves_already_compatible_grid(self):
        result = symmetry_compatible_grid_divisions(
            np.asarray([20, 20, 20], dtype=int),
            self.stabilizer,
        )
        self.assertEqual(tuple(result), (20, 20, 20))


if __name__ == "__main__":
    unittest.main()
