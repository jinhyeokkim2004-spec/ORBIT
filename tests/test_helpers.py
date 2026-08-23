from pathlib import Path
import tempfile
import unittest

import numpy as np

from orbit.config import load_project_config
from orbit.helpers import _anchor_displacement, _choose_winner
from orbit.project import initialize_project


class HelperIterationTests(unittest.TestCase):
    def test_helper_defaults_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            initialize_project(root, Path("H2O.cif"))
            config = load_project_config(root)
            self.assertAlmostEqual(config.helper.target_gap_eV, 0.3)
            self.assertAlmostEqual(config.helper.neighbor_cutoff_angstrom, 3.5)
            self.assertEqual(config.helper.scan_axes, "xyz")
            self.assertEqual(config.helper.scan_steps_per_axis, 5)
            self.assertAlmostEqual(config.helper.taper_half_width_images, 8.0)

    def test_winner_tie_break_prefers_smaller_displacement(self):
        rows = [
            {"gap_eV": 1.0, "disp_norm_A": 0.3, "point_index": 0},
            {"gap_eV": 1.0, "disp_norm_A": 0.1, "point_index": 1},
            {"gap_eV": 0.9, "disp_norm_A": 0.0, "point_index": 2},
        ]
        winner = _choose_winner(rows)
        self.assertEqual(winner["point_index"], 1)

    def test_anchor_interpolation_preserves_anchors_and_tapers(self):
        nat = 2
        left = np.zeros((nat, 3), dtype=float)
        right = np.zeros((nat, 3), dtype=float)
        left[1, 0] = 0.2
        right[1, 0] = 0.4
        anchors = [
            {"image": 10, "displacements_A": left.tolist()},
            {"image": 20, "displacements_A": right.tolist()},
        ]
        at_left, meta_left = _anchor_displacement(10, anchors, nat, 8.0)
        at_mid, meta_mid = _anchor_displacement(15, anchors, nat, 8.0)
        outside, meta_out = _anchor_displacement(1, anchors, nat, 8.0)

        np.testing.assert_allclose(at_left, left)
        self.assertEqual(meta_left["region"], "anchor")
        np.testing.assert_allclose(at_mid[1, 0], 0.3)
        self.assertEqual(meta_mid["region"], "between_anchors")
        np.testing.assert_allclose(outside, np.zeros((nat, 3)))
        self.assertEqual(meta_out["region"], "left_taper")


if __name__ == "__main__":
    unittest.main()
