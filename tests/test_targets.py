import tempfile
from pathlib import Path
import unittest

from orbit.structure import SiteRecord
from orbit.targets import select_targets, write_targets_manifest


def site(site_id, element, orbit, representative):
    return SiteRecord(
        site_id=site_id,
        ase_index_zero_based=0,
        cif_label=site_id,
        element=element,
        element_occurrence=1,
        frac_x=0.0,
        frac_y=0.0,
        frac_z=0.0,
        cart_x_A=0.0,
        cart_y_A=0.0,
        cart_z_A=0.0,
        symmetry_orbit_id=orbit,
        orbit_representative_site_id=representative,
        is_orbit_representative=site_id == representative,
        orbit_size=2 if site_id != "Ti1" else 1,
        wyckoff="a",
    )


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.sites = [
            site("O1", "O", "orbit_001", "O1"),
            site("O2", "O", "orbit_001", "O1"),
            site("Ti1", "Ti", "orbit_002", "Ti1"),
        ]

    def test_element_selection_uses_one_orbit_representative(self):
        chosen, mode = select_targets(self.sites, elements=["O"])
        self.assertEqual([value.site_id for value in chosen], ["O1"])
        self.assertEqual(mode, "inequivalent_elements")

    def test_explicit_site_selection_preserves_order(self):
        chosen, mode = select_targets(self.sites, site_ids=["Ti1", "O2"])
        self.assertEqual([value.site_id for value in chosen], ["Ti1", "O2"])
        self.assertEqual(mode, "explicit_sites")

    def test_target_manifest_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "targets.csv"
            write_targets_manifest(path, self.sites[:1], "test")
            text = path.read_text(encoding="utf-8")
            self.assertIn("target_id", text)
            self.assertIn("O1", text)


if __name__ == "__main__":
    unittest.main()

