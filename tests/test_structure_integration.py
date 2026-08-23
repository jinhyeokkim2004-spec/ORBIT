from pathlib import Path
import tempfile
import unittest

try:
    import ase  # noqa: F401
    import spglib  # noqa: F401
except ImportError:
    HAS_STRUCTURE_DEPS = False
else:
    HAS_STRUCTURE_DEPS = True

from orbit.structure import inspect_structure


def cif_text(rows):
    return """data_H2O
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 3.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
""" + "\n".join(rows) + "\n"


@unittest.skipUnless(HAS_STRUCTURE_DEPS, "ASE and spglib are not installed")
class StructureIntegrationTests(unittest.TestCase):
    def test_labels_stay_with_physical_sites_when_cif_rows_are_reordered(self):
        rows = [
            "O O1 1 0 0 0",
            "H H1 1 0 0.25 0",
            "H H2 1 0 0 0.333333333333",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.cif"
            second = root / "second.cif"
            first.write_text(cif_text(rows), encoding="utf-8")
            second.write_text(cif_text(list(reversed(rows))), encoding="utf-8")
            a = inspect_structure(first, 1.0e-3)
            b = inspect_structure(second, 1.0e-3)
            a_sites = {site.site_id: site.frac for site in a.sites}
            b_sites = {site.site_id: site.frac for site in b.sites}
            self.assertEqual(a_sites, b_sites)
            self.assertEqual(set(a_sites), {"O1", "H1", "H2"})
            self.assertEqual(a.structure_fingerprint, b.structure_fingerprint)


if __name__ == "__main__":
    unittest.main()

