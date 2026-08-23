import csv
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    import ase  # noqa: F401
    import spglib  # noqa: F401
except ImportError:
    HAS_STRUCTURE_DEPS = False
else:
    HAS_STRUCTURE_DEPS = True

from orbit.config import load_project_config, set_target_site_ids
from orbit.project import ProjectLayout, initialize_project
from orbit.sampling import build_sampling_plans, write_sampling_plan
from orbit.structure import inspect_structure, write_structure_manifests


FIXTURE = Path(__file__).parent / "fixtures" / "H2O.cif"


def initialized_project(root: Path, spacing: float = 0.4):
    shutil.copy2(FIXTURE, root / "H2O.cif")
    layout, _ = initialize_project(root, Path("H2O.cif"))
    if spacing != 0.4:
        text = layout.config.read_text(encoding="utf-8")
        layout.config.write_text(
            text.replace("spacing_angstrom = 0.4", f"spacing_angstrom = {spacing}"),
            encoding="utf-8",
        )
    config = load_project_config(root)
    inspection = inspect_structure(config.structure, config.symprec_angstrom)
    write_structure_manifests(inspection, layout.manifests, root)
    set_target_site_ids(layout.config, ["O1", "H1", "H2"])
    return layout, load_project_config(root)


@unittest.skipUnless(HAS_STRUCTURE_DEPS, "ASE and spglib are not installed")
class SamplingIntegrationTests(unittest.TestCase):
    def test_h2o_reproduces_original_grid_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, config = initialized_project(Path(temporary))
            _atoms, plans = build_sampling_plans(config, layout)
            self.assertEqual([plan.target.site_id for plan in plans], ["O1", "H1", "H2"])
            for plan in plans:
                self.assertEqual(plan.divisions, (10, 10, 8))
                self.assertEqual(plan.raw_grid_count, 800)
                self.assertEqual(plan.site_stabilizer_order, 2)
                self.assertEqual(plan.representative_count, 480)
                self.assertEqual(sum(point.is_equilibrium for point in plan.points), 1)

    def test_writes_structures_and_machine_readable_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, config = initialized_project(Path(temporary), spacing=2.0)
            atoms, plans = build_sampling_plans(config, layout, ["O1"])
            output, action = write_sampling_plan(config, layout, atoms, plans[0])
            self.assertEqual(action, "written")
            self.assertTrue((output / "sampling.json").is_file())
            with (output / "samples.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), plans[0].representative_count)
            self.assertEqual(
                len(list((output / "structures").glob("sample_*.cif"))),
                plans[0].representative_count,
            )
            self.assertEqual(sum(row["is_equilibrium"] == "1" for row in rows), 1)
            same_output, second_action = write_sampling_plan(
                config, layout, atoms, plans[0]
            )
            self.assertEqual(same_output, output)
            self.assertEqual(second_action, "skipped_identical")

    def test_direct_overlap_is_recorded_but_has_no_cif(self):
        with tempfile.TemporaryDirectory() as temporary:
            layout, config = initialized_project(Path(temporary), spacing=1.0)
            atoms, plans = build_sampling_plans(config, layout, ["O1"])
            plan = plans[0]
            self.assertGreaterEqual(plan.invalid_count, 1)
            output, _ = write_sampling_plan(config, layout, atoms, plan)
            with (output / "samples.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rejected = [row for row in rows if row["geometry_status"] == "DIRECT_OVERLAP"]
            self.assertEqual(len(rejected), plan.invalid_count)
            self.assertTrue(all(row["structure_file"] == "" for row in rejected))
            self.assertTrue(all(row["invalid_reason"] for row in rejected))
            self.assertEqual(
                len(list((output / "structures").glob("sample_*.cif"))),
                plan.valid_count,
            )


if __name__ == "__main__":
    unittest.main()
