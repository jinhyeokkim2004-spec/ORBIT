import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HAS_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("ase", "spglib")
)

from orbit.config import load_project_config, set_target_site_ids
from orbit.gaps import extract_target
from orbit.pathfinding import find_target_path, widest_winding_path
from orbit.path_gaps import extract_path_target
from orbit.path_plotting import write_path_viewer
from orbit.project import ProjectLayout, initialize_project
from orbit.qe.path_calculations import (
    build_prepared_path_target,
    path_calculation_output_state,
    write_prepared_path_target,
)
from orbit.qe.resources import resolve_qe_resources
from orbit.sampling import build_sampling_plans, write_sampling_plan
from orbit.structure import inspect_structure, write_structure_manifests


FIXTURE = Path(__file__).parent / "fixtures" / "H2O.cif"


@unittest.skipUnless(HAS_DEPS, "ASE and spglib are not installed")
class PathfindingIntegrationTests(unittest.TestCase):
    def test_widest_path_avoids_missing_wall(self):
        import numpy as np

        field = np.ones((3, 3, 3), dtype=float)
        live = np.ones((3, 3, 3), dtype=bool)
        live[1, 0, 0] = False
        threshold, states = widest_winding_path(
            field, live, np.array([3, 3, 3]), (0, 0, 0), (1, 0, 0), 1
        )
        self.assertEqual(threshold, 1.0)
        self.assertIsNotNone(states)
        self.assertNotIn(((1, 0, 0), (0, 0, 0)), states)

    def test_complete_extraction_writes_selected_path_and_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(FIXTURE, root / "H2O.cif")
            layout, _ = initialize_project(root, Path("H2O.cif"))
            layout.config.write_text(
                layout.config.read_text(encoding="utf-8").replace(
                    "spacing_angstrom = 0.4", "spacing_angstrom = 2.0"
                ),
                encoding="utf-8",
            )
            config = load_project_config(root)
            inspection = inspect_structure(config.structure, config.symprec_angstrom)
            write_structure_manifests(inspection, layout.manifests, root)
            set_target_site_ids(layout.config, ["O1"])
            config = load_project_config(root)
            atoms, plans = build_sampling_plans(config, layout, ["O1"])
            write_sampling_plan(config, layout, atoms, plans[0])

            sample_root = root / "samples/O1"
            sampling = json.loads((sample_root / "sampling.json").read_text())
            with (sample_root / "samples.csv").open(newline="", encoding="utf-8") as handle:
                samples = list(csv.DictReader(handle))
            calculation_root = root / "calculations/samples/O1"
            calculation_root.mkdir(parents=True)
            (calculation_root / "preparation.json").write_text(json.dumps({
                "target_id": "O1",
                "sampling_signature": sampling["sampling_signature"],
                "preparation_signature": "path-test",
            }))
            with (calculation_root / "calculations.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                fields = ("sample_name", "preparation_status", "invalid_reason", "qe_output")
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for sample in samples:
                    name = sample["sample_name"]
                    output = f"{name}/espresso_scf.pwo"
                    writer.writerow({
                        "sample_name": name,
                        "preparation_status": "VALID",
                        "invalid_reason": "",
                        "qe_output": output,
                    })
                    folder = calculation_root / name
                    folder.mkdir()
                    (folder / "espresso_scf.pwo").write_text(
                        "highest occupied, lowest unoccupied level (ev): -2.0 -1.0\n"
                        "JOB DONE.\n",
                        encoding="utf-8",
                    )
            extract_target(config, ProjectLayout(root), "O1")
            result = find_target_path(
                config,
                ProjectLayout(root),
                "O1",
                theta_min_eV=0.3,
                n_images=50,
                winding_padding=1,
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual(result.selected_destination, "a")
            self.assertEqual(result.theta_star_eV, 1.0)
            self.assertEqual(len(list((result.run_directory / "cifs").glob("*.cif"))), 50)
            with (result.run_directory / "candidates.csv").open() as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(len(candidates), 7)
            self.assertEqual(candidates[0]["selected"], "1")
            self.assertTrue((result.run_directory / "path.csv").is_file())
            self.assertTrue((result.run_directory / "path.json").is_file())
            transcript = (result.run_directory / "pathfinding.log").read_text()
            self.assertIn("[R=abc]", transcript)
            self.assertIn("[best] O1: R=a", transcript)

            pseudo = root / "pseudo"
            library = pseudo / "library"
            library.mkdir(parents=True)
            (library / "O.test.upf").write_text(
                '<UPF><PP_HEADER z_valence="6.0"/></UPF>\n', encoding="utf-8"
            )
            (library / "H.test.upf").write_text(
                "1.0 Z valence\n", encoding="utf-8"
            )
            (pseudo / "cutoffs.json").write_text(json.dumps({
                "O": {"cutoff_wfc": 45, "cutoff_rho": 360},
                "H": {"cutoff_wfc": 30, "cutoff_rho": 240},
            }))
            layout.config.write_text(
                layout.config.read_text(encoding="utf-8").replace(
                    'root = "../pseudo"', 'root = "pseudo"'
                ),
                encoding="utf-8",
            )
            path_config = load_project_config(root)
            resources = resolve_qe_resources(
                path_config, list(atoms.get_chemical_symbols())
            )
            prepared = build_prepared_path_target(
                path_config,
                ProjectLayout(root),
                resources,
                "O1",
                run_id=None,
                max_concurrent=3,
            )
            self.assertEqual(prepared.path_run_id, result.run_id)
            self.assertEqual(prepared.valid_count, 50)
            self.assertEqual(prepared.invalid_count, 0)
            output, action = write_prepared_path_target(
                path_config, resources, prepared
            )
            self.assertEqual(action, "written")
            self.assertEqual(path_calculation_output_state(prepared), "identical")
            self.assertTrue((output / "image_000/espresso_scf.pwi").is_file())
            self.assertEqual(
                subprocess.run(["bash", "-n", str(output / "run_array.sh")]).returncode,
                0,
            )
            self.assertEqual(
                subprocess.run(["bash", "-n", str(output / "submit.sh")]).returncode,
                0,
            )
            same_output, second_action = write_prepared_path_target(
                path_config, resources, prepared
            )
            self.assertEqual(same_output, output)
            self.assertEqual(second_action, "skipped_identical")

            completed = output / "image_000/espresso_scf.pwo"
            completed.write_text("JOB DONE.\n", encoding="utf-8")
            scheduler_only = build_prepared_path_target(
                path_config,
                ProjectLayout(root),
                resources,
                "O1",
                run_id=result.run_id,
                max_concurrent=4,
            )
            with self.assertRaises(FileExistsError):
                write_prepared_path_target(path_config, resources, scheduler_only)
            write_prepared_path_target(
                path_config, resources, scheduler_only, overwrite=True
            )
            self.assertTrue(completed.is_file())

            (library / "O.test.upf").write_text(
                '<UPF><PP_HEADER z_valence="6.0"/></UPF>\n<!-- changed -->\n',
                encoding="utf-8",
            )
            changed_resources = resolve_qe_resources(
                path_config, list(atoms.get_chemical_symbols())
            )
            physics_changed = build_prepared_path_target(
                path_config,
                ProjectLayout(root),
                changed_resources,
                "O1",
                run_id=result.run_id,
                max_concurrent=4,
            )
            write_prepared_path_target(
                path_config, changed_resources, physics_changed, overwrite=True
            )
            self.assertFalse(completed.exists())
            self.assertEqual(
                len(list(completed.parent.glob("espresso_scf.pwo.stale_*"))), 1
            )

            for image in range(50):
                folder = output / f"image_{image:03d}"
                (folder / "espresso_scf.pwo").write_text(
                    "highest occupied, lowest unoccupied level (ev): "
                    f"-2.0 {-1.0 + 0.01 * image}\nJOB DONE.\n",
                    encoding="utf-8",
                )
            extracted_path = extract_path_target(
                path_config,
                ProjectLayout(root),
                "O1",
                run_id=result.run_id,
            )
            self.assertEqual(extracted_path.complete_count, 50)
            viewer, viewer_manifest, viewer_payload = write_path_viewer(
                path_config,
                ProjectLayout(root),
                ["O1"],
                None,
                run_id=result.run_id,
            )
            self.assertTrue(viewer.is_file())
            self.assertTrue(viewer_manifest.is_file())
            viewer_text = viewer.read_text(encoding="utf-8")
            self.assertIn("HEATMAP", viewer_text)
            self.assertIn("PATH", viewer_text)
            self.assertIn("Play", viewer_text)
            self.assertIn("Pause", viewer_text)
            self.assertIn("Heatmap ON", viewer_text)
            self.assertIn("Heatmap OFF", viewer_text)
            self.assertIn("Path step: ", viewer_text)
            self.assertIn('"visible":false', viewer_text)
            self.assertIn("#7B2CBF", viewer_text)
            self.assertIn("#FFD166", viewer_text)
            self.assertIn("#f4f6f8", viewer_text)
            self.assertIn("#244b74", viewer_text)
            self.assertIn("LIGHT", viewer_text)
            self.assertIn("DARK", viewer_text)
            self.assertIn("setTheme('light')", viewer_text)
            self.assertIn("setTheme('dark')", viewer_text)
            self.assertIn("data-theme=\"dark\"", viewer_text)
            self.assertIn("Plotly.relayout", viewer_text)
            self.assertIn("rgb(244, 250, 255)", viewer_text)
            self.assertIn("fill:#1570EF!important", viewer_text)
            self.assertIn('"type":"mesh3d"', viewer_text)
            self.assertIn("plotly.js", viewer_text.lower())
            self.assertEqual(viewer_payload["targets"][0]["complete_count"], 50)
            self.assertGreater(
                viewer_payload["targets"][0]["atom_sphere_count"],
                len(atoms),
            )
            self.assertGreater(
                viewer_payload["targets"][0]["target_periodic_sphere_count"],
                1,
            )
            self.assertAlmostEqual(
                viewer_payload["targets"][0]["minimum_gap_eV"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
