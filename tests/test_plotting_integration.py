import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HAS_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("ase", "spglib", "plotly")
)

from orbit.config import load_project_config, set_target_site_ids
from orbit.gaps import extract_target
from orbit.plotting import write_gap_heatmap
from orbit.project import ProjectLayout, initialize_project
from orbit.sampling import build_sampling_plans, write_sampling_plan
from orbit.structure import inspect_structure, write_structure_manifests


FIXTURE = Path(__file__).parent / "fixtures" / "H2O.cif"


@unittest.skipUnless(HAS_DEPS, "ASE, spglib, and Plotly are not installed")
class PlottingIntegrationTests(unittest.TestCase):
    def test_extract_then_write_standalone_heatmap(self):
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

            sample_root = root / "samples" / "O1"
            sampling = json.loads(
                (sample_root / "sampling.json").read_text(encoding="utf-8")
            )
            with (sample_root / "samples.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                samples = list(csv.DictReader(handle))

            calculation_root = root / "calculations" / "samples" / "O1"
            calculation_root.mkdir(parents=True, exist_ok=True)
            (calculation_root / "preparation.json").write_text(
                json.dumps(
                    {
                        "target_id": "O1",
                        "sampling_signature": sampling["sampling_signature"],
                        "preparation_signature": "plot-test",
                    }
                ),
                encoding="utf-8",
            )
            with (calculation_root / "calculations.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                fields = (
                    "sample_name", "preparation_status", "invalid_reason", "qe_output"
                )
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, sample in enumerate(samples):
                    name = sample["sample_name"]
                    output = f"{name}/espresso_scf.pwo"
                    writer.writerow(
                        {
                            "sample_name": name,
                            "preparation_status": "VALID",
                            "invalid_reason": "",
                            "qe_output": output,
                        }
                    )
                    run = calculation_root / name
                    run.mkdir()
                    lumo = -1.0 + 0.01 * index
                    (run / "espresso_scf.pwo").write_text(
                        "highest occupied, lowest unoccupied level (ev): "
                        f"-5.0 {lumo}\nJOB DONE.\n",
                        encoding="utf-8",
                    )

            extracted = extract_target(config, ProjectLayout(root), "O1")
            self.assertEqual(extracted.complete_count, len(samples))
            html, manifest, payload = write_gap_heatmap(
                config, ProjectLayout(root), ["O1"]
            )
            self.assertTrue(html.is_file())
            self.assertTrue(manifest.is_file())
            text = html.read_text(encoding="utf-8")
            self.assertIn("plotly.js", text.lower())
            self.assertIn("O1: sampled HOMO", text)
            self.assertEqual(payload["targets"][0]["complete_count"], len(samples))


if __name__ == "__main__":
    unittest.main()
