import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from orbit.config import load_project_config, set_target_site_ids
from orbit.gaps import GAP_FIELDS, extract_target, parse_qe_output
from orbit.project import ProjectLayout, initialize_project


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class GapExtractionTests(unittest.TestCase):
    def test_parser_uses_final_pair_and_fortran_exponents(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "espresso_scf.pwo"
            output.write_text(
                "highest occupied, lowest unoccupied level (ev): -5.0 -1.0\n"
                "highest occupied, lowest unoccupied level (ev): -4.2D+00 -1.1D+00\n"
                "the Fermi energy is -2.0 ev\nJOB DONE.\n",
                encoding="utf-8",
            )
            parsed = parse_qe_output(output)
            self.assertEqual(parsed.status, "COMPLETE")
            self.assertAlmostEqual(parsed.homo_ev, -4.2)
            self.assertAlmostEqual(parsed.lumo_ev, -1.1)
            self.assertAlmostEqual(parsed.gap_ev, 3.1)

    def test_levels_without_job_done_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "espresso_scf.pwo"
            output.write_text(
                "highest occupied, lowest unoccupied level (ev): -4.0 -2.0\n",
                encoding="utf-8",
            )
            parsed = parse_qe_output(output)
            self.assertEqual(parsed.status, "INCOMPLETE_OUTPUT")
            self.assertFalse(parsed.job_done)
            self.assertEqual(parsed.gap_ev, 2.0)

    def test_extraction_keeps_complete_incomplete_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structure = root / "H2O.cif"
            structure.write_text("test structure bytes\n", encoding="utf-8")
            layout, _ = initialize_project(root, Path("H2O.cif"))
            set_target_site_ids(layout.config, ["O1"])
            layout.manifests.mkdir(exist_ok=True)
            layout.structure_manifest.write_text(
                json.dumps(
                    {
                        "source_sha256": hashlib.sha256(structure.read_bytes()).hexdigest(),
                        "structure_fingerprint": "fixture",
                    }
                ),
                encoding="utf-8",
            )

            sample_root = root / "samples" / "O1"
            sample_root.mkdir(parents=True)
            (sample_root / "sampling.json").write_text(
                json.dumps({"target_id": "O1", "sampling_signature": "sample-sig"}),
                encoding="utf-8",
            )
            sample_fields = (
                "sample_id", "sample_name", "target_id", "element",
                "frac_x", "frac_y", "frac_z", "cart_x_A", "cart_y_A", "cart_z_A",
                "displacement_x_A", "displacement_y_A", "displacement_z_A",
                "displacement_norm_A", "orbit_size", "point_stabilizer_order",
                "is_equilibrium",
            )
            sample_rows = []
            for index in range(3):
                sample_rows.append(
                    {
                        "sample_id": index,
                        "sample_name": f"sample_{index:05d}",
                        "target_id": "O1",
                        "element": "O",
                        "frac_x": index / 3,
                        "frac_y": 0,
                        "frac_z": 0,
                        "cart_x_A": index,
                        "cart_y_A": 0,
                        "cart_z_A": 0,
                        "displacement_x_A": index,
                        "displacement_y_A": 0,
                        "displacement_z_A": 0,
                        "displacement_norm_A": index,
                        "orbit_size": 1,
                        "point_stabilizer_order": 2,
                        "is_equilibrium": int(index == 0),
                    }
                )
            _write_csv(sample_root / "samples.csv", sample_fields, sample_rows)

            calculation_root = root / "calculations" / "samples" / "O1"
            calculation_root.mkdir(parents=True)
            (calculation_root / "preparation.json").write_text(
                json.dumps(
                    {
                        "target_id": "O1",
                        "sampling_signature": "sample-sig",
                        "preparation_signature": "prep-sig",
                    }
                ),
                encoding="utf-8",
            )
            calculation_fields = (
                "sample_name", "preparation_status", "invalid_reason", "qe_output"
            )
            calculation_rows = []
            for index in range(3):
                name = f"sample_{index:05d}"
                calculation_rows.append(
                    {
                        "sample_name": name,
                        "preparation_status": "VALID",
                        "invalid_reason": "",
                        "qe_output": f"{name}/espresso_scf.pwo",
                    }
                )
            _write_csv(
                calculation_root / "calculations.csv",
                calculation_fields,
                calculation_rows,
            )
            for index, completed in ((0, True), (1, False)):
                run = calculation_root / f"sample_{index:05d}"
                run.mkdir()
                text = "highest occupied, lowest unoccupied level (ev): -5.0 -1.0\n"
                if completed:
                    text += "JOB DONE.\n"
                (run / "espresso_scf.pwo").write_text(text, encoding="utf-8")

            config = load_project_config(root)
            result = extract_target(config, ProjectLayout(root), "O1")
            self.assertEqual(result.total_count, 3)
            self.assertEqual(
                result.status_counts,
                {"COMPLETE": 1, "INCOMPLETE_OUTPUT": 1, "MISSING_OUTPUT": 1},
            )
            with result.gaps_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), GAP_FIELDS)
            self.assertEqual([row["output_status"] for row in rows], [
                "COMPLETE", "INCOMPLETE_OUTPUT", "MISSING_OUTPUT"
            ])


if __name__ == "__main__":
    unittest.main()
