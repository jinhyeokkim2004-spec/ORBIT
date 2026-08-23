import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from orbit.records.pathfinding import CandidateRecord, PathRunRecorder


def candidate(order, destination, theta):
    found = theta is not None
    return CandidateRecord(
        candidate_order=order,
        destination_id=destination,
        displacement_frac_a=1.0,
        displacement_frac_b=0.0,
        displacement_frac_c=0.0,
        exit_grid_i=0,
        exit_grid_j=0,
        exit_grid_k=0,
        winding_a=1,
        winding_b=0,
        winding_c=0,
        route_found=found,
        theta_star_eV=theta,
        theta_min_eV=0.3,
        feasibility="INSULATING" if found and theta > 0.3 else "NO_ROUTE",
        exact_node_count=18 if found else None,
        reason="",
    )


class PathRunRecorderTests(unittest.TestCase):
    def test_records_candidates_transcript_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            stream = io.StringIO()
            with PathRunRecorder(
                Path(temporary) / "paths",
                "O1",
                inputs={"gap_table": "results/gaps/O1/gaps.csv"},
                parameters={"theta_min_eV": 0.3},
                run_id="test-run",
                stream=stream,
            ) as recorder:
                recorder.set_grid_summary(shape=[10, 10, 10], live_nodes=950)
                recorder.add_candidate(candidate(0, "a", 0.45))
                recorder.add_candidate(candidate(1, "b", None))
                recorder.finish("SUCCEEDED", selected_destination="a")

            run_dir = Path(temporary) / "paths/O1/runs/test-run"
            manifest = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(manifest["status"], "SUCCEEDED")
            self.assertEqual(manifest["candidate_count"], 2)
            self.assertEqual(manifest["selected_destination"], "a")

            with (run_dir / "candidates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["destination_id"] for row in rows], ["a", "b"])
            self.assertEqual(rows[0]["theta_star_eV"], "0.45")
            self.assertEqual(rows[0]["selected"], "1")
            self.assertEqual(rows[1]["theta_star_eV"], "")

            transcript = (run_dir / "pathfinding.log").read_text()
            self.assertIn("[R=a] theta* = 0.450000 eV -> INSULATING", transcript)
            self.assertIn("[R=b] theta* = -inf eV -> no route", transcript)
            self.assertIn("[finish] status=SUCCEEDED; selected=a", transcript)
            self.assertIn("[R=a]", stream.getvalue())

            with (Path(temporary) / "paths/O1/index.csv").open() as handle:
                index_rows = list(csv.DictReader(handle))
            self.assertEqual(index_rows[0]["run_id"], "test-run")

    def test_exception_marks_run_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "numerical failure"):
                with PathRunRecorder(
                    Path(temporary) / "paths",
                    "Ti1",
                    inputs={},
                    parameters={},
                    run_id="failed-run",
                    stream=io.StringIO(),
                ):
                    raise RuntimeError("numerical failure")

            run_dir = Path(temporary) / "paths/Ti1/runs/failed-run"
            manifest = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertIn("numerical failure", manifest["message"])


if __name__ == "__main__":
    unittest.main()
