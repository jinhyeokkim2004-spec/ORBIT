from pathlib import Path

import numpy as np

from orbit.qe.path_response_analysis import (
    _closest_branch_vector,
    _minimum_image_displacements,
    _nearest_integer,
    parse_asr_born_charges,
    parse_polarization_output,
)


def test_nearest_integer_ties_away_from_zero():
    assert _nearest_integer(0.5) == 1
    assert _nearest_integer(-0.5) == -1


def test_minimum_image_displacements():
    cell = np.eye(3) * 4.0
    previous = np.array([[3.9, 0.0, 0.0], [1.0, 1.0, 1.0]])
    current = np.array([[0.1, 0.0, 0.0], [0.8, 1.2, 1.0]])
    result = _minimum_image_displacements(previous, current, cell)
    assert np.allclose(result, [[0.2, 0.0, 0.0], [-0.2, 0.2, 0.0]])


def test_joint_branch_selection_in_nonorthogonal_cell():
    unit_direct = np.array(
        [[1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    raw = np.array([0.1, 0.2, 0.3])
    quantum = np.ones(3)
    expected_indices = np.array([1, -1, 0])
    expected_coefficients = raw + expected_indices * quantum
    estimate = expected_coefficients @ unit_direct + np.array([0.01, -0.01, 0.0])
    indices, coefficients, cartesian = _closest_branch_vector(
        raw, quantum, estimate, unit_direct
    )
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(coefficients, expected_coefficients)
    assert np.allclose(cartesian, expected_coefficients @ unit_direct)


def test_qe_response_parsers(tmp_path: Path):
    polarization = tmp_path / "espresso_pol_gdir1.pwo"
    polarization.write_text(
        "P = -1.5 (mod 3.0) C/m^2\nJOB DONE.\n", encoding="utf-8"
    )
    raw, quantum, printed, _ = parse_polarization_output(
        polarization, quantum_divisor=2.0
    )
    assert (raw, quantum, printed) == (-1.5, 1.5, 3.0)

    ph = tmp_path / "espresso_ph.pwo"
    ph.write_text(
        """Effective charges with asr applied:
 atom 1 Ba Mean Z*: 2.0
 E*x ( 2.0 0.0 0.0 )
 E*y ( 0.0 2.0 0.0 )
 E*z ( 0.0 0.0 2.0 )
 atom 2 O Mean Z*: -2.0
 E*x ( -2.0 0.0 0.0 )
 E*y ( 0.0 -2.0 0.0 )
 E*z ( 0.0 0.0 -2.0 )
 JOB DONE.
""",
        encoding="utf-8",
    )
    tensors = parse_asr_born_charges(ph)
    assert sorted(tensors) == [1, 2]
    assert np.allclose(tensors[2]["tensor"], -2.0 * np.eye(3))
