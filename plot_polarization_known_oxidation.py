#!/usr/bin/env python3
"""Plot Berry polarization using continuity plus a known oxidation-state endpoint.

This is intentionally separate from ORBIT's Z*-guided path-response analysis.
It reads only the three Berry-polarization outputs for each path image and
writes results to ``response/analysis_known_oxidation``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from ase.io import read as ase_read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbit.config import load_project_config
from orbit.qe.path_response import _image_directories, _resolve_source
from orbit.qe.path_response_analysis import (
    ELEMENTARY_CHARGE_C,
    _nearest_integer,
    _unit_direct_vectors,
    parse_polarization_output,
)


SCHEMA_VERSION = 1


def _project_structure_path(config: Any) -> Path:
    project = config.raw.get("project")
    if not isinstance(project, dict):
        raise ValueError("orbit.toml is missing [project]")
    raw = project.get("structure")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("[project].structure must be a nonempty path")
    path = Path(raw)
    return path if path.is_absolute() else config.root / path


def _path_metadata(config: Any, target_id: str, run_id: str) -> tuple[dict[str, Any], np.ndarray]:
    path = config.root / "paths" / target_id / "runs" / run_id / "path.json"
    if not path.is_file():
        raise FileNotFoundError(f"Selected-path metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target_id") != target_id or payload.get("run_id") != run_id:
        raise ValueError(f"Selected-path metadata identity is inconsistent: {path}")
    raw = payload.get("requested_displacement_frac")
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in raw)
    ):
        raise ValueError(f"Invalid requested_displacement_frac in {path}")
    winding = np.asarray(raw, dtype=int)
    if not np.any(winding):
        raise ValueError("The requested lattice displacement is zero")
    return payload, winding


def _analysis_settings(config: Any) -> tuple[float | None, int, np.ndarray]:
    response = config.raw.get("response", {})
    response = response if isinstance(response, dict) else {}
    analysis = response.get("analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    divisor = analysis.get("polarization_quantum_divisor", "auto")
    if isinstance(divisor, str):
        if divisor.lower() != "auto":
            raise ValueError("polarization_quantum_divisor must be 'auto' or positive")
        resolved_divisor = None
    elif isinstance(divisor, bool) or not isinstance(divisor, (int, float)) or divisor <= 0:
        raise ValueError("polarization_quantum_divisor must be 'auto' or positive")
    else:
        resolved_divisor = float(divisor)
    branch_range = analysis.get("branch_range", 4)
    if isinstance(branch_range, bool) or not isinstance(branch_range, int) or branch_range < 1:
        raise ValueError("[response.analysis].branch_range must be a positive integer")
    initial = analysis.get("initial_branch", [0, 0, 0])
    if (
        not isinstance(initial, list)
        or len(initial) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in initial)
    ):
        raise ValueError("[response.analysis].initial_branch must contain three integers")
    return resolved_divisor, branch_range, np.asarray(initial, dtype=int)


def _states(center: np.ndarray, radius: int) -> np.ndarray:
    axes = [np.arange(int(value) - radius, int(value) + radius + 1) for value in center]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3).astype(int)


def _branch_cartesian(
    raw_coefficients: np.ndarray,
    quanta: np.ndarray,
    states: np.ndarray,
    unit_direct: np.ndarray,
) -> np.ndarray:
    return (raw_coefficients[None, :] + states * quanta[None, :]) @ unit_direct


def _unwrap_componentwise_nearest(
    raw_coefficients: np.ndarray,
    quanta: np.ndarray,
    initial_branch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    # Greedy/local Berry unwrapping independently for P1, P2, P3.
    raw = np.asarray(raw_coefficients, dtype=float)
    q = np.asarray(quanta, dtype=float)
    initial = np.asarray(initial_branch, dtype=int)

    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("raw_coefficients must have shape (n_images, 3)")
    if q.shape != raw.shape:
        raise ValueError("quanta must have the same shape as raw_coefficients")
    if initial.shape != (3,):
        raise ValueError("initial_branch must contain three integers")
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(q)):
        raise ValueError("Non-finite polarization values or quanta")
    if np.any(np.abs(q) <= 1.0e-15):
        raise ValueError("Polarization quantum is zero")

    image_count = raw.shape[0]
    branches = np.empty((image_count, 3), dtype=int)
    selected = np.empty((image_count, 3), dtype=float)
    exact_ties: list[dict[str, Any]] = []

    branches[0] = initial
    selected[0] = raw[0] + branches[0] * q[0]

    for step in range(1, image_count):
        for component in range(3):
            previous_value = float(selected[step - 1, component])
            previous_branch = int(branches[step - 1, component])
            raw_value = float(raw[step, component])
            quantum = float(q[step, component])

            # The real-valued optimum branch coordinate.  The closest integer
            # must be floor(center) or ceil(center), so the search is exact and
            # does not depend on branch_range or the known oxidation state.
            center = (previous_value - raw_value) / quantum
            lower = int(np.floor(center))
            upper = int(np.ceil(center))
            candidates = sorted({lower, upper})

            errors = [
                abs(raw_value + candidate * quantum - previous_value)
                for candidate in candidates
            ]
            minimum_error = min(errors)
            best = [
                candidate
                for candidate, error in zip(candidates, errors)
                if np.isclose(
                    error,
                    minimum_error,
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
            ]

            if len(best) > 1:
                exact_ties.append(
                    {
                        "step": int(step),
                        "component": int(component + 1),
                        "previous_selected_C_per_m2": previous_value,
                        "raw_C_per_m2": raw_value,
                        "quantum_C_per_m2": quantum,
                        "candidate_branches": [int(value) for value in best],
                    }
                )

            # Deterministic tie rule:
            # 1. smallest change in branch index from previous image;
            # 2. smaller integer branch index.
            branch = min(
                best,
                key=lambda candidate: (
                    abs(int(candidate) - previous_branch),
                    int(candidate),
                ),
            )

            branches[step, component] = int(branch)
            selected[step, component] = raw_value + branch * quantum

    return branches, selected, exact_ties


def _transported_charge(
    initial_cartesian: np.ndarray,
    final_cartesian: np.ndarray,
    volume_A3: float,
    displacement_A: np.ndarray,
) -> np.ndarray:
    delta = final_cartesian - initial_cartesian
    displacement_m = displacement_A * 1.0e-10
    volume_m3 = volume_A3 * 1.0e-30
    denominator = ELEMENTARY_CHARGE_C * float(displacement_m @ displacement_m)
    if denominator <= 0.0:
        raise ValueError("Cannot normalize transported charge by a zero displacement")
    return volume_m3 * (delta @ displacement_m) / denominator


def _predecessor_indices(states: np.ndarray, delta: np.ndarray) -> np.ndarray:
    lookup = {tuple(state): index for index, state in enumerate(states.tolist())}
    return np.asarray(
        [lookup.get(tuple((state - delta).tolist()), -1) for state in states],
        dtype=int,
    )


def _unwrap_dynamic_program(
    raw_coefficients: np.ndarray,
    quanta: np.ndarray,
    unit_direct: np.ndarray,
    states: np.ndarray,
    initial_branch: np.ndarray,
    jump_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DP costs, predecessor table, and per-image branch Cartesian vectors."""
    image_count = len(raw_coefficients)
    state_count = len(states)
    cartesian = np.asarray(
        [
            _branch_cartesian(raw_coefficients[step], quanta[step], states, unit_direct)
            for step in range(image_count)
        ]
    )
    initial_matches = np.flatnonzero(np.all(states == initial_branch[None, :], axis=1))
    if len(initial_matches) != 1:
        raise ValueError("The requested initial branch is outside the DP state space")

    costs = np.full((image_count, state_count), np.inf, dtype=float)
    predecessors = np.full((image_count, state_count), -1, dtype=np.int32)
    costs[0, initial_matches[0]] = 0.0

    deltas = np.asarray(
        np.meshgrid(
            np.arange(-jump_radius, jump_radius + 1),
            np.arange(-jump_radius, jump_radius + 1),
            np.arange(-jump_radius, jump_radius + 1),
            indexing="ij",
        )
    ).reshape(3, -1).T
    predecessor_maps = [_predecessor_indices(states, delta) for delta in deltas]
    quantum_vectors = quanta[0][:, None] * unit_direct
    normalization = max(float(np.min(np.linalg.norm(quantum_vectors, axis=1))), 1.0e-15)

    for step in range(1, image_count):
        best = np.full(state_count, np.inf, dtype=float)
        best_predecessor = np.full(state_count, -1, dtype=np.int32)
        raw_delta_cartesian = (raw_coefficients[step] - raw_coefficients[step - 1]) @ unit_direct
        for delta, predecessor_map in zip(deltas, predecessor_maps):
            valid = predecessor_map >= 0
            if not np.any(valid):
                continue
            transition = raw_delta_cartesian + (delta * quanta[step]) @ unit_direct
            transition_cost = float(transition @ transition) / (normalization * normalization)
            candidate = np.full(state_count, np.inf, dtype=float)
            candidate[valid] = costs[step - 1, predecessor_map[valid]] + transition_cost
            improved = candidate < best
            best[improved] = candidate[improved]
            best_predecessor[improved] = predecessor_map[improved]
        costs[step] = best
        predecessors[step] = best_predecessor
    return costs, predecessors, cartesian


def _trace(predecessors: np.ndarray, endpoint: int) -> np.ndarray:
    result = np.empty(len(predecessors), dtype=np.int32)
    result[-1] = endpoint
    for step in range(len(predecessors) - 1, 0, -1):
        result[step - 1] = predecessors[step, result[step]]
        if result[step - 1] < 0:
            raise RuntimeError("Dynamic-programming branch trace is incomplete")
    return result


def _select_endpoints(
    costs: np.ndarray,
    cartesian: np.ndarray,
    initial_state_index: int,
    volume_A3: float,
    displacement_A: np.ndarray,
    expected_oxidation: float,
) -> tuple[int, int, np.ndarray]:
    endpoint_n = _transported_charge(
        cartesian[0, initial_state_index], cartesian[-1], volume_A3, displacement_A
    )
    finite = np.isfinite(costs[-1])
    if not np.any(finite):
        raise RuntimeError("No branch path reaches the final image")

    natural_cost = float(np.min(costs[-1, finite]))
    natural_candidates = np.flatnonzero(finite & np.isclose(costs[-1], natural_cost, atol=1.0e-12, rtol=1.0e-12))
    natural_endpoint = int(natural_candidates[np.argmin(np.abs(endpoint_n[natural_candidates]))])

    mismatch = np.abs(endpoint_n - expected_oxidation)
    minimum_mismatch = float(np.min(mismatch[finite]))
    guided_candidates = np.flatnonzero(
        finite & np.isclose(mismatch, minimum_mismatch, atol=1.0e-10, rtol=1.0e-10)
    )
    guided_endpoint = int(guided_candidates[np.argmin(costs[-1, guided_candidates])])
    return guided_endpoint, natural_endpoint, endpoint_n


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(
    rows: list[dict[str, Any]],
    expected_oxidation: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    steps = np.asarray([row["step"] for row in rows], dtype=int)
    titles = (
        "Contribution along direct vector a1",
        "Contribution along direct vector a2",
        "Contribution along direct vector a3",
    )
    for component in (1, 2, 3):
        axis = axes[component - 1]
        axis.plot(
            steps,
            [row[f"P{component}_raw_C_per_m2"] for row in rows],
            color="#9aa0a6",
            linewidth=1.0,
            marker="o",
            markersize=3,
            label=f"P{component} raw QE branch",
        )
        axis.plot(
            steps,
            [row[f"P{component}_guided_C_per_m2"] for row in rows],
            color="#1565c0",
            linewidth=1.1,
            linestyle=":",
            label=f"P{component} oxidation-guided",
        )
        axis.plot(
            steps,
            [row[f"P{component}_natural_C_per_m2"] for row in rows],
            color="#ef6c00",
            linewidth=2.8,
            linestyle="-",
            marker="o",
            markersize=3.5,
            label=f"P{component} continuity-only (componentwise nearest)",
        )
        axis.set_title(titles[component - 1])
        axis.set_ylabel("C/m²")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)

    n_axis = axes[3]
    n_axis.plot(
        steps,
        [row["N_guided"] for row in rows],
        color="#1565c0",
        linewidth=1.1,
        linestyle=":",
        label="N oxidation-guided",
    )
    n_axis.plot(
        steps,
        [row["N_natural"] for row in rows],
        color="#ef6c00",
        linewidth=2.8,
        linestyle="-",
        marker="o",
        markersize=3.5,
        label="N continuity-only (componentwise nearest)",
    )
    n_axis.axhline(
        expected_oxidation,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"known oxidation state = {expected_oxidation:g}",
    )
    n_axis.set_title("Transported charge along the target lattice translation")
    n_axis.set_xlabel("Path image")
    n_axis.set_ylabel("N")
    n_axis.grid(alpha=0.25)
    n_axis.legend(loc="best", fontsize=9)
    guided_final = float(rows[-1]["N_guided"])
    natural_final = float(rows[-1]["N_natural"])
    figure.suptitle(
            "Known-oxidation Berry-branch check "
            f"(guided final N={guided_final:.6f}; continuity-only final N={natural_final:.6f})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = load_project_config(args.root)
    source, resolved_iteration, source_root = _resolve_source(
        config, args.target, args.run_id, args.source, args.iteration
    )
    images = _image_directories(source_root)
    path_payload, winding = _path_metadata(config, args.target, args.run_id)
    target_index = int(path_payload.get("target_index_zero_based", -1))
    divisor, configured_range, initial_branch = _analysis_settings(config)

    initialized_atoms = ase_read(_project_structure_path(config))
    initialized_cell = np.asarray(initialized_atoms.cell.array, dtype=float)
    volume_A3 = abs(float(np.linalg.det(initialized_cell)))
    if volume_A3 <= 0.0:
        raise ValueError("The initialized CIF has zero cell volume")
    unit_direct = _unit_direct_vectors(initialized_cell)
    displacement_A = winding.astype(float) @ initialized_cell

    raw_rows: list[dict[str, Any]] = []
    structures: list[Any] = []
    problems: list[str] = []
    for step, image in enumerate(images):
        try:
            atoms = ase_read(image / "structure.cif")
            cell = np.asarray(atoms.cell.array, dtype=float)
            if not np.allclose(cell, initialized_cell, atol=1.0e-7, rtol=1.0e-9):
                raise ValueError("path cell differs from the initialized CIF cell")
            if target_index < 0 or target_index >= len(atoms):
                raise ValueError(f"target index {target_index} is outside structure")
            structures.append(atoms)
            row: dict[str, Any] = {"step": step, "image_name": image.name}
            for component in (1, 2, 3):
                value, _, printed_modulo, source_line = parse_polarization_output(
                    image / f"espresso_pol_gdir{component}.pwo", quantum_divisor=1.0
                )
                cell_quantum = (
                    ELEMENTARY_CHARGE_C
                    * 1.0e20
                    * float(np.linalg.norm(initialized_cell[component - 1]))
                    / volume_A3
                )
                quantum = cell_quantum if divisor is None else printed_modulo / divisor
                row[f"P{component}_raw_C_per_m2"] = value
                row[f"P{component}_quantum_C_per_m2"] = quantum
                row[f"P{component}_cell_quantum_C_per_m2"] = cell_quantum
                row[f"P{component}_qe_modulo_C_per_m2"] = printed_modulo
                row[f"P{component}_source_line"] = source_line
            raw_rows.append(row)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{image.name}: {exc}")
    if problems:
        detail = "\n  ".join(problems[:12])
        suffix = "" if len(problems) <= 12 else f"\n  ... and {len(problems) - 12} more"
        raise ValueError(
            "Known-oxidation analysis requires three complete polarization outputs "
            f"per image; found {len(problems)} problems:\n  {detail}{suffix}"
        )

    raw_coefficients = np.asarray(
        [[row[f"P{i}_raw_C_per_m2"] for i in (1, 2, 3)] for row in raw_rows], dtype=float
    )
    quanta = np.asarray(
        [[row[f"P{i}_quantum_C_per_m2"] for i in (1, 2, 3)] for row in raw_rows], dtype=float
    )
    if not np.allclose(quanta, quanta[0], atol=1.0e-10, rtol=1.0e-8):
        raise ValueError("Polarization quanta vary along a fixed-cell path")

    required_range = int(np.ceil(abs(args.oxidation_state) * np.max(np.abs(winding)))) + 2
    branch_range = args.branch_range or max(configured_range, required_range)
    state_vectors = _states(initial_branch, branch_range)
    initial_state_index = int(np.flatnonzero(np.all(state_vectors == initial_branch, axis=1))[0])
    costs, predecessors, cartesian = _unwrap_dynamic_program(
        raw_coefficients,
        quanta,
        unit_direct,
        state_vectors,
        initial_branch,
        args.jump_radius,
    )
    guided_endpoint, _unused_natural_endpoint, endpoint_n = _select_endpoints(
        costs,
        cartesian,
        initial_state_index,
        volume_A3,
        displacement_A,
        args.oxidation_state,
    )
    guided_trace = _trace(predecessors, guided_endpoint)
    guided_cartesian = cartesian[np.arange(len(images)), guided_trace]
    guided_branches = state_vectors[guided_trace]

    # Reference-independent continuity-only branch selection:
    # unwrap each direct-vector component independently using only the
    # previously selected value of that same component.
    (
        natural_branches,
        natural_coefficients,
        natural_exact_ties,
    ) = _unwrap_componentwise_nearest(
        raw_coefficients,
        quanta,
        initial_branch,
    )
    natural_cartesian = natural_coefficients @ unit_direct

    guided_n = _transported_charge(
        guided_cartesian[0], guided_cartesian, volume_A3, displacement_A
    )
    natural_n = _transported_charge(
        natural_cartesian[0], natural_cartesian, volume_A3, displacement_A
    )
    guided_jumps = np.r_[0.0, np.linalg.norm(np.diff(guided_cartesian, axis=0), axis=1)]
    natural_jumps = np.r_[0.0, np.linalg.norm(np.diff(natural_cartesian, axis=0), axis=1)]
    quantum_vector_norms = np.linalg.norm(quanta[0][:, None] * unit_direct, axis=1)
    minimum_quantum_norm = float(np.min(quantum_vector_norms))

    rows: list[dict[str, Any]] = []
    for step, raw in enumerate(raw_rows):
        guided_coefficients = raw_coefficients[step] + guided_branches[step] * quanta[step]
        natural_coefficients_step = natural_coefficients[step]
        row: dict[str, Any] = {"step": step, "image_name": raw["image_name"]}
        for component in (1, 2, 3):
            index = component - 1
            row[f"P{component}_raw_C_per_m2"] = float(raw_coefficients[step, index])
            row[f"P{component}_quantum_C_per_m2"] = float(quanta[step, index])
            row[f"P{component}_guided_branch_index"] = int(guided_branches[step, index])
            row[f"P{component}_guided_C_per_m2"] = float(guided_coefficients[index])
            row[f"P{component}_natural_branch_index"] = int(natural_branches[step, index])
            row[f"P{component}_natural_C_per_m2"] = float(natural_coefficients_step[index])
        for index, axis in enumerate("xyz"):
            row[f"P_guided_{axis}_C_per_m2"] = float(guided_cartesian[step, index])
            row[f"P_natural_{axis}_C_per_m2"] = float(natural_cartesian[step, index])
        row["N_guided"] = float(guided_n[step])
        row["N_natural"] = float(natural_n[step])
        row["guided_step_jump_C_per_m2"] = float(guided_jumps[step])
        row["natural_step_jump_C_per_m2"] = float(natural_jumps[step])
        rows.append(row)

    target_positions = np.asarray([atoms.positions[target_index] for atoms in structures])
    cumulative_target = np.zeros(3, dtype=float)
    for step in range(1, len(structures)):
        fractional_delta = (target_positions[step] - target_positions[step - 1]) @ np.linalg.inv(initialized_cell)
        fractional_delta -= np.floor(fractional_delta + 0.5)
        cumulative_target += fractional_delta @ initialized_cell
    closure_error_A = float(np.linalg.norm(cumulative_target - displacement_A))

    output_root = source_root / "response" / "analysis_known_oxidation"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "known_oxidation_polarization.csv"
    plot_path = output_root / "known_oxidation_polarization.png"
    manifest_path = output_root / "analysis.json"
    _write_csv(csv_path, rows)
    _make_figure(rows, args.oxidation_state, plot_path)

    final_guided_n = float(guided_n[-1])
    final_natural_n = float(natural_n[-1])
    guided_max_jump_fraction = float(np.max(guided_jumps) / minimum_quantum_norm)
    natural_max_jump_fraction = float(np.max(natural_jumps) / minimum_quantum_norm)
    natural_nearest = _nearest_integer(final_natural_n)
    independently_consistent = natural_nearest == _nearest_integer(args.oxidation_state)
    forced_branch_difference = bool(np.any(guided_branches != natural_branches))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "known_oxidation_endpoint_plus_componentwise_greedy_continuity",
        "target_id": args.target,
        "run_id": args.run_id,
        "source": source,
        "iteration": resolved_iteration,
        "image_count": len(images),
        "known_oxidation_state": args.oxidation_state,
        "requested_displacement_frac": winding.tolist(),
        "total_lattice_displacement_A": displacement_A.tolist(),
        "initialized_cif_volume_A3": volume_A3,
        "target_unwrapped_net_displacement_A": cumulative_target.tolist(),
        "target_winding_closure_error_A": closure_error_A,
        "polarization_quantum_source": (
            "initialized_cif_e_times_direct_vector_over_volume"
            if divisor is None
            else "qe_printed_modulo_divided_by_configured_override"
        ),
        "branch_range": branch_range,
        "branch_jump_radius_per_image": args.jump_radius,
        "initial_branch": initial_branch.tolist(),
        "guided_final_N": final_guided_n,
        "guided_oxidation_mismatch": final_guided_n - args.oxidation_state,
        "continuity_only_final_N": final_natural_n,
        "continuity_only_nearest_integer_N": natural_nearest,
        "continuity_only_matches_known_oxidation": independently_consistent,
        "guided_path_differs_from_continuity_only": forced_branch_difference,
        "guided_max_step_jump_fraction_of_smallest_quantum": guided_max_jump_fraction,
        "continuity_only_max_step_jump_fraction_of_smallest_quantum": natural_max_jump_fraction,
        "continuity_only_selection_rule": (
            "greedy nearest branch to previous selected value independently "
            "for P1/P2/P3; components recombined only after unwrapping"
        ),
        "continuity_only_reference_independent": True,
        "continuity_only_exact_tie_count": len(natural_exact_ties),
        "continuity_only_exact_ties": natural_exact_ties,
        "interpretation": (
            "SUPPORTED: the componentwise greedy continuity path independently reaches the known oxidation integer"
            if independently_consistent
            else "GUIDED ONLY: the known integer differs from the componentwise greedy continuity result"
        ),
        "files": {"csv": csv_path.name, "plot": plot_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "csv_path": csv_path, "plot_path": plot_path, "manifest_path": manifest_path}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Temporary PH-free Berry-branch plotter guided by a known signed oxidation state. "
            "The existing ORBIT Z*-based analyzer is not modified."
        )
    )
    result.add_argument("--root", type=Path, default=Path.cwd(), help="ORBIT material-project root")
    result.add_argument("--target", required=True, help="target site ID, e.g. Ti1")
    result.add_argument("--run-id", required=True, help="selected path run ID")
    result.add_argument(
        "--oxidation-state",
        required=True,
        type=float,
        help="known signed oxidation state, e.g. 4 for Ti(IV) or -2 for O(II-)",
    )
    result.add_argument("--source", choices=("path", "helper"), default="path")
    result.add_argument("--iteration", type=int, help="helper iteration; default is latest")
    result.add_argument(
        "--branch-range",
        type=int,
        help="absolute branch radius about initial_branch; default expands automatically",
    )
    result.add_argument(
        "--jump-radius",
        type=int,
        default=1,
        help="guided-path DP branch-index jump radius only; continuity-only ignores this (default: 1)",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.branch_range is not None and args.branch_range < 1:
        print("ERROR: --branch-range must be positive", file=sys.stderr)
        return 2
    if args.jump_radius < 1:
        print("ERROR: --jump-radius must be positive", file=sys.stderr)
        return 2
    try:
        result = analyze(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"[known oxidation plot {result['target_id']}] run={result['run_id']}; "
        f"winding={result['requested_displacement_frac']}; "
        f"R_A={result['total_lattice_displacement_A']}; "
        f"volume={result['initialized_cif_volume_A3']:.8f} A^3"
    )
    print(
        f"[guided] known={result['known_oxidation_state']:g}; "
        f"final N={result['guided_final_N']:.8f}; "
        f"mismatch={result['guided_oxidation_mismatch']:+.3e}"
    )
    print(
        f"[continuity only] final N={result['continuity_only_final_N']:.8f}; "
        f"nearest integer={result['continuity_only_nearest_integer_N']}; "
        f"matches known={result['continuity_only_matches_known_oxidation']}"
    )
    print(f"[interpretation] {result['interpretation']}")
    print(f"[done] CSV: {result['csv_path']}")
    print(f"[done] plot: {result['plot_path']}")
    print(f"[done] manifest: {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
