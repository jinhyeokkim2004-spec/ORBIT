"""Extract and plot Berry polarization and transported charge for a final path."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from ase.io import read as ase_read
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ..config import ProjectConfig
from .path_response import _image_directories, _resolve_source, _table


SCHEMA_VERSION = 1
ELEMENTARY_CHARGE_C = 1.602176634e-19
NUMBER_TEXT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
ASR_HEADER_PATTERN = re.compile(
    r"Effective\s+charges.*?with\s+asr\s+applied\s*:", re.IGNORECASE
)
ATOM_HEADER_PATTERN = re.compile(
    rf"atom\s+(\d+)\s+([A-Za-z][A-Za-z0-9]*)"
    rf"\s+Mean\s+Z\*:\s*({NUMBER_TEXT})",
    re.IGNORECASE,
)
TENSOR_ROW_PATTERN = re.compile(
    rf"E\*([xyz])\s*\(\s*({NUMBER_TEXT})\s+({NUMBER_TEXT})\s+"
    rf"({NUMBER_TEXT})\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResponseAnalysis:
    target_id: str
    run_id: str
    source: str
    iteration: int | None
    image_count: int
    output_root: Path
    polarization_csv: Path
    selected_csv: Path
    transported_charge_csv: Path
    polarization_plot: Path
    transported_charge_plot: Path
    final_n: float
    nearest_integer_n: int
    requested_displacement_frac: tuple[int, int, int]


def _qe_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _complete_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "JOB DONE" not in text:
        raise ValueError(f"{label} did not finish with JOB DONE: {path}")
    return text


def _has_c_per_m2(line: str) -> bool:
    normalized = line.lower().replace(" ", "").replace("**", "^").replace("²", "^2")
    return any(marker in normalized for marker in ("c/m^2", "c/m2", "cm^-2", "c.m^-2"))


def _parse_polarization_line(line: str) -> tuple[float, float]:
    value_match = re.search(rf"\bP\s*=\s*({NUMBER_TEXT})", line, re.IGNORECASE)
    modulo_match = re.search(rf"\bmod\s+({NUMBER_TEXT})", line, re.IGNORECASE)
    if modulo_match is None:
        modulo_match = re.search(
            rf"\bmod(?:ulo)?\s*[=:]?\s*({NUMBER_TEXT})", line, re.IGNORECASE
        )
    if value_match is None or modulo_match is None:
        raise ValueError("line does not contain a parseable P/modulo pair")
    value = _qe_float(value_match.group(1))
    printed_modulo = abs(_qe_float(modulo_match.group(1)))
    if printed_modulo == 0.0:
        raise ValueError("printed polarization modulo is zero")
    return value, printed_modulo


def parse_polarization_output(
    path: Path, *, quantum_divisor: float
) -> tuple[float, float, float, str]:
    """Return raw P, physical branch quantum, QE modulo, and source line."""
    text = _complete_text(path, "Polarization output")
    p_lines = [line.strip() for line in text.splitlines() if re.search(r"\bP\s*=", line, re.I)]
    if not p_lines:
        raise ValueError(f"No lines containing 'P =' were found in {path}")
    candidates = [line for line in p_lines if _has_c_per_m2(line)]
    if len(p_lines) >= 3:
        candidates.append(p_lines[2])
    candidates.extend(reversed(p_lines))
    seen: set[str] = set()
    errors: list[str] = []
    for line in candidates:
        if line in seen:
            continue
        seen.add(line)
        try:
            value, printed_modulo = _parse_polarization_line(line)
            return value, printed_modulo / quantum_divisor, printed_modulo, line
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(f"No parseable polarization line in {path}: {'; '.join(errors[-3:])}")


def parse_asr_born_charges(path: Path) -> dict[int, dict[str, Any]]:
    """Parse the final ASR-corrected Born tensor for every atom in ph.x output."""
    text = _complete_text(path, "PH output")
    headers = list(ASR_HEADER_PATTERN.finditer(text))
    if not headers:
        raise ValueError(f"No ASR-corrected effective-charge block was found in {path}")
    block = text[headers[-1].end():]
    timing = block.find("\n     PHONON")
    if timing >= 0:
        block = block[:timing]
    atoms = list(ATOM_HEADER_PATTERN.finditer(block))
    if not atoms:
        raise ValueError(f"No atoms were found in the ASR-corrected block in {path}")
    result: dict[int, dict[str, Any]] = {}
    for index, match in enumerate(atoms):
        atom_number = int(match.group(1))
        end = atoms[index + 1].start() if index + 1 < len(atoms) else len(block)
        rows: dict[str, list[float]] = {}
        for row in TENSOR_ROW_PATTERN.finditer(block[match.end():end]):
            rows[row.group(1).lower()] = [_qe_float(row.group(i)) for i in (2, 3, 4)]
        if set(rows) != {"x", "y", "z"}:
            raise ValueError(
                f"Atom {atom_number} {match.group(2)} has incomplete Born tensor rows: "
                f"{sorted(rows)}"
            )
        if atom_number in result:
            raise ValueError(f"Duplicate atom {atom_number} in ASR Born-charge block")
        result[atom_number] = {
            "symbol": match.group(2),
            "mean": _qe_float(match.group(3)),
            "tensor": np.asarray([rows[axis] for axis in "xyz"], dtype=float),
        }
    return result


def _nearest_integer(value: float) -> int:
    return int(np.floor(value + 0.5)) if value >= 0 else int(np.ceil(value - 0.5))


def _unit_direct_vectors(cell: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Cell contains a zero-length direct lattice vector")
    return cell / lengths[:, None]


def _closest_branch_vector(
    raw: np.ndarray,
    quantum: np.ndarray,
    estimate_cartesian: np.ndarray,
    unit_direct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the nearby integer branch triplet closest in Cartesian norm."""
    estimate_coefficients = estimate_cartesian @ np.linalg.inv(unit_direct)
    centers = np.asarray(
        [_nearest_integer(value) for value in (estimate_coefficients - raw) / quantum],
        dtype=int,
    )
    best: tuple[float, tuple[int, int, int], np.ndarray, np.ndarray] | None = None
    for m1 in range(centers[0] - 2, centers[0] + 3):
        for m2 in range(centers[1] - 2, centers[1] + 3):
            for m3 in range(centers[2] - 2, centers[2] + 3):
                indices = np.asarray([m1, m2, m3], dtype=int)
                coefficients = raw + indices * quantum
                cartesian = coefficients @ unit_direct
                error = float(np.linalg.norm(cartesian - estimate_cartesian))
                key = (error, (m1, m2, m3), coefficients, cartesian)
                if best is None or key[:2] < best[:2]:
                    best = key
    assert best is not None
    return np.asarray(best[1], dtype=int), best[2], best[3]


def _minimum_image_displacements(
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    inverse = np.linalg.inv(cell)
    delta_fractional = current_positions @ inverse - previous_positions @ inverse
    delta_fractional -= np.floor(delta_fractional + 0.5)
    return delta_fractional @ cell


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validated_settings(
    config: ProjectConfig,
    quantum_divisor: float | None,
    branch_range: int | None,
) -> tuple[float | None, int, tuple[int, int, int]]:
    response = _table(config.raw.get("response"), "[response]")
    analysis = _table(response.get("analysis"), "[response.analysis]")
    divisor: object = analysis.get("polarization_quantum_divisor", "auto") if quantum_divisor is None else quantum_divisor
    branches = analysis.get("branch_range", 4) if branch_range is None else branch_range
    initial = analysis.get("initial_branch", [0, 0, 0])
    if isinstance(divisor, str):
        if divisor.lower() != "auto":
            raise ValueError("polarization quantum divisor must be 'auto' or positive")
        resolved_divisor = None
    elif isinstance(divisor, bool) or not isinstance(divisor, (int, float)) or divisor <= 0:
        raise ValueError("polarization quantum divisor must be 'auto' or positive")
    else:
        resolved_divisor = float(divisor)
    if isinstance(branches, bool) or not isinstance(branches, int) or branches < 1:
        raise ValueError("branch range must be a positive integer")
    if not isinstance(initial, list) or len(initial) != 3 or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in initial
    ):
        raise ValueError("[response.analysis].initial_branch must contain three integers")
    return resolved_divisor, branches, tuple(initial)


def _path_metadata(
    config: ProjectConfig, target_id: str, run_id: str
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    path = config.root / "paths" / target_id / "runs" / run_id / "path.json"
    if not path.is_file():
        raise FileNotFoundError(f"Original selected-path metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target_id") != target_id or payload.get("run_id") != run_id:
        raise ValueError(f"Selected-path metadata identity is inconsistent: {path}")
    raw = payload.get("requested_displacement_frac")
    if not isinstance(raw, list) or len(raw) != 3 or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw
    ):
        raise ValueError(f"Invalid requested_displacement_frac in {path}")
    winding = tuple(raw)
    if winding == (0, 0, 0):
        raise ValueError("Transported-charge lattice displacement is zero")
    return payload, winding


def _polarization_figure(
    rows: list[dict[str, Any]], branch_range: int, output: Path
) -> None:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            "Contribution along direct vector a1",
            "Contribution along direct vector a2",
            "Contribution along direct vector a3",
            "Cartesian polarization-vector magnitude",
        ),
    )
    steps = [row["step"] for row in rows]
    colors = [
        f"hsl({240 - 240 * (index + branch_range) / (2 * branch_range)},55%,55%)"
        for index in range(-branch_range, branch_range + 1)
    ]
    for component in (1, 2, 3):
        for branch, color in zip(range(-branch_range, branch_range + 1), colors):
            figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=[row[f"P{component}_raw_C_per_m2"] + branch * row[f"P{component}_quantum_C_per_m2"] for row in rows],
                    mode="lines",
                    line={"color": color, "width": 0.8},
                    opacity=0.38,
                    name=f"P{component} {branch:+d}Q{component}",
                    legendgroup=f"branches-{component}",
                    showlegend=branch in (-branch_range, 0, branch_range),
                ),
                row=component,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[row[f"P{component}_selected_C_per_m2"] for row in rows],
                mode="lines+markers",
                line={"width": 2.3},
                marker={"size": 7, "symbol": "circle-open"},
                name=f"P{component} selected branch",
            ),
            row=component,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=[row[f"P{component}_zstar_estimate_C_per_m2"] for row in rows],
                mode="lines+markers",
                line={"color": "black", "width": 1.8, "dash": "dot"},
                marker={"size": 5},
                name=f"P{component} Z*·du estimate",
            ),
            row=component,
            col=1,
        )
        figure.update_yaxes(title_text="C/m²", row=component, col=1)
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=[row["P_selected_magnitude_C_per_m2"] for row in rows],
            mode="lines+markers",
            line={"width": 2.3},
            marker={"size": 7, "symbol": "circle-open"},
            name="|P| selected",
        ),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=[row["P_zstar_estimate_magnitude_C_per_m2"] for row in rows],
            mode="lines+markers",
            line={"color": "black", "width": 1.8, "dash": "dot"},
            marker={"size": 5},
            name="|P| Z*·du estimate",
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="C/m²", row=4, col=1)
    figure.update_xaxes(title_text="Path image", row=4, col=1)
    figure.update_layout(
        title=(
            "Berry-phase directional contributions, selected vector branch, "
            "and Born-charge prediction"
        ),
        template="plotly_white",
        height=1150,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.075},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)


def _n_figure(rows: list[dict[str, Any]], output: Path) -> None:
    final_n = float(rows[-1]["N"])
    nearest = _nearest_integer(final_n)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row["step"] for row in rows],
            y=[row["N"] for row in rows],
            mode="lines+markers",
            line={"width": 2.2},
            marker={"size": 7},
            name="Transported charge N",
        )
    )
    figure.add_hline(
        y=nearest,
        line_dash="dash",
        line_color="black",
        annotation_text=f"nearest integer = {nearest}",
    )
    figure.update_layout(
        title=f"Transported charge along the winding vector (final N = {final_n:.6f})",
        xaxis_title="Path image",
        yaxis_title="N = (Ω/e)(ΔP·R)/|R|²",
        template="plotly_white",
        height=620,
        hovermode="x unified",
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)


def analyze_path_response(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    *,
    source: str = "path",
    iteration: int | None = None,
    quantum_divisor: float | None = None,
    branch_range: int | None = None,
) -> ResponseAnalysis:
    divisor, branches, initial_branch = _validated_settings(
        config, quantum_divisor, branch_range
    )
    source, resolved_iteration, source_root = _resolve_source(
        config, target_id, run_id, source, iteration
    )
    images = _image_directories(source_root)
    path_payload, winding = _path_metadata(config, target_id, run_id)
    target_index = int(path_payload.get("target_index_zero_based", -1))

    structures: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    born_by_step: list[dict[int, dict[str, Any]]] = []
    problems: list[str] = []
    for step, image in enumerate(images):
        try:
            atoms = ase_read(image / "structure.cif")
            cell = np.asarray(atoms.cell.array, dtype=float)
            positions = np.asarray(atoms.positions, dtype=float)
            symbols = list(atoms.get_chemical_symbols())
            if target_index < 0 or target_index >= len(symbols):
                raise ValueError(f"target index {target_index} is outside structure")
            structures.append({"cell": cell, "positions": positions, "symbols": symbols})
            born = parse_asr_born_charges(image / "espresso_ph.pwo")
            expected_numbers = set(range(1, len(symbols) + 1))
            if set(born) != expected_numbers:
                raise ValueError(
                    f"Born tensors cover atoms {sorted(born)}, expected {sorted(expected_numbers)}"
                )
            for number, symbol in enumerate(symbols, start=1):
                if born[number]["symbol"].lower() != symbol.lower():
                    raise ValueError(
                        f"atom {number} symbol mismatch: PH={born[number]['symbol']}, structure={symbol}"
                    )
            born_by_step.append(born)
            row: dict[str, Any] = {"step": step, "image_name": image.name}
            volume = abs(float(np.linalg.det(cell)))
            for component in (1, 2, 3):
                value, _, printed_modulo, source_line = parse_polarization_output(
                    image / f"espresso_pol_gdir{component}.pwo",
                    quantum_divisor=1.0,
                )
                cell_quantum = (
                    ELEMENTARY_CHARGE_C
                    * 1.0e20
                    * float(np.linalg.norm(cell[component - 1]))
                    / volume
                )
                quantum = (
                    cell_quantum
                    if divisor is None
                    else printed_modulo / divisor
                )
                row[f"P{component}_raw_C_per_m2"] = value
                row[f"P{component}_quantum_C_per_m2"] = quantum
                row[f"P{component}_cell_quantum_C_per_m2"] = cell_quantum
                row[f"P{component}_qe_modulo_C_per_m2"] = printed_modulo
                row[f"P{component}_qe_modulo_over_cell_quantum"] = (
                    printed_modulo / cell_quantum
                )
                row[f"P{component}_source_line"] = source_line
                for branch in range(-branches, branches + 1):
                    row[f"P{component}_branch_{branch:+d}_C_per_m2"] = value + branch * quantum
            for number in sorted(born):
                tensor = born[number]["tensor"]
                prefix = f"Zstar_atom{number:03d}_{born[number]['symbol']}"
                row[f"{prefix}_mean"] = born[number]["mean"]
                for i, axis_i in enumerate("xyz"):
                    for j, axis_j in enumerate("xyz"):
                        row[f"{prefix}_{axis_i}{axis_j}"] = float(tensor[i, j])
            raw_rows.append(row)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{image.name}: {exc}")
    if problems:
        detail = "\n  ".join(problems[:12])
        suffix = "" if len(problems) <= 12 else f"\n  ... and {len(problems) - 12} more"
        raise ValueError(
            f"Response analysis requires complete PH and three-direction polarization "
            f"outputs for every image; found {len(problems)} problems:\n  {detail}{suffix}"
        )

    selected_rows: list[dict[str, Any]] = []
    previous_selected_cartesian: np.ndarray | None = None
    cumulative_displacement = np.zeros_like(structures[0]["positions"])
    for step, raw in enumerate(raw_rows):
        unit_direct = _unit_direct_vectors(structures[step]["cell"])
        raw_coefficients = np.asarray(
            [raw[f"P{component}_raw_C_per_m2"] for component in (1, 2, 3)],
            dtype=float,
        )
        quanta = np.asarray(
            [raw[f"P{component}_quantum_C_per_m2"] for component in (1, 2, 3)],
            dtype=float,
        )
        if step == 0:
            delta_p_cartesian = np.zeros(3, dtype=float)
            estimate_coefficients = raw_coefficients + quanta * np.asarray(
                initial_branch, dtype=int
            )
            estimate_cartesian = estimate_coefficients @ unit_direct
        else:
            current = structures[step]
            previous = structures[step - 1]
            if not np.allclose(current["cell"], previous["cell"], atol=1.0e-8, rtol=1.0e-8):
                raise ValueError("Response analysis currently requires a fixed path cell")
            du = _minimum_image_displacements(
                previous["positions"], current["positions"], current["cell"]
            )
            cumulative_displacement += du
            volume = abs(float(np.linalg.det(current["cell"])))
            delta_p_cartesian = np.zeros(3, dtype=float)
            previous_born = born_by_step[step - 1]
            for component in range(3):
                total_z_du = sum(
                    float(previous_born[atom]["tensor"][component] @ du[atom - 1])
                    for atom in sorted(previous_born)
                )
                delta_p_cartesian[component] = (
                    ELEMENTARY_CHARGE_C * 1.0e20 * total_z_du / volume
                )
            assert previous_selected_cartesian is not None
            estimate_cartesian = previous_selected_cartesian + delta_p_cartesian
            estimate_coefficients = estimate_cartesian @ np.linalg.inv(unit_direct)

        branch_indices, selected_coefficients, selected_cartesian = (
            _closest_branch_vector(
                raw_coefficients,
                quanta,
                estimate_cartesian,
                unit_direct,
            )
        )
        delta_coefficients = delta_p_cartesian @ np.linalg.inv(unit_direct)
        row = {
            "step": step,
            "image_name": raw["image_name"],
            **{f"P{component}_raw_C_per_m2": raw[f"P{component}_raw_C_per_m2"] for component in (1, 2, 3)},
            **{f"P{component}_quantum_C_per_m2": raw[f"P{component}_quantum_C_per_m2"] for component in (1, 2, 3)},
            **{f"P{component}_zstar_delta_C_per_m2": float(delta_coefficients[component - 1]) for component in (1, 2, 3)},
            **{f"P{component}_zstar_estimate_C_per_m2": float(estimate_coefficients[component - 1]) for component in (1, 2, 3)},
            **{f"P{component}_selected_branch_index": branch_indices[component - 1] for component in (1, 2, 3)},
            **{f"P{component}_selected_C_per_m2": float(selected_coefficients[component - 1]) for component in (1, 2, 3)},
            **{f"P_zstar_estimate_{axis}_C_per_m2": float(estimate_cartesian[index]) for index, axis in enumerate("xyz")},
            **{f"P_selected_{axis}_C_per_m2": float(selected_cartesian[index]) for index, axis in enumerate("xyz")},
            "P_zstar_estimate_magnitude_C_per_m2": float(np.linalg.norm(estimate_cartesian)),
            "P_selected_magnitude_C_per_m2": float(np.linalg.norm(selected_cartesian)),
            "branch_selection_error_C_per_m2": float(np.linalg.norm(selected_cartesian - estimate_cartesian)),
        }
        selected_rows.append(row)
        previous_selected_cartesian = selected_cartesian

    initial_selected_cartesian = np.asarray(
        [selected_rows[0][f"P_selected_{axis}_C_per_m2"] for axis in "xyz"],
        dtype=float,
    )
    n_rows: list[dict[str, Any]] = []
    for step, row in enumerate(selected_rows):
        cell = structures[step]["cell"]
        volume = abs(float(np.linalg.det(cell)))
        displacement_vector = np.asarray(winding, dtype=float) @ cell
        norm_squared = float(displacement_vector @ displacement_vector)
        selected_cartesian = np.asarray(
            [row[f"P_selected_{axis}_C_per_m2"] for axis in "xyz"],
            dtype=float,
        )
        delta_cartesian = selected_cartesian - initial_selected_cartesian
        numerator = volume * 1.0e-30 * float(
            delta_cartesian @ (displacement_vector * 1.0e-10)
        )
        denominator = ELEMENTARY_CHARGE_C * norm_squared * 1.0e-20
        n_value = numerator / denominator
        n_rows.append(
            {
                "step": step,
                "image_name": row["image_name"],
                **{f"P{component}_selected_C_per_m2": row[f"P{component}_selected_C_per_m2"] for component in (1, 2, 3)},
                **{f"P_{axis}_selected_C_per_m2": float(selected_cartesian[index]) for index, axis in enumerate("xyz")},
                **{f"delta_P_{axis}_C_per_m2": float(delta_cartesian[index]) for index, axis in enumerate("xyz")},
                "R_x_A": float(displacement_vector[0]),
                "R_y_A": float(displacement_vector[1]),
                "R_z_A": float(displacement_vector[2]),
                "volume_A3": volume,
                "N": n_value,
            }
        )

    output_root = source_root / "response" / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    polarization_csv = output_root / "polarization_and_born_charges.csv"
    selected_csv = output_root / "polarization_vector_all_atoms.csv"
    n_csv = output_root / "transported_charge_N.csv"
    polarization_plot = output_root / "polarization.html"
    n_plot = output_root / "transported_charge_N.html"
    raw_fields = list(raw_rows[0])
    selected_fields = list(selected_rows[0])
    n_fields = list(n_rows[0])
    _write_csv(polarization_csv, raw_rows, raw_fields)
    _write_csv(selected_csv, selected_rows, selected_fields)
    _write_csv(n_csv, n_rows, n_fields)
    _polarization_figure(selected_rows, branches, polarization_plot)
    _n_figure(n_rows, n_plot)

    target_net = cumulative_displacement[target_index]
    expected_target_net = np.asarray(winding, dtype=float) @ structures[0]["cell"]
    helper_net_max = max(
        float(np.linalg.norm(vector))
        for atom, vector in enumerate(cumulative_displacement)
        if atom != target_index
    ) if len(cumulative_displacement) > 1 else 0.0
    final_n = float(n_rows[-1]["N"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_id": target_id,
        "run_id": run_id,
        "source": source,
        "iteration": resolved_iteration,
        "image_count": len(images),
        "all_atoms_used_for_zstar_prediction": True,
        "polarization_quantum_divisor": divisor,
        "polarization_quantum_source": (
            "cell_e_times_direct_vector_over_volume"
            if divisor is None
            else "qe_printed_modulo_divided_by_override"
        ),
        "polarization_component_convention": (
            "gdir scalar contribution along the corresponding unit direct lattice vector"
        ),
        "initial_branch": initial_branch,
        "branch_plot_range": [-branches, branches],
        "requested_displacement_frac": winding,
        "target_net_displacement_A": target_net.tolist(),
        "expected_target_net_displacement_A": expected_target_net.tolist(),
        "target_winding_closure_error_A": float(np.linalg.norm(target_net - expected_target_net)),
        "maximum_helper_closure_error_A": helper_net_max,
        "final_N": final_n,
        "nearest_integer_N": _nearest_integer(final_n),
        "files": {
            "polarization_and_born_charges": polarization_csv.name,
            "polarization_vector_all_atoms": selected_csv.name,
            "transported_charge": n_csv.name,
            "polarization_plot": polarization_plot.name,
            "transported_charge_plot": n_plot.name,
        },
    }
    (output_root / "analysis.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return ResponseAnalysis(
        target_id=target_id,
        run_id=run_id,
        source=source,
        iteration=resolved_iteration,
        image_count=len(images),
        output_root=output_root,
        polarization_csv=polarization_csv,
        selected_csv=selected_csv,
        transported_charge_csv=n_csv,
        polarization_plot=polarization_plot,
        transported_charge_plot=n_plot,
        final_n=final_n,
        nearest_integer_n=_nearest_integer(final_n),
        requested_displacement_frac=winding,
    )
