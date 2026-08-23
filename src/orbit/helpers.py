"""Single-helper iterative refinement of calculated target-displacement paths."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write

from .config import ProjectConfig
from .gaps import parse_qe_output
from .periodic import minimum_image_displacement, minimum_image_distance
from .project import ProjectLayout
from .qe.input import InvalidGeometryError, render_scf_input, validate_scf_geometry
from .qe.path_calculations import resolve_path_run
from .qe.resources import resolve_qe_resources
from .scheduler.slurm import render_array_script, render_submit_script
from .structure import load_site_records, validate_manifest_source


HELPER_SCHEMA_VERSION = 1
ITERATION_PREFIX = "iteration_"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."} or any(ch in run_id for ch in ("/", "\\")):
        raise ValueError("helper --run-id must be a nonempty path-safe identifier")
    return run_id


def _selected_target(config: ProjectConfig, target_id: str) -> str:
    if target_id not in config.target_site_ids:
        raise ValueError(f"Requested helper target is not configured: {target_id}")
    return target_id


def resolve_helper_run_id(
    config: ProjectConfig,
    target_id: str,
    run_id: str | None = None,
) -> str:
    """Resolve the parent path run for a helper campaign."""
    target_id = _selected_target(config, target_id)

    if run_id is not None:
        return _safe_run_id(run_id)

    helper_target = config.root / "helpers" / target_id
    if helper_target.is_dir():
        for campaign in sorted(
            (p for p in helper_target.iterdir()
             if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name,
            reverse=True,
        ):
            candidate = _safe_run_id(campaign.name)
            if (config.root / "paths" / target_id / "runs" / candidate).is_dir():
                return candidate

    path_root = config.root / "paths" / target_id / "runs"
    if path_root.is_dir():
        for run in sorted(
            (p for p in path_root.iterdir()
             if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name,
            reverse=True,
        ):
            if (run / "path.json").is_file() and (run / "cifs").is_dir():
                return _safe_run_id(run.name)

    raise FileNotFoundError(
        f"No helper campaign or constructed path run exists for {target_id}. "
        f"Run `orbit path --target {target_id}` first, or specify --run-id."
    )


def _campaign_root(config: ProjectConfig, target_id: str, run_id: str) -> Path:
    return config.root / "helpers" / target_id / _safe_run_id(run_id)


def _iteration_root(config: ProjectConfig, target_id: str, run_id: str, iteration: int) -> Path:
    return _campaign_root(config, target_id, run_id) / f"{ITERATION_PREFIX}{iteration:03d}"


def _scan_root(config: ProjectConfig, target_id: str, run_id: str, iteration: int) -> Path:
    return (
        config.root
        / "calculations"
        / "helpers"
        / target_id
        / _safe_run_id(run_id)
        / f"{ITERATION_PREFIX}{iteration:03d}"
        / "scan"
    )


def _helper_path_root(
    config: ProjectConfig, target_id: str, run_id: str, iteration: int
) -> Path:
    return (
        config.root
        / "calculations"
        / "helpers"
        / target_id
        / _safe_run_id(run_id)
        / f"{ITERATION_PREFIX}{iteration:03d}"
        / "path"
    )


def _result_root(config: ProjectConfig, target_id: str, run_id: str, iteration: int) -> Path:
    return (
        config.root
        / "results"
        / "helper_gaps"
        / target_id
        / _safe_run_id(run_id)
        / f"{ITERATION_PREFIX}{iteration:03d}"
    )


def _plot_root(config: ProjectConfig, target_id: str, run_id: str) -> Path:
    return config.root / "plots" / "helpers" / target_id / _safe_run_id(run_id)


def _base_path_calc_root(config: ProjectConfig, target_id: str, run_id: str) -> Path:
    return config.root / "calculations" / "paths" / target_id / _safe_run_id(run_id)


def _base_path_gap_csv(config: ProjectConfig, target_id: str, run_id: str) -> Path:
    return config.root / "results" / "path_gaps" / target_id / _safe_run_id(run_id) / "gaps.csv"


def _iteration_numbers(campaign_root: Path) -> list[int]:
    if not campaign_root.is_dir():
        return []
    found: list[int] = []
    for child in campaign_root.iterdir():
        if not child.is_dir() or not child.name.startswith(ITERATION_PREFIX):
            continue
        suffix = child.name[len(ITERATION_PREFIX):]
        if suffix.isdigit():
            found.append(int(suffix))
    return sorted(found)


def _minimum_all_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _minimum_helper_distance(atoms: Any, helper_index: int) -> float:
    if not 0 <= helper_index < len(atoms):
        raise ValueError(f"helper index {helper_index} outside atom range")
    frac = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    return min(
        minimum_image_distance(frac[helper_index], frac[index], cell)
        for index in range(len(atoms))
        if index != helper_index
    )


def _site_maps(layout: ProjectLayout) -> tuple[dict[str, Any], dict[int, Any]]:
    sites = load_site_records(layout.sites_manifest)
    by_id = {site.site_id: site for site in sites}
    by_index = {site.ase_index_zero_based: site for site in sites}
    if len(by_id) != len(sites) or len(by_index) != len(sites):
        raise ValueError("Site manifest contains duplicate stable IDs or ASE indices")
    return by_id, by_index


def _complete_gap_rows(path: Path, description: str) -> list[dict[str, str]]:
    rows = _read_csv(path, description)
    if not rows:
        raise ValueError(f"{description} is empty: {path}")
    incomplete = [row for row in rows if row.get("output_status") != "COMPLETE"]
    if incomplete:
        counts: dict[str, int] = {}
        for row in rows:
            status = row.get("output_status", "")
            counts[status] = counts.get(status, 0) + 1
        summary = ", ".join(f"{key or 'UNKNOWN'}={value}" for key, value in sorted(counts.items()))
        raise ValueError(
            f"{description} is not complete ({summary}); finish/extract the full path first"
        )
    for row in rows:
        try:
            gap = float(row["gap_eV"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{description} contains a nonnumeric gap") from exc
        if not math.isfinite(gap):
            raise ValueError(f"{description} contains a nonfinite gap")
    return rows


def _path_image_name(row: dict[str, str]) -> str:
    name = row.get("image_name")
    if name:
        return name
    return f"image_{int(row['image']):03d}"


def _bottleneck(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: (float(row["gap_eV"]), int(row["image"])))


def _effective_scan_parameters(
    config: ProjectConfig,
    *,
    neighbor_cutoff: float | None,
    radius: float | None,
    steps: int | None,
    axes: str | None,
) -> tuple[float, float, int, str]:
    cutoff = config.helper.neighbor_cutoff_angstrom if neighbor_cutoff is None else float(neighbor_cutoff)
    scan_radius = config.helper.scan_radius_angstrom if radius is None else float(radius)
    scan_steps = config.helper.scan_steps_per_axis if steps is None else int(steps)
    scan_axes = config.helper.scan_axes if axes is None else axes.lower()
    if cutoff <= 0:
        raise ValueError("--neighbor-cutoff must be positive")
    if scan_radius < 0:
        raise ValueError("--radius must be nonnegative")
    if scan_steps <= 0 or scan_steps % 2 == 0:
        raise ValueError("--steps must be a positive odd integer")
    if not scan_axes or any(axis not in "xyz" for axis in scan_axes):
        raise ValueError("--axes must be a nonempty subset of xyz")
    if len(set(scan_axes)) != len(scan_axes):
        raise ValueError("--axes may not contain repeated coordinates")
    return cutoff, scan_radius, scan_steps, scan_axes


def _resolve_iteration_for_new_scan(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    *,
    force: bool,
) -> tuple[int, Path, Path, list[dict[str, Any]]]:
    campaign = _campaign_root(config, target_id, run_id)
    numbers = _iteration_numbers(campaign)
    if not numbers:
        return (
            1,
            _base_path_calc_root(config, target_id, run_id),
            _base_path_gap_csv(config, target_id, run_id),
            [],
        )

    latest = numbers[-1]
    latest_root = _iteration_root(config, target_id, run_id, latest)
    state = _read_json(latest_root / "iteration.json", "Latest helper iteration manifest")
    status = str(state.get("status", ""))
    if status == "CONVERGED" and not force:
        raise ValueError(
            f"Helper campaign is already converged at iteration {latest}: "
            f"minimum gap={state.get('minimum_gap_eV')} eV >= "
            f"{state.get('target_gap_eV')} eV. Use --force only to continue intentionally."
        )
    if status != "READY_FOR_NEXT" and not force:
        raise ValueError(
            f"Iteration {latest} is in state {status!r}. The default manual-review "
            "workflow requires helper extract and helper plot before the next scan."
        )
    if not (latest_root / "plot.json").is_file() and not force:
        raise ValueError(
            f"Iteration {latest} has not been plotted/reviewed; run orbit helper plot first."
        )
    previous_anchors = _read_json(
        latest_root / "anchors.json", "Previous helper anchor state"
    ).get("anchors", [])
    if not isinstance(previous_anchors, list):
        raise ValueError("Previous helper anchor state has a malformed anchors list")
    return (
        latest + 1,
        _helper_path_root(config, target_id, run_id, latest),
        _result_root(config, target_id, run_id, latest) / "gaps.csv",
        previous_anchors,
    )


def _prepare_submission_root(
    root: Path,
    label: str,
    config: ProjectConfig,
    max_concurrent: int,
    valid_names: list[str],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "valid_calculations.txt").write_text(
        "".join(f"{name}\n" for name in valid_names), encoding="utf-8"
    )
    array = root / "run_array.sh"
    submit = root / "submit.sh"
    array.write_text(render_array_script(label, config.slurm), encoding="utf-8")
    submit.write_text(render_submit_script(label, max_concurrent), encoding="utf-8")
    array.chmod(array.stat().st_mode | 0o111)
    submit.chmod(submit.stat().st_mode | 0o111)


def _disp_token(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.3f}".replace(".", "p")


def _scan_point_name(dx: float, dy: float, dz: float) -> str:
    return f"d_{_disp_token(dx)}_{_disp_token(dy)}_{_disp_token(dz)}"


def prepare_helper_scan(
    config: ProjectConfig,
    layout: ProjectLayout,
    target_id: str,
    run_id: str,
    *,
    neighbor_cutoff: float | None = None,
    radius: float | None = None,
    steps: int | None = None,
    axes: str | None = None,
    helper_site_ids: list[str] | None = None,
    max_concurrent: int | None = None,
    check_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Plan/write the next single-helper scan around the current global bottleneck."""
    target_id = _selected_target(config, target_id)
    run_id = _safe_run_id(run_id)
    validate_manifest_source(layout.structure_manifest, config.structure)
    resolve_path_run(layout, target_id, run_id)

    cutoff, scan_radius, scan_steps, scan_axes = _effective_scan_parameters(
        config,
        neighbor_cutoff=neighbor_cutoff,
        radius=radius,
        steps=steps,
        axes=axes,
    )
    concurrency = config.slurm.max_concurrent if max_concurrent is None else max_concurrent
    if concurrency <= 0:
        raise ValueError("--max-concurrent must be positive")

    iteration, baseline_calc_root, baseline_gap_csv, inherited_anchors = (
        _resolve_iteration_for_new_scan(config, target_id, run_id, force=force)
    )
    if not baseline_calc_root.is_dir():
        raise FileNotFoundError(f"Baseline helper path calculation directory not found: {baseline_calc_root}")
    gap_rows = _complete_gap_rows(baseline_gap_csv, "Baseline path gap table")
    bottleneck = _bottleneck(gap_rows)
    center = int(bottleneck["image"])
    center_name = _path_image_name(bottleneck)
    baseline_gap = float(bottleneck["gap_eV"])

    source_structure = baseline_calc_root / center_name / "structure.cif"
    if not source_structure.is_file():
        raise FileNotFoundError(f"Baseline bottleneck structure not found: {source_structure}")
    atoms = ase_read(source_structure)
    resources = resolve_qe_resources(config, list(atoms.get_chemical_symbols()))

    by_id, by_index = _site_maps(layout)
    if target_id not in by_id:
        raise ValueError(f"Target {target_id} is absent from the site manifest")
    target_site = by_id[target_id]
    target_index = int(target_site.ase_index_zero_based)
    symbols = list(atoms.get_chemical_symbols())
    if not 0 <= target_index < len(atoms):
        raise ValueError(f"Target index {target_index} is outside the bottleneck structure")
    if symbols[target_index] != target_site.element:
        raise ValueError(
            f"Target site manifest says {target_id} is {target_site.element}, "
            f"but path atom index {target_index} is {symbols[target_index]}"
        )

    frac = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    neighbors: list[dict[str, Any]] = []
    for helper_index in range(len(atoms)):
        if helper_index == target_index:
            continue
        site = by_index.get(helper_index)
        if site is None:
            raise ValueError(f"ASE index {helper_index} is absent from the site manifest")
        distance = minimum_image_distance(frac[helper_index], frac[target_index], cell)
        vector = minimum_image_displacement(frac[helper_index], frac[target_index], cell)
        neighbors.append(
            {
                "helper_site_id": site.site_id,
                "helper_index": helper_index,
                "helper_element": site.element,
                "target_distance_A": distance,
                "vector_x_A": float(vector[0]),
                "vector_y_A": float(vector[1]),
                "vector_z_A": float(vector[2]),
            }
        )
    neighbors.sort(key=lambda row: (row["target_distance_A"], row["helper_index"]))

    if helper_site_ids is None:
        selected = [
            row for row in neighbors
            if float(row["target_distance_A"]) <= cutoff + 1.0e-12
        ]
        selection_rule = f"distance <= {cutoff:.6f} A"
    else:
        requested = list(helper_site_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("--helper-site contains duplicates")
        if target_id in requested:
            raise ValueError("The displacement target itself cannot be a helper")
        unknown = [site_id for site_id in requested if site_id not in by_id]
        if unknown:
            raise ValueError("Unknown helper site IDs: " + ", ".join(unknown))
        selected_ids = set(requested)
        selected = [row for row in neighbors if row["helper_site_id"] in selected_ids]
        if len(selected) != len(requested):
            found = {row["helper_site_id"] for row in selected}
            raise ValueError(
                "Requested helper sites are not present in this structure: "
                + ", ".join(site_id for site_id in requested if site_id not in found)
            )
        selection_rule = "explicit --helper-site"
    if not selected:
        nearest = neighbors[0]["target_distance_A"] if neighbors else math.nan
        raise ValueError(
            f"No helper atoms selected; nearest non-target atom is {nearest:.6f} A. "
            "Increase [helper].neighbor_cutoff_angstrom or pass --helper-site."
        )

    values = np.linspace(-scan_radius, scan_radius, scan_steps)
    axis_values = [values if axis in scan_axes else np.array([0.0]) for axis in "xyz"]
    grid = list(itertools.product(*axis_values))
    nominal_per_helper = len(grid)
    estimated = nominal_per_helper * len(selected) - len(selected)  # zero uses baseline control

    summary = {
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": iteration,
        "center_image": center,
        "center_image_name": center_name,
        "baseline_gap_eV": baseline_gap,
        "target_gap_eV": config.helper.target_gap_eV,
        "baseline_calculation_root": str(baseline_calc_root),
        "baseline_gap_csv": str(baseline_gap_csv),
        "neighbor_cutoff_A": cutoff,
        "selection_rule": selection_rule,
        "candidate_helpers": selected,
        "scan_axes": scan_axes,
        "scan_radius_A": scan_radius,
        "scan_steps_per_axis": scan_steps,
        "nominal_points_per_helper": nominal_per_helper,
        "nominal_new_scf_count_before_geometry_veto": estimated,
        "check_only": check_only,
    }
    if check_only:
        return summary

    iteration_root = _iteration_root(config, target_id, run_id, iteration)
    scan_root = _scan_root(config, target_id, run_id, iteration)
    if iteration_root.exists() or scan_root.exists():
        raise FileExistsError(
            f"Helper iteration {iteration} already exists for {target_id}/{run_id}; "
            "finish that iteration instead of overwriting it"
        )
    iteration_root.mkdir(parents=True)
    scan_root.mkdir(parents=True)

    candidate_fields = [
        "helper_site_id", "helper_index", "helper_element", "target_distance_A",
        "vector_x_A", "vector_y_A", "vector_z_A",
    ]
    _write_csv(iteration_root / "neighbor_candidates.csv", selected, candidate_fields)

    scan_rows: list[dict[str, Any]] = []
    valid_names: list[str] = []
    vetoed = 0
    for helper in selected:
        helper_index = int(helper["helper_index"])
        helper_site_id = str(helper["helper_site_id"])
        helper_root = scan_root / helper_site_id
        helper_root.mkdir(parents=True)
        baseline_helper_min = _minimum_helper_distance(atoms, helper_index)
        configured_floor = config.helper.minimum_helper_distance_angstrom
        allowed_floor = (
            -math.inf
            if configured_floor <= 0
            else min(baseline_helper_min, configured_floor)
        )
        for point_index, (dx, dy, dz) in enumerate(grid):
            displacement = np.array([dx, dy, dz], dtype=float)
            magnitude = float(np.linalg.norm(displacement))
            point_name = _scan_point_name(float(dx), float(dy), float(dz))
            relative_calc = f"{helper_site_id}/{point_name}"
            common = {
                "iteration": iteration,
                "center_image": center,
                "baseline_gap_eV": baseline_gap,
                "target_site_id": target_id,
                "target_index": target_index,
                "helper_site_id": helper_site_id,
                "helper_index": helper_index,
                "helper_element": helper["helper_element"],
                "target_to_helper_distance_A": helper["target_distance_A"],
                "point_index": point_index,
                "dx_A": float(dx),
                "dy_A": float(dy),
                "dz_A": float(dz),
                "disp_norm_A": magnitude,
                "baseline_helper_min_distance_A": baseline_helper_min,
                "required_helper_min_distance_A": (
                    "" if not math.isfinite(allowed_floor) else allowed_floor
                ),
                "calculation_directory": relative_calc,
            }
            if magnitude <= 1.0e-12:
                scan_rows.append(
                    {
                        **common,
                        "preparation_status": "BASELINE_CONTROL",
                        "invalid_reason": "",
                        "minimum_helper_distance_A": baseline_helper_min,
                        "qe_input": "",
                        "qe_output": "",
                    }
                )
                continue

            generated = atoms.copy()
            positions = np.asarray(generated.get_positions(), dtype=float)
            positions[helper_index] += displacement
            generated.set_positions(positions)
            helper_min = _minimum_helper_distance(generated, helper_index)
            if helper_min < allowed_floor - 1.0e-10:
                vetoed += 1
                scan_rows.append(
                    {
                        **common,
                        "preparation_status": "COLLISION_VETO",
                        "invalid_reason": (
                            f"helper minimum distance {helper_min:.6f} A is below "
                            f"allowed {allowed_floor:.6f} A"
                        ),
                        "minimum_helper_distance_A": helper_min,
                        "qe_input": "",
                        "qe_output": "",
                    }
                )
                continue
            try:
                validate_scf_geometry(
                    generated,
                    resources,
                    config.qe.exact_overlap_tolerance_angstrom,
                )
                input_text = render_scf_input(
                    generated,
                    f"{target_id}_helper_i{iteration}",
                    f"{helper_site_id}_{point_name}",
                    config,
                    resources,
                )
            except InvalidGeometryError as exc:
                vetoed += 1
                scan_rows.append(
                    {
                        **common,
                        "preparation_status": "INVALID_GEOMETRY",
                        "invalid_reason": str(exc),
                        "minimum_helper_distance_A": helper_min,
                        "qe_input": "",
                        "qe_output": "",
                    }
                )
                continue

            folder = scan_root / relative_calc
            folder.mkdir(parents=True)
            (folder / "tmp").mkdir()
            ase_write(folder / "structure.cif", generated)
            (folder / "espresso_scf.pwi").write_text(input_text, encoding="utf-8")
            valid_names.append(relative_calc)
            scan_rows.append(
                {
                    **common,
                    "preparation_status": "VALID",
                    "invalid_reason": "",
                    "minimum_helper_distance_A": helper_min,
                    "qe_input": f"{relative_calc}/espresso_scf.pwi",
                    "qe_output": f"{relative_calc}/espresso_scf.pwo",
                }
            )

    if not valid_names:
        raise ValueError("No nonzero helper scan geometry survived validation")
    _write_csv(iteration_root / "scan_manifest.csv", scan_rows)
    _prepare_submission_root(
        scan_root,
        f"{target_id}_helper_i{iteration}_scan",
        config,
        concurrency,
        valid_names,
    )

    campaign = _campaign_root(config, target_id, run_id)
    campaign.mkdir(parents=True, exist_ok=True)
    campaign_manifest = campaign / "campaign.json"
    if not campaign_manifest.is_file():
        _write_json(
            campaign_manifest,
            {
                "schema_version": HELPER_SCHEMA_VERSION,
                "created_at": _utc_now(),
                "target_id": target_id,
                "path_run_id": run_id,
                "reference_path_calculation_root": str(
                    _base_path_calc_root(config, target_id, run_id)
                ),
                "reference_path_gap_csv": str(
                    _base_path_gap_csv(config, target_id, run_id)
                ),
            },
        )

    iteration_manifest = {
        "schema_version": HELPER_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "SCAN_PREPARED",
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": iteration,
        "center_image": center,
        "center_image_name": center_name,
        "baseline_gap_eV": baseline_gap,
        "target_gap_eV": config.helper.target_gap_eV,
        "baseline_calculation_root": str(baseline_calc_root),
        "baseline_gap_csv": str(baseline_gap_csv),
        "reference_path_calculation_root": str(
            _base_path_calc_root(config, target_id, run_id)
        ),
        "scan_root": str(scan_root),
        "scan_manifest": str(iteration_root / "scan_manifest.csv"),
        "candidate_helper_count": len(selected),
        "valid_scan_count": len(valid_names),
        "collision_or_geometry_veto_count": vetoed,
        "scan_axes": scan_axes,
        "scan_radius_A": scan_radius,
        "scan_steps_per_axis": scan_steps,
        "neighbor_cutoff_A": cutoff,
        "minimum_helper_distance_A": config.helper.minimum_helper_distance_angstrom,
        "inherited_anchor_count": len(inherited_anchors),
        "max_concurrent": concurrency,
    }
    _write_json(iteration_root / "iteration.json", iteration_manifest)
    summary.update(
        {
            "scan_root": str(scan_root),
            "iteration_root": str(iteration_root),
            "valid_scan_count": len(valid_names),
            "vetoed_count": vetoed,
        }
    )
    return summary


def submit_helper_scan(
    config: ProjectConfig, target_id: str, run_id: str, iteration: int
) -> None:
    root = _scan_root(config, target_id, run_id, iteration)
    submit = root / "submit.sh"
    if not submit.is_file():
        raise FileNotFoundError(f"Helper scan submission script not found: {submit}")
    result = subprocess.run(["bash", str(submit)], cwd=root, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Helper scan submission failed with exit status {result.returncode}")


def _boundary_axes(row: dict[str, Any], axes: str, radius: float, tol: float = 1.0e-10) -> list[str]:
    found: list[str] = []
    for axis in axes:
        value = float(row[f"d{axis}_A"])
        if abs(value + radius) <= tol:
            found.append(f"{axis}-")
        if abs(value - radius) <= tol:
            found.append(f"{axis}+")
    return found


def _choose_winner(rows: list[dict[str, Any]], tie_tol: float = 1.0e-8) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot choose a helper winner from an empty row set")
    largest = max(float(row["gap_eV"]) for row in rows)
    tied = [row for row in rows if largest - float(row["gap_eV"]) <= tie_tol]
    return min(
        tied,
        key=lambda row: (
            float(row["disp_norm_A"]),
            int(row.get("point_index", 0)),
        ),
    )


def _raised_cosine(image: int, center: int, half_width: float) -> float:
    if half_width <= 0:
        raise ValueError("helper taper half width must be positive")
    distance = abs(image - center)
    if distance >= half_width:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * distance / half_width))


def _smoothstep(t: float) -> float:
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"smoothstep argument outside [0,1]: {t}")
    return t * t * (3.0 - 2.0 * t)


def _anchor_displacement(
    image: int,
    anchors: list[dict[str, Any]],
    nat: int,
    half_width: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    zero = np.zeros((nat, 3), dtype=float)
    if not anchors:
        return zero, {
            "region": "zero",
            "left_anchor": "",
            "right_anchor": "",
            "weight": 0.0,
        }
    ordered = sorted(anchors, key=lambda item: int(item["image"]))
    images = [int(item["image"]) for item in ordered]
    for anchor in ordered:
        array = np.asarray(anchor["displacements_A"], dtype=float)
        if array.shape != (nat, 3) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"Anchor image {anchor.get('image')} has invalid displacement array"
            )
    if image in images:
        anchor = ordered[images.index(image)]
        return np.asarray(anchor["displacements_A"], dtype=float), {
            "region": "anchor",
            "left_anchor": image,
            "right_anchor": image,
            "weight": 1.0,
        }
    if image < images[0]:
        weight = _raised_cosine(image, images[0], half_width)
        return weight * np.asarray(ordered[0]["displacements_A"], dtype=float), {
            "region": "left_taper",
            "left_anchor": "",
            "right_anchor": images[0],
            "weight": weight,
        }
    if image > images[-1]:
        weight = _raised_cosine(image, images[-1], half_width)
        return weight * np.asarray(ordered[-1]["displacements_A"], dtype=float), {
            "region": "right_taper",
            "left_anchor": images[-1],
            "right_anchor": "",
            "weight": weight,
        }
    for left, right in zip(ordered[:-1], ordered[1:]):
        li = int(left["image"])
        ri = int(right["image"])
        if li < image < ri:
            t = (image - li) / (ri - li)
            weight = _smoothstep(t)
            left_d = np.asarray(left["displacements_A"], dtype=float)
            right_d = np.asarray(right["displacements_A"], dtype=float)
            return (1.0 - weight) * left_d + weight * right_d, {
                "region": "between_anchors",
                "left_anchor": li,
                "right_anchor": ri,
                "weight": weight,
            }
    return zero, {
        "region": "zero",
        "left_anchor": "",
        "right_anchor": "",
        "weight": 0.0,
    }


def _full_displacements(reference_atoms: Any, geometry_atoms: Any) -> np.ndarray:
    if list(reference_atoms.get_chemical_symbols()) != list(geometry_atoms.get_chemical_symbols()):
        raise ValueError("Reference and helper geometries have different atom ordering/species")
    ref_cell = np.asarray(reference_atoms.cell.array, dtype=float)
    geo_cell = np.asarray(geometry_atoms.cell.array, dtype=float)
    if not np.allclose(ref_cell, geo_cell, atol=1.0e-9, rtol=1.0e-9):
        raise ValueError("Reference and helper geometries have different cells")
    ref_frac = np.asarray(reference_atoms.get_scaled_positions(wrap=False), dtype=float)
    geo_frac = np.asarray(geometry_atoms.get_scaled_positions(wrap=False), dtype=float)
    return np.asarray(
        [
            minimum_image_displacement(geo_frac[index], ref_frac[index], ref_cell)
            for index in range(len(reference_atoms))
        ],
        dtype=float,
    )


def analyze_helper_scan(
    config: ProjectConfig,
    layout: ProjectLayout,
    target_id: str,
    run_id: str,
    *,
    allow_nonimproving: bool = False,
    max_concurrent: int | None = None,
) -> dict[str, Any]:
    """Extract the latest scan, choose the best path-safe helper, and build its path."""
    target_id = _selected_target(config, target_id)
    run_id = _safe_run_id(run_id)
    campaign = _campaign_root(config, target_id, run_id)
    numbers = _iteration_numbers(campaign)
    if not numbers:
        raise FileNotFoundError("No helper iteration exists; run orbit helper scan first")
    iteration = numbers[-1]
    iteration_root = _iteration_root(config, target_id, run_id, iteration)
    state = _read_json(iteration_root / "iteration.json", "Helper iteration manifest")
    if state.get("status") != "SCAN_PREPARED":
        raise ValueError(
            f"Iteration {iteration} is in state {state.get('status')!r}, not SCAN_PREPARED"
        )
    scan_root = _scan_root(config, target_id, run_id, iteration)
    manifest = _read_csv(iteration_root / "scan_manifest.csv", "Helper scan manifest")
    baseline_gap = float(state["baseline_gap_eV"])

    result_rows: list[dict[str, Any]] = []
    incomplete = 0
    for row in manifest:
        prepared = row["preparation_status"]
        output_status = prepared
        homo: float | str = ""
        lumo: float | str = ""
        gap: float | str = ""
        note = row.get("invalid_reason", "")
        output_hash = ""
        if prepared == "BASELINE_CONTROL":
            output_status = "CONTROL"
            gap = baseline_gap
            note = "baseline bottleneck gap reused as zero-displacement control"
        elif prepared == "VALID":
            output_path = scan_root / row["qe_output"]
            if not output_path.is_file():
                output_status = "MISSING_OUTPUT"
                note = "QE output file not found"
                incomplete += 1
            else:
                parsed = parse_qe_output(output_path)
                output_status = parsed.status
                if parsed.homo_ev is not None:
                    homo = parsed.homo_ev
                if parsed.lumo_ev is not None:
                    lumo = parsed.lumo_ev
                if parsed.gap_ev is not None:
                    gap = parsed.gap_ev
                note = parsed.note
                output_hash = _sha256(output_path)
                if parsed.status != "COMPLETE":
                    incomplete += 1
        result_rows.append(
            {
                **row,
                "output_status": output_status,
                "homo_eV": homo,
                "lumo_eV": lumo,
                "gap_eV": gap,
                "qe_output_sha256": output_hash,
                "note": note,
            }
        )
    _write_csv(iteration_root / "scan_results.csv", result_rows)

    # Missing/failed scan points do not block the whole iteration. They remain
    # visible in scan_results.csv and are excluded from optimization.
    usable = [
        row for row in result_rows
        if row["output_status"] in {"COMPLETE", "CONTROL"} and row["gap_eV"] != ""
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in usable:
        grouped.setdefault(str(row["helper_site_id"]), []).append(row)
    if not grouped:
        raise ValueError("No completed helper scan points are available for ranking")

    # Preserve the familiar per-helper electronic ranking for diagnosis.
    rankings: list[dict[str, Any]] = []
    radius = float(state["scan_radius_A"])
    axes = str(state["scan_axes"])
    for helper_site_id, rows in sorted(grouped.items()):
        helper_winner = _choose_winner(rows)
        best_gap = float(helper_winner["gap_eV"])
        boundaries = _boundary_axes(helper_winner, axes, radius)
        rankings.append(
            {
                "rank": 0,
                "helper_site_id": helper_site_id,
                "helper_index": helper_winner["helper_index"],
                "helper_element": helper_winner["helper_element"],
                "target_to_helper_distance_A": helper_winner["target_to_helper_distance_A"],
                "best_gap_eV": best_gap,
                "control_gap_eV": baseline_gap,
                "gap_improvement_eV": best_gap - baseline_gap,
                "dx_A": helper_winner["dx_A"],
                "dy_A": helper_winner["dy_A"],
                "dz_A": helper_winner["dz_A"],
                "disp_norm_A": helper_winner["disp_norm_A"],
                "winner_on_scan_boundaries": ",".join(boundaries),
                "winner_calculation_directory": helper_winner["calculation_directory"],
            }
        )
    rankings.sort(
        key=lambda row: (
            -float(row["best_gap_eV"]),
            float(row["disp_norm_A"]),
            int(row["helper_index"]),
        )
    )
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank
    _write_csv(iteration_root / "helper_rankings.csv", rankings)

    center = int(state["center_image"])
    center_name = str(state["center_image_name"])
    reference_root = _base_path_calc_root(config, target_id, run_id)
    baseline_root = Path(str(state["baseline_calculation_root"]))
    reference_center = reference_root / center_name / "structure.cif"
    if not reference_center.is_file():
        raise FileNotFoundError(f"Reference center structure is missing: {reference_center}")
    reference_atoms = ase_read(reference_center)

    by_id, _by_index = _site_maps(layout)
    target_index = int(by_id[target_id].ase_index_zero_based)

    inherited_anchors: list[dict[str, Any]] = []
    if iteration > 1:
        previous_root = _iteration_root(config, target_id, run_id, iteration - 1)
        previous_state = _read_json(previous_root / "anchors.json", "Previous anchor state")
        raw_anchors = previous_state.get("anchors", [])
        if not isinstance(raw_anchors, list):
            raise ValueError("Previous anchor state has malformed anchors")
        inherited_anchors = list(raw_anchors)

    resources = resolve_qe_resources(config, list(reference_atoms.get_chemical_symbols()))
    concurrency = config.slurm.max_concurrent if max_concurrent is None else max_concurrent
    if concurrency <= 0:
        raise ValueError("--max-concurrent must be positive")

    reference_rows = _complete_gap_rows(
        _base_path_gap_csv(config, target_id, run_id),
        "Reference target-only path gap table",
    )
    baseline_rows = _complete_gap_rows(
        Path(str(state["baseline_gap_csv"])), "Current baseline gap table"
    )
    ref_names = [_path_image_name(row) for row in reference_rows]
    baseline_names = [_path_image_name(row) for row in baseline_rows]
    if ref_names != baseline_names:
        raise ValueError("Reference and current baseline paths have different image sets")

    # Load the fixed path geometry once. Candidate helper scan points are then
    # screened in descending gap order by reconstructing the *entire*
    # interpolated helper path. This enforces the STO-derived rule that a large
    # center-gap improvement is rejected if its smooth interpolation creates a
    # worse short contact elsewhere.
    geometry: list[tuple[int, str, Any, Any]] = []
    for image, image_name in enumerate(ref_names):
        reference_structure = reference_root / image_name / "structure.cif"
        baseline_structure = baseline_root / image_name / "structure.cif"
        if not reference_structure.is_file() or not baseline_structure.is_file():
            raise FileNotFoundError(
                f"Missing reference/baseline structure for helper path image {image_name}"
            )
        geometry.append(
            (
                image,
                image_name,
                ase_read(reference_structure),
                ase_read(baseline_structure),
            )
        )

    candidates = [
        row for row in usable
        if row["output_status"] == "COMPLETE"
        and row.get("preparation_status") == "VALID"
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["gap_eV"]),
            float(row["disp_norm_A"]),
            int(row["helper_index"]),
            str(row["calculation_directory"]),
        )
    )
    if not candidates:
        raise ValueError(
            "No completed nonzero helper displacement is available; only controls "
            "or unusable scan points remain."
        )

    screening_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_anchors: list[dict[str, Any]] | None = None
    selected_full: np.ndarray | None = None
    rejected_higher_gap = 0

    for candidate_rank, candidate in enumerate(candidates, start=1):
        candidate_gap = float(candidate["gap_eV"])
        improvement = candidate_gap - baseline_gap
        if (
            improvement <= config.helper.improvement_tolerance_eV
            and not allow_nonimproving
        ):
            # Candidates are sorted by gap, so every later point is no better.
            break

        helper_index = int(candidate["helper_index"])
        winner_structure = scan_root / candidate["calculation_directory"] / "structure.cif"
        if not winner_structure.is_file():
            screening_rows.append(
                {
                    "candidate_rank": candidate_rank,
                    "helper_site_id": candidate["helper_site_id"],
                    "helper_index": helper_index,
                    "gap_eV": candidate_gap,
                    "gap_improvement_eV": improvement,
                    "dx_A": candidate["dx_A"],
                    "dy_A": candidate["dy_A"],
                    "dz_A": candidate["dz_A"],
                    "disp_norm_A": candidate["disp_norm_A"],
                    "path_safe": False,
                    "first_unsafe_image": "",
                    "generated_minimum_distance_A": "",
                    "allowed_minimum_distance_A": "",
                    "reason": f"missing candidate structure: {winner_structure}",
                }
            )
            rejected_higher_gap += 1
            continue

        candidate_atoms = ase_read(winner_structure)
        full = _full_displacements(reference_atoms, candidate_atoms)
        target_error = float(np.linalg.norm(full[target_index]))
        if target_error > 1.0e-7:
            screening_rows.append(
                {
                    "candidate_rank": candidate_rank,
                    "helper_site_id": candidate["helper_site_id"],
                    "helper_index": helper_index,
                    "gap_eV": candidate_gap,
                    "gap_improvement_eV": improvement,
                    "dx_A": candidate["dx_A"],
                    "dy_A": candidate["dy_A"],
                    "dz_A": candidate["dz_A"],
                    "disp_norm_A": candidate["disp_norm_A"],
                    "path_safe": False,
                    "first_unsafe_image": center,
                    "generated_minimum_distance_A": "",
                    "allowed_minimum_distance_A": "",
                    "reason": (
                        "helper scan moved the target relative to the reference "
                        f"path by {target_error:.3e} A"
                    ),
                }
            )
            rejected_higher_gap += 1
            continue

        anchors = [
            anchor
            for anchor in inherited_anchors
            if int(anchor["image"]) != center
        ]
        anchors.append(
            {
                "image": center,
                "iteration": iteration,
                "helper_site_id": candidate["helper_site_id"],
                "helper_index": helper_index,
                "helper_element": candidate["helper_element"],
                "center_gap_before_eV": baseline_gap,
                "center_gap_after_scan_eV": candidate_gap,
                "gap_improvement_eV": improvement,
                "incremental_displacement_A": [
                    float(candidate["dx_A"]),
                    float(candidate["dy_A"]),
                    float(candidate["dz_A"]),
                ],
                "displacements_A": full.tolist(),
            }
        )
        anchors.sort(key=lambda anchor: int(anchor["image"]))

        safe = True
        unsafe_image: int | str = ""
        unsafe_generated: float | str = ""
        unsafe_allowed: float | str = ""
        unsafe_reason = ""
        for image, image_name, ref_atoms, baseline_atoms in geometry:
            correction, _interpolation = _anchor_displacement(
                image,
                anchors,
                len(ref_atoms),
                config.helper.taper_half_width_images,
            )
            generated = ref_atoms.copy()
            generated.set_positions(
                np.asarray(generated.get_positions(), dtype=float) + correction
            )
            baseline_min = _minimum_all_distance(baseline_atoms)
            generated_min = _minimum_all_distance(generated)
            configured_floor = config.helper.minimum_helper_distance_angstrom
            allowed_floor = (
                min(baseline_min, configured_floor)
                if configured_floor > 0
                else -math.inf
            )
            if configured_floor > 0 and generated_min < allowed_floor - 1.0e-10:
                safe = False
                unsafe_image = image
                unsafe_generated = generated_min
                unsafe_allowed = allowed_floor
                unsafe_reason = (
                    f"{image_name}: minimum distance {generated_min:.6f} A below "
                    f"allowed {allowed_floor:.6f} A"
                )
                break
            try:
                validate_scf_geometry(
                    generated,
                    resources,
                    config.qe.exact_overlap_tolerance_angstrom,
                )
            except InvalidGeometryError as exc:
                safe = False
                unsafe_image = image
                unsafe_generated = generated_min
                unsafe_allowed = (
                    "" if not math.isfinite(allowed_floor) else allowed_floor
                )
                unsafe_reason = f"{image_name}: {exc}"
                break

        screening_rows.append(
            {
                "candidate_rank": candidate_rank,
                "helper_site_id": candidate["helper_site_id"],
                "helper_index": helper_index,
                "gap_eV": candidate_gap,
                "gap_improvement_eV": improvement,
                "dx_A": candidate["dx_A"],
                "dy_A": candidate["dy_A"],
                "dz_A": candidate["dz_A"],
                "disp_norm_A": candidate["disp_norm_A"],
                "path_safe": safe,
                "first_unsafe_image": unsafe_image,
                "generated_minimum_distance_A": unsafe_generated,
                "allowed_minimum_distance_A": unsafe_allowed,
                "reason": unsafe_reason,
            }
        )
        if safe:
            selected = candidate
            selected_anchors = anchors
            selected_full = full
            break
        rejected_higher_gap += 1

    screening_path = iteration_root / "path_candidate_screening.csv"
    if screening_rows:
        _write_csv(screening_path, screening_rows)

    if selected is None or selected_anchors is None or selected_full is None:
        best_improvement = max(
            (float(row["gap_eV"]) - baseline_gap for row in candidates),
            default=-math.inf,
        )
        if (
            best_improvement <= config.helper.improvement_tolerance_eV
            and not allow_nonimproving
        ):
            raise ValueError(
                f"Best individual helper improves the bottleneck by only "
                f"{best_improvement:+.6e} eV, not more than "
                f"{config.helper.improvement_tolerance_eV:.6e} eV. "
                "No helper path was built."
            )
        detail = screening_rows[0]["reason"] if screening_rows else "no candidate was screened"
        raise ValueError(
            "No improving completed helper scan point produces a geometry-safe "
            f"interpolated path. First rejection: {detail}. "
            f"See {screening_path}."
        )

    winner = selected
    anchors = selected_anchors
    full = selected_full
    improvement = float(winner["gap_eV"]) - baseline_gap
    helper_index = int(winner["helper_index"])

    path_root = _helper_path_root(config, target_id, run_id, iteration)
    # A failed analyze can leave a partially written path directory while
    # iteration.json remains SCAN_PREPARED. Such data are uncommitted and safe
    # to discard so analyze is restartable.
    if path_root.exists():
        shutil.rmtree(path_root)
    path_root.mkdir(parents=True)

    helper_path_rows: list[dict[str, Any]] = []
    valid_names: list[str] = []
    for image, image_name, ref_atoms, baseline_atoms in geometry:
        correction, interpolation = _anchor_displacement(
            image, anchors, len(ref_atoms), config.helper.taper_half_width_images
        )
        generated = ref_atoms.copy()
        generated.set_positions(
            np.asarray(generated.get_positions(), dtype=float) + correction
        )

        baseline_min = _minimum_all_distance(baseline_atoms)
        generated_min = _minimum_all_distance(generated)
        configured_floor = config.helper.minimum_helper_distance_angstrom
        if configured_floor > 0:
            allowed_floor = min(baseline_min, configured_floor)
            if generated_min < allowed_floor - 1.0e-10:
                raise RuntimeError(
                    f"Internal path-safety inconsistency at {image_name}: "
                    f"{generated_min:.6f} A below {allowed_floor:.6f} A after "
                    "the candidate passed preflight."
                )
        validate_scf_geometry(
            generated,
            resources,
            config.qe.exact_overlap_tolerance_angstrom,
        )
        folder = path_root / image_name
        folder.mkdir()
        (folder / "tmp").mkdir()
        ase_write(folder / "structure.cif", generated)
        input_text = render_scf_input(
            generated,
            f"{target_id}_helper_i{iteration}",
            image_name,
            config,
            resources,
        )
        (folder / "espresso_scf.pwi").write_text(input_text, encoding="utf-8")
        valid_names.append(image_name)
        norms = np.linalg.norm(correction, axis=1)
        helper_path_rows.append(
            {
                "image": image,
                "image_name": image_name,
                "iteration": iteration,
                "center_image": center,
                "new_helper_site_id": winner["helper_site_id"],
                "new_helper_index": helper_index,
                "interpolation_region": interpolation["region"],
                "left_anchor": interpolation["left_anchor"],
                "right_anchor": interpolation["right_anchor"],
                "interpolation_weight": interpolation["weight"],
                "active_atom_count": int(np.count_nonzero(norms > 1.0e-10)),
                "maximum_atomic_correction_A": float(np.max(norms, initial=0.0)),
                "minimum_distance_A": generated_min,
                "baseline_minimum_distance_A": baseline_min,
                "structure_file": f"{image_name}/structure.cif",
                "qe_input": f"{image_name}/espresso_scf.pwi",
                "qe_output": f"{image_name}/espresso_scf.pwo",
            }
        )
    _write_csv(iteration_root / "helper_path.csv", helper_path_rows)
    _prepare_submission_root(
        path_root,
        f"{target_id}_helper_i{iteration}_path",
        config,
        concurrency,
        valid_names,
    )

    anchor_payload = {
        "schema_version": HELPER_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "target_id": target_id,
        "path_run_id": run_id,
        "latest_iteration": iteration,
        "taper_half_width_images": config.helper.taper_half_width_images,
        "anchors": anchors,
    }
    _write_json(iteration_root / "anchors.json", anchor_payload)

    best_payload = {
        "schema_version": HELPER_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": iteration,
        "center_image": center,
        "helper_site_id": winner["helper_site_id"],
        "helper_index": helper_index,
        "helper_element": winner["helper_element"],
        "best_gap_eV": float(winner["gap_eV"]),
        "control_gap_eV": baseline_gap,
        "gap_improvement_eV": improvement,
        "displacement_A": {
            "dx": float(winner["dx_A"]),
            "dy": float(winner["dy_A"]),
            "dz": float(winner["dz_A"]),
        },
        "winner_on_scan_boundaries": _boundary_axes(
            winner, str(state["scan_axes"]), float(state["scan_radius_A"])
        ),
        "selection_mode": "maximum_gap_subject_to_full_path_geometry",
        "rejected_higher_gap_path_candidates": rejected_higher_gap,
        "path_candidate_screening_file": str(screening_path),
    }
    _write_json(iteration_root / "best_helper.json", best_payload)

    state.update(
        {
            "updated_at": _utc_now(),
            "status": "PATH_PREPARED",
            "best_helper_site_id": winner["helper_site_id"],
            "best_helper_index": helper_index,
            "best_helper_gap_eV": float(winner["gap_eV"]),
            "best_helper_gap_improvement_eV": improvement,
            "helper_path_root": str(path_root),
            "helper_path_manifest": str(iteration_root / "helper_path.csv"),
            "anchor_file": str(iteration_root / "anchors.json"),
            "path_image_count": len(valid_names),
            "max_concurrent": concurrency,
            "skipped_incomplete_scan_count": incomplete,
            "rejected_higher_gap_path_candidates": rejected_higher_gap,
            "path_candidate_screening_file": str(screening_path),
        }
    )
    _write_json(iteration_root / "iteration.json", state)

    return {
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": iteration,
        "center_image": center,
        "baseline_gap_eV": baseline_gap,
        "best_helper_site_id": winner["helper_site_id"],
        "best_helper_index": helper_index,
        "best_gap_eV": float(winner["gap_eV"]),
        "gap_improvement_eV": improvement,
        "displacement_A": [
            float(winner["dx_A"]),
            float(winner["dy_A"]),
            float(winner["dz_A"]),
        ],
        "winner_on_scan_boundaries": best_payload["winner_on_scan_boundaries"],
        "rankings": rankings,
        "path_root": str(path_root),
        "skipped_incomplete_scan_count": incomplete,
        "rejected_higher_gap_path_candidates": rejected_higher_gap,
        "path_candidate_screening_file": str(screening_path),
    }


def helper_path_scf(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    *,
    submit: bool = False,
) -> dict[str, Any]:
    target_id = _selected_target(config, target_id)
    run_id = _safe_run_id(run_id)
    numbers = _iteration_numbers(_campaign_root(config, target_id, run_id))
    if not numbers:
        raise FileNotFoundError("No helper iteration exists")
    iteration = numbers[-1]
    iteration_root = _iteration_root(config, target_id, run_id, iteration)
    state = _read_json(iteration_root / "iteration.json", "Helper iteration manifest")
    if state.get("status") != "PATH_PREPARED":
        raise ValueError(
            f"Iteration {iteration} is in state {state.get('status')!r}; "
            "run helper analyze before helper path-scf"
        )
    path_root = _helper_path_root(config, target_id, run_id, iteration)
    names = [
        line.strip()
        for line in (path_root / "valid_calculations.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if submit:
        submit_script = path_root / "submit.sh"
        result = subprocess.run(["bash", str(submit_script)], cwd=path_root, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Helper path submission failed with exit status {result.returncode}"
            )
    return {
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": iteration,
        "path_root": str(path_root),
        "image_count": len(names),
        "submitted": submit,
    }


def extract_helper_path(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
) -> dict[str, Any]:
    target_id = _selected_target(config, target_id)
    run_id = _safe_run_id(run_id)
    numbers = _iteration_numbers(_campaign_root(config, target_id, run_id))
    if not numbers:
        raise FileNotFoundError("No helper iteration exists")
    iteration = numbers[-1]
    iteration_root = _iteration_root(config, target_id, run_id, iteration)
    state = _read_json(iteration_root / "iteration.json", "Helper iteration manifest")
    if state.get("status") not in {"PATH_PREPARED", "NEEDS_REVIEW", "READY_FOR_NEXT", "CONVERGED"}:
        raise ValueError(
            f"Iteration {iteration} is in state {state.get('status')!r}; "
            "no helper path is ready for extraction"
        )
    path_root = _helper_path_root(config, target_id, run_id, iteration)
    path_rows = _read_csv(iteration_root / "helper_path.csv", "Helper path manifest")
    output_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    complete_gaps: list[tuple[int, float]] = []
    for row in path_rows:
        image_name = row["image_name"]
        output = path_root / image_name / "espresso_scf.pwo"
        output_hash = ""
        if not output.is_file():
            status = "MISSING_OUTPUT"
            parsed = None
            note = "QE output file not found"
        else:
            parsed = parse_qe_output(output)
            status = parsed.status
            note = parsed.note
            output_hash = _sha256(output)
        status_counts[status] = status_counts.get(status, 0) + 1
        gap = "" if parsed is None or parsed.gap_ev is None else parsed.gap_ev
        homo = "" if parsed is None or parsed.homo_ev is None else parsed.homo_ev
        lumo = "" if parsed is None or parsed.lumo_ev is None else parsed.lumo_ev
        if status == "COMPLETE":
            complete_gaps.append((int(row["image"]), float(gap)))
        output_rows.append(
            {
                "image": row["image"],
                "image_name": image_name,
                "target_id": target_id,
                "path_run_id": run_id,
                "iteration": iteration,
                "output_status": status,
                "job_done": int(parsed.job_done) if parsed is not None else 0,
                "homo_eV": homo,
                "lumo_eV": lumo,
                "gap_eV": gap,
                "qe_output": str(output),
                "qe_output_sha256": output_hash,
                "note": note,
            }
        )
    result_root = _result_root(config, target_id, run_id, iteration)
    result_root.mkdir(parents=True, exist_ok=True)
    gaps_csv = result_root / "gaps.csv"
    _write_csv(gaps_csv, output_rows)
    all_complete = len(complete_gaps) == len(path_rows)
    minimum_image = None
    minimum_gap = None
    if all_complete and complete_gaps:
        minimum_image, minimum_gap = min(complete_gaps, key=lambda item: (item[1], item[0]))
        new_status = (
            "CONVERGED"
            if minimum_gap >= config.helper.target_gap_eV
            else "NEEDS_REVIEW"
        )
    else:
        new_status = "PATH_PREPARED"
    extraction = {
        "schema_version": HELPER_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "target_id": target_id,
        "path_run_id": run_id,
        "iteration": iteration,
        "total_count": len(path_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "complete_count": len(complete_gaps),
        "minimum_gap_eV": minimum_gap,
        "minimum_gap_image": minimum_image,
        "target_gap_eV": config.helper.target_gap_eV,
        "converged": bool(all_complete and minimum_gap is not None and minimum_gap >= config.helper.target_gap_eV),
        "gaps_file": str(gaps_csv),
        "gaps_file_sha256": _sha256(gaps_csv),
    }
    _write_json(result_root / "extraction.json", extraction)
    state.update(
        {
            "updated_at": _utc_now(),
            "status": new_status,
            "complete_path_image_count": len(complete_gaps),
            "minimum_gap_eV": minimum_gap,
            "minimum_gap_image": minimum_image,
            "target_gap_eV": config.helper.target_gap_eV,
            "converged": extraction["converged"],
            "result_root": str(result_root),
        }
    )
    _write_json(iteration_root / "iteration.json", state)
    return extraction


def write_helper_plot(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    *,
    output: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Plot all completed helper iterations without blocking on a running one.

    If the newest discovered helper iteration is still scanning, has a prepared
    path, is running path SCFs, or otherwise lacks a complete extracted gaps.csv,
    plot only through the newest plot-ready iteration.  A running iteration is
    never marked reviewed by this command.
    """
    target_id = _selected_target(config, target_id)
    run_id = _safe_run_id(run_id)
    numbers = _iteration_numbers(_campaign_root(config, target_id, run_id))
    if not numbers:
        raise FileNotFoundError("No helper iteration exists")

    newest_iteration = numbers[-1]

    # Find the newest helper iteration with a complete extracted path-gap table.
    # Missing/empty/incomplete later iterations are normal while a campaign is
    # running and must not prevent inspection of earlier completed iterations.
    plot_iteration: int | None = None
    skipped: list[dict[str, Any]] = []
    for candidate in reversed(numbers):
        result_csv = _result_root(config, target_id, run_id, candidate) / "gaps.csv"
        try:
            _complete_gap_rows(
                result_csv,
                f"Helper iteration {candidate} path gap table",
            )
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(
                {
                    "iteration": candidate,
                    "reason": str(exc),
                }
            )
            continue
        plot_iteration = candidate
        break

    if plot_iteration is None:
        raise ValueError(
            "No completed helper iteration is plot-ready yet. "
            "Wait for at least one helper path to finish and run helper extract."
        )

    plot_root = _iteration_root(config, target_id, run_id, plot_iteration)
    plot_state = _read_json(
        plot_root / "iteration.json",
        "Helper iteration manifest",
    )

    from .helper_plotting import write_helper_iteration_viewers

    output_path, manifest = write_helper_iteration_viewers(
        config,
        target_id,
        run_id,
        plot_iteration,
        output=output,
    )
    manifest.update(
        {
            "created_at": _utc_now(),
            "plot_file": str(output_path),
            "plot_sha256": _sha256(output_path),
            "newest_discovered_iteration": newest_iteration,
            "latest_plotted_iteration": plot_iteration,
            "skipped_newer_iterations": list(reversed(skipped)),
        }
    )
    _write_json(plot_root / "plot.json", manifest)

    # Manual-review semantics apply only to the iteration actually plotted.
    # In particular, plotting iteration 3 while iteration 4 is running must not
    # mutate iteration 4's state.
    if plot_state.get("status") == "NEEDS_REVIEW":
        plot_state["status"] = "READY_FOR_NEXT"
        plot_state["reviewed_at"] = _utc_now()
        plot_state["updated_at"] = _utc_now()
        _write_json(plot_root / "iteration.json", plot_state)

    return output_path, manifest


def helper_status(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
) -> dict[str, Any]:
    target_id = _selected_target(config, target_id)
    run_id = _safe_run_id(run_id)
    campaign = _campaign_root(config, target_id, run_id)
    numbers = _iteration_numbers(campaign)
    rows = []
    for iteration in numbers:
        root = _iteration_root(config, target_id, run_id, iteration)
        state = _read_json(root / "iteration.json", f"Helper iteration {iteration} manifest")
        rows.append(
            {
                "iteration": iteration,
                "status": state.get("status"),
                "center_image": state.get("center_image"),
                "baseline_gap_eV": state.get("baseline_gap_eV"),
                "best_helper_site_id": state.get("best_helper_site_id"),
                "best_helper_gap_eV": state.get("best_helper_gap_eV"),
                "minimum_gap_eV": state.get("minimum_gap_eV"),
                "minimum_gap_image": state.get("minimum_gap_image"),
                "converged": state.get("converged", False),
                "reviewed": bool(state.get("reviewed_at")),
            }
        )
    next_action = "helper scan"
    if rows:
        status = rows[-1]["status"]
        next_action = {
            "SCAN_PREPARED": "helper analyze",
            "PATH_PREPARED": "helper path-scf --submit, then helper extract",
            "NEEDS_REVIEW": "helper plot",
            "READY_FOR_NEXT": "helper scan",
            "CONVERGED": "done",
        }.get(str(status), "inspect iteration state")
    return {
        "target_id": target_id,
        "path_run_id": run_id,
        "target_gap_eV": config.helper.target_gap_eV,
        "iterations": rows,
        "next_action": next_action,
    }
