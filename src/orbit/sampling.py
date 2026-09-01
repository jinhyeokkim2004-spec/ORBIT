"""Deterministic full-cell one-atom displacement sampling."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from .config import ProjectConfig
from .periodic import (
    fractional_to_cartesian,
    minimum_image_displacement,
    minimum_image_distance,
    wrap_fractional,
)
from .project import ProjectLayout
from .structure import SiteRecord, load_site_records, validate_manifest_source
from .symmetry import (
    SymmetryOperations,
    structure_symmetry_operations,
    target_site_stabilizer,
)


SAMPLING_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SamplePoint:
    sample_id: int
    sample_name: str
    raw_grid_index: int
    frac_x: float
    frac_y: float
    frac_z: float
    cart_x_A: float
    cart_y_A: float
    cart_z_A: float
    displacement_x_A: float
    displacement_y_A: float
    displacement_z_A: float
    displacement_norm_A: float
    orbit_size: int
    point_stabilizer_order: int
    is_equilibrium: bool
    geometry_status: str
    invalid_reason: str
    minimum_target_distance_A: float


@dataclass(frozen=True)
class SamplingPlan:
    target: SiteRecord
    target_index_zero_based: int
    requested_divisions: tuple[int, int, int]
    divisions: tuple[int, int, int]
    raw_grid_count: int
    full_symmetry_order: int
    site_stabilizer_order: int
    spacing_angstrom: float
    fold_by_symmetry: bool
    overlap_tolerance_angstrom: float
    points: tuple[SamplePoint, ...]

    @property
    def representative_count(self) -> int:
        return len(self.points)

    @property
    def reduction_factor(self) -> float:
        return self.raw_grid_count / max(1, self.representative_count)

    @property
    def valid_count(self) -> int:
        return sum(point.geometry_status == "VALID" for point in self.points)

    @property
    def invalid_count(self) -> int:
        return self.representative_count - self.valid_count


SAMPLE_FIELDS = (
    "sample_id",
    "sample_name",
    "target_id",
    "target_index_zero_based",
    "element",
    "raw_grid_index",
    "frac_x",
    "frac_y",
    "frac_z",
    "cart_x_A",
    "cart_y_A",
    "cart_z_A",
    "displacement_x_A",
    "displacement_y_A",
    "displacement_z_A",
    "displacement_norm_A",
    "orbit_size",
    "point_stabilizer_order",
    "is_equilibrium",
    "geometry_status",
    "invalid_reason",
    "minimum_target_distance_A",
    "structure_file",
)


def grid_divisions(cell: Any, spacing_angstrom: float) -> Any:
    """Choose full-cell divisions, rounded up and made even when > 1."""
    import numpy as np

    if spacing_angstrom <= 0:
        raise ValueError("sampling spacing must be positive")
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    divisions = np.maximum(1, np.ceil(lengths / spacing_angstrom).astype(int))
    return np.asarray(
        [value + value % 2 if value > 1 else value for value in divisions],
        dtype=np.int64,
    )


def symmetry_compatible_grid_divisions(
    requested_divisions: Any,
    operations: SymmetryOperations,
) -> Any:
    """Return the smallest no-coarser grid closed under the supplied rotations.

    For fractional-grid divisions n_i, closure under a crystallographic
    rotation R requires n_j to divide R_ij*n_i for every nonzero R_ij.
    Only divisions at least as fine as the spacing-derived request are
    considered. The uniform grid with all divisions equal to the largest
    requested division guarantees a finite fallback.
    """
    import numpy as np

    requested = np.asarray(requested_divisions, dtype=np.int64)

    if requested.shape != (3,) or bool(np.any(requested <= 0)):
        raise ValueError(
            "requested grid divisions must be three positive integers"
        )

    def compatible(candidate: tuple[int, int, int]) -> bool:
        for rotation in operations.rotations:
            for i in range(3):
                for j in range(3):
                    coefficient = abs(int(rotation[i, j]))
                    if coefficient == 0:
                        continue
                    if (
                        coefficient * int(candidate[i])
                    ) % int(candidate[j]) != 0:
                        return False
        return True

    requested_tuple = tuple(int(value) for value in requested)
    if compatible(requested_tuple):
        return requested.copy()

    common = int(max(requested))

    def axis_candidates(lower: int) -> tuple[int, ...]:
        if common == 1:
            return (1,)
        values: list[int] = []
        if lower == 1:
            values.append(1)
            lower = 2
        if lower % 2:
            lower += 1
        values.extend(range(lower, common + 1, 2))
        return tuple(values)

    axes = [axis_candidates(int(value)) for value in requested]

    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int, tuple[int, int, int]] | None = None

    for candidate in itertools.product(*axes):
        candidate = tuple(int(value) for value in candidate)
        if not compatible(candidate):
            continue

        key = (
            int(np.prod(candidate)),
            sum(
                candidate[axis] - int(requested[axis])
                for axis in range(3)
            ),
            candidate,
        )

        if best_key is None or key < best_key:
            best = candidate
            best_key = key

    if best is None:
        raise RuntimeError(
            "Could not construct a symmetry-compatible sampling grid from "
            f"requested divisions {requested.tolist()}"
        )

    return np.asarray(best, dtype=np.int64)


def build_anchor_grid(anchor: Any, divisions: Any) -> Any:
    import numpy as np

    anchor = wrap_fractional(anchor)
    divisions = np.asarray(divisions, dtype=np.int64)
    if divisions.shape != (3,) or bool(np.any(divisions <= 0)):
        raise ValueError("grid divisions must be three positive integers")
    axes = [
        np.mod(anchor[axis] + np.arange(divisions[axis]) / divisions[axis], 1.0)
        for axis in range(3)
    ]
    return np.asarray(list(itertools.product(*axes)), dtype=float)


def _fold_grid(
    grid: Any,
    divisions: Any,
    operations: SymmetryOperations,
    tolerance_angstrom: float,
    cell: Any,
    anchor: Any,
) -> tuple[list[int], list[int], list[int]]:
    """Return representative raw indices, orbit sizes, and point stabilizers."""
    import numpy as np

    grid = np.asarray(grid, dtype=float)
    divisions = np.asarray(divisions, dtype=np.int64)
    cell = np.asarray(cell, dtype=float)
    anchor = wrap_fractional(anchor)

    def mesh_key(point: Any, operation_number: int | None = None) -> tuple[int, int, int]:
        point = wrap_fractional(point)
        scaled = np.mod(point - anchor, 1.0) * divisions
        nearest = np.rint(scaled).astype(np.int64)
        key_array = nearest % divisions
        reconstructed = wrap_fractional(anchor + key_array / divisions)
        residual = minimum_image_distance(point, reconstructed, cell)
        if residual > tolerance_angstrom:
            operation_text = (
                "" if operation_number is None
                else f" under site-stabilizer operation {operation_number}"
            )
            raise RuntimeError(
                "Sampling grid is not closed under target-site symmetry"
                f"{operation_text}: off-grid residual={residual:.6e} A, "
                f"divisions={divisions.tolist()}"
            )
        return tuple(int(value) for value in key_array)

    index_of: dict[tuple[int, int, int], int] = {}
    for index, point in enumerate(grid):
        key = mesh_key(point)
        if key in index_of:
            raise RuntimeError(f"Duplicate sampling-grid key: {key}")
        index_of[key] = index

    assigned = -np.ones(len(grid), dtype=np.int64)
    representative_indices: list[int] = []
    orbit_sizes: list[int] = []
    point_stabilizers: list[int] = []
    group_order = len(operations)
    for index, point in enumerate(grid):
        if assigned[index] >= 0:
            continue
        point_key = mesh_key(point)
        images: set[int] = set()
        point_stabilizer_order = 0
        for operation_number, (rotation, translation) in enumerate(
            zip(operations.rotations, operations.translations)
        ):
            image_key = mesh_key(
                rotation @ point + translation,
                operation_number=operation_number,
            )
            if image_key == point_key:
                point_stabilizer_order += 1
            if image_key not in index_of:
                raise RuntimeError(f"Symmetry image grid key is absent: {image_key}")
            images.add(index_of[image_key])
        images.add(index)
        if any(assigned[image] >= 0 for image in images):
            raise RuntimeError("A new sampling orbit overlaps an assigned orbit")
        if len(images) * point_stabilizer_order != group_order:
            raise RuntimeError(
                "Orbit-stabilizer consistency failed: "
                f"orbit={len(images)}, point_stabilizer={point_stabilizer_order}, "
                f"site_stabilizer={group_order}"
            )
        orbit_number = len(representative_indices)
        for image in images:
            assigned[image] = orbit_number
        representative_indices.append(index)
        orbit_sizes.append(len(images))
        point_stabilizers.append(point_stabilizer_order)
    if bool(np.any(assigned < 0)):
        raise RuntimeError("At least one sampling point was not assigned to an orbit")
    return representative_indices, orbit_sizes, point_stabilizers


def _load_atoms(config: ProjectConfig):
    try:
        import numpy as np
        from ase.data import atomic_numbers
        from ase.io import read as ase_read
    except ImportError as exc:
        raise RuntimeError("Sampling requires numpy, ASE, and spglib") from exc
    atoms = ase_read(config.structure)
    cell = np.asarray(atoms.cell.array, dtype=float)
    frac = wrap_fractional(atoms.get_scaled_positions(wrap=False))
    numbers = [atomic_numbers[symbol] for symbol in atoms.get_chemical_symbols()]
    return atoms, cell, frac, numbers


def build_sampling_plans(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested_targets: list[str] | None = None,
) -> tuple[Any, list[SamplingPlan]]:
    import numpy as np

    validate_manifest_source(layout.structure_manifest, config.structure)
    sites = load_site_records(layout.sites_manifest)
    by_id = {site.site_id: site for site in sites}
    configured = list(config.target_site_ids)
    if not configured:
        raise ValueError("No targets are configured; run orbit targets set first")
    selected = configured if requested_targets is None else requested_targets
    if len(selected) != len(set(selected)):
        raise ValueError("Requested sampling targets contain duplicates")
    unconfigured = [site_id for site_id in selected if site_id not in configured]
    if unconfigured:
        raise ValueError(
            "Requested targets are not configured: " + ", ".join(unconfigured)
        )
    missing = [site_id for site_id in selected if site_id not in by_id]
    if missing:
        raise ValueError("Target IDs are absent from sites.csv: " + ", ".join(missing))

    atoms, cell, equilibrium_frac, numbers = _load_atoms(config)
    operations = structure_symmetry_operations(
        cell, equilibrium_frac, numbers, config.symprec_angstrom
    )
    requested_divisions = grid_divisions(
        cell,
        config.sampling_spacing_angstrom,
    )
    plans: list[SamplingPlan] = []
    for site_id in selected:
        target = by_id[site_id]
        target_index = target.ase_index_zero_based
        if target_index >= len(atoms):
            raise ValueError(f"Stored ASE index is invalid for target {site_id}")
        target_frac = equilibrium_frac[target_index]
        stabilizer = target_site_stabilizer(
            operations, target_frac, cell, config.symprec_angstrom
        )
        divisions = np.asarray(requested_divisions, dtype=np.int64)
        if config.fold_by_symmetry:
            divisions = symmetry_compatible_grid_divisions(
                requested_divisions,
                stabilizer,
            )
        grid = build_anchor_grid(target_frac, divisions)
        if config.fold_by_symmetry:
            indices, orbit_sizes, point_stabilizers = _fold_grid(
                grid,
                divisions,
                stabilizer,
                config.symprec_angstrom,
                cell,
                target_frac,
            )
        else:
            indices = list(range(len(grid)))
            orbit_sizes = [1] * len(grid)
            point_stabilizers = [1] * len(grid)
        if sum(orbit_sizes) != len(grid):
            raise RuntimeError(
                f"Target {site_id} orbit sizes sum to {sum(orbit_sizes)}, "
                f"but the raw grid contains {len(grid)} points"
            )

        points: list[SamplePoint] = []
        equilibrium_samples = 0
        for sample_id, (raw_index, orbit_size, point_stabilizer) in enumerate(
            zip(indices, orbit_sizes, point_stabilizers)
        ):
            representative = grid[raw_index]
            cart = fractional_to_cartesian(representative, cell)
            displacement = minimum_image_displacement(
                representative, target_frac, cell
            )
            norm = float(np.linalg.norm(displacement))
            is_equilibrium = norm <= 1.0e-10
            equilibrium_samples += int(is_equilibrium)
            other_indices = [
                index for index in range(len(atoms)) if index != target_index
            ]
            minimum_target_distance = (
                min(
                    minimum_image_distance(
                        representative, equilibrium_frac[index], cell
                    )
                    for index in other_indices
                )
                if other_indices else float("inf")
            )
            invalid = (
                minimum_target_distance
                <= config.sampling_overlap_tolerance_angstrom
            )
            geometry_status = "DIRECT_OVERLAP" if invalid else "VALID"
            invalid_reason = (
                "target-to-atom minimum periodic distance "
                f"{minimum_target_distance:.12g} A is at or below sampling "
                f"overlap tolerance {config.sampling_overlap_tolerance_angstrom:.12g} A"
                if invalid else ""
            )
            points.append(
                SamplePoint(
                    sample_id=sample_id,
                    sample_name=f"sample_{sample_id:05d}",
                    raw_grid_index=raw_index,
                    frac_x=float(representative[0]),
                    frac_y=float(representative[1]),
                    frac_z=float(representative[2]),
                    cart_x_A=float(cart[0]),
                    cart_y_A=float(cart[1]),
                    cart_z_A=float(cart[2]),
                    displacement_x_A=float(displacement[0]),
                    displacement_y_A=float(displacement[1]),
                    displacement_z_A=float(displacement[2]),
                    displacement_norm_A=norm,
                    orbit_size=orbit_size,
                    point_stabilizer_order=point_stabilizer,
                    is_equilibrium=is_equilibrium,
                    geometry_status=geometry_status,
                    invalid_reason=invalid_reason,
                    minimum_target_distance_A=minimum_target_distance,
                )
            )
        if equilibrium_samples != 1:
            raise RuntimeError(
                f"Target {site_id} has {equilibrium_samples} equilibrium representatives; expected one"
            )
        plans.append(
            SamplingPlan(
                target=target,
                target_index_zero_based=target_index,
                requested_divisions=tuple(
                    int(value) for value in requested_divisions
                ),
                divisions=tuple(int(value) for value in divisions),
                raw_grid_count=len(grid),
                full_symmetry_order=len(operations),
                site_stabilizer_order=len(stabilizer),
                spacing_angstrom=config.sampling_spacing_angstrom,
                fold_by_symmetry=config.fold_by_symmetry,
                overlap_tolerance_angstrom=(
                    config.sampling_overlap_tolerance_angstrom
                ),
                points=tuple(points),
            )
        )
    return atoms, plans


def _sampling_signature(config: ProjectConfig, plan: SamplingPlan, structure_fingerprint: str) -> str:
    payload = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "structure_fingerprint": structure_fingerprint,
        "target_id": plan.target.site_id,
        "spacing_angstrom": plan.spacing_angstrom,
        "fold_by_symmetry": plan.fold_by_symmetry,
        "symprec_angstrom": config.symprec_angstrom,
        "overlap_tolerance_angstrom": plan.overlap_tolerance_angstrom,
        "divisions": plan.divisions,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sampling_output_state(
    config: ProjectConfig,
    layout: ProjectLayout,
    plan: SamplingPlan,
) -> str:
    """Return `missing`, `identical`, or `conflict` for a target output."""
    structure_payload = json.loads(
        layout.structure_manifest.read_text(encoding="utf-8")
    )
    signature = _sampling_signature(
        config, plan, str(structure_payload["structure_fingerprint"])
    )
    output = layout.samples / plan.target.site_id
    manifest_path = output / "sampling.json"
    if not output.exists():
        return "missing"
    if not output.is_dir():
        return "conflict"
    if not any(output.iterdir()):
        return "missing"
    if not manifest_path.is_file():
        return "conflict"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    if old.get("sampling_signature") != signature:
        return "conflict"
    expected = int(old.get("valid_geometry_count", -1))
    structures = output / "structures"
    present = (
        len(list(structures.glob("sample_*.cif"))) if structures.is_dir() else 0
    )
    return (
        "identical"
        if expected == plan.valid_count == present
        else "conflict"
    )


def write_sampling_plan(
    config: ProjectConfig,
    layout: ProjectLayout,
    atoms: Any,
    plan: SamplingPlan,
    *,
    overwrite: bool = False,
) -> tuple[Path, str]:
    import numpy as np
    from ase.io import write as ase_write

    structure_payload = json.loads(layout.structure_manifest.read_text(encoding="utf-8"))
    fingerprint = str(structure_payload["structure_fingerprint"])
    signature = _sampling_signature(config, plan, fingerprint)
    output = layout.samples / plan.target.site_id
    state = sampling_output_state(config, layout, plan)
    if state == "identical" and not overwrite:
        return output, "skipped_identical"
    if state == "conflict" and not overwrite:
        raise FileExistsError(
            f"Sampling output conflicts for {plan.target.site_id}: {output}; "
            "use --overwrite to replace it"
        )

    staging = layout.samples / f".{plan.target.site_id}.staging-{uuid4().hex}"
    structures_dir = staging / "structures"
    structures_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    base_frac = wrap_fractional(atoms.get_scaled_positions(wrap=False))
    for point in plan.points:
        relative_structure = ""
        if point.geometry_status == "VALID":
            configuration = np.array(base_frac, copy=True)
            configuration[plan.target_index_zero_based] = [
                point.frac_x,
                point.frac_y,
                point.frac_z,
            ]
            sample_atoms = atoms.copy()
            sample_atoms.set_scaled_positions(wrap_fractional(configuration))
            relative_structure = (
                Path("structures") / f"{point.sample_name}.cif"
            ).as_posix()
            try:
                ase_write(
                    structures_dir / f"{point.sample_name}.cif",
                    sample_atoms,
                    format="cif",
                )
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        row = asdict(point)
        row.update(
            {
                "target_id": plan.target.site_id,
                "target_index_zero_based": plan.target_index_zero_based,
                "element": plan.target.element,
                "is_equilibrium": int(point.is_equilibrium),
                "structure_file": relative_structure,
            }
        )
        rows.append(row)

    samples_path = staging / "samples.csv"
    try:
        with samples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in SAMPLE_FIELDS} for row in rows
            )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "sampling_signature": signature,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "target_id": plan.target.site_id,
        "target_index_zero_based": plan.target_index_zero_based,
        "element": plan.target.element,
        "structure_fingerprint": fingerprint,
        "source_sha256": structure_payload["source_sha256"],
        "spacing_angstrom": plan.spacing_angstrom,
        "symprec_angstrom": config.symprec_angstrom,
        "fold_by_symmetry": plan.fold_by_symmetry,
        "geometry_filter": "direct_overlap_veto",
        "overlap_tolerance_angstrom": plan.overlap_tolerance_angstrom,
        "requested_grid_divisions": plan.requested_divisions,
        "grid_divisions": plan.divisions,
        "grid_divisions_adjusted_for_symmetry": (
            plan.divisions != plan.requested_divisions
        ),
        "grid_division_policy": (
            "smallest_no_coarser_site_stabilizer_compatible_even_grid"
        ),
        "raw_grid_count": plan.raw_grid_count,
        "representative_count": plan.representative_count,
        "valid_geometry_count": plan.valid_count,
        "direct_overlap_count": plan.invalid_count,
        "reduction_factor": plan.reduction_factor,
        "full_symmetry_order": plan.full_symmetry_order,
        "site_stabilizer_order": plan.site_stabilizer_order,
        "equilibrium_sample_id": next(
            point.sample_id for point in plan.points if point.is_equilibrium
        ),
        "samples_file": "samples.csv",
        "structures_directory": "structures",
    }
    try:
        (staging / "sampling.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    backup = None
    try:
        if output.exists():
            backup = layout.samples / f".{plan.target.site_id}.backup-{uuid4().hex}"
            output.replace(backup)
        staging.replace(output)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if not output.exists() and backup is not None and backup.exists():
            backup.replace(output)
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output, "written"
