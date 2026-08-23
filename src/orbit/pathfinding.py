"""Exact-grid widest paths through symmetry-unfolded sampled gaps."""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .gaps import selected_targets
from .periodic import wrap_fractional
from .project import ProjectLayout
from .records.pathfinding import CandidateRecord, PathRunRecorder
from .sampling import SamplingPlan, build_sampling_plans
from .symmetry import structure_symmetry_operations, target_site_stabilizer


PATH_SCHEMA_VERSION = 1
DESTINATIONS = (
    ("a", (1, 0, 0)),
    ("b", (0, 1, 0)),
    ("c", (0, 0, 1)),
    ("ab", (1, 1, 0)),
    ("bc", (0, 1, 1)),
    ("ac", (1, 0, 1)),
    ("abc", (1, 1, 1)),
)


@dataclass(frozen=True)
class PathResult:
    target_id: str
    run_id: str
    run_directory: Path
    status: str
    selected_destination: str | None
    theta_star_eV: float | None


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


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_key(frac: Any, anchor: Any, shape: Any, cell: Any, tolerance_A: float):
    import numpy as np

    frac = wrap_fractional(frac)
    anchor = wrap_fractional(anchor)
    shape = np.asarray(shape, dtype=int)
    scaled = wrap_fractional(frac - anchor) * shape
    nearest = np.rint(scaled).astype(int)
    key = nearest % shape
    rebuilt = wrap_fractional(anchor + key / shape)
    delta = (frac - rebuilt + 0.5) % 1.0 - 0.5
    residual = float(np.linalg.norm(delta @ cell))
    if residual > tolerance_A:
        raise ValueError(
            f"Point {frac.tolist()} is off the authoritative sample grid by "
            f"{residual:.6e} A"
        )
    return tuple(int(value) for value in key)


def _build_field(config: ProjectConfig, layout: ProjectLayout, atoms: Any, plan: SamplingPlan):
    import numpy as np
    from ase.data import atomic_numbers

    gap_root = config.root / "results" / "gaps" / plan.target.site_id
    gap_path = gap_root / "gaps.csv"
    extraction_path = gap_root / "extraction.json"
    extraction = _read_json(extraction_path, "Gap extraction manifest")
    sampling_path = layout.samples / plan.target.site_id / "sampling.json"
    sampling = _read_json(sampling_path, "Sampling manifest")
    if extraction.get("target_id") != plan.target.site_id:
        raise ValueError("Gap extraction target does not match requested target")
    if extraction.get("sampling_signature") != sampling.get("sampling_signature"):
        raise ValueError(
            f"Gap extraction for {plan.target.site_id} does not match the current "
            "sampling dataset; rerun orbit scf and orbit extract"
        )
    if extraction.get("gaps_file_sha256") != _sha256(gap_path):
        raise ValueError(
            f"Extracted gap table changed after extraction: {gap_path}; "
            "rerun orbit extract"
        )

    rows = _read_csv(gap_path, "Extracted gap table")
    by_name = {row.get("sample_name", ""): row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError(f"Duplicate sample names in {gap_path}")
    expected = {point.sample_name for point in plan.points}
    if set(by_name) != expected:
        raise ValueError("Extracted gap rows do not match the authoritative sampling plan")

    cell = np.asarray(atoms.cell.array, dtype=float)
    all_frac = wrap_fractional(atoms.get_scaled_positions(wrap=False))
    numbers = [atomic_numbers[symbol] for symbol in atoms.get_chemical_symbols()]
    operations = structure_symmetry_operations(
        cell, all_frac, numbers, config.symprec_angstrom
    )
    anchor = all_frac[plan.target_index_zero_based]
    stabilizer = target_site_stabilizer(
        operations, anchor, cell, config.symprec_angstrom
    )
    if len(stabilizer) != plan.site_stabilizer_order:
        raise ValueError("Current site stabilizer differs from the sampling manifest")

    shape = np.asarray(plan.divisions, dtype=int)
    field = np.full(tuple(shape), -np.inf, dtype=float)
    live = np.zeros(tuple(shape), dtype=bool)
    source = np.full(tuple(shape), -1, dtype=int)
    geometry = np.zeros(tuple(shape), dtype=bool)
    accepted_representatives = 0
    for representative_index, point in enumerate(plan.points):
        row = by_name[point.sample_name]
        status = row.get("output_status", "")
        gap = None
        if status == "COMPLETE" and point.geometry_status == "VALID":
            try:
                gap = float(row["gap_ev"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"COMPLETE row {point.sample_name} lacks a numeric gap"
                ) from exc
            if not np.isfinite(gap):
                raise ValueError(f"COMPLETE row {point.sample_name} has nonfinite gap")
            accepted_representatives += 1

        representative = np.asarray([point.frac_x, point.frac_y, point.frac_z])
        image_keys: set[tuple[int, int, int]] = set()
        for rotation, translation in zip(stabilizer.rotations, stabilizer.translations):
            image = wrap_fractional(rotation @ representative + translation)
            key = _grid_key(image, anchor, shape, cell, config.symprec_angstrom)
            image_keys.add(key)
        if len(image_keys) != point.orbit_size:
            raise ValueError(
                f"Orbit size mismatch for {point.sample_name}: manifest="
                f"{point.orbit_size}, unfolded={len(image_keys)}"
            )
        for key in image_keys:
            geometry[key] = True
            if gap is None:
                continue
            if live[key] and abs(float(field[key]) - gap) > 1.0e-9:
                raise ValueError(f"Conflicting gaps unfold onto grid node {key}")
            field[key] = gap
            live[key] = True
            source[key] = representative_index

    expected_nodes = int(np.prod(shape))
    if int(geometry.sum()) != expected_nodes:
        missing = np.argwhere(~geometry)
        preview = [tuple(int(value) for value in row) for row in missing[:10]]
        raise ValueError(
            f"Symmetry unfolding covers {int(geometry.sum())}/{expected_nodes} "
            f"grid nodes; first missing keys: {preview}"
        )
    entry = _grid_key(anchor, anchor, shape, cell, config.symprec_angstrom)
    return {
        "field": field,
        "live": live,
        "source": source,
        "shape": shape,
        "cell": cell,
        "anchor": anchor,
        "entry": entry,
        "accepted_representatives": accepted_representatives,
        "representative_count": len(plan.points),
        "site_stabilizer_order": len(stabilizer),
        "gaps_csv": gap_path,
        "extraction_json": extraction_path,
        "sampling_json": sampling_path,
        "extraction": extraction,
    }


def lifted_neighbors(state: Any, shape: Any, winding_lo: Any, winding_hi: Any):
    node, winding = state
    for axis in range(3):
        for sign in (-1, 1):
            next_node = list(node)
            next_winding = list(winding)
            raw = next_node[axis] + sign
            if raw < 0:
                next_node[axis] = int(shape[axis] - 1)
                next_winding[axis] -= 1
            elif raw >= shape[axis]:
                next_node[axis] = 0
                next_winding[axis] += 1
            else:
                next_node[axis] = raw
            if winding_lo[axis] <= next_winding[axis] <= winding_hi[axis]:
                yield tuple(next_node), tuple(next_winding)


def widest_winding_path(
    field: Any,
    live: Any,
    shape: Any,
    entry: Any,
    goal_winding: Any,
    winding_padding: int,
):
    """Maximize minimum node gap, then minimize edge count at that threshold."""
    import numpy as np

    start = (tuple(entry), (0, 0, 0))
    goal = (tuple(entry), tuple(int(value) for value in goal_winding))
    if not live[start[0]]:
        return None, None
    winding_lo = tuple(min(0, goal[1][axis]) - winding_padding for axis in range(3))
    winding_hi = tuple(max(0, goal[1][axis]) + winding_padding for axis in range(3))
    start_score = float(field[start[0]])
    best = {start: start_score}
    heap = [(-start_score, start)]
    while heap:
        negative_score, state = heapq.heappop(heap)
        score = -negative_score
        if score < best.get(state, -np.inf):
            continue
        if state == goal:
            break
        for neighbor in lifted_neighbors(state, shape, winding_lo, winding_hi):
            node = neighbor[0]
            if not live[node]:
                continue
            candidate = min(score, float(field[node]))
            if candidate > best.get(neighbor, -np.inf):
                best[neighbor] = candidate
                heapq.heappush(heap, (-candidate, neighbor))
    if goal not in best:
        return None, None

    threshold = float(best[goal])
    previous = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == goal:
            break
        for neighbor in lifted_neighbors(state, shape, winding_lo, winding_hi):
            if neighbor in previous:
                continue
            node = neighbor[0]
            if live[node] and float(field[node]) >= threshold:
                previous[neighbor] = state
                queue.append(neighbor)
    if goal not in previous:
        raise RuntimeError("Optimal bottleneck found, but route recovery failed")
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = previous[state]
    path.reverse()
    return threshold, path


def _lifted_frac(state: Any, shape: Any, anchor: Any):
    import numpy as np

    node, winding = state
    return (
        np.asarray(anchor, dtype=float)
        + np.asarray(node, dtype=float) / np.asarray(shape, dtype=float)
        + np.asarray(winding, dtype=float)
    )


def _resample_path(states: Any, field: Any, shape: Any, anchor: Any, cell: Any, n_images: int):
    import numpy as np

    vertices = np.asarray([_lifted_frac(state, shape, anchor) for state in states])
    gaps = np.asarray([float(field[state[0]]) for state in states])
    lengths = np.linalg.norm(np.diff(vertices, axis=0) @ cell, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("Selected winding path has zero Cartesian length")
    distances = np.linspace(0.0, total, n_images)
    image_frac = np.empty((n_images, 3), dtype=float)
    image_gap = np.empty(n_images, dtype=float)
    for image, distance in enumerate(distances):
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = min(max(segment, 0), len(lengths) - 1)
        fraction = 0.0 if lengths[segment] == 0 else (
            (distance - cumulative[segment]) / lengths[segment]
        )
        image_frac[image] = (
            (1.0 - fraction) * vertices[segment]
            + fraction * vertices[segment + 1]
        )
        image_gap[image] = min(gaps[segment], gaps[segment + 1])
    image_frac[0] = vertices[0]
    image_frac[-1] = vertices[-1]
    image_gap[0] = gaps[0]
    image_gap[-1] = gaps[-1]
    return image_frac, image_gap, distances, vertices, gaps, total


PATH_FIELDS = (
    "target_id", "target_index_zero_based", "destination_id", "image_name",
    "image", "path_fraction", "wrapped_frac_x", "wrapped_frac_y", "wrapped_frac_z",
    "unwrapped_frac_x", "unwrapped_frac_y", "unwrapped_frac_z", "winding_a",
    "winding_b", "winding_c", "unwrapped_cart_x_A", "unwrapped_cart_y_A",
    "unwrapped_cart_z_A", "arc_length_A", "gap_proxy_eV", "gap_proxy_kind",
    "is_nearest_bottleneck_image", "structure_file",
)


def _write_selected_path(
    recorder: PathRunRecorder,
    config: ProjectConfig,
    atoms: Any,
    plan: SamplingPlan,
    field_data: dict[str, Any],
    destination_id: str,
    goal_winding: Any,
    theta_star: float,
    states: Any,
    n_images: int,
    write_cifs: bool,
) -> None:
    import numpy as np
    from ase.io import write as ase_write

    field = field_data["field"]
    shape = field_data["shape"]
    anchor = field_data["anchor"]
    cell = field_data["cell"]
    image_frac, image_gap, distances, raw_frac, raw_gap, total = _resample_path(
        states, field, shape, anchor, cell, n_images
    )
    minimum = float(raw_gap.min())
    tied = np.flatnonzero(raw_gap <= minimum + 1.0e-12)
    bottleneck_raw_index = int(tied[len(tied) // 2])
    bottleneck_frac = raw_frac[bottleneck_raw_index]
    bottleneck_image = int(
        np.linalg.norm((image_frac - bottleneck_frac) @ cell, axis=1).argmin()
    )

    rows = []
    base_frac = wrap_fractional(atoms.get_scaled_positions(wrap=False))
    for image in range(n_images):
        unwrapped = image_frac[image]
        wrapped = wrap_fractional(unwrapped)
        cart = unwrapped @ cell
        winding = np.floor(unwrapped - anchor + 1.0e-12).astype(int)
        structure_file = f"cifs/image_{image:03d}.cif" if write_cifs else ""
        rows.append({
            "target_id": plan.target.site_id,
            "target_index_zero_based": plan.target_index_zero_based,
            "destination_id": destination_id,
            "image_name": f"image_{image:03d}",
            "image": image,
            "path_fraction": image / (n_images - 1),
            "wrapped_frac_x": wrapped[0], "wrapped_frac_y": wrapped[1], "wrapped_frac_z": wrapped[2],
            "unwrapped_frac_x": unwrapped[0], "unwrapped_frac_y": unwrapped[1], "unwrapped_frac_z": unwrapped[2],
            "winding_a": winding[0], "winding_b": winding[1], "winding_c": winding[2],
            "unwrapped_cart_x_A": cart[0], "unwrapped_cart_y_A": cart[1], "unwrapped_cart_z_A": cart[2],
            "arc_length_A": distances[image],
            "gap_proxy_eV": image_gap[image],
            "gap_proxy_kind": "conservative_adjacent_exact_node_min",
            "is_nearest_bottleneck_image": int(image == bottleneck_image),
            "structure_file": structure_file,
        })
        if write_cifs:
            image_atoms = atoms.copy()
            positions = np.array(base_frac, copy=True)
            positions[plan.target_index_zero_based] = wrapped
            image_atoms.set_scaled_positions(positions)
            ase_write(recorder.run_dir / structure_file, image_atoms, format="cif")

    with (recorder.run_dir / "path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    raw_sources = [int(field_data["source"][state[0]]) for state in states]
    payload = {
        "schema_version": PATH_SCHEMA_VERSION,
        "target_id": plan.target.site_id,
        "target_index_zero_based": plan.target_index_zero_based,
        "run_id": recorder.run_id,
        "destination_id": destination_id,
        "requested_displacement_frac": list(goal_winding),
        "theta_star_eV": theta_star,
        "theta_min_eV": field_data["theta_min_eV"],
        "feasible": theta_star > field_data["theta_min_eV"],
        "n_images": n_images,
        "total_arc_length_A": total,
        "raw_exact_node_count": len(states),
        "raw_path_node": [list(state[0]) for state in states],
        "raw_path_winding": [list(state[1]) for state in states],
        "raw_path_frac_unwrapped": raw_frac.tolist(),
        "raw_path_gap_eV": raw_gap.tolist(),
        "raw_path_source_representative_index": raw_sources,
        "bottleneck_raw_index": bottleneck_raw_index,
        "bottleneck_gap_eV": minimum,
        "bottleneck_frac_unwrapped": bottleneck_frac.tolist(),
        "bottleneck_frac_wrapped": wrap_fractional(bottleneck_frac).tolist(),
        "tied_bottleneck_raw_indices": [int(value) for value in tied],
        "nearest_bottleneck_image": bottleneck_image,
        "path_file": "path.csv",
        "cifs_directory": "cifs" if write_cifs else None,
        "gap_proxy_rule": "minimum of adjacent exact sampled nodes",
    }
    (recorder.run_dir / "path.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )



def straight_path_images(anchor: Any, goal_winding: Any, cell: Any, n_images: int):
    """Return a straight unwrapped fractional path and cumulative distance."""
    import numpy as np

    if n_images < 2:
        raise ValueError("n_images must be at least 2")
    anchor = np.asarray(anchor, dtype=float)
    winding = np.asarray(goal_winding, dtype=float)
    cell = np.asarray(cell, dtype=float)
    fractions = np.linspace(0.0, 1.0, n_images)
    image_frac = anchor[None, :] + fractions[:, None] * winding[None, :]
    cart = image_frac @ cell
    lengths = np.linalg.norm(np.diff(cart, axis=0), axis=1)
    distances = np.concatenate([[0.0], np.cumsum(lengths)])
    return image_frac, distances


def _write_straight_path(
    recorder: PathRunRecorder,
    config: ProjectConfig,
    atoms: Any,
    plan: SamplingPlan,
    destination_id: str,
    goal_winding: Any,
    n_images: int,
    write_cifs: bool,
) -> None:
    import numpy as np
    from ase.io import write as ase_write

    base_frac = wrap_fractional(atoms.get_scaled_positions(wrap=False))
    anchor = np.asarray(base_frac[plan.target_index_zero_based], dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    image_frac, distances = straight_path_images(anchor, goal_winding, cell, n_images)
    total = float(distances[-1])

    rows = []
    for image in range(n_images):
        unwrapped = image_frac[image]
        wrapped = wrap_fractional(unwrapped)
        cart = unwrapped @ cell
        winding = np.floor(unwrapped - anchor + 1.0e-12).astype(int)
        structure_file = f"cifs/image_{image:03d}.cif" if write_cifs else ""
        rows.append({
            "target_id": plan.target.site_id,
            "target_index_zero_based": plan.target_index_zero_based,
            "destination_id": destination_id,
            "image_name": f"image_{image:03d}",
            "image": image,
            "path_fraction": image / (n_images - 1),
            "wrapped_frac_x": wrapped[0],
            "wrapped_frac_y": wrapped[1],
            "wrapped_frac_z": wrapped[2],
            "unwrapped_frac_x": unwrapped[0],
            "unwrapped_frac_y": unwrapped[1],
            "unwrapped_frac_z": unwrapped[2],
            "winding_a": winding[0],
            "winding_b": winding[1],
            "winding_c": winding[2],
            "unwrapped_cart_x_A": cart[0],
            "unwrapped_cart_y_A": cart[1],
            "unwrapped_cart_z_A": cart[2],
            "arc_length_A": distances[image],
            "gap_proxy_eV": float("nan"),
            "gap_proxy_kind": "unavailable_straight_path",
            "is_nearest_bottleneck_image": 0,
            "structure_file": structure_file,
        })
        if write_cifs:
            image_atoms = atoms.copy()
            positions = np.array(base_frac, copy=True)
            positions[plan.target_index_zero_based] = wrapped
            image_atoms.set_scaled_positions(positions)
            ase_write(recorder.run_dir / structure_file, image_atoms, format="cif")

    with (recorder.run_dir / "path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": PATH_SCHEMA_VERSION,
        "target_id": plan.target.site_id,
        "target_index_zero_based": plan.target_index_zero_based,
        "run_id": recorder.run_id,
        "path_mode": "straight",
        "destination_id": destination_id,
        "requested_displacement_frac": [int(v) for v in goal_winding],
        "theta_star_eV": None,
        "theta_min_eV": None,
        "feasible": None,
        "n_images": n_images,
        "total_arc_length_A": total,
        "raw_exact_node_count": None,
        "raw_path_node": None,
        "raw_path_winding": [[0, 0, 0], [int(v) for v in goal_winding]],
        "raw_path_frac_unwrapped": [image_frac[0].tolist(), image_frac[-1].tolist()],
        "raw_path_gap_eV": None,
        "raw_path_source_representative_index": None,
        "bottleneck_raw_index": None,
        "bottleneck_gap_eV": None,
        "bottleneck_frac_unwrapped": None,
        "bottleneck_frac_wrapped": None,
        "tied_bottleneck_raw_indices": [],
        "nearest_bottleneck_image": None,
        "path_file": "path.csv",
        "cifs_directory": "cifs" if write_cifs else None,
        "gap_proxy_rule": "unavailable_for_prescribed_straight_path",
    }
    (recorder.run_dir / "path.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def construct_straight_target_path(
    config: ProjectConfig,
    layout: ProjectLayout,
    target_id: str,
    *,
    destination_id: str,
    n_images: int,
    write_cifs: bool = True,
) -> PathResult:
    destination_map = dict(DESTINATIONS)
    if destination_id not in destination_map:
        raise ValueError(
            f"Unknown destination {destination_id!r}; choose one of {', '.join(destination_map)}"
        )
    if n_images < 2:
        raise ValueError("n_images must be at least 2")

    atoms, plans = build_sampling_plans(config, layout, [target_id])
    plan = plans[0]
    winding = destination_map[destination_id]
    inputs = {
        "structure": str(config.structure),
        "sampling_manifest": str(layout.samples / plan.target.site_id / "sampling.json"),
        "prescribed_path": "straight_fractional_translation",
    }
    parameters = {
        "path_mode": "straight",
        "destination": destination_id,
        "requested_displacement_frac": list(winding),
        "n_images": n_images,
    }

    with PathRunRecorder(
        layout.paths,
        target_id,
        inputs=inputs,
        parameters=parameters,
        algorithm="prescribed_straight_fractional_translation",
    ) as recorder:
        recorder.emit(
            f"[straight] {target_id}: destination={destination_id}; "
            f"displacement_frac={list(winding)}; images={n_images}"
        )

        # Prescribed straight paths still need a candidate record because
        # PathRunRecorder requires selected_destination to reference one.
        recorder.add_candidate(CandidateRecord(
            candidate_order=0,
            destination_id=destination_id,
            displacement_frac_a=float(winding[0]),
            displacement_frac_b=float(winding[1]),
            displacement_frac_c=float(winding[2]),
            exit_grid_i=0,
            exit_grid_j=0,
            exit_grid_k=0,
            winding_a=int(winding[0]),
            winding_b=int(winding[1]),
            winding_c=int(winding[2]),
            route_found=True,
            theta_star_eV=None,
            theta_min_eV=0.0,
            feasibility="PRESCRIBED",
            exact_node_count=None,
            reason="user-prescribed straight periodic path; electronic feasibility not evaluated",
        ))

        _write_straight_path(
            recorder, config, atoms, plan, destination_id, winding, n_images, write_cifs
        )
        recorder.emit(
            f"[done] wrote path.csv, path.json, and {n_images if write_cifs else 0} CIFs"
        )
        recorder.finish("CONSTRUCTED", selected_destination=destination_id)
        return PathResult(
            target_id, recorder.run_id, recorder.run_dir,
            "CONSTRUCTED", destination_id, None,
        )


def construct_straight_paths(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested: list[str] | None,
    *,
    destination_id: str,
    n_images: int | None = None,
    write_cifs: bool = True,
) -> list[PathResult]:
    images = config.pathfinding.n_images if n_images is None else n_images
    if images < 2:
        raise ValueError("n_images must be at least 2")
    return [
        construct_straight_target_path(
            config, layout, target_id,
            destination_id=destination_id,
            n_images=images,
            write_cifs=write_cifs,
        )
        for target_id in selected_targets(config, requested)
    ]

def find_target_path(
    config: ProjectConfig,
    layout: ProjectLayout,
    target_id: str,
    *,
    theta_min_eV: float,
    n_images: int,
    winding_padding: int,
    write_cifs: bool = True,
) -> PathResult:
    atoms, plans = build_sampling_plans(config, layout, [target_id])
    plan = plans[0]
    field_data = _build_field(config, layout, atoms, plan)
    field_data["theta_min_eV"] = theta_min_eV
    inputs = {
        "structure": str(config.structure),
        "sampling_manifest": str(field_data["sampling_json"]),
        "gap_table": str(field_data["gaps_csv"]),
        "extraction_manifest": str(field_data["extraction_json"]),
        "sampling_signature": field_data["extraction"].get("sampling_signature"),
        "preparation_signature": field_data["extraction"].get("preparation_signature"),
        "gap_table_sha256": _sha256(field_data["gaps_csv"]),
    }
    parameters = {
        "theta_min_eV": theta_min_eV,
        "n_images": n_images,
        "winding_padding": winding_padding,
        "destinations": [name for name, _ in DESTINATIONS],
        "missing_gap_policy": "impassable_wall",
        "neighbor_connectivity": 6,
    }
    with PathRunRecorder(
        layout.paths,
        target_id,
        inputs=inputs,
        parameters=parameters,
        algorithm="exact_grid_widest_path_dijkstra_then_shortest_route",
    ) as recorder:
        shape = field_data["shape"]
        live = field_data["live"]
        entry = field_data["entry"]
        recorder.emit(
            f"[data] representatives={field_data['representative_count']}; "
            f"accepted_gaps={field_data['accepted_representatives']}"
        )
        recorder.emit(
            f"[grid] shape={shape.tolist()}; exact_nodes={int(shape.prod())}; "
            f"live_gap_nodes={int(live.sum())}; missing_gap_walls={int((~live).sum())}"
        )
        recorder.set_grid_summary(
            shape=[int(value) for value in shape],
            exact_nodes=int(shape.prod()),
            live_gap_nodes=int(live.sum()),
            missing_gap_walls=int((~live).sum()),
            representative_count=field_data["representative_count"],
            accepted_representative_count=field_data["accepted_representatives"],
            site_stabilizer_order=field_data["site_stabilizer_order"],
            entry_node=list(entry),
            entry_gap_eV=float(field_data["field"][entry]) if live[entry] else None,
        )
        if not live[entry]:
            recorder.emit(f"[entry] equilibrium node {entry} has no accepted finite gap")
            for order, (destination, winding) in enumerate(DESTINATIONS):
                recorder.add_candidate(CandidateRecord(
                    candidate_order=order,
                    destination_id=destination,
                    displacement_frac_a=float(winding[0]),
                    displacement_frac_b=float(winding[1]),
                    displacement_frac_c=float(winding[2]),
                    exit_grid_i=int(entry[0]),
                    exit_grid_j=int(entry[1]),
                    exit_grid_k=int(entry[2]),
                    winding_a=int(winding[0]),
                    winding_b=int(winding[1]),
                    winding_c=int(winding[2]),
                    route_found=False,
                    theta_star_eV=None,
                    theta_min_eV=theta_min_eV,
                    feasibility="NO_ROUTE",
                    exact_node_count=None,
                    reason="equilibrium entry is a missing-gap wall",
                ))
            recorder.finish("NO_ROUTE", message="equilibrium entry is a missing-gap wall")
            return PathResult(target_id, recorder.run_id, recorder.run_dir, "NO_ROUTE", None, None)

        results = []
        for order, (destination, winding) in enumerate(DESTINATIONS):
            threshold, states = widest_winding_path(
                field_data["field"], live, shape, entry, winding, winding_padding
            )
            found = states is not None
            feasibility = (
                "NO_ROUTE" if not found else
                "INSULATING" if float(threshold) > theta_min_eV else "CLOSES_GAP"
            )
            recorder.add_candidate(CandidateRecord(
                candidate_order=order,
                destination_id=destination,
                displacement_frac_a=float(winding[0]),
                displacement_frac_b=float(winding[1]),
                displacement_frac_c=float(winding[2]),
                exit_grid_i=int(entry[0]), exit_grid_j=int(entry[1]), exit_grid_k=int(entry[2]),
                winding_a=int(winding[0]), winding_b=int(winding[1]), winding_c=int(winding[2]),
                route_found=found,
                theta_star_eV=None if threshold is None else float(threshold),
                theta_min_eV=theta_min_eV,
                feasibility=feasibility,
                exact_node_count=None if states is None else len(states),
                reason="no connected live-node route" if states is None else "",
            ))
            results.append((destination, winding, threshold, states))

        found_results = [item for item in results if item[3] is not None]
        if not found_results:
            recorder.finish("NO_ROUTE", message="no connected winding destination")
            return PathResult(target_id, recorder.run_id, recorder.run_dir, "NO_ROUTE", None, None)
        # Stable first-destination tie behavior preserves the original workflow.
        best = max(found_results, key=lambda item: float(item[2]))
        destination, winding, threshold, states = best
        status = "SUCCEEDED" if float(threshold) > theta_min_eV else "INFEASIBLE"
        recorder.emit(
            f"[best] {target_id}: R={destination}; theta*={float(threshold):.6f} eV; "
            f"theta_min={theta_min_eV:.6f} eV -> {status}"
        )
        _write_selected_path(
            recorder, config, atoms, plan, field_data, destination, winding,
            float(threshold), states, n_images, write_cifs,
        )
        recorder.emit(f"[done] wrote path.csv, path.json, and {n_images if write_cifs else 0} CIFs")
        recorder.finish(status, selected_destination=destination)
        return PathResult(
            target_id, recorder.run_id, recorder.run_dir, status,
            destination, float(threshold),
        )


def find_paths(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested: list[str] | None,
    *,
    theta_min_eV: float | None = None,
    n_images: int | None = None,
    winding_padding: int | None = None,
    write_cifs: bool = True,
) -> list[PathResult]:
    theta = config.pathfinding.theta_min_eV if theta_min_eV is None else theta_min_eV
    images = config.pathfinding.n_images if n_images is None else n_images
    padding = config.pathfinding.winding_padding if winding_padding is None else winding_padding
    if theta < 0:
        raise ValueError("theta_min_eV must be nonnegative")
    if images < 2:
        raise ValueError("n_images must be at least 2")
    if padding < 0:
        raise ValueError("winding_padding must be nonnegative")
    return [
        find_target_path(
            config, layout, target_id, theta_min_eV=theta,
            n_images=images, winding_padding=padding, write_cifs=write_cifs,
        )
        for target_id in selected_targets(config, requested)
    ]
