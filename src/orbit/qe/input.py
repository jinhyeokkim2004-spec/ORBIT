"""Generic Quantum ESPRESSO SCF input validation and rendering."""

from __future__ import annotations

import re
from typing import Any

from ..config import ProjectConfig
from .resources import QEResources


class InvalidGeometryError(ValueError):
    """A sampled geometry is intentionally excluded from QE preparation."""


def validate_scf_geometry(
    atoms: Any,
    resources: QEResources,
    exact_overlap_tolerance_angstrom: float,
) -> float:
    import numpy as np

    if len(atoms) == 0:
        raise InvalidGeometryError("structure contains no atoms")
    if not bool(np.all(atoms.get_pbc())):
        raise InvalidGeometryError("structure is not periodic in all directions")
    cell = np.asarray(atoms.cell.array, dtype=float)
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise InvalidGeometryError("structure has a singular lattice")
    supported = set(resources.pseudopotentials)
    unsupported = sorted(set(atoms.get_chemical_symbols()) - supported)
    if unsupported:
        raise InvalidGeometryError(f"unsupported elements: {unsupported}")
    if len(atoms) == 1:
        return float("inf")
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    minimum_distance = float(distances.min())
    if minimum_distance < exact_overlap_tolerance_angstrom:
        raise InvalidGeometryError(
            f"minimum nuclear distance {minimum_distance:.6e} A is below "
            f"the exact-overlap tolerance {exact_overlap_tolerance_angstrom:.6e} A"
        )
    return minimum_distance


def _prefix(target_id: str, sample_name: str) -> str:
    raw = f"gf_{target_id}_{sample_name}"
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    return cleaned[:80]


def render_scf_input(
    atoms: Any,
    target_id: str,
    sample_name: str,
    config: ProjectConfig,
    resources: QEResources,
) -> str:
    import numpy as np
    from ase.data import atomic_masses, atomic_numbers

    validate_scf_geometry(
        atoms,
        resources,
        config.qe.exact_overlap_tolerance_angstrom,
    )
    symbols = list(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.get_positions(wrap=True), dtype=float)
    species_order = [item.element for item in resources.elements]

    hubbard = config.qe.hubbard
    if hubbard is not None:
        missing = sorted({entry.element for entry in hubbard.u} - set(species_order))
        if missing:
            raise ValueError(
                "Hubbard U is configured for elements not present in the "
                f"structure/resources: {missing}"
            )

    lines = [
        "&CONTROL",
        "    calculation      = 'scf'",
        "    tstress          = .true.",
        "    tprnfor          = .true.",
        "    outdir           = './tmp'",
        f"    prefix           = '{_prefix(target_id, sample_name)}'",
        f"    pseudo_dir       = '{resources.pseudo_dir}'",
        "    verbosity        = 'high'",
        "/",
        "",
        "&SYSTEM",
        f"    ecutwfc          = {resources.ecutwfc_Ry:.10g}",
        f"    ecutrho          = {resources.ecutrho_Ry:.10g}",
        f"    occupations      = '{config.qe.occupations}'",
        f"    ntyp             = {len(species_order)}",
        f"    nat              = {len(atoms)}",
        "    ibrav            = 0",
        f"    nbnd             = {resources.nbnd}",
        "/",
        "",
        "&ELECTRONS",
        f"    conv_thr         = {config.qe.conv_thr:.10g}",
        f"    mixing_beta      = {config.qe.mixing_beta:.10g}",
        "/",
        "",
        "ATOMIC_SPECIES",
    ]
    for element in species_order:
        mass = float(atomic_masses[atomic_numbers[element]])
        lines.append(
            f"{element:<3s} {mass:.6f} {resources.pseudopotentials[element]}"
        )
    lines.extend(["", "K_POINTS automatic"])
    lines.append(
        "  "
        + " ".join(
            str(value) for value in (*config.qe.kpoints, *config.qe.offsets)
        )
    )
    lines.extend(["", "CELL_PARAMETERS angstrom"])
    lines.extend(
        "  " + " ".join(f"{coordinate:.14f}" for coordinate in vector)
        for vector in cell
    )
    lines.extend(["", "ATOMIC_POSITIONS angstrom"])
    for symbol, position in zip(symbols, positions):
        lines.append(
            f"{symbol:<3s} "
            + " ".join(f"{coordinate:.10f}" for coordinate in position)
        )

    if hubbard is not None:
        lines.extend(["", f"HUBBARD {{{hubbard.projector}}}"])
        for entry in hubbard.u:
            lines.append(f"U {entry.element}-{entry.manifold} {entry.value_eV}")

    lines.append("")
    return "\n".join(lines)

