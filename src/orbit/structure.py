"""Generic periodic-structure inspection and stable atom identities."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any


STRUCTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SiteRecord:
    site_id: str
    ase_index_zero_based: int
    cif_label: str
    element: str
    element_occurrence: int
    frac_x: float
    frac_y: float
    frac_z: float
    cart_x_A: float
    cart_y_A: float
    cart_z_A: float
    symmetry_orbit_id: str
    orbit_representative_site_id: str
    is_orbit_representative: bool
    orbit_size: int
    wyckoff: str

    @property
    def frac(self) -> tuple[float, float, float]:
        return (self.frac_x, self.frac_y, self.frac_z)

    def csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["is_orbit_representative"] = int(self.is_orbit_representative)
        return row


SITE_FIELDS = tuple(SiteRecord.__dataclass_fields__)


@dataclass(frozen=True)
class StructureInspection:
    source_path: Path
    source_sha256: str
    structure_fingerprint: str
    formula: str
    cell_A: tuple[tuple[float, float, float], ...]
    volume_A3: float
    space_group_symbol: str
    space_group_number: int
    symmetry_operation_count: int
    symprec_angstrom: float
    sites: tuple[SiteRecord, ...]


def _dataset_value(dataset: Any, name: str) -> Any:
    if hasattr(dataset, name):
        return getattr(dataset, name)
    return dataset[name]


def _clean_token(token: str) -> str:
    return token.strip().strip("'\"")


def _parse_cif_atom_labels(path: Path) -> list[str] | None:
    """Return asymmetric-unit labels in CIF loop order when available."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for loop_index, line in enumerate(lines):
        if line.strip().lower() != "loop_":
            continue
        headers: list[str] = []
        cursor = loop_index + 1
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_"):
            headers.append(lines[cursor].split()[0].lower())
            cursor += 1
        if "_atom_site_label" not in headers:
            continue
        label_column = headers.index("_atom_site_label")
        labels: list[str] = []
        for line in lines[cursor:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if labels:
                    break
                continue
            if stripped.startswith("_") or stripped.lower() == "loop_" or stripped.lower().startswith("data_"):
                break
            try:
                tokens = shlex.split(stripped, comments=True, posix=True)
            except ValueError:
                return None
            if len(tokens) < len(headers):
                return None
            labels.append(_clean_token(tokens[label_column]))
        return labels or None
    return None


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    return cleaned or "site"


def _fractional_key(symbol: str, frac: Any) -> tuple[Any, ...]:
    wrapped = [float(value) % 1.0 for value in frac]
    wrapped = [0.0 if abs(value - 1.0) < 1.0e-10 else value for value in wrapped]
    return (symbol, *(round(value, 10) for value in wrapped))


def _assign_site_ids(symbols: list[str], frac: Any, labels: list[str | None]) -> list[str]:
    bases = [_safe_id(label) if label else symbol for symbol, label in zip(symbols, labels)]
    grouped: dict[str, list[int]] = {}
    for index, base in enumerate(bases):
        grouped.setdefault(base, []).append(index)

    proposed: dict[int, str] = {}
    for base, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: _fractional_key(symbols[index], frac[index]))
        if len(ordered) == 1:
            proposed[ordered[0]] = base
        else:
            for occurrence, index in enumerate(ordered, start=1):
                proposed[index] = f"{base}_{occurrence}"

    used: set[str] = set()
    result: list[str] = []
    for index in range(len(symbols)):
        candidate = proposed[index]
        unique = candidate
        suffix = 2
        while unique in used:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        used.add(unique)
        result.append(unique)
    return result


def inspect_structure(path: Path, symprec_angstrom: float) -> StructureInspection:
    try:
        import numpy as np
        import spglib
        from ase.data import atomic_numbers
        from ase.io import read as ase_read
    except ImportError as exc:
        raise RuntimeError(
            "Structure inspection requires numpy, ASE, and spglib. Reinstall "
            "ORBIT in the active environment with: python -m pip install -e PATH"
        ) from exc

    path = path.expanduser().resolve()
    atoms = ase_read(path)
    if len(atoms) == 0:
        raise ValueError(f"ASE read zero atoms from {path}")
    if not bool(np.all(atoms.get_pbc())):
        raise ValueError(f"Structure is not periodic in all three directions: {path}")

    cell = np.asarray(atoms.cell.array, dtype=float)
    frac = np.mod(np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float), 1.0)
    cart = frac @ cell
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError(f"Invalid or singular lattice in {path}")
    symbols = list(atoms.get_chemical_symbols())
    numbers = [atomic_numbers[symbol] for symbol in symbols]
    dataset = spglib.get_symmetry_dataset((cell, frac, numbers), symprec=symprec_angstrom)
    if dataset is None:
        raise ValueError(f"spglib could not determine symmetry for {path}")

    raw_labels = _parse_cif_atom_labels(path)
    kinds = atoms.arrays.get("spacegroup_kinds")
    labels: list[str | None]
    if raw_labels and len(raw_labels) == len(atoms):
        labels = list(raw_labels)
    elif raw_labels and kinds is not None and len(kinds) == len(atoms) and max(kinds) < len(raw_labels):
        labels = [raw_labels[int(kind)] for kind in kinds]
    else:
        labels = [None] * len(atoms)

    site_ids = _assign_site_ids(symbols, frac, labels)
    element_order = sorted(
        range(len(atoms)),
        key=lambda index: (symbols[index], site_ids[index], _fractional_key(symbols[index], frac[index])),
    )
    element_occurrences: dict[int, int] = {}
    counts: dict[str, int] = {}
    for index in element_order:
        counts[symbols[index]] = counts.get(symbols[index], 0) + 1
        element_occurrences[index] = counts[symbols[index]]

    equivalent = [int(value) for value in _dataset_value(dataset, "equivalent_atoms")]
    raw_groups: dict[int, list[int]] = {}
    for index, group in enumerate(equivalent):
        raw_groups.setdefault(group, []).append(index)
    group_members = sorted(raw_groups.values(), key=lambda members: min(site_ids[index] for index in members))
    group_info: dict[int, tuple[str, int, str]] = {}
    for group_number, members in enumerate(group_members, start=1):
        representative = min(members, key=lambda index: site_ids[index])
        orbit_id = f"orbit_{group_number:03d}"
        for index in members:
            group_info[index] = (orbit_id, representative, site_ids[representative])

    wyckoffs = [str(value) for value in _dataset_value(dataset, "wyckoffs")]
    sites: list[SiteRecord] = []
    for index in range(len(atoms)):
        orbit_id, representative, representative_id = group_info[index]
        members = raw_groups[equivalent[index]]
        sites.append(
            SiteRecord(
                site_id=site_ids[index],
                ase_index_zero_based=index,
                cif_label=labels[index] or "",
                element=symbols[index],
                element_occurrence=element_occurrences[index],
                frac_x=float(frac[index, 0]),
                frac_y=float(frac[index, 1]),
                frac_z=float(frac[index, 2]),
                cart_x_A=float(cart[index, 0]),
                cart_y_A=float(cart[index, 1]),
                cart_z_A=float(cart[index, 2]),
                symmetry_orbit_id=orbit_id,
                orbit_representative_site_id=representative_id,
                is_orbit_representative=index == representative,
                orbit_size=len(members),
                wyckoff=wyckoffs[index],
            )
        )

    canonical_atoms = sorted(
        (symbols[index], *(round(float(value), 12) for value in frac[index]))
        for index in range(len(atoms))
    )
    fingerprint_payload = {
        "cell_A": [[round(float(value), 12) for value in row] for row in cell],
        "atoms": canonical_atoms,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rotations = _dataset_value(dataset, "rotations")
    return StructureInspection(
        source_path=path,
        source_sha256=source_sha,
        structure_fingerprint=fingerprint,
        formula=str(atoms.get_chemical_formula()),
        cell_A=tuple(tuple(float(value) for value in row) for row in cell),
        volume_A3=abs(float(np.linalg.det(cell))),
        space_group_symbol=str(_dataset_value(dataset, "international")),
        space_group_number=int(_dataset_value(dataset, "number")),
        symmetry_operation_count=len(rotations),
        symprec_angstrom=float(symprec_angstrom),
        sites=tuple(sites),
    )


def write_structure_manifests(inspection: StructureInspection, manifest_dir: Path, project_root: Path) -> tuple[Path, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    structure_path = manifest_dir / "structure.json"
    sites_path = manifest_dir / "sites.csv"
    try:
        source = str(inspection.source_path.relative_to(project_root))
    except ValueError:
        source = str(inspection.source_path)
    payload = {
        "schema_version": STRUCTURE_SCHEMA_VERSION,
        "source_path": source,
        "source_sha256": inspection.source_sha256,
        "structure_fingerprint": inspection.structure_fingerprint,
        "formula": inspection.formula,
        "atom_count": len(inspection.sites),
        "cell_A": inspection.cell_A,
        "volume_A3": inspection.volume_A3,
        "space_group_symbol": inspection.space_group_symbol,
        "space_group_number": inspection.space_group_number,
        "symmetry_operation_count": inspection.symmetry_operation_count,
        "symprec_angstrom": inspection.symprec_angstrom,
        "sites_file": "sites.csv",
    }
    temporary = structure_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(structure_path)
    temporary_sites = sites_path.with_suffix(".csv.tmp")
    with temporary_sites.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SITE_FIELDS)
        writer.writeheader()
        writer.writerows(site.csv_row() for site in inspection.sites)
    temporary_sites.replace(sites_path)
    return structure_path, sites_path


def load_site_records(path: Path) -> list[SiteRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Site manifest not found: {path}; run orbit inspect")
    sites: list[SiteRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sites.append(
                SiteRecord(
                    site_id=row["site_id"],
                    ase_index_zero_based=int(row["ase_index_zero_based"]),
                    cif_label=row["cif_label"],
                    element=row["element"],
                    element_occurrence=int(row["element_occurrence"]),
                    frac_x=float(row["frac_x"]),
                    frac_y=float(row["frac_y"]),
                    frac_z=float(row["frac_z"]),
                    cart_x_A=float(row["cart_x_A"]),
                    cart_y_A=float(row["cart_y_A"]),
                    cart_z_A=float(row["cart_z_A"]),
                    symmetry_orbit_id=row["symmetry_orbit_id"],
                    orbit_representative_site_id=row["orbit_representative_site_id"],
                    is_orbit_representative=row["is_orbit_representative"] in {"1", "true", "True"},
                    orbit_size=int(row["orbit_size"]),
                    wyckoff=row["wyckoff"],
                )
            )
    return sites


def validate_manifest_source(structure_manifest: Path, structure_path: Path) -> None:
    if not structure_manifest.is_file():
        raise FileNotFoundError(
            f"Structure manifest not found: {structure_manifest}; run orbit inspect"
        )
    payload = json.loads(structure_manifest.read_text(encoding="utf-8"))
    current_sha = hashlib.sha256(structure_path.read_bytes()).hexdigest()
    if payload.get("source_sha256") != current_sha:
        raise ValueError(
            "The configured structure changed after inspection; run orbit inspect again"
        )
