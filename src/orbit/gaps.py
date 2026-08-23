"""Auditable extraction of HOMO/LUMO gaps from QE sample calculations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .config import ProjectConfig
from .project import ProjectLayout
from .structure import validate_manifest_source


EXTRACTION_SCHEMA_VERSION = 1
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
RE_HOMO_LUMO = re.compile(
    rf"highest occupied,\s*lowest unoccupied level\s*\(ev\)\s*:\s*"
    rf"({NUMBER})\s+({NUMBER})",
    re.IGNORECASE,
)
RE_HOMO_ONLY = re.compile(
    rf"highest occupied level\s*\(ev\)\s*:\s*({NUMBER})",
    re.IGNORECASE,
)
RE_FERMI = re.compile(
    rf"the Fermi energy is\s*({NUMBER})\s*ev",
    re.IGNORECASE,
)
RE_JOB_DONE = re.compile(r"JOB DONE", re.IGNORECASE)

GAP_FIELDS = (
    "sample_id",
    "sample_name",
    "target_id",
    "element",
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
    "preparation_status",
    "output_status",
    "job_done",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "fermi_ev",
    "qe_output",
    "qe_output_sha256",
    "note",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qe_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class ParsedQEOutput:
    job_done: bool
    homo_ev: float | None
    lumo_ev: float | None
    gap_ev: float | None
    fermi_ev: float | None
    status: str
    note: str


@dataclass(frozen=True)
class ExtractionResult:
    target_id: str
    output_directory: Path
    gaps_csv: Path
    manifest: Path
    total_count: int
    status_counts: dict[str, int]

    @property
    def complete_count(self) -> int:
        return self.status_counts.get("COMPLETE", 0)


def parse_qe_output(path: Path) -> ParsedQEOutput:
    """Parse the final printed QE level summary and completion marker."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ParsedQEOutput(
            False, None, None, None, None, "UNREADABLE_OUTPUT", str(exc)
        )

    homo = None
    lumo = None
    fermi = None
    for match in RE_HOMO_LUMO.finditer(text):
        homo = _qe_float(match.group(1))
        lumo = _qe_float(match.group(2))
    if homo is None:
        for match in RE_HOMO_ONLY.finditer(text):
            homo = _qe_float(match.group(1))
    for match in RE_FERMI.finditer(text):
        fermi = _qe_float(match.group(1))

    job_done = bool(RE_JOB_DONE.search(text))
    gap = None if homo is None or lumo is None else lumo - homo
    if not job_done:
        note = "QE output does not contain JOB DONE"
        if gap is not None:
            note += "; levels were parsed but are not accepted as complete"
        return ParsedQEOutput(
            False, homo, lumo, gap, fermi, "INCOMPLETE_OUTPUT", note
        )
    if gap is None:
        note = (
            "JOB DONE was found, but QE did not print a HOMO/LUMO pair"
            if homo is None
            else "JOB DONE was found, but only the occupied level was printed"
        )
        return ParsedQEOutput(
            True, homo, lumo, None, fermi, "NO_GAP_RECORD", note
        )
    note = "negative HOMO-LUMO difference" if gap < 0 else ""
    return ParsedQEOutput(True, homo, lumo, gap, fermi, "COMPLETE", note)


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


def _selected_targets(config: ProjectConfig, requested: list[str] | None) -> list[str]:
    targets = list(config.target_site_ids if requested is None else requested)
    if not targets:
        raise ValueError("No targets selected; run orbit targets set first")
    unknown = [item for item in targets if item not in config.target_site_ids]
    if unknown:
        raise ValueError(
            "Requested targets are not configured: " + ", ".join(unknown)
        )
    if len(targets) != len(set(targets)):
        raise ValueError("Target selection contains duplicates")
    return targets


def extract_target(
    config: ProjectConfig,
    layout: ProjectLayout,
    target_id: str,
) -> ExtractionResult:
    """Extract one row per authoritative sample, retaining all failures."""
    validate_manifest_source(layout.structure_manifest, config.structure)
    sample_root = layout.samples / target_id
    calculation_root = config.root / "calculations" / "samples" / target_id
    output_root = config.root / "results" / "gaps" / target_id

    sampling = _read_json(sample_root / "sampling.json", "Sampling manifest")
    preparation = _read_json(
        calculation_root / "preparation.json", "QE preparation manifest"
    )
    for payload, name in ((sampling, "sampling"), (preparation, "preparation")):
        if payload.get("target_id") != target_id:
            raise ValueError(
                f"{name} manifest target is {payload.get('target_id')!r}, "
                f"not {target_id!r}"
            )
    if preparation.get("sampling_signature") != sampling.get("sampling_signature"):
        raise ValueError(
            f"QE preparation for {target_id} does not match the current sampling dataset"
        )

    samples = _read_csv(sample_root / "samples.csv", "Sampling table")
    calculations = _read_csv(
        calculation_root / "calculations.csv", "QE calculation table"
    )
    calculation_by_name = {row.get("sample_name", ""): row for row in calculations}
    if len(calculation_by_name) != len(calculations):
        raise ValueError(f"Duplicate sample names in {calculation_root / 'calculations.csv'}")
    sample_names = [row.get("sample_name", "") for row in samples]
    if set(sample_names) != set(calculation_by_name):
        missing = sorted(set(sample_names) - set(calculation_by_name))
        extra = sorted(set(calculation_by_name) - set(sample_names))
        raise ValueError(
            f"Sample/calculation manifest mismatch for {target_id}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    output_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    inventory: list[dict[str, str]] = []
    for sample in samples:
        sample_name = sample["sample_name"]
        calculation = calculation_by_name[sample_name]
        preparation_status = calculation.get("preparation_status", "")
        relative_output = calculation.get("qe_output", "")
        output_path = calculation_root / relative_output if relative_output else None
        output_hash = ""

        if preparation_status != "VALID":
            parsed = ParsedQEOutput(
                False,
                None,
                None,
                None,
                None,
                "INVALID_GEOMETRY",
                calculation.get("invalid_reason", "geometry rejected during preparation"),
            )
        elif output_path is None or not output_path.is_file():
            parsed = ParsedQEOutput(
                False,
                None,
                None,
                None,
                None,
                "MISSING_OUTPUT",
                "QE output file not found",
            )
        else:
            output_hash = _sha256_file(output_path)
            parsed = parse_qe_output(output_path)

        status_counts[parsed.status] = status_counts.get(parsed.status, 0) + 1
        inventory.append(
            {
                "sample_name": sample_name,
                "status": parsed.status,
                "sha256": output_hash,
            }
        )
        output_rows.append(
            {
                "sample_id": sample.get("sample_id", ""),
                "sample_name": sample_name,
                "target_id": target_id,
                "element": sample.get("element", ""),
                "frac_x": sample.get("frac_x", ""),
                "frac_y": sample.get("frac_y", ""),
                "frac_z": sample.get("frac_z", ""),
                "cart_x_A": sample.get("cart_x_A", ""),
                "cart_y_A": sample.get("cart_y_A", ""),
                "cart_z_A": sample.get("cart_z_A", ""),
                "displacement_x_A": sample.get("displacement_x_A", ""),
                "displacement_y_A": sample.get("displacement_y_A", ""),
                "displacement_z_A": sample.get("displacement_z_A", ""),
                "displacement_norm_A": sample.get("displacement_norm_A", ""),
                "orbit_size": sample.get("orbit_size", ""),
                "point_stabilizer_order": sample.get("point_stabilizer_order", ""),
                "is_equilibrium": sample.get("is_equilibrium", ""),
                "preparation_status": preparation_status,
                "output_status": parsed.status,
                "job_done": int(parsed.job_done),
                "homo_ev": "" if parsed.homo_ev is None else parsed.homo_ev,
                "lumo_ev": "" if parsed.lumo_ev is None else parsed.lumo_ev,
                "gap_ev": "" if parsed.gap_ev is None else parsed.gap_ev,
                "fermi_ev": "" if parsed.fermi_ev is None else parsed.fermi_ev,
                "qe_output": relative_output,
                "qe_output_sha256": output_hash,
                "note": parsed.note,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    gaps_path = output_root / "gaps.csv"
    temporary_csv = output_root / "gaps.csv.tmp"
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAP_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_csv.replace(gaps_path)

    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = output_root / "extraction.json"
    manifest_payload = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "target_id": target_id,
        "sampling_signature": sampling.get("sampling_signature"),
        "preparation_signature": preparation.get("preparation_signature"),
        "output_inventory_sha256": inventory_hash,
        "total_count": len(output_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "gaps_file": "gaps.csv",
        "gaps_file_sha256": _sha256_file(gaps_path),
        "accepted_gap_rule": "output_status == COMPLETE",
    }
    temporary_manifest = output_root / "extraction.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return ExtractionResult(
        target_id=target_id,
        output_directory=output_root,
        gaps_csv=gaps_path,
        manifest=manifest_path,
        total_count=len(output_rows),
        status_counts=dict(sorted(status_counts.items())),
    )


def extract_targets(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested: list[str] | None,
) -> list[ExtractionResult]:
    return [
        extract_target(config, layout, target_id)
        for target_id in _selected_targets(config, requested)
    ]


def selected_targets(config: ProjectConfig, requested: list[str] | None) -> list[str]:
    """Public target validation shared with result consumers."""
    return _selected_targets(config, requested)
