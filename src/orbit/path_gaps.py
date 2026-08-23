"""Auditable extraction of calculated gaps along constructed paths."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .gaps import parse_qe_output, selected_targets
from .project import ProjectLayout
from .qe.path_calculations import resolve_path_run


PATH_EXTRACTION_SCHEMA_VERSION = 1
PATH_GAP_FIELDS = (
    "image", "image_name", "target_id", "path_run_id", "destination_id",
    "path_fraction", "wrapped_frac_x", "wrapped_frac_y", "wrapped_frac_z",
    "unwrapped_frac_x", "unwrapped_frac_y", "unwrapped_frac_z", "arc_length_A",
    "gap_proxy_eV", "is_nearest_bottleneck_image", "preparation_status",
    "output_status", "job_done", "homo_eV", "lumo_eV", "gap_eV", "fermi_eV",
    "qe_output", "qe_output_sha256", "note",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@dataclass(frozen=True)
class PathExtractionResult:
    target_id: str
    path_run_id: str
    destination_id: str
    output_directory: Path
    gaps_csv: Path
    manifest: Path
    total_count: int
    status_counts: dict[str, int]

    @property
    def complete_count(self) -> int:
        return self.status_counts.get("COMPLETE", 0)


def extract_path_target(
    config: ProjectConfig,
    layout: ProjectLayout,
    target_id: str,
    *,
    run_id: str | None,
) -> PathExtractionResult:
    run_dir = resolve_path_run(layout, target_id, run_id)
    run_manifest = _read_json(run_dir / "run.json", "Path-run manifest")
    path_manifest = _read_json(run_dir / "path.json", "Selected-path manifest")
    resolved_run_id = str(run_manifest.get("run_id", ""))
    if resolved_run_id != run_dir.name:
        raise ValueError("Path-run directory and manifest IDs disagree")
    destination = str(path_manifest.get("destination_id", ""))
    path_csv = run_dir / "path.csv"
    path_rows = _read_csv(path_csv, "Selected path table")

    calculation_root = config.root / "calculations" / "paths" / target_id / resolved_run_id
    preparation = _read_json(
        calculation_root / "preparation.json", "Path QE preparation manifest"
    )
    if preparation.get("target_id") != target_id or preparation.get("path_run_id") != resolved_run_id:
        raise ValueError("Path QE preparation identity is inconsistent")
    if preparation.get("path_csv_sha256") != _sha256(path_csv):
        raise ValueError("Constructed path changed after QE preparation")
    calculation_rows = _read_csv(
        calculation_root / "calculations.csv", "Path QE calculation table"
    )
    by_name = {row.get("image_name", ""): row for row in calculation_rows}
    if len(by_name) != len(calculation_rows):
        raise ValueError("Duplicate image names in path calculation table")
    expected = {row.get("image_name", "") for row in path_rows}
    if set(by_name) != expected:
        raise ValueError("Path and calculation image sets disagree")

    output_rows = []
    status_counts: dict[str, int] = {}
    inventory = []
    for path_row in path_rows:
        name = path_row["image_name"]
        calculation = by_name[name]
        prep_status = calculation.get("preparation_status", "")
        relative_output = calculation.get("qe_output", "")
        output_path = calculation_root / relative_output if relative_output else None
        output_hash = ""
        if prep_status != "VALID":
            from .gaps import ParsedQEOutput
            parsed = ParsedQEOutput(
                False, None, None, None, None, "INVALID_GEOMETRY",
                calculation.get("invalid_reason", "geometry rejected during preparation"),
            )
        elif output_path is None or not output_path.is_file():
            from .gaps import ParsedQEOutput
            parsed = ParsedQEOutput(
                False, None, None, None, None, "MISSING_OUTPUT", "QE output file not found"
            )
        else:
            output_hash = _sha256(output_path)
            parsed = parse_qe_output(output_path)
        status_counts[parsed.status] = status_counts.get(parsed.status, 0) + 1
        inventory.append({"image_name": name, "status": parsed.status, "sha256": output_hash})
        output_rows.append({
            "image": path_row["image"],
            "image_name": name,
            "target_id": target_id,
            "path_run_id": resolved_run_id,
            "destination_id": destination,
            "path_fraction": path_row["path_fraction"],
            "wrapped_frac_x": path_row["wrapped_frac_x"],
            "wrapped_frac_y": path_row["wrapped_frac_y"],
            "wrapped_frac_z": path_row["wrapped_frac_z"],
            "unwrapped_frac_x": path_row["unwrapped_frac_x"],
            "unwrapped_frac_y": path_row["unwrapped_frac_y"],
            "unwrapped_frac_z": path_row["unwrapped_frac_z"],
            "arc_length_A": path_row["arc_length_A"],
            "gap_proxy_eV": path_row["gap_proxy_eV"],
            "is_nearest_bottleneck_image": path_row["is_nearest_bottleneck_image"],
            "preparation_status": prep_status,
            "output_status": parsed.status,
            "job_done": int(parsed.job_done),
            "homo_eV": "" if parsed.homo_ev is None else parsed.homo_ev,
            "lumo_eV": "" if parsed.lumo_ev is None else parsed.lumo_ev,
            "gap_eV": "" if parsed.gap_ev is None else parsed.gap_ev,
            "fermi_eV": "" if parsed.fermi_ev is None else parsed.fermi_ev,
            "qe_output": relative_output,
            "qe_output_sha256": output_hash,
            "note": parsed.note,
        })

    output_root = config.root / "results" / "path_gaps" / target_id / resolved_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    gaps_path = output_root / "gaps.csv"
    temporary = gaps_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_GAP_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(gaps_path)
    manifest_path = output_root / "extraction.json"
    payload = {
        "schema_version": PATH_EXTRACTION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "target_id": target_id,
        "path_run_id": resolved_run_id,
        "destination_id": destination,
        "path_csv_sha256": _sha256(path_csv),
        "path_preparation_signature": preparation.get("preparation_signature"),
        "output_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "total_count": len(output_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "gaps_file": "gaps.csv",
        "gaps_file_sha256": _sha256(gaps_path),
        "accepted_gap_rule": "output_status == COMPLETE",
    }
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_manifest.replace(manifest_path)
    return PathExtractionResult(
        target_id, resolved_run_id, destination, output_root, gaps_path,
        manifest_path, len(output_rows), dict(sorted(status_counts.items())),
    )


def extract_path_targets(
    config: ProjectConfig,
    layout: ProjectLayout,
    requested: list[str] | None,
    *,
    run_id: str | None,
) -> list[PathExtractionResult]:
    targets = selected_targets(config, requested)
    if run_id is not None and len(targets) != 1:
        raise ValueError("--run-id requires exactly one --target")
    return [
        extract_path_target(config, layout, target, run_id=run_id if len(targets) == 1 else None)
        for target in targets
    ]
