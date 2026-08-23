"""Prepare and record QE calculations for constructed path images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from ase.io import read as ase_read

from ..config import ProjectConfig, qe_hubbard_payload
from ..project import ProjectLayout
from ..scheduler.slurm import render_array_script, render_submit_script
from .input import InvalidGeometryError, render_scf_input, validate_scf_geometry
from .resources import QEResources


PATH_PREPARATION_SCHEMA_VERSION = 2
PATH_CALCULATION_FIELDS = (
    "image",
    "image_name",
    "target_id",
    "path_run_id",
    "destination_id",
    "path_fraction",
    "gap_proxy_eV",
    "preparation_status",
    "invalid_reason",
    "minimum_distance_A",
    "source_structure",
    "calculation_directory",
    "qe_input",
    "qe_output",
    "input_sha256",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _resources_payload(resources: QEResources) -> dict[str, Any]:
    return {
        "pseudo_dir": str(resources.pseudo_dir),
        "cutoff_file": str(resources.cutoff_file),
        "cutoff_file_sha256": resources.cutoff_file_sha256,
        "elements": [
            {
                "element": item.element,
                "pseudopotential": item.pseudopotential,
                "pseudopotential_sha256": item.pseudopotential_sha256,
                "cutoff_wfc_Ry": item.cutoff_wfc_Ry,
                "cutoff_rho_Ry": item.cutoff_rho_Ry,
                "z_valence": item.z_valence,
            }
            for item in resources.elements
        ],
        "ecutwfc_Ry": resources.ecutwfc_Ry,
        "ecutrho_Ry": resources.ecutrho_Ry,
        "total_valence_electrons": resources.total_valence_electrons,
        "occupied_bands": resources.occupied_bands,
        "nbnd": resources.nbnd,
        "nbnd_source": resources.nbnd_source,
    }


def _safe_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."} or any(
        character in run_id for character in ("/", "\\")
    ):
        raise ValueError("path run ID must be a nonempty path-safe identifier")
    return run_id


def resolve_path_run(
    layout: ProjectLayout,
    target_id: str,
    run_id: str | None = None,
) -> Path:
    """Resolve an explicit run, or the newest selected path-bearing run."""
    target_root = layout.paths / target_id
    if run_id is not None:
        run_dir = target_root / "runs" / _safe_run_id(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Path run not found: {run_dir}")
        return run_dir

    index_path = target_root / "index.csv"
    rows = _read_csv(index_path, "Path-run index")
    resolved_target_root = target_root.resolve()
    for row in reversed(rows):
        if row.get("status") not in {"SUCCEEDED", "INFEASIBLE", "CONSTRUCTED"}:
            continue
        candidate = (target_root / row.get("run_directory", "")).resolve()
        if not candidate.is_relative_to(resolved_target_root):
            continue
        if candidate.is_dir() and (candidate / "path.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"No selected constructed path exists for {target_id}; "
        "run orbit path first or specify --run-id"
    )


@dataclass(frozen=True)
class PreparedPathCalculation:
    image: int
    image_name: str
    target_id: str
    path_run_id: str
    destination_id: str
    path_fraction: float
    gap_proxy_eV: float
    preparation_status: str
    invalid_reason: str
    minimum_distance_A: float | None
    source_structure: Path
    input_text: str | None
    input_sha256: str

    def csv_row(self) -> dict[str, Any]:
        valid = self.preparation_status == "VALID"
        return {
            "image": self.image,
            "image_name": self.image_name,
            "target_id": self.target_id,
            "path_run_id": self.path_run_id,
            "destination_id": self.destination_id,
            "path_fraction": self.path_fraction,
            "gap_proxy_eV": self.gap_proxy_eV,
            "preparation_status": self.preparation_status,
            "invalid_reason": self.invalid_reason,
            "minimum_distance_A": (
                "" if self.minimum_distance_A is None else self.minimum_distance_A
            ),
            "source_structure": str(self.source_structure),
            "calculation_directory": self.image_name if valid else "",
            "qe_input": f"{self.image_name}/espresso_scf.pwi" if valid else "",
            "qe_output": f"{self.image_name}/espresso_scf.pwo" if valid else "",
            "input_sha256": self.input_sha256,
        }


@dataclass(frozen=True)
class PreparedPathTarget:
    target_id: str
    path_run_id: str
    destination_id: str
    path_run_directory: Path
    output_directory: Path
    path_csv_sha256: str
    path_json_sha256: str
    calculation_signature: str
    preparation_signature: str
    calculations: tuple[PreparedPathCalculation, ...]
    max_concurrent: int

    @property
    def valid_count(self) -> int:
        return sum(item.preparation_status == "VALID" for item in self.calculations)

    @property
    def invalid_count(self) -> int:
        return len(self.calculations) - self.valid_count


def _calculation_signature(
    config: ProjectConfig,
    resources: QEResources,
    target_id: str,
    run_id: str,
    path_csv_sha256: str,
    path_json_sha256: str,
    cif_inventory: list[dict[str, str]],
) -> str:
    payload = {
        "schema_version": PATH_PREPARATION_SCHEMA_VERSION,
        "target_id": target_id,
        "path_run_id": run_id,
        "path_csv_sha256": path_csv_sha256,
        "path_json_sha256": path_json_sha256,
        "path_cifs": cif_inventory,
        "qe": {
            "kpoints": config.qe.kpoints,
            "offsets": config.qe.offsets,
            "conv_thr": config.qe.conv_thr,
            "mixing_beta": config.qe.mixing_beta,
            "occupations": config.qe.occupations,
            "hubbard": qe_hubbard_payload(config.qe),
            "exact_overlap_tolerance_angstrom": config.qe.exact_overlap_tolerance_angstrom,
        },
        "resources": _resources_payload(resources),
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _preparation_signature(
    calculation_signature: str,
    config: ProjectConfig,
    max_concurrent: int,
) -> str:
    payload = {
        "calculation_signature": calculation_signature,
        "slurm": {
            "partition": config.slurm.partition,
            "nodes": config.slurm.nodes,
            "tasks_per_node": config.slurm.tasks_per_node,
            "time": config.slurm.time,
            "modules": config.slurm.modules,
            "launcher": config.slurm.launcher,
            "pw_command": config.slurm.pw_command,
            "max_concurrent": max_concurrent,
        }
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def build_prepared_path_target(
    config: ProjectConfig,
    layout: ProjectLayout,
    resources: QEResources,
    target_id: str,
    *,
    run_id: str | None,
    max_concurrent: int,
) -> PreparedPathTarget:
    if target_id not in config.target_site_ids:
        raise ValueError(f"Requested target is not configured: {target_id}")
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    run_dir = resolve_path_run(layout, target_id, run_id)
    run_manifest = _read_json(run_dir / "run.json", "Path-run manifest")
    path_manifest = _read_json(run_dir / "path.json", "Selected-path manifest")
    if run_manifest.get("target_id") != target_id or path_manifest.get("target_id") != target_id:
        raise ValueError("Path-run target identity is inconsistent")
    resolved_run_id = str(run_manifest.get("run_id", ""))
    if resolved_run_id != run_dir.name:
        raise ValueError("Path-run directory and manifest IDs disagree")
    if run_manifest.get("status") not in {"SUCCEEDED", "INFEASIBLE", "CONSTRUCTED"}:
        raise ValueError(
            f"Path run {resolved_run_id} has status {run_manifest.get('status')!r} "
            "and has no submit-ready selected path"
        )

    path_csv = run_dir / "path.csv"
    path_json = run_dir / "path.json"
    rows = _read_csv(path_csv, "Selected path table")
    n_images = int(path_manifest.get("n_images", -1))
    if n_images < 2 or len(rows) != n_images:
        raise ValueError(
            f"Path table contains {len(rows)} rows but path.json declares {n_images} images"
        )
    destination = str(path_manifest.get("destination_id", ""))
    calculations: list[PreparedPathCalculation] = []
    cif_inventory: list[dict[str, str]] = []
    for image, row in enumerate(rows):
        image_name = f"image_{image:03d}"
        if row.get("image_name") != image_name or int(row.get("image", -1)) != image:
            raise ValueError(f"Path row {image} is not the expected {image_name}")
        expected_relative = f"cifs/{image_name}.cif"
        if row.get("structure_file") != expected_relative:
            raise ValueError(
                f"Path row {image} must reference {expected_relative}, got "
                f"{row.get('structure_file')!r}"
            )
        source = run_dir / expected_relative
        if not source.is_file():
            raise FileNotFoundError(f"Path image CIF not found: {source}")
        source_hash = _sha256_file(source)
        cif_inventory.append({"image_name": image_name, "sha256": source_hash})
        atoms = ase_read(source)
        try:
            minimum_distance = validate_scf_geometry(
                atoms, resources, config.qe.exact_overlap_tolerance_angstrom
            )
            input_text = render_scf_input(
                atoms, f"{target_id}_path", image_name, config, resources
            )
            status = "VALID"
            reason = ""
            input_hash = _sha256_bytes(input_text.encode("utf-8"))
        except InvalidGeometryError as exc:
            minimum_distance = None
            input_text = None
            status = "INVALID_GEOMETRY"
            reason = str(exc)
            input_hash = ""
        calculations.append(PreparedPathCalculation(
            image=image,
            image_name=image_name,
            target_id=target_id,
            path_run_id=resolved_run_id,
            destination_id=destination,
            path_fraction=float(row["path_fraction"]),
            gap_proxy_eV=float(row["gap_proxy_eV"]),
            preparation_status=status,
            invalid_reason=reason,
            minimum_distance_A=minimum_distance,
            source_structure=source,
            input_text=input_text,
            input_sha256=input_hash,
        ))
    if not any(item.preparation_status == "VALID" for item in calculations):
        raise ValueError(f"No valid path-image geometries remain for {target_id}")

    path_csv_hash = _sha256_file(path_csv)
    path_json_hash = _sha256_file(path_json)
    calculation_signature = _calculation_signature(
        config, resources, target_id, resolved_run_id, path_csv_hash,
        path_json_hash, cif_inventory,
    )
    signature = _preparation_signature(
        calculation_signature, config, max_concurrent,
    )
    output = config.root / "calculations" / "paths" / target_id / resolved_run_id
    return PreparedPathTarget(
        target_id=target_id,
        path_run_id=resolved_run_id,
        destination_id=destination,
        path_run_directory=run_dir,
        output_directory=output,
        path_csv_sha256=path_csv_hash,
        path_json_sha256=path_json_hash,
        calculation_signature=calculation_signature,
        preparation_signature=signature,
        calculations=tuple(calculations),
        max_concurrent=max_concurrent,
    )


def path_calculation_output_state(prepared: PreparedPathTarget) -> str:
    root = prepared.output_directory
    if not root.exists() or (root.is_dir() and not any(root.iterdir())):
        return "missing"
    if not root.is_dir():
        return "conflict"
    required = (
        root / "preparation.json",
        root / "calculations.csv",
        root / "valid_calculations.txt",
        root / "run_array.sh",
        root / "submit.sh",
    )
    if not all(path.is_file() for path in required):
        return "conflict"
    manifest = _read_json(root / "preparation.json", "Path QE preparation manifest")
    if manifest.get("preparation_signature") != prepared.preparation_signature:
        return "conflict"
    rows = _read_csv(root / "calculations.csv", "Path QE calculation table")
    if len(rows) != len(prepared.calculations):
        return "conflict"
    expected = {item.image_name: item for item in prepared.calculations}
    for row in rows:
        item = expected.get(row.get("image_name", ""))
        if item is None or row.get("preparation_status") != item.preparation_status:
            return "conflict"
        if item.preparation_status == "VALID":
            input_path = root / item.image_name / "espresso_scf.pwi"
            structure_path = root / item.image_name / "structure.cif"
            if not input_path.is_file() or not structure_path.is_file():
                return "conflict"
            if _sha256_file(input_path) != item.input_sha256:
                return "conflict"
    return "identical"


def _stale_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.stale_{stamp}")
    if not candidate.exists():
        return candidate
    return path.with_name(f"{path.name}.stale_{stamp}_{uuid4().hex[:8]}")


def write_prepared_path_target(
    config: ProjectConfig,
    resources: QEResources,
    prepared: PreparedPathTarget,
    *,
    overwrite: bool = False,
) -> tuple[Path, str]:
    root = prepared.output_directory
    state = path_calculation_output_state(prepared)
    if state == "identical" and not overwrite:
        return root, "skipped_identical"
    if state == "conflict" and not overwrite:
        raise FileExistsError(
            f"Conflicting path QE preparation exists for {prepared.target_id} "
            f"run {prepared.path_run_id}; use --overwrite to refresh inputs"
        )
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_manifest = (
        _read_json(root / "preparation.json", "Existing path QE preparation manifest")
        if (root / "preparation.json").is_file() else {}
    )
    physics_changed = (
        old_manifest.get("calculation_signature")
        != prepared.calculation_signature
    )
    valid_names: list[str] = []
    for calculation in prepared.calculations:
        folder = root / calculation.image_name
        if physics_changed and folder.is_dir():
            for name in ("espresso_scf.pwo", "tmp"):
                old = folder / name
                if old.exists():
                    old.replace(_stale_path(old, stamp))
        if calculation.preparation_status != "VALID":
            continue
        assert calculation.input_text is not None
        valid_names.append(calculation.image_name)
        folder.mkdir(parents=True, exist_ok=True)
        input_path = folder / "espresso_scf.pwi"
        old_hash = _sha256_file(input_path) if input_path.is_file() else None
        if (
            not physics_changed
            and old_hash is not None
            and old_hash != calculation.input_sha256
        ):
            output_path = folder / "espresso_scf.pwo"
            if output_path.exists():
                output_path.replace(_stale_path(output_path, stamp))
            tmp_path = folder / "tmp"
            if tmp_path.exists():
                tmp_path.replace(_stale_path(tmp_path, stamp))
        shutil.copy2(calculation.source_structure, folder / "structure.cif")
        input_path.write_text(calculation.input_text, encoding="utf-8")
        (folder / "tmp").mkdir(exist_ok=True)

    table_tmp = root / "calculations.csv.tmp"
    with table_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_CALCULATION_FIELDS)
        writer.writeheader()
        writer.writerows(item.csv_row() for item in prepared.calculations)
    table_tmp.replace(root / "calculations.csv")
    valid_tmp = root / "valid_calculations.txt.tmp"
    valid_tmp.write_text("".join(f"{name}\n" for name in valid_names), encoding="utf-8")
    valid_tmp.replace(root / "valid_calculations.txt")

    label = f"{prepared.target_id}_path_{prepared.path_run_id[-8:]}"
    array_path = root / "run_array.sh"
    submit_path = root / "submit.sh"
    array_path.write_text(render_array_script(label, config.slurm), encoding="utf-8")
    submit_path.write_text(
        render_submit_script(label, prepared.max_concurrent), encoding="utf-8"
    )
    array_path.chmod(array_path.stat().st_mode | 0o111)
    submit_path.chmod(submit_path.stat().st_mode | 0o111)

    manifest = {
        "schema_version": PATH_PREPARATION_SCHEMA_VERSION,
        "preparation_signature": prepared.preparation_signature,
        "calculation_signature": prepared.calculation_signature,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "target_id": prepared.target_id,
        "path_run_id": prepared.path_run_id,
        "destination_id": prepared.destination_id,
        "source_path_run_directory": str(prepared.path_run_directory),
        "path_csv_sha256": prepared.path_csv_sha256,
        "path_json_sha256": prepared.path_json_sha256,
        "image_count": len(prepared.calculations),
        "valid_count": prepared.valid_count,
        "invalid_geometry_count": prepared.invalid_count,
        "qe_resources": _resources_payload(resources),
        "qe": {
            "kpoints": config.qe.kpoints,
            "offsets": config.qe.offsets,
            "conv_thr": config.qe.conv_thr,
            "mixing_beta": config.qe.mixing_beta,
            "occupations": config.qe.occupations,
            "hubbard": qe_hubbard_payload(config.qe),
            "exact_overlap_tolerance_angstrom": config.qe.exact_overlap_tolerance_angstrom,
        },
        "slurm": {
            "partition": config.slurm.partition,
            "nodes": config.slurm.nodes,
            "tasks_per_node": config.slurm.tasks_per_node,
            "time": config.slurm.time,
            "max_concurrent": prepared.max_concurrent,
            "modules": config.slurm.modules,
            "launcher": config.slurm.launcher,
            "pw_command": config.slurm.pw_command,
        },
        "calculations_file": "calculations.csv",
        "valid_calculations_file": "valid_calculations.txt",
        "array_script": "run_array.sh",
        "submit_script": "submit.sh",
    }
    manifest_tmp = root / "preparation.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_tmp.replace(root / "preparation.json")
    return root, "written"
