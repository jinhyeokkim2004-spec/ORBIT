"""Prepare auditable QE sample calculations from ORBIT sampling datasets."""

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
from ..sampling import SamplingPlan, sampling_output_state
from ..scheduler.slurm import render_array_script, render_submit_script
from .input import InvalidGeometryError, render_scf_input, validate_scf_geometry
from .resources import QEResources


PREPARATION_SCHEMA_VERSION = 2
CALCULATION_FIELDS = (
    "sample_id",
    "sample_name",
    "target_id",
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


@dataclass(frozen=True)
class PreparedCalculation:
    sample_id: int
    sample_name: str
    target_id: str
    preparation_status: str
    invalid_reason: str
    minimum_distance_A: float | None
    source_structure: Path | None
    input_text: str | None
    input_sha256: str

    def csv_row(self) -> dict[str, Any]:
        valid = self.preparation_status == "VALID"
        return {
            "sample_id": self.sample_id,
            "sample_name": self.sample_name,
            "target_id": self.target_id,
            "preparation_status": self.preparation_status,
            "invalid_reason": self.invalid_reason,
            "minimum_distance_A": (
                "" if self.minimum_distance_A is None else self.minimum_distance_A
            ),
            "source_structure": (
                "" if self.source_structure is None else str(self.source_structure)
            ),
            "calculation_directory": self.sample_name if valid else "",
            "qe_input": f"{self.sample_name}/espresso_scf.pwi" if valid else "",
            "qe_output": f"{self.sample_name}/espresso_scf.pwo" if valid else "",
            "input_sha256": self.input_sha256,
        }


@dataclass(frozen=True)
class PreparedTarget:
    plan: SamplingPlan
    sampling_signature: str
    preparation_signature: str
    calculations: tuple[PreparedCalculation, ...]
    max_concurrent: int

    @property
    def valid_count(self) -> int:
        return sum(item.preparation_status == "VALID" for item in self.calculations)

    @property
    def invalid_count(self) -> int:
        return len(self.calculations) - self.valid_count


def _preparation_signature(
    config: ProjectConfig,
    plan: SamplingPlan,
    sampling_signature: str,
    resources: QEResources,
    max_concurrent: int,
) -> str:
    payload = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "target_id": plan.target.site_id,
        "sampling_signature": sampling_signature,
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
        "slurm": {
            "partition": config.slurm.partition,
            "nodes": config.slurm.nodes,
            "tasks_per_node": config.slurm.tasks_per_node,
            "time": config.slurm.time,
            "modules": config.slurm.modules,
            "launcher": config.slurm.launcher,
            "pw_command": config.slurm.pw_command,
            "max_concurrent": max_concurrent,
        },
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def build_prepared_targets(
    config: ProjectConfig,
    layout: ProjectLayout,
    plans: list[SamplingPlan],
    resources: QEResources,
    max_concurrent: int,
) -> list[PreparedTarget]:
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    prepared_targets: list[PreparedTarget] = []
    for plan in plans:
        if sampling_output_state(config, layout, plan) != "identical":
            raise ValueError(
                f"Sampling dataset for {plan.target.site_id} is missing, "
                "incomplete, or inconsistent; run orbit sample first"
            )
        sample_root = layout.samples / plan.target.site_id
        sampling_manifest = json.loads(
            (sample_root / "sampling.json").read_text(encoding="utf-8")
        )
        sampling_signature = str(sampling_manifest["sampling_signature"])
        calculations: list[PreparedCalculation] = []
        for point in plan.points:
            source = sample_root / "structures" / f"{point.sample_name}.cif"
            if point.geometry_status != "VALID":
                calculations.append(
                    PreparedCalculation(
                        sample_id=point.sample_id,
                        sample_name=point.sample_name,
                        target_id=plan.target.site_id,
                        preparation_status="INVALID_GEOMETRY",
                        invalid_reason=point.invalid_reason,
                        minimum_distance_A=point.minimum_target_distance_A,
                        source_structure=None,
                        input_text=None,
                        input_sha256="",
                    )
                )
                continue
            if not source.is_file():
                raise FileNotFoundError(f"Sample structure not found: {source}")
            atoms = ase_read(source)
            try:
                minimum_distance = validate_scf_geometry(
                    atoms,
                    resources,
                    config.qe.exact_overlap_tolerance_angstrom,
                )
                input_text = render_scf_input(
                    atoms,
                    plan.target.site_id,
                    point.sample_name,
                    config,
                    resources,
                )
            except InvalidGeometryError as exc:
                calculations.append(
                    PreparedCalculation(
                        sample_id=point.sample_id,
                        sample_name=point.sample_name,
                        target_id=plan.target.site_id,
                        preparation_status="INVALID_GEOMETRY",
                        invalid_reason=str(exc),
                        minimum_distance_A=None,
                        source_structure=source,
                        input_text=None,
                        input_sha256="",
                    )
                )
                continue
            calculations.append(
                PreparedCalculation(
                    sample_id=point.sample_id,
                    sample_name=point.sample_name,
                    target_id=plan.target.site_id,
                    preparation_status="VALID",
                    invalid_reason="",
                    minimum_distance_A=minimum_distance,
                    source_structure=source,
                    input_text=input_text,
                    input_sha256=_sha256_bytes(input_text.encode("utf-8")),
                )
            )
        if not any(item.preparation_status == "VALID" for item in calculations):
            raise ValueError(f"No valid QE geometries remain for {plan.target.site_id}")
        signature = _preparation_signature(
            config, plan, sampling_signature, resources, max_concurrent
        )
        prepared_targets.append(
            PreparedTarget(
                plan=plan,
                sampling_signature=sampling_signature,
                preparation_signature=signature,
                calculations=tuple(calculations),
                max_concurrent=max_concurrent,
            )
        )
    return prepared_targets


def calculation_output_state(layout: ProjectLayout, prepared: PreparedTarget) -> str:
    root = layout.root / "calculations" / "samples" / prepared.plan.target.site_id
    if not root.exists():
        return "missing"
    if not root.is_dir():
        return "conflict"
    if not any(root.iterdir()):
        return "missing"
    manifest_path = root / "preparation.json"
    records_path = root / "calculations.csv"
    valid_path = root / "valid_calculations.txt"
    if not all(path.is_file() for path in (manifest_path, records_path, valid_path)):
        return "conflict"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preparation_signature") != prepared.preparation_signature:
        return "conflict"
    with records_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(prepared.calculations):
        return "conflict"
    expected_by_name = {item.sample_name: item for item in prepared.calculations}
    for row in rows:
        expected = expected_by_name.get(row.get("sample_name", ""))
        if expected is None or row.get("preparation_status") != expected.preparation_status:
            return "conflict"
        if expected.preparation_status == "VALID":
            input_path = root / expected.sample_name / "espresso_scf.pwi"
            structure_path = root / expected.sample_name / "structure.cif"
            if not input_path.is_file() or not structure_path.is_file():
                return "conflict"
            if _sha256_file(input_path) != expected.input_sha256:
                return "conflict"
    if not (root / "run_array.sh").is_file() or not (root / "submit.sh").is_file():
        return "conflict"
    return "identical"


def _stale_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.stale_{stamp}")
    if not candidate.exists():
        return candidate
    return path.with_name(f"{path.name}.stale_{stamp}_{uuid4().hex[:8]}")


def write_prepared_target(
    config: ProjectConfig,
    layout: ProjectLayout,
    resources: QEResources,
    prepared: PreparedTarget,
    *,
    overwrite: bool = False,
) -> tuple[Path, str]:
    root = layout.root / "calculations" / "samples" / prepared.plan.target.site_id
    state = calculation_output_state(layout, prepared)
    if state == "identical" and not overwrite:
        return root, "skipped_identical"
    if state == "conflict" and not overwrite:
        raise FileExistsError(
            f"Conflicting QE preparation exists for {prepared.plan.target.site_id}; "
            "use --overwrite to refresh inputs"
        )
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    valid_names: list[str] = []
    for calculation in prepared.calculations:
        if calculation.preparation_status != "VALID":
            continue
        assert calculation.input_text is not None
        assert calculation.source_structure is not None
        valid_names.append(calculation.sample_name)
        folder = root / calculation.sample_name
        folder.mkdir(parents=True, exist_ok=True)
        input_path = folder / "espresso_scf.pwi"
        old_hash = _sha256_file(input_path) if input_path.is_file() else None
        if old_hash is not None and old_hash != calculation.input_sha256:
            output_path = folder / "espresso_scf.pwo"
            if output_path.exists():
                output_path.replace(_stale_path(output_path, stamp))
            tmp_path = folder / "tmp"
            if tmp_path.exists():
                tmp_path.replace(_stale_path(tmp_path, stamp))
        shutil.copy2(calculation.source_structure, folder / "structure.cif")
        input_path.write_text(calculation.input_text, encoding="utf-8")
        (folder / "tmp").mkdir(exist_ok=True)

    records_tmp = root / "calculations.csv.tmp"
    with records_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALCULATION_FIELDS)
        writer.writeheader()
        writer.writerows(item.csv_row() for item in prepared.calculations)
    records_tmp.replace(root / "calculations.csv")
    valid_tmp = root / "valid_calculations.txt.tmp"
    valid_tmp.write_text("".join(f"{name}\n" for name in valid_names), encoding="utf-8")
    valid_tmp.replace(root / "valid_calculations.txt")

    array_path = root / "run_array.sh"
    submit_path = root / "submit.sh"
    array_path.write_text(
        render_array_script(prepared.plan.target.site_id, config.slurm),
        encoding="utf-8",
    )
    submit_path.write_text(
        render_submit_script(
            prepared.plan.target.site_id, prepared.max_concurrent
        ),
        encoding="utf-8",
    )
    array_path.chmod(array_path.stat().st_mode | 0o111)
    submit_path.chmod(submit_path.stat().st_mode | 0o111)

    manifest = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "preparation_signature": prepared.preparation_signature,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "SUCCEEDED",
        "target_id": prepared.plan.target.site_id,
        "sampling_signature": prepared.sampling_signature,
        "sample_count": len(prepared.calculations),
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
