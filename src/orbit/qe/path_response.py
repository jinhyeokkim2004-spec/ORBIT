"""Prepare PH and Berry-polarization jobs for a calculated final path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

from ..config import ProjectConfig


SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=")


@dataclass(frozen=True)
class ResponsePreparation:
    target_id: str
    run_id: str
    source: str
    iteration: int | None
    source_root: Path
    response_root: Path
    image_directories: tuple[Path, ...]
    gdirs: tuple[int, ...]
    max_concurrent: int
    ph_count: int
    polarization_count: int


def _table(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    return dict(value)


def _safe_identifier(value: str, label: str) -> str:
    if not value or not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a nonempty path-safe identifier")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _triplet(value: object, label: str, *, positive: bool) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three integers")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{label} must contain exactly three integers")
    lower = 1 if positive else 0
    if any(item < lower for item in value):
        word = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} entries must be {word}")
    return tuple(value)


def _qe_value(value: object) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("QE namelist values must be finite")
        return f"{value:.12g}"
    raise ValueError(f"Unsupported QE namelist value {value!r}")


def _validated_options(value: object, label: str) -> dict[str, Any]:
    options = _table(value, label)
    for key, item in options.items():
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"{label} contains an invalid QE key: {key!r}")
        _qe_value(item)
    return options


def _read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _prefix_and_outdir(scf_text: str) -> tuple[str, str]:
    prefix_match = re.search(
        r"^\s*prefix\s*=\s*['\"]([^'\"]+)['\"]", scf_text,
        re.IGNORECASE | re.MULTILINE,
    )
    outdir_match = re.search(
        r"^\s*outdir\s*=\s*['\"]([^'\"]+)['\"]", scf_text,
        re.IGNORECASE | re.MULTILINE,
    )
    if prefix_match is None or outdir_match is None:
        raise ValueError("SCF input does not contain parseable prefix and outdir values")
    return prefix_match.group(1), outdir_match.group(1)


def _update_namelist(text: str, name: str, updates: dict[str, Any]) -> str:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines)
         if re.match(rf"^\s*&{re.escape(name)}\b", line, re.IGNORECASE)),
        None,
    )
    if start is None:
        raise ValueError(f"QE input is missing the &{name.upper()} namelist")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip() == "/"),
        None,
    )
    if end is None:
        raise ValueError(f"QE input has an unterminated &{name.upper()} namelist")
    normalized = {key.lower(): (key, value) for key, value in updates.items()}
    retained: list[str] = []
    for line in lines[start + 1:end]:
        match = _ASSIGNMENT.match(line)
        if match is None or match.group(1).lower() not in normalized:
            retained.append(line)
    inserted = [f"    {key:<16s} = {_qe_value(value)}" for key, value in updates.items()]
    lines[start + 1:end] = retained + inserted
    return "\n".join(lines) + "\n"


def _remove_namelist_keys(text: str, name: str, keys: set[str]) -> str:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines)
         if re.match(rf"^\s*&{re.escape(name)}\b", line, re.IGNORECASE)),
        None,
    )
    if start is None:
        raise ValueError(f"QE input is missing the &{name.upper()} namelist")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip() == "/"),
        None,
    )
    if end is None:
        raise ValueError(f"QE input has an unterminated &{name.upper()} namelist")
    normalized = {key.lower() for key in keys}
    retained: list[str] = []
    for line in lines[start + 1:end]:
        match = _ASSIGNMENT.match(line)
        if match is None or match.group(1).lower() not in normalized:
            retained.append(line)
    lines[start + 1:end] = retained
    return "\n".join(lines) + "\n"


def _update_kpoints(text: str, kpoints: tuple[int, int, int], offsets: tuple[int, int, int]) -> str:
    lines = text.splitlines()
    marker = next(
        (index for index, line in enumerate(lines)
         if re.match(r"^\s*K_POINTS\s+automatic\s*$", line, re.IGNORECASE)),
        None,
    )
    if marker is None:
        raise ValueError("SCF input must use K_POINTS automatic")
    data = marker + 1
    while data < len(lines) and not lines[data].strip():
        data += 1
    if data >= len(lines):
        raise ValueError("K_POINTS automatic has no grid line")
    lines[data] = "  " + " ".join(str(value) for value in (*kpoints, *offsets))
    return "\n".join(lines) + "\n"


def _render_polarization_input(
    scf_text: str,
    gdir: int,
    nppstr: int,
    response_outdir: str,
    kpoints: tuple[int, int, int],
    offsets: tuple[int, int, int],
    control_options: dict[str, Any],
    system_options: dict[str, Any],
    electrons_options: dict[str, Any],
) -> str:
    forbidden = {"calculation", "prefix", "outdir", "lberry", "gdir", "nppstr"}
    overlap = forbidden.intersection(key.lower() for key in control_options)
    if overlap:
        raise ValueError(
            "[qe.polarization.control] may not override controlled keys: "
            + ", ".join(sorted(overlap))
        )
    updates = dict(control_options)
    updates.update({
        "calculation": "nscf",
        "outdir": response_outdir,
        "lberry": True,
        "gdir": gdir,
        "nppstr": nppstr,
    })
    text = _update_namelist(scf_text, "control", updates)
    if any(key.lower() == "nbnd" for key in system_options):
        raise ValueError(
            "[qe.polarization.system] may not set nbnd; polarization uses "
            "QE's default occupied-band count"
        )
    if system_options:
        text = _update_namelist(text, "system", system_options)
    text = _remove_namelist_keys(text, "system", {"nbnd"})
    if electrons_options:
        text = _update_namelist(text, "electrons", electrons_options)
    return _update_kpoints(text, kpoints, offsets)


def _render_ph_input(
    prefix: str,
    outdir: str,
    qpoint: tuple[float, float, float],
    inputph_options: dict[str, Any],
) -> str:
    forbidden = {"prefix", "outdir"}
    overlap = forbidden.intersection(key.lower() for key in inputph_options)
    if overlap:
        raise ValueError(
            "[qe.ph.inputph] may not override controlled keys: "
            + ", ".join(sorted(overlap))
        )
    options: dict[str, Any] = {
        "tr2_ph": 1.0e-14,
        "epsil": True,
        "zeu": True,
        "trans": False,
    }
    options.update(inputph_options)
    # These controls are unnecessary for the single Gamma-point dielectric
    # calculation prepared by ORBIT.  Drop legacy material-config entries
    # too, so regenerating inputs removes the lines without a TOML migration.
    options.pop("ldisp", None)
    options.pop("fildyn", None)
    options = {"prefix": prefix, "outdir": outdir, **options}
    lines = ["&INPUTPH"]
    lines.extend(f"    {key:<16s} = {_qe_value(value)}" for key, value in options.items())
    lines.extend(["/", "  " + " ".join(f"{value:.12g}" for value in qpoint), ""])
    return "\n".join(lines)


def _resolve_source(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    source: str,
    iteration: int | None,
) -> tuple[str, int | None, Path]:
    if source == "path":
        if iteration is not None:
            raise ValueError("--iteration is allowed only with --source helper")
        root = config.root / "calculations" / "paths" / target_id / run_id
        if not root.is_dir():
            raise FileNotFoundError(f"Calculated path run not found: {root}")
        return source, None, root
    if source != "helper":
        raise ValueError("source must be 'path' or 'helper'")
    campaign = config.root / "helpers" / target_id / run_id
    if not campaign.is_dir():
        raise FileNotFoundError(f"Helper campaign not found: {campaign}")
    available = sorted(
        int(match.group(1))
        for child in campaign.iterdir()
        if child.is_dir() and (match := re.fullmatch(r"iteration_(\d+)", child.name))
    )
    if not available:
        raise FileNotFoundError(f"No helper iterations found in {campaign}")
    resolved_iteration = available[-1] if iteration is None else iteration
    if resolved_iteration not in available:
        raise FileNotFoundError(
            f"Helper iteration {resolved_iteration} not found; available={available}"
        )
    root = (
        config.root / "calculations" / "helpers" / target_id / run_id
        / f"iteration_{resolved_iteration:03d}" / "path"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Calculated helper path not found: {root}")
    return source, resolved_iteration, root


def _image_directories(source_root: Path) -> tuple[Path, ...]:
    valid_file = source_root / "valid_calculations.txt"
    names = [line.strip() for line in _read_text(valid_file, "Valid calculation list").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No valid path images are listed in {valid_file}")
    images: list[Path] = []
    for expected, name in enumerate(names):
        if name != f"image_{expected:03d}":
            raise ValueError(
                f"Expected contiguous image_{expected:03d}, found {name!r} in {valid_file}"
            )
        image = source_root / name
        if not image.is_dir():
            raise FileNotFoundError(f"Path-image calculation directory not found: {image}")
        images.append(image.resolve())
    return tuple(images)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_input(path: Path, text: str, *, overwrite: bool) -> str:
    if path.is_file():
        old = path.read_text(encoding="utf-8", errors="replace")
        if old == text:
            return "identical"
        if not overwrite:
            raise FileExistsError(f"Conflicting response input exists: {path}; use --overwrite")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = path.with_suffix(".pwo")
        if output.exists():
            output.replace(output.with_name(f"{output.name}.stale_{stamp}"))
    path.write_text(text, encoding="utf-8")
    return "written"


def _module_lines(config: ProjectConfig) -> str:
    return "\n".join(f"module load {shlex.quote(module)}" for module in config.slurm.modules)


def _execution(command: str, input_name: str, output_name: str, launcher: str) -> str:
    command = shlex.quote(command)
    if launcher == "mpirun":
        return f'mpirun -np "${{SLURM_NTASKS}}" {command} -in {input_name} > {output_name}'
    if launcher == "srun":
        return f"srun {command} -in {input_name} > {output_name}"
    raise ValueError(f"Unsupported [slurm].launcher {launcher!r}; use mpirun or srun")


def _render_ph_array(config: ProjectConfig, label: str, ph_command: str) -> str:
    execution = _execution(ph_command, "espresso_ph.pwi", "espresso_ph.pwo", config.slurm.launcher)
    return f"""#!/bin/bash
#SBATCH --job-name=gf_{label}_ph
#SBATCH --partition={config.slurm.partition}
#SBATCH --nodes={config.slurm.nodes}
#SBATCH --ntasks-per-node={config.slurm.tasks_per_node}
#SBATCH --time={config.slurm.time}

set -euo pipefail
CALC_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{GAPFLOW_RESPONSE_LIST}}")"
[[ -d "${{CALC_DIR}}" ]] || {{ echo "ERROR: invalid image directory: ${{CALC_DIR}}"; exit 1; }}
cd "${{CALC_DIR}}"
[[ -f espresso_scf.pwo ]] && grep -q "JOB DONE" espresso_scf.pwo || {{ echo "ERROR: SCF incomplete in ${{CALC_DIR}}"; exit 1; }}
module purge
{_module_lines(config)}
{execution}
grep -q "JOB DONE" espresso_ph.pwo || {{ echo "ERROR: ph.x ended without JOB DONE"; exit 1; }}
"""


def _render_pol_array(
    config: ProjectConfig,
    label: str,
    pw_command: str,
    gdirs: tuple[int, ...],
) -> str:
    if config.slurm.launcher == "mpirun":
        execution = 'mpirun -np "${SLURM_NTASKS}" ' + shlex.quote(pw_command) + ' -in "${INPUT}" > "${OUTPUT}"'
    elif config.slurm.launcher == "srun":
        execution = 'srun ' + shlex.quote(pw_command) + ' -in "${INPUT}" > "${OUTPUT}"'
    else:
        raise ValueError(f"Unsupported [slurm].launcher {config.slurm.launcher!r}")
    return f"""#!/bin/bash
#SBATCH --job-name=gf_{label}_pol
#SBATCH --partition={config.slurm.partition}
#SBATCH --nodes={config.slurm.nodes}
#SBATCH --ntasks-per-node={config.slurm.tasks_per_node}
#SBATCH --time={config.slurm.time}

set -euo pipefail
CALC_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{GAPFLOW_RESPONSE_LIST}}")"
[[ -d "${{CALC_DIR}}" ]] || {{ echo "ERROR: invalid image directory: ${{CALC_DIR}}"; exit 1; }}
cd "${{CALC_DIR}}"
[[ -f espresso_scf.pwo ]] && grep -q "JOB DONE" espresso_scf.pwo || {{ echo "ERROR: SCF incomplete in ${{CALC_DIR}}"; exit 1; }}
module purge
{_module_lines(config)}

# One array task owns one image. If all three gdirs are already complete, exit.
ALL_POL_COMPLETE=1
for GDIR in {' '.join(str(value) for value in gdirs)}; do
    OUTPUT="espresso_pol_gdir${{GDIR}}.pwo"
    if [[ ! -f "${{OUTPUT}}" ]] || ! grep -q "JOB DONE" "${{OUTPUT}}"; then
        ALL_POL_COMPLETE=0
        break
    fi
done
if (( ALL_POL_COMPLETE == 1 )); then
    echo "All polarization gdirs already complete in ${{CALC_DIR}}"
    exit 0
fi

# Rebuild a fresh SCF scratch once per image before any unfinished Berry NSCF.
# This reproduces the successful manual test: fresh SCF -> gdir1 -> gdir2 -> gdir3.
[[ -f espresso_scf.pwi ]] || {{ echo "ERROR: missing espresso_scf.pwi in ${{CALC_DIR}}"; exit 1; }}
echo "Refreshing SCF scratch for polarization in ${{CALC_DIR}}"
rm -rf ./tmp
INPUT="espresso_scf.pwi"
OUTPUT="espresso_response_seed_scf.pwo"
{execution}
grep -q "JOB DONE" "${{OUTPUT}}" || {{ echo "ERROR: response seed SCF ended without JOB DONE"; exit 1; }}

# gdirs are strictly sequential within this image task; different images may run concurrently.
for GDIR in {' '.join(str(value) for value in gdirs)}; do
    INPUT="espresso_pol_gdir${{GDIR}}.pwi"
    OUTPUT="espresso_pol_gdir${{GDIR}}.pwo"
    [[ -f "${{INPUT}}" ]] || {{ echo "ERROR: missing ${{INPUT}}"; exit 1; }}
    if [[ -f "${{OUTPUT}}" ]] && grep -q "JOB DONE" "${{OUTPUT}}"; then
        echo "Skipping completed polarization gdir=${{GDIR}} in ${{CALC_DIR}}"
        continue
    fi
    echo "Starting polarization gdir=${{GDIR}} in ${{CALC_DIR}}"
    {execution}
    grep -q "JOB DONE" "${{OUTPUT}}" || {{ echo "ERROR: polarization NSCF ended without JOB DONE"; exit 1; }}
    echo "Completed polarization gdir=${{GDIR}} in ${{CALC_DIR}}"
done
"""

def _render_all_array(
    config: ProjectConfig,
    label: str,
    ph_command: str,
    pw_command: str,
    gdirs: tuple[int, ...],
) -> str:
    ph_execution = _execution(
        ph_command, "espresso_ph.pwi", "espresso_ph.pwo", config.slurm.launcher
    )
    if config.slurm.launcher == "mpirun":
        pol_execution = 'mpirun -np "${SLURM_NTASKS}" ' + shlex.quote(pw_command) + ' -in "${INPUT}" > "${OUTPUT}"'
    elif config.slurm.launcher == "srun":
        pol_execution = 'srun ' + shlex.quote(pw_command) + ' -in "${INPUT}" > "${OUTPUT}"'
    else:
        raise ValueError(f"Unsupported [slurm].launcher {config.slurm.launcher!r}")
    return f"""#!/bin/bash
#SBATCH --job-name=gf_{label}_response
#SBATCH --partition={config.slurm.partition}
#SBATCH --nodes={config.slurm.nodes}
#SBATCH --ntasks-per-node={config.slurm.tasks_per_node}
#SBATCH --time={config.slurm.time}

set -euo pipefail
CALC_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{GAPFLOW_RESPONSE_LIST}}")"
[[ -d "${{CALC_DIR}}" ]] || {{ echo "ERROR: invalid image directory: ${{CALC_DIR}}"; exit 1; }}
cd "${{CALC_DIR}}"
[[ -f espresso_scf.pwo ]] && grep -q "JOB DONE" espresso_scf.pwo || {{ echo "ERROR: SCF incomplete in ${{CALC_DIR}}"; exit 1; }}
module purge
{_module_lines(config)}
if [[ -f espresso_ph.pwo ]] && grep -q "JOB DONE" espresso_ph.pwo; then
    echo "Skipping completed ph.x in ${{CALC_DIR}}"
else
    {ph_execution}
    grep -q "JOB DONE" espresso_ph.pwo || {{ echo "ERROR: ph.x ended without JOB DONE"; exit 1; }}
fi
for GDIR in {' '.join(str(value) for value in gdirs)}; do
    INPUT="espresso_pol_gdir${{GDIR}}.pwi"
    OUTPUT="espresso_pol_gdir${{GDIR}}.pwo"
    [[ -f "${{INPUT}}" ]] || {{ echo "ERROR: missing ${{INPUT}}"; exit 1; }}
    if [[ -f "${{OUTPUT}}" ]] && grep -q "JOB DONE" "${{OUTPUT}}"; then
        echo "Skipping completed polarization gdir=${{GDIR}} in ${{CALC_DIR}}"
        continue
    fi
    {pol_execution}
    grep -q "JOB DONE" "${{OUTPUT}}" || {{ echo "ERROR: polarization NSCF ended without JOB DONE"; exit 1; }}
done
"""


def _render_submit_all(label: str, max_concurrent: int) -> str:
    return f"""#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SOURCE="${{ROOT}}/all_calculations.txt"
COUNT="$(grep -cve '^$' "${{SOURCE}}")"
if (( COUNT == 0 )); then echo "No response calculations are listed."; exit 0; fi
JOB_ID=$(sbatch --parsable --array="0-$((COUNT - 1))%{max_concurrent}" \
    --output="${{ROOT}}/slurm_logs/all_%A_%a.out" \
    --error="${{ROOT}}/slurm_logs/all_%A_%a.err" \
    --export="ALL,GAPFLOW_RESPONSE_LIST=${{SOURCE}}" "${{ROOT}}/run_response_array.sh")
echo "Submitted {label} combined response array job ${{JOB_ID}}"
"""


def _render_submit(label: str, kind: str, max_concurrent: int) -> str:
    if kind == "ph":
        list_name, runner = "ph_calculations.txt", "run_ph_array.sh"
        completion = '[[ -f "${CALC_DIR}/espresso_ph.pwo" ]] && grep -q "JOB DONE" "${CALC_DIR}/espresso_ph.pwo"'
        parse = 'CALC_DIR="${ENTRY}"'
    else:
        list_name, runner = "polarization_calculations.txt", "run_polarization_array.sh"
        parse = 'CALC_DIR="${ENTRY}"'
        # The polarization runner performs per-gdir completion checks.  Keep
        # every image in the resubmitted array so partially complete images
        # resume safely without concurrent writers in the same QE outdir.
        completion = 'false'
    return f"""#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SOURCE="${{ROOT}}/{list_name}"
REMAINING="${{ROOT}}/remaining_{kind}_$(date +%Y%m%d_%H%M%S).txt"
: > "${{REMAINING}}"
while IFS= read -r ENTRY; do
    [[ -n "${{ENTRY}}" ]] || continue
    {parse}
    if {completion}; then
        echo "Skipping completed: ${{ENTRY}}"
    else
        echo "${{ENTRY}}" >> "${{REMAINING}}"
    fi
done < "${{SOURCE}}"
COUNT="$(wc -l < "${{REMAINING}}")"
if (( COUNT == 0 )); then echo "No unfinished {kind} calculations remain."; exit 0; fi
JOB_ID=$(sbatch --parsable --array="0-$((COUNT - 1))%{max_concurrent}" \
    --output="${{ROOT}}/slurm_logs/{kind}_%A_%a.out" \
    --error="${{ROOT}}/slurm_logs/{kind}_%A_%a.err" \
    --export="ALL,GAPFLOW_RESPONSE_LIST=${{REMAINING}}" "${{ROOT}}/{runner}")
echo "Submitted {label} {kind} array job ${{JOB_ID}}"
"""


def prepare_path_response(
    config: ProjectConfig,
    target_id: str,
    run_id: str,
    *,
    source: str,
    iteration: int | None,
    max_concurrent: int | None,
    check_only: bool,
    overwrite: bool,
    submit: bool,
    kind: str,
    separate: bool = False,
) -> ResponsePreparation:
    if target_id not in config.target_site_ids:
        raise ValueError(f"Requested target is not configured: {target_id}")
    target_id = _safe_identifier(target_id, "target ID")
    run_id = _safe_identifier(run_id, "run ID")
    if check_only and (overwrite or submit):
        raise ValueError("--check-only cannot be combined with --overwrite or --submit")
    if kind not in {"all", "ph", "polarization"}:
        raise ValueError("kind must be all, ph, or polarization")
    if separate and kind == "all":
        raise ValueError("A separate response command must select ph or polarization")
    # The legacy combined command still prepares both families.  The dedicated
    # commands validate and write only their selected family.
    make_ph = not separate or kind == "ph"
    make_pol = not separate or kind == "polarization"
    source, resolved_iteration, source_root = _resolve_source(
        config, target_id, run_id, source, iteration
    )
    images = _image_directories(source_root)

    qe_raw = _table(config.raw.get("qe"), "[qe]")
    inputph: dict[str, Any] = {}
    qpoint: tuple[float, ...] = ()
    if make_ph:
        ph_raw = _table(qe_raw.get("ph"), "[qe.ph]")
        inputph = _validated_options(ph_raw.get("inputph"), "[qe.ph.inputph]")
        qpoint_raw = ph_raw.get("qpoint", [0.0, 0.0, 0.0])
        if not isinstance(qpoint_raw, list) or len(qpoint_raw) != 3 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in qpoint_raw
        ):
            raise ValueError("[qe.ph].qpoint must contain exactly three numbers")
        qpoint = tuple(float(value) for value in qpoint_raw)

    control: dict[str, Any] = {}
    system: dict[str, Any] = {}
    electrons: dict[str, Any] = {}
    gdirs: tuple[int, ...] = ()
    kpoints_by_gdir: dict[int, tuple[int, int, int]] = {}
    offsets: tuple[int, ...] = ()
    nppstr: tuple[int, ...] = ()
    if make_pol:
        pol_raw = _table(qe_raw.get("polarization"), "[qe.polarization]")
        control = _validated_options(pol_raw.get("control"), "[qe.polarization.control]")
        system = _validated_options(pol_raw.get("system"), "[qe.polarization.system]")
        electrons = _validated_options(
            pol_raw.get("electrons"), "[qe.polarization.electrons]"
        )
        gdirs_raw = pol_raw.get("gdirs", [1, 2, 3])
        if not isinstance(gdirs_raw, list) or not gdirs_raw or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in {1, 2, 3}
            for value in gdirs_raw
        ) or len(gdirs_raw) != len(set(gdirs_raw)):
            raise ValueError(
                "[qe.polarization].gdirs must be a unique nonempty subset of [1,2,3]"
            )
        gdirs = tuple(gdirs_raw)
        base_kpoints = _triplet(
            pol_raw.get("kpoints", list(config.qe.kpoints)),
            "[qe.polarization].kpoints",
            positive=True,
        )
        for gdir in gdirs:
            key = f"kpoints_gdir{gdir}"
            kpoints_by_gdir[gdir] = _triplet(
                pol_raw.get(key, list(base_kpoints)),
                f"[qe.polarization].{key}",
                positive=True,
            )
        offsets = _triplet(
            pol_raw.get("offsets", list(config.qe.offsets)),
            "[qe.polarization].offsets",
            positive=False,
        )
        if any(value not in {0, 1} for value in offsets):
            raise ValueError("[qe.polarization].offsets entries must be 0 or 1")
        default_nppstr = [
            kpoints_by_gdir.get(gdir, base_kpoints)[gdir - 1]
            for gdir in (1, 2, 3)
        ]
        nppstr_raw = pol_raw.get("nppstr", default_nppstr)
        if isinstance(nppstr_raw, int) and not isinstance(nppstr_raw, bool):
            nppstr = (nppstr_raw, nppstr_raw, nppstr_raw)
        else:
            nppstr = _triplet(
                nppstr_raw, "[qe.polarization].nppstr", positive=True
            )
        for gdir in gdirs:
            along_string = kpoints_by_gdir[gdir][gdir - 1]
            if nppstr[gdir - 1] != along_string:
                raise ValueError(
                    f"[qe.polarization].nppstr entry for gdir={gdir} must equal "
                    f"the active-direction grid size {along_string}, got "
                    f"{nppstr[gdir - 1]}"
                )

    response_raw = _table(config.raw.get("response"), "[response]")
    concurrency = config.slurm.max_concurrent if max_concurrent is None else max_concurrent
    concurrency = _positive_int(concurrency, "max_concurrent")
    ph_command = str(response_raw.get("ph_command", "ph.x")).strip()
    pw_command = str(response_raw.get("pw_command", config.slurm.pw_command)).strip()
    if not ph_command or not pw_command:
        raise ValueError("[response] commands must be nonempty")

    incomplete: list[str] = []
    rendered: list[tuple[Path, str]] = []
    ph_list: list[str] = []
    pol_list: list[str] = []
    input_hashes: dict[str, str] = {}
    for image in images:
        scf_input = image / "espresso_scf.pwi"
        scf_output = image / "espresso_scf.pwo"
        scf_text = _read_text(scf_input, "Path SCF input")
        prefix, outdir = _prefix_and_outdir(scf_text)
        save_dir = (image / outdir / f"{prefix}.save").resolve()
        if not scf_output.is_file() or "JOB DONE" not in scf_output.read_text(errors="replace"):
            incomplete.append(f"{image.name}: espresso_scf.pwo is missing or incomplete")
        elif not save_dir.is_dir():
            incomplete.append(f"{image.name}: SCF save directory is missing: {save_dir}")
        # PH and polarization both read the exact saved-state directory
        # written by the corresponding ground-state SCF, but dedicated
        # commands render only their selected calculation family.
        if make_ph:
            ph_text = _render_ph_input(prefix, outdir, qpoint, inputph)
            rendered.append((image / "espresso_ph.pwi", ph_text))
            ph_list.append(str(image))
            input_hashes[str(image / "espresso_ph.pwi")] = _sha256(ph_text)
        if make_pol:
            for gdir in gdirs:
                pol_text = _render_polarization_input(
                    scf_text, gdir, nppstr[gdir - 1], outdir,
                    kpoints_by_gdir[gdir], offsets,
                    control, system, electrons,
                )
                path = image / f"espresso_pol_gdir{gdir}.pwi"
                rendered.append((path, pol_text))
                input_hashes[str(path)] = _sha256(pol_text)
            pol_list.append(str(image))
    if incomplete:
        details = "\n  ".join(incomplete[:12])
        suffix = "" if len(incomplete) <= 12 else f"\n  ... and {len(incomplete) - 12} more"
        raise ValueError(
            f"Response calculations require completed reusable SCFs for every image; "
            f"found {len(incomplete)} problems:\n  {details}{suffix}"
        )

    response_root = source_root / "response"
    if separate:
        response_root = response_root / kind
    result = ResponsePreparation(
        target_id=target_id, run_id=run_id, source=source,
        iteration=resolved_iteration, source_root=source_root,
        response_root=response_root, image_directories=images,
        gdirs=gdirs, max_concurrent=concurrency,
        ph_count=len(ph_list), polarization_count=len(pol_list) * len(gdirs),
    )
    if check_only:
        return result

    for path, text in rendered:
        # Berry-polarization NSCFs use CG for numerical robustness.
        # Ordinary/sample/path SCFs are unchanged.
        if path.name.startswith("espresso_pol_gdir") and path.suffix == ".pwi":
            if re.search(r"(?im)^\s*diagonalization\s*=", text):
                text = re.sub(
                    r"(?im)^\s*diagonalization\s*=.*$",
                    "    diagonalization = 'cg'",
                    text,
                    count=1,
                )
            else:
                marker = "&ELECTRONS\n"
                if marker not in text:
                    raise RuntimeError(
                        f"polarization input lacks &ELECTRONS: {path}"
                    )
                text = text.replace(
                    marker,
                    marker + "    diagonalization = 'cg'\n",
                    1,
                )
        # POLARIZATION_CG_DEFAULT
        # Berry-phase NSCFs use CG for numerical robustness.
        # Ordinary/sample/path SCFs remain unchanged.
        if path.name.startswith("espresso_pol_gdir") and path.suffix == ".pwi":
            pol_lines = text.splitlines()
            diag_found = False
            for j, pol_line in enumerate(pol_lines):
                if pol_line.strip().lower().startswith("diagonalization"):
                    pol_lines[j] = "    diagonalization = 'cg'"
                    diag_found = True
                    break
            if not diag_found:
                for j, pol_line in enumerate(pol_lines):
                    if pol_line.strip().upper() == "&ELECTRONS":
                        pol_lines.insert(j + 1, "    diagonalization = 'cg'")
                        diag_found = True
                        break
            if not diag_found:
                raise RuntimeError(
                    f"polarization input lacks &ELECTRONS: {path}"
                )
            text = "\n".join(pol_lines) + "\n"
        _write_input(path, text, overwrite=overwrite)
    response_root.mkdir(parents=True, exist_ok=True)
    (response_root / "slurm_logs").mkdir(exist_ok=True)
    if make_ph:
        (response_root / "ph_calculations.txt").write_text(
            "\n".join(ph_list) + "\n", encoding="utf-8"
        )
    if make_pol:
        (response_root / "polarization_calculations.txt").write_text(
            "\n".join(pol_list) + "\n", encoding="utf-8"
        )
    if not separate:
        (response_root / "all_calculations.txt").write_text(
            "\n".join(ph_list) + "\n", encoding="utf-8"
        )
    label = re.sub(r"[^A-Za-z0-9]+", "_", f"{target_id}_{run_id[-8:]}").lower()
    scripts: dict[str, str] = {}
    if make_ph:
        scripts["run_ph_array.sh"] = _render_ph_array(config, label, ph_command)
        scripts["submit_ph.sh"] = _render_submit(label, "ph", concurrency)
    if make_pol:
        scripts["run_polarization_array.sh"] = _render_pol_array(
            config, label, pw_command, gdirs
        )
        scripts["submit_polarization.sh"] = _render_submit(
            label, "polarization", concurrency
        )
    if not separate:
        scripts["run_response_array.sh"] = _render_all_array(
            config, label, ph_command, pw_command, gdirs
        )
        scripts["submit_all.sh"] = _render_submit_all(label, concurrency)
    for name, text in scripts.items():
        path = response_root / name
        path.write_text(text, encoding="utf-8")
        path.chmod(path.stat().st_mode | 0o111)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_id": target_id,
        "run_id": run_id,
        "source": source,
        "iteration": resolved_iteration,
        "prepared_kind": "all" if not separate else kind,
        "response_root": str(response_root),
        "source_root": str(source_root),
        "image_count": len(images),
        "ph_count": len(ph_list),
        "polarization_count": len(pol_list) * len(gdirs),
        "gdirs": gdirs,
        "kpoints_by_gdir": {
            str(gdir): kpoints_by_gdir[gdir]
            for gdir in gdirs
        },
        "offsets": offsets,
        "nppstr": nppstr,
        "qpoint": qpoint,
        "max_concurrent": concurrency,
        "input_sha256": input_hashes,
    }
    (response_root / "preparation.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if submit:
        selected = (kind,)
        for selected_kind in selected:
            script = response_root / f"submit_{selected_kind}.sh"
            completed = subprocess.run(["bash", str(script)], cwd=response_root, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{selected_kind} submission failed with exit status {completed.returncode}"
                )
    return result
