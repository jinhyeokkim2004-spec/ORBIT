"""Restart-aware Slurm array scripts for prepared QE calculations."""

from __future__ import annotations

import re
import shlex

from ..config import SlurmConfig


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"Cannot construct a Slurm identifier from {value!r}")
    return slug


def render_array_script(target_id: str, config: SlurmConfig) -> str:
    module_lines = "\n".join(
        f"module load {shlex.quote(module)}" for module in config.modules
    )
    env_lines = "\n".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in sorted(config.environment.items())
    )
    launcher_args = " ".join(config.launcher_args)
    pw_command = shlex.quote(config.pw_command)
    if config.launcher == "mpirun":
        execution = (
            f'mpirun -np "${{SLURM_NTASKS}}" {pw_command} '
            "-in espresso_scf.pwi > espresso_scf.pwo"
        )
    elif config.launcher == "srun":
        execution = f"srun {launcher_args} {pw_command} -in espresso_scf.pwi > espresso_scf.pwo"
    else:
        raise ValueError(
            f"Unsupported [slurm].launcher {config.launcher!r}; use mpirun or srun"
        )
    if launcher_args:
        execution = execution.replace(f"srun {launcher_args} ", f"srun {launcher_args} ")
    account_line = f"#SBATCH --account={config.account}" if config.account else ""
    constraint_line = f"#SBATCH --constraint={config.constraint}" if config.constraint else ""
    qos_line = f"#SBATCH --qos={config.qos}" if config.qos else ""
    purge_line = "module purge" if config.module_purge else ""
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=gf_{_slug(target_id)}_scf",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks-per-node={config.tasks_per_node}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --time={config.time}",
    ]
    if account_line:
        script_lines.append(account_line)
    if constraint_line:
        script_lines.append(constraint_line)
    if qos_line:
        script_lines.append(qos_line)
    script_lines.extend([
        "",
        "set -euo pipefail",
        "",
        "if [[ -z \"${GAPFLOW_CALC_LIST:-}\" || ! -f \"${GAPFLOW_CALC_LIST:-}\" ]]; then",
        "    echo \"ERROR: GAPFLOW_CALC_LIST is missing: ${GAPFLOW_CALC_LIST:-not set}\"",
        "    exit 1",
        "fi",
        "if [[ -z \"${SLURM_ARRAY_TASK_ID:-}\" ]]; then",
        "    echo \"ERROR: This script must run as a Slurm array.\"",
        "    exit 1",
        "fi",
        "",
        "CALC_DIR=\"$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" \"${GAPFLOW_CALC_LIST}\")\"",
        "if [[ -z \"${CALC_DIR}\" || ! -d \"${CALC_DIR}\" ]]; then",
        "    echo \"ERROR: Invalid calculation directory: ${CALC_DIR:-empty}\"",
        "    exit 1",
        "fi",
        "cd \"${CALC_DIR}\"",
        "if [[ ! -f espresso_scf.pwi ]]; then",
        "    echo \"ERROR: Missing ${CALC_DIR}/espresso_scf.pwi\"",
        "    exit 1",
        "fi",
        "mkdir -p tmp",
        "",
        "echo \"Calculation: $(basename \"${CALC_DIR}\")\"",
        "echo \"Job: ${SLURM_ARRAY_JOB_ID:-unknown} task ${SLURM_ARRAY_TASK_ID}\"",
        "echo \"Node: ${SLURMD_NODENAME:-unknown}\"",
        "echo \"Started: $(date)\"",
        "",
    ])
    if purge_line:
        script_lines.append(purge_line)
    if module_lines:
        script_lines.append(module_lines)
    if env_lines:
        script_lines.append(env_lines)
    script_lines.extend([
        "",
        execution,
        "",
        "if grep -q \"JOB DONE\" espresso_scf.pwo; then",
        "    echo \"Completed successfully: $(basename \"${CALC_DIR}\")\"",
        "else",
        "    echo \"ERROR: pw.x ended without JOB DONE: $(basename \"${CALC_DIR}\")\"",
        "    exit 1",
        "fi",
        "echo \"Finished: $(date)\"",
        "",
    ])
    return "\n".join(script_lines) + "\n"


def render_submit_script(target_id: str, max_concurrent: int) -> str:
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    slug = _slug(target_id)
    return f"""#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VALID_LIST="${{ROOT_DIR}}/valid_calculations.txt"
ARRAY_SCRIPT="${{ROOT_DIR}}/run_array.sh"
LOG_DIR="${{ROOT_DIR}}/slurm_logs"

if [[ ! -f "${{VALID_LIST}}" ]]; then
    echo "ERROR: Missing valid calculation list: ${{VALID_LIST}}"
    exit 1
fi
if [[ ! -f "${{ARRAY_SCRIPT}}" ]]; then
    echo "ERROR: Missing array script: ${{ARRAY_SCRIPT}}"
    exit 1
fi
mkdir -p "${{LOG_DIR}}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
REMAINING="${{LOG_DIR}}/remaining_{slug}_${{RUN_TAG}}.txt"
: > "${{REMAINING}}"
while IFS= read -r calc_name; do
    [[ -n "${{calc_name}}" ]] || continue
    calc_dir="${{ROOT_DIR}}/${{calc_name}}"
    if [[ ! -d "${{calc_dir}}" || ! -f "${{calc_dir}}/espresso_scf.pwi" ]]; then
        echo "ERROR: Prepared calculation is missing: ${{calc_dir}}"
        exit 1
    fi
    if [[ -f "${{calc_dir}}/espresso_scf.pwo" ]] && \
       grep -q "JOB DONE" "${{calc_dir}}/espresso_scf.pwo"; then
        echo "Skipping completed: $(basename "${{calc_dir}}")"
    else
        echo "${{calc_dir}}" >> "${{REMAINING}}"
    fi
done < "${{VALID_LIST}}"

COUNT="$(wc -l < "${{REMAINING}}")"
if (( COUNT == 0 )); then
    echo "No unfinished {target_id} calculations remain."
    exit 0
fi

echo "Submitting ${{COUNT}} unfinished {target_id} calculations."
echo "Maximum simultaneous tasks: {max_concurrent}"

# Whoville rejects very large Slurm arrays.  Split large calculation
# inventories into safe chunks and chain them with afterany.  Chaining,
# rather than submitting all chunks independently, preserves the requested
# global max_concurrent limit.
MAX_ARRAY_TASKS=900

if (( COUNT <= MAX_ARRAY_TASKS )); then
    JOB_ID=$(
        sbatch --parsable \
            --array="0-$((COUNT - 1))%{max_concurrent}" \
            --output="${{LOG_DIR}}/{slug}_%A_%a.out" \
            --error="${{LOG_DIR}}/{slug}_%A_%a.err" \
            --export="ALL,GAPFLOW_CALC_LIST=${{REMAINING}}" \
            "${{ARRAY_SCRIPT}}"
    )

    echo "Submitted {target_id} array job ${{JOB_ID}}"
    echo "Monitor with: squeue -j ${{JOB_ID}}"

else
    CHUNK_PREFIX="${{LOG_DIR}}/remaining_{slug}_${{RUN_TAG}}_chunk_"

    split \
        -l "${{MAX_ARRAY_TASKS}}" \
        -d \
        -a 4 \
        "${{REMAINING}}" \
        "${{CHUNK_PREFIX}}"

    PREVIOUS_JOB_ID=""
    TERMINAL_JOB_ID=""
    CHUNK_NUMBER=0
    TOTAL_CHUNKS=$(( (COUNT + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS ))

    for CHUNK in "${{CHUNK_PREFIX}}"*; do
        [[ -s "${{CHUNK}}" ]] || continue

        CHUNK_NUMBER=$((CHUNK_NUMBER + 1))
        CHUNK_COUNT="$(wc -l < "${{CHUNK}}")"

        echo "Submitting chunk ${{CHUNK_NUMBER}}/${{TOTAL_CHUNKS}}: ${{CHUNK_COUNT}} calculations"

        if [[ -z "${{PREVIOUS_JOB_ID}}" ]]; then
            JOB_ID=$(
                sbatch --parsable \
                    --array="0-$((CHUNK_COUNT - 1))%{max_concurrent}" \
                    --output="${{LOG_DIR}}/{slug}_%A_%a.out" \
                    --error="${{LOG_DIR}}/{slug}_%A_%a.err" \
                    --export="ALL,GAPFLOW_CALC_LIST=${{CHUNK}}" \
                    "${{ARRAY_SCRIPT}}"
            )
        else
            JOB_ID=$(
                sbatch --parsable \
                    --dependency="afterany:${{PREVIOUS_JOB_ID}}" \
                    --array="0-$((CHUNK_COUNT - 1))%{max_concurrent}" \
                    --output="${{LOG_DIR}}/{slug}_%A_%a.out" \
                    --error="${{LOG_DIR}}/{slug}_%A_%a.err" \
                    --export="ALL,GAPFLOW_CALC_LIST=${{CHUNK}}" \
                    "${{ARRAY_SCRIPT}}"
            )
        fi

        echo "  chunk job = ${{JOB_ID}}"
        PREVIOUS_JOB_ID="${{JOB_ID}}"
        TERMINAL_JOB_ID="${{JOB_ID}}"
    done

    if [[ -z "${{TERMINAL_JOB_ID}}" ]]; then
        echo "ERROR: no Slurm chunk was submitted"
        exit 1
    fi

    # Keep this exact message format for compatibility with existing
    # ORBIT automation, which parses the final dependency job ID.
    echo "Submitted {target_id} array job ${{TERMINAL_JOB_ID}}"
    echo "Monitor terminal job with: squeue -j ${{TERMINAL_JOB_ID}}"
fi
"""
