# ORBIT

ORBIT (Oxidation-state Retrieval Band-gap Insulating path Tracer) is a reproducible workflow for sampling one-atom displacements in a
periodic crystal, running Quantum ESPRESSO (QE) SCF calculations, extracting
HOMO-LUMO gaps, finding a maximum-bottleneck winding path, calculating every
path image, and viewing the result in an interactive HTML plot.

This README is the user guide. Users wishing to customize scripts for their personal usage should also read
[`DEVELOPER_PATCH_GUIDE.md`](DEVELOPER_PATCH_GUIDE.md).

## What the workflow does

For each selected crystallographic site, ORBIT proceeds through the following stages.

### 1. Initialize and inspect the structure

Initialize an ORBIT project from a CIF structure:

```bash
orbit init
```

Inspect the structure, crystallographic symmetry, stable site IDs, and
symmetry-inequivalent sites:

```bash
orbit inspect
```

List the currently selected displacement targets:

```bash
orbit targets list
```

Select the site or sites to study:

```bash
orbit targets set Ti1
```

For multiple targets:

```bash
orbit targets set Ti1 Sr1
```

### 2. Build the displacement sampling grid

For a selected target, ORBIT builds the full-cell displacement grid, folds it
by the stabilizer symmetry of that target site, and applies the configured
geometry exclusions.

```bash
orbit sample --target Ti1
```

The relevant sampling controls are specified in `orbit.toml`, for example:

```toml
[sampling]
spacing_angstrom = 0.3
fold_by_symmetry = true
symprec_angstrom = 0.001
overlap_tolerance_angstrom = 1.0e-5
minimum_interatomic_distance_angstrom = 0.5
```

Rejected geometries remain unavailable to the pathfinding graph rather than
being interpolated or assigned artificial gap values.

### 3. Prepare and run sampled-point SCF calculations

Prepare Quantum ESPRESSO SCF inputs for the valid sampled configurations:

```bash
orbit scf --target Ti1
```

To also submit the generated Slurm array:

```bash
orbit scf --target Ti1 --submit
```

ORBIT never submits calculations unless `--submit` is explicitly supplied.

### 4. Extract sampled band gaps

After the sampled SCFs finish, extract the electronic results:

```bash
orbit extract --target Ti1
```

Only successfully completed and electronically usable SCF calculations are
accepted as live gap nodes. Missing, failed, geometrically rejected, or
non-converged calculations remain unavailable to the pathfinding graph.

### 5. Construct the periodic insulating path

Use the sampled gap field to search for the periodic target-atom trajectory
whose bottleneck gap is maximized:

```bash
orbit path --target Ti1
```

ORBIT searches the periodic sampling graph and records the selected
destination, raw graph path, bottleneck gap, and interpolated path images.

The generated path receives a unique run ID, for example:

```text
20260822T183544Z-4603a350
```

This run ID can be supplied explicitly to later commands when needed.

### 6. Prepare and run SCFs along the selected path

Prepare SCF calculations for all interpolated path images:

```bash
orbit path-scf --target Ti1
```

Submit them with:

```bash
orbit path-scf --target Ti1 --submit
```

If a specific path run should be used:

```bash
orbit path-scf \
    --target Ti1 \
    --run-id 20260822T183544Z-4603a350 \
    --submit
```

### 7. Extract the calculated path gaps

After all path-image SCFs finish:

```bash
orbit path-extract --target Ti1
```

Or, for a specific run:

```bash
orbit path-extract \
    --target Ti1 \
    --run-id 20260822T183544Z-4603a350
```

This produces the calculated band gap as a function of path image and records
the minimum gap along the trajectory.

### 8. Plot and inspect the path

Generate the interactive path/gap visualization:

```bash
orbit plot --target Ti1
```

Or, for a specific run:

```bash
orbit plot \
    --target Ti1 \
    --run-id 20260822T183544Z-4603a350
```

The viewer allows the atomic configuration and calculated band gap to be
inspected along the displacement trajectory.

### 9. Refine a path with helper-atom motion when necessary

If the target-only path does not meet the desired insulating threshold, ORBIT
can search nearby atoms for helper displacements that increase the gap at the
current bottleneck.

Inspect the helper campaign state:

```bash
orbit helper status --target Ti1
```

Prepare a helper scan:

```bash
orbit helper scan --target Ti1 --check-only
```

Prepare and submit the helper scan:

```bash
orbit helper scan --target Ti1 --submit
```

After the helper scan calculations finish, select the best electronically
improving, geometry-safe helper configuration and construct the refined path:

```bash
orbit helper analyze --target Ti1
```

Submit SCFs along the refined helper path:

```bash
orbit helper path-scf --target Ti1 --submit
```

Extract the resulting helper-path gaps:

```bash
orbit helper extract --target Ti1
```

Plot the helper iterations:

```bash
orbit helper plot --target Ti1
```

If the resulting minimum gap is still below the configured helper target,
repeat the helper cycle:

```text
helper scan
    ↓
helper analyze
    ↓
helper path-scf
    ↓
helper extract
    ↓
helper plot
    ↓
next helper scan
```

The target insulating threshold and helper search parameters are configured in
`orbit.toml`, for example:

```toml
[helper]
target_gap_eV = 0.15
neighbor_cutoff_angstrom = 2.00
scan_axes = "xyz"
scan_radius_angstrom = 0.50
scan_steps_per_axis = 5
minimum_helper_distance_angstrom = 0.5
taper_half_width_images = 8.0
improvement_tolerance_eV = 1.0e-6
```

### 10. Calculate polarization along the final insulating path

Once the final path is accepted, prepare the Berry-phase polarization
calculations:

```bash
orbit path-polarization --target Ti1
```

Submit them with:

```bash
orbit path-polarization --target Ti1 --submit
```

When a helper-refined path is the final trajectory, ORBIT uses that helper
iteration as the polarization source.

### Typical command sequence

For a target `Ti1`, a complete target-only workflow is:

```bash
orbit inspect
orbit targets set Ti1

orbit sample --target Ti1

orbit scf --target Ti1
orbit scf --target Ti1 --submit

orbit extract --target Ti1

orbit path --target Ti1

orbit path-scf --target Ti1
orbit path-scf --target Ti1 --submit

orbit path-extract --target Ti1
orbit plot --target Ti1
```

If helper refinement is required:

```bash
orbit helper scan --target Ti1 --check-only
orbit helper scan --target Ti1 --submit

# Wait for the helper scan SCFs to finish.

orbit helper analyze --target Ti1
orbit helper path-scf --target Ti1 --submit

# Wait for the helper-path SCFs to finish.

orbit helper extract --target Ti1
orbit helper plot --target Ti1
orbit helper status --target Ti1
```

Repeat the helper cycle until the configured target gap is reached, then run
the final polarization calculation.

ORBIT never submits a job unless `--submit` is explicitly supplied.

## Requirements

- Python 3.11 or newer
- Quantum ESPRESSO `pw.x`
- A Slurm cluster using either `mpirun` or `srun`
- One UPF pseudopotential per element
- A `cutoffs.json` database containing wavefunction and density cutoffs

Python dependencies (`ase`, `numpy`, `plotly`, and `spglib`) are installed by
the package.

## 1. Install or update ORBIT

Clone or unpack the release, enter the project directory, and install it in
editable mode:

```bash
cd /path/to/orbit
python -m pip install -e .
orbit --version
orbit --help
```

Editable installation means later edits under `src/orbit/` take effect
without reinstalling. Existing terminals can continue to use the `orbit`
command.

## 1a. Activate ORBIT in a new terminal

A fresh terminal does not automatically know about the Conda environment.
Open a new shell and run:

```bash
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate base
orbit --version
orbit --help
```

This is the command sequence required for ORBIT to be available in a new
window. If you want the activation to happen automatically for every new
bash session, add the same two lines to your shell startup file:

```bash
printf '\nsource /path/to/miniforge3/etc/profile.d/conda.sh\nconda activate base\n' >> ~/.bashrc
source ~/.bashrc
orbit --help
```

After activation, run ORBIT from any project directory, for example:

```bash
cd /path/to/project
orbit init --root . --structure crystal.cif
```

## 2. Prepare the pseudopotential library

The configured pseudopotential root must contain this layout:

```text
pseudo/
├── cutoffs.json
└── library/
    ├── H....upf
    ├── O....upf
    └── ...
```

A minimal `cutoffs.json` has one entry per element:

```json
{
  "H": {"cutoff_wfc": 30, "cutoff_rho": 240},
  "O": {"cutoff_wfc": 45, "cutoff_rho": 540}
}
```

ORBIT automatically selects `Element*.upf` only when exactly one match is
present. If several UPFs exist for an element, select the intended filenames
in `orbit.toml`:

```toml
[pseudopotentials]
root = "../pseudo"

[pseudopotentials.files]
H = "H.us.pbe.upf"
O = "O.paw.pbe.upf"
```

**Reference pseudopotential set used for testing**

ORBIT development and many benchmark calculations were tested using the
SSSP PBE pseudopotential set from the Standard Solid-State
Pseudopotentials (SSSP) library.

The pseudopotential files are not distributed with ORBIT. Users should
download the desired SSSP PBE pseudopotentials separately and place them in
the configured pseudo/library/ directory.

ORBIT is not restricted to SSSP nor PBE pseudopotentials. Other Quantum ESPRESSO
compatible UPF pseudopotentials may be used by configuring the corresponding
files and cutoff values in orbit.toml and cutoffs.json.


## 3. Initialize a crystal project

Place the CIF in its project directory, then initialize the stable ORBIT
layout. This does not sample, prepare QE inputs, or submit jobs.

```bash
cd /path/to/project
orbit init --root . --structure crystal.cif
```

`orbit init` preserves an existing `orbit.toml`. Use `--force-config` only
when you intentionally want to replace that configuration:

```bash
orbit init --root . --structure crystal.cif --force-config
```

## 4. Configure sampling, QE, and Slurm

Edit `orbit.toml` before running the workflow. A complete example is:

```toml
schema_version = 1

[project]
structure = "H2O.cif"

[sampling]
spacing_angstrom = 0.4
fold_by_symmetry = true
symprec_angstrom = 0.001
overlap_tolerance_angstrom = 1.0e-5

[targets]
site_ids = []

[pathfinding]
theta_min_eV = 0.3
n_images = 50
winding_padding = 1

[qe]
kpoints = [6, 6, 6]
offsets = [0, 0, 0]
conv_thr = 1.0e-10
mixing_beta = 0.30
occupations = "fixed"
# nbnd = 16
exact_overlap_tolerance_angstrom = 1.0e-5

# Optional DFT+U example:
# [qe.hubbard]
# projector = "atomic"
#
# [[qe.hubbard.u]]
# element = "O"
# manifold = "2p"
# value_eV = 6.0

[pseudopotentials]
root = "../pseudo"

[slurm]
partition = "regular"
nodes = 1
tasks_per_node = 48
time = "02:00:00"
max_concurrent = 6
modules = [
    "intel/2024.2",
    "impi/2021.13",
    "mkl/2024.2",
    "espresso/7.1.0-intel-mkl",
]
launcher = "mpirun"
pw_command = "pw.x"
```

Important settings:

| Setting | Meaning |
| --- | --- |
| `spacing_angstrom` | Approximate Cartesian spacing of the target-position grid. Smaller values increase the number of SCFs. |
| `fold_by_symmetry` | Keeps one representative per target-site-stabilizer orbit. |
| `symprec_angstrom` | Tolerance used by spglib for symmetry detection. |
| `overlap_tolerance_angstrom` | Sampling-stage tolerance for direct nuclear coincidence. |
| `theta_min_eV` | Minimum acceptable bottleneck gap for a successful path. |
| `n_images` | Number of interpolated structures written for the selected path. |
| `winding_padding` | Number of periodic cells available to the winding-path search. |
| `kpoints`, `offsets` | QE Monkhorst-Pack grid and offsets. |
| `nbnd` | Optional explicit band count. If omitted, it is derived from UPF valence counts. |
| `hubbard.projector` | Optional QE Hubbard-projector type. |
| `hubbard.u` | One or more element/manifold Hubbard U values in eV. |
| `exact_overlap_tolerance_angstrom` | Final geometry check before writing a QE input. |
| `max_concurrent` | Maximum simultaneous tasks in each Slurm array. |
| `launcher` | Must be `mpirun` or `srun`. |

With automatic `nbnd`, ORBIT computes the occupied count as
`N_occ = N_e / 2`, where `N_e` is the neutral-cell UPF valence-electron count.
It then adds at least eight empty bands (or 25% of `N_occ` for larger systems)
and rounds upward to a multiple of eight. Automatic setup currently requires
an even valence-electron count and does not configure spin polarization.

## 5. Inspect the structure

```bash
orbit inspect
```

This writes:

- `manifests/structure.json`: structure hash, cell, formula, and symmetry;
- `manifests/sites.csv`: stable site IDs, elements, coordinates, orbits, and
  ASE provenance.

Always rerun `orbit inspect` after changing the CIF. Downstream commands
verify the CIF hash and reject stale manifests.

## 6. Select displacement targets

First list all sites and their stable IDs:

```bash
orbit targets list
```

Choose exactly one of these selection modes:

```bash
# One representative from every crystallographic site orbit
orbit targets set --all-inequivalent

# All inequivalent sites belonging to selected elements
orbit targets set --elements O Ti

# Explicit stable site IDs, preserved in the given order
orbit targets set --sites O1 H1 H2
```

Confirm the selection:

```bash
orbit targets list
```

The command updates `[targets].site_ids` in `orbit.toml` and writes
`manifests/targets.csv`.

## 7. Validate and write displacement samples

Check grid sizes and overlap counts without writing structures:

```bash
orbit sample --check-only
```

Generate all configured targets:

```bash
orbit sample
```

Or generate only one configured target:

```bash
orbit sample --target O1
```

Each target is written under `samples/TARGET/` with `sampling.json`,
`samples.csv`, and valid CIFs under `structures/`. Direct-overlap rows remain
in `samples.csv` with status `DIRECT_OVERLAP`, but receive no CIF. These rows
become impassable walls during pathfinding.

An identical complete dataset is skipped. A conflicting or incomplete dataset
requires explicit replacement:

```bash
orbit sample --target O1 --overwrite
```

## 8. Validate, prepare, and submit sampled SCFs

First validate pseudopotentials, cutoffs, band count, every geometry, every QE
input, and the Slurm configuration without writing files:

```bash
orbit scf --check-only
```

Prepare inputs and scripts:

```bash
orbit scf
```

Submit all unfinished valid calculations:

```bash
orbit scf --submit
```

For one target and a different concurrency limit:

```bash
orbit scf --target O1 --max-concurrent 8 --submit
```

Prepared files live under `calculations/samples/TARGET/`. `submit.sh` creates a
restart-aware list, skips outputs already containing `JOB DONE`, and calls
`sbatch` only when unfinished calculations remain.

Monitor the cluster normally:

```bash
squeue -u "$USER"
```

If QE settings or pseudopotentials change, conflicting prepared inputs are
rejected. Refresh them explicitly:

```bash
orbit scf --target O1 --overwrite
```

Old outputs and `tmp` directories are renamed with a `.stale_TIMESTAMP` suffix
before changed inputs are written.

## 9. Extract sampled gaps

Run extraction after the sampled SCFs finish:

```bash
orbit extract
```

Or extract one target:

```bash
orbit extract --target O1
```

This writes `results/gaps/TARGET/gaps.csv` and `extraction.json`. Every
authoritative sample receives a row. Missing, incomplete, unreadable, invalid,
and gapless outputs remain visible through `output_status`; they are not
silently discarded.

## 10. Plot the sampled heatmap

```bash
orbit plot --target O1
```

The default output is `plots/gap_heatmap_O1.html`. Calculated representatives
are circles, symmetry images are translucent diamonds, and the equilibrium
target is shown separately.

Useful variants:

```bash
# Hide symmetry-equivalent heatmap points
orbit plot --target O1 --representatives-only

# Put several configured targets into one tabbed HTML file
orbit plot --target O1 --target H1 --target H2

# Choose an output path
orbit plot --target O1 --output plots/my_O1_heatmap.html
```

## 11. Construct the maximum-bottleneck path

```bash
orbit path --target O1
```

The search evaluates the periodic destinations `a`, `b`, `c`, `ab`, `bc`,
`ac`, and `abc`. Missing gaps, incomplete calculations, invalid geometries,
and direct overlaps are walls. Every attempt receives an immutable directory
under `paths/O1/runs/RUN_ID/`, including failed or infeasible attempts.

Override path settings for one run when needed:

```bash
orbit path --target O1 --theta-min 0.3 --n-images 50 --winding-padding 1
```

For a diagnostic run that does not need path CIFs:

```bash
orbit path --target O1 --no-cifs
```

Do not use `--no-cifs` for a run that will proceed to path SCFs.

The selected run contains `run.json`, `candidates.csv`, `pathfinding.log`,
`path.csv`, `path.json`, and `cifs/`.

## 12. Validate, prepare, and submit path-image SCFs

ORBIT selects the latest successful path by default:

```bash
orbit path-scf --target O1 --check-only
orbit path-scf --target O1
orbit path-scf --target O1 --submit
```

To use a specific historical run:

```bash
orbit path-scf --target O1 --run-id RUN_ID --check-only
orbit path-scf --target O1 --run-id RUN_ID
orbit path-scf --target O1 --run-id RUN_ID --submit
```

`--run-id` is allowed only when exactly one target is requested. Path
calculations are isolated under `calculations/paths/TARGET/RUN_ID/`. Submission
again skips completed outputs containing `JOB DONE`.

Monitor jobs:

```bash
squeue -u "$USER"
```

Resubmitting unfinished images is safe:

```bash
orbit path-scf --target O1 --run-id RUN_ID --submit
```

## 13. Extract path gaps

After every path-image SCF finishes:

```bash
orbit path-extract --target O1
```

For an explicit run:

```bash
orbit path-extract --target O1 --run-id RUN_ID
```

This writes `results/path_gaps/TARGET/RUN_ID/gaps.csv` and `extraction.json`.

## 14. Generate the combined heatmap/path viewer

```bash
orbit plot-path --target O1
```

For an explicit run or output path:

```bash
orbit plot-path --target O1 --run-id RUN_ID
orbit plot-path --target O1 --output plots/O1_final.html
```

For a tabbed multi-target viewer using each target's latest successful run:

```bash
orbit plot-path --target O1 --target H1 --target H2
```

The default single-target output is `plots/gap_path_viewer_O1.html`. The viewer
contains:

- `HEATMAP` and `PATH` modes;
- a manually draggable path-step slider;
- animation that remains paused until `Play` is clicked;
- a synchronized calculated-gap curve, current-step cursor, and minimum marker;
- atoms rendered as spheres at their equilibrium positions plus periodic
  boundary copies;
- a heatmap visibility toggle in path mode; and
- `LIGHT` and `DARK` themes, with light mode selected initially.

In dark mode, the active Plotly button uses a blue background so its white text
remains visible.

## Complete command sequence for one target

After editing `orbit.toml`, the full workflow for `O1` is:

```bash
cd /global/workdir/jkim068/H2O

orbit inspect
orbit targets set --sites O1
orbit targets list

orbit sample --target O1 --check-only
orbit sample --target O1

orbit scf --target O1 --check-only
orbit scf --target O1
orbit scf --target O1 --submit

# Wait for sampled SCFs to finish.
squeue -u "$USER"

orbit extract --target O1
orbit plot --target O1

orbit path --target O1

orbit path-scf --target O1 --check-only
orbit path-scf --target O1
orbit path-scf --target O1 --submit

# Wait for path-image SCFs to finish.
squeue -u "$USER"

orbit path-extract --target O1
orbit plot-path --target O1
```

## Complete user command reference

This section lists every user command and every supported option. The ordered
workflow above shows when to run them; this reference explains exactly what
each form does.

### Command notation and common behavior

- Text in `UPPER_CASE` is a value supplied by the user.
- Brackets in syntax descriptions mean an option is optional; do not type the
  brackets.
- `--root PROJECT_DIRECTORY` is accepted by every workflow command. It selects
  the directory containing `orbit.toml`. The default is the current
  directory.
- Commands with repeatable `--target SITE_ID` accept the option more than once,
  for example `--target O1 --target H1`. If omitted, all configured targets are
  used.
- A requested target must already appear in `[targets].site_ids`.
- `--check-only` validates and reports without writing stage outputs.
- `--submit` is the only option that permits ORBIT to invoke `sbatch`.
- `--overwrite` never means “delete everything.” It permits a conflicting
  stage to be refreshed according to that stage's preservation rules.
- All expected user/configuration failures print `ERROR: ...` and return exit
  status 2.

### General help and version commands

```bash
orbit --help
orbit --version
orbit COMMAND --help
orbit targets --help
orbit targets list --help
orbit targets set --help
```

| Command | Explanation |
| --- | --- |
| `orbit --help` | Lists all top-level ORBIT commands. |
| `orbit --version` | Prints the installed package version. |
| `orbit COMMAND --help` | Lists the options for one top-level command. |
| `orbit targets ... --help` | Shows help for the nested target-selection commands. |

These commands read no project data, write nothing, and submit nothing.

### Task: initialize a project — `orbit init`

Purpose: create the standard directory tree and, when absent, a starter
`orbit.toml`.

```text
orbit init --structure STRUCTURE_PATH [--root PROJECT_DIRECTORY] [--force-config]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--structure STRUCTURE_PATH` | Yes | Path stored under `[project].structure`. It may be absolute or relative to `--root`. Initialization records the path but does not inspect the file. |
| `--root PROJECT_DIRECTORY` | No | Project directory to create or reuse. Default: current directory. |
| `--force-config` | No | Replace an existing `orbit.toml` with the starter configuration. Other project files are not removed. |

Examples:

```bash
orbit init --root /global/workdir/jkim068/H2O --structure H2O.cif
orbit init --root . --structure H2O.cif
```

Inputs: the structure path supplied on the command line.

Outputs: `orbit.toml` when needed, plus the standard `manifests/`, `samples/`,
`calculations/`, `results/`, `paths/`, `plots/`, and `logs/` directories.

Next task: edit `orbit.toml`, then run `orbit inspect`.

### Task: inspect the structure — `orbit inspect`

Purpose: read the configured structure, detect symmetry, assign stable site
IDs, and bind downstream work to the current CIF bytes.

```text
orbit inspect [--root PROJECT_DIRECTORY]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Directory containing `orbit.toml`. Default: current directory. |

Example:

```bash
orbit inspect
```

Inputs: `orbit.toml` and the configured structure.

Outputs: `manifests/structure.json` and `manifests/sites.csv`; the terminal also
shows formula, volume, space group, site IDs, orbit information, fractional
coordinates, and any current target selection.

Run this command again whenever the CIF changes. Then review target IDs before
continuing.

Next task: choose targets with `orbit targets set`.

### Task: view target IDs and selection — `orbit targets list`

Purpose: display every inspected site and mark the currently configured
targets.

```text
orbit targets list [--root PROJECT_DIRECTORY]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Directory containing the current site manifest and configuration. |

Example:

```bash
orbit targets list
```

Inputs: `orbit.toml`, `manifests/structure.json`, and `manifests/sites.csv`.

Outputs: terminal table only. The command writes nothing and verifies that all
configured target IDs exist in the current site manifest.

### Task: select displacement targets — `orbit targets set`

Purpose: replace the current target list. Exactly one selection mode is
required.

```text
orbit targets set [--root PROJECT_DIRECTORY] \
  (--all-inequivalent | --elements ELEMENT [ELEMENT ...] | --sites SITE_ID [SITE_ID ...])
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Directory containing the inspected project. |
| `--all-inequivalent` | One mode required | Select one representative from every crystallographic site orbit. |
| `--elements ELEMENT ...` | One mode required | Select every inequivalent site whose element is listed. Element symbols are case-sensitive chemical symbols such as `O`, `Ti`, or `Sr`. |
| `--sites SITE_ID ...` | One mode required | Select explicit stable IDs in the supplied order. Use IDs printed by `orbit targets list`. |

Examples:

```bash
orbit targets set --all-inequivalent
orbit targets set --elements O Ti
orbit targets set --sites O1 H1 H2
```

Inputs: the inspected site manifest.

Outputs: replaces `[targets].site_ids` in `orbit.toml`, writes
`manifests/targets.csv`, and creates empty per-target workflow directories.

The three selection modes cannot be combined. Running the command again
replaces—not appends to—the previous selection.

Next task: confirm with `orbit targets list`, then validate sampling.

### Task: validate or generate displacement samples — `orbit sample`

Purpose: build the full-cell target-position grid, reduce it by target-site
stabilizer symmetry, identify exact overlaps, and optionally write the sampled
structures.

```text
orbit sample [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--check-only | --overwrite]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Restrict sampling to configured targets. Omit to process all configured targets. |
| `--check-only` | No | Compute and validate grids, representatives, valid structures, and overlap walls without writing the dataset. |
| `--overwrite` | No | Atomically replace a conflicting target sampling dataset. |

Examples:

```bash
orbit sample --check-only
orbit sample
orbit sample --target O1 --check-only
orbit sample --target O1
orbit sample --target O1 --target H1
orbit sample --target O1 --overwrite
```

`--check-only` and `--overwrite` cannot be combined.

Inputs: current structure/site manifests, configured targets, and `[sampling]`
settings.

Outputs when not checking only: `samples/TARGET/sampling.json`,
`samples/TARGET/samples.csv`, and valid sampled structures under
`samples/TARGET/structures/`. Direct overlaps stay in the CSV but have no CIF.

Next task: validate sampled QE inputs with `orbit scf --check-only`.

### Task: validate, prepare, submit, or resume sampled SCFs — `orbit scf`

Purpose: resolve QE resources, validate every sampled geometry, write QE SCF
inputs and restart-aware Slurm scripts, and optionally submit unfinished work.

```text
orbit scf [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--check-only] [--overwrite] [--submit] [--max-concurrent COUNT]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Restrict preparation/submission to configured targets. Omit for all. |
| `--check-only` | No | Validate UPFs, cutoffs, valence count, `nbnd`, geometries, QE input rendering, and Slurm settings without writing or submitting. |
| `--overwrite` | No | Refresh conflicting prepared inputs. Changed outputs and `tmp` directories are preserved with `.stale_TIMESTAMP` names. |
| `--submit` | No | After preparing, execute each target's `submit.sh`, which calls `sbatch` only for unfinished valid calculations. |
| `--max-concurrent COUNT` | No | Override `[slurm].max_concurrent` for the generated array. `COUNT` must be positive. |

Examples:

```bash
orbit scf --check-only
orbit scf
orbit scf --submit
orbit scf --target O1 --check-only
orbit scf --target O1 --max-concurrent 8
orbit scf --target O1 --max-concurrent 8 --submit
orbit scf --target O1 --overwrite
orbit scf --target O1 --submit      # Resume unfinished work safely
```

Invalid combinations: `--check-only --submit` and
`--check-only --overwrite`.

Inputs: sampled datasets, structure, UPFs, `cutoffs.json`, and the `[qe]`,
`[pseudopotentials]`, and `[slurm]` configuration sections.

Outputs when not checking only: `calculations/samples/TARGET/preparation.json`,
`calculations.csv`, `valid_calculations.txt`, `run_array.sh`, `submit.sh`, and
one directory per valid sample containing `espresso_scf.pwi`.

Submission behavior: calculations with an existing `espresso_scf.pwo`
containing `JOB DONE` are skipped. Monitor submitted jobs with
`squeue -u "$USER"`.

Next task after the array finishes: `orbit extract`.

### Task: extract sampled QE gaps — `orbit extract`

Purpose: parse every authoritative sampled calculation into an auditable
status/HOMO/LUMO/gap table.

```text
orbit extract [--root PROJECT_DIRECTORY] [--target SITE_ID ...]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Extract selected configured targets. Omit for all. |

Examples:

```bash
orbit extract
orbit extract --target O1
orbit extract --target O1 --target H1
```

Inputs: sample preparation records and `espresso_scf.pwo` files.

Outputs: `results/gaps/TARGET/gaps.csv` and `extraction.json`. A row is
`COMPLETE` only when QE printed a usable HOMO/LUMO pair and `JOB DONE`.

If the summary includes missing or incomplete rows, finish/resubmit the SCFs
and run extraction again before pathfinding.

Next tasks: inspect the heatmap with `orbit plot`, then construct a path.

### Task: plot sampled-gap heatmaps — `orbit plot`

Purpose: create a standalone interactive 3D HTML heatmap from accepted sampled
gaps.

```text
orbit plot [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--output HTML_PATH] [--representatives-only]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Plot selected configured targets. Omit for all. Several targets become tabs and share one gap-color range. |
| `--output HTML_PATH` | No | Choose the HTML output. A relative path is resolved from the project root; an absolute path is used directly. |
| `--representatives-only` | No | Hide symmetry-equivalent images and show only calculated representatives. |

Examples:

```bash
orbit plot --target O1
orbit plot --target O1 --representatives-only
orbit plot --target O1 --target H1 --target H2
orbit plot --target O1 --output plots/O1_custom.html
```

Inputs: extracted sampled gaps and the current structure/site manifests.

Outputs: a standalone HTML file and a same-basename JSON manifest under
`plots/` by default. Plotting does not submit or rerun QE.

### Task: construct an insulating winding path — `orbit path`

Purpose: evaluate the seven supported positive lattice windings, solve the
maximum-bottleneck path problem using exact sampled gaps, and interpolate the
selected path into CIF images.

```text
orbit path [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--theta-min EV] [--n-images COUNT] [--winding-padding COUNT] [--no-cifs]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Construct paths for selected configured targets. Omit for all. Each target receives its own run. |
| `--theta-min EV` | No | Override `[pathfinding].theta_min_eV` for this attempt. The selected bottleneck must strictly exceed this threshold for `SUCCEEDED`. |
| `--n-images COUNT` | No | Override `[pathfinding].n_images`. Must be an integer of at least 2. |
| `--winding-padding COUNT` | No | Override `[pathfinding].winding_padding`, the periodic search-box padding. Must be a nonnegative integer; zero uses only the cells required by the requested winding. |
| `--no-cifs` | No | Record the path search without writing interpolated path CIFs. Such a run cannot be prepared with `path-scf`. |

Examples:

```bash
orbit path --target O1
orbit path
orbit path --target O1 --theta-min 0.3 --n-images 50
orbit path --target O1 --winding-padding 2
orbit path --target O1 --no-cifs
```

Inputs: sampling topology, extracted sampled gaps, symmetry, and pathfinding
configuration.

Outputs: a new immutable `paths/TARGET/runs/RUN_ID/` containing `run.json`,
`candidates.csv`, `pathfinding.log`, and, after success, `path.csv`,
`path.json`, and normally `cifs/`. `paths/TARGET/index.csv` receives one run
summary row.

The command may finish with `INFEASIBLE` without destroying earlier runs.
Inspect `candidates.csv` and `pathfinding.log` before changing thresholds.

Next task after a successful run with CIFs: `orbit path-scf --check-only`.

### Task: validate, prepare, submit, or resume path-image SCFs — `orbit path-scf`

Purpose: select a successful constructed path, validate every path image,
prepare QE inputs and restart-aware Slurm scripts, and optionally submit
unfinished images.

```text
orbit path-scf [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--run-id RUN_ID] [--check-only] [--overwrite] [--submit] \
  [--max-concurrent COUNT]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Prepare selected configured targets. Omit for all. |
| `--run-id RUN_ID` | No | Pin one target to a particular path attempt. Without it, the latest successful run is used. Allowed only with exactly one requested target. |
| `--check-only` | No | Validate the path run, resources, every CIF/geometry, QE input rendering, and Slurm configuration without writing or submitting. |
| `--overwrite` | No | Refresh conflicting path inputs while preserving changed outputs and `tmp` directories as stale timestamped copies. |
| `--submit` | No | Prepare, then execute `submit.sh` for each path and submit only unfinished valid images. |
| `--max-concurrent COUNT` | No | Override the Slurm array concurrency cap. Must be positive. |

Examples:

```bash
orbit path-scf --target O1 --check-only
orbit path-scf --target O1
orbit path-scf --target O1 --submit
orbit path-scf --target O1 --max-concurrent 6 --submit
orbit path-scf --target O1 --run-id RUN_ID --check-only
orbit path-scf --target O1 --run-id RUN_ID
orbit path-scf --target O1 --run-id RUN_ID --submit
orbit path-scf --target O1 --run-id RUN_ID --overwrite
orbit path-scf --target O1 --run-id RUN_ID --submit  # Resume unfinished images
```

Invalid combinations: `--check-only --submit` and
`--check-only --overwrite`. `--run-id` with zero/multiple targets is invalid.

Inputs: selected path run and CIFs, structure, UPFs, cutoffs, and QE/Slurm
configuration.

Outputs when not checking only: isolated
`calculations/paths/TARGET/RUN_ID/` preparation records, calculation table,
valid list, Slurm scripts, and one input directory per valid image.

Next task after the array finishes: `orbit path-extract`.

### Task: extract calculated path gaps — `orbit path-extract`

Purpose: parse every calculated path image and join its electronic result to
the constructed path coordinates and sampled-grid proxy metadata.

```text
orbit path-extract [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--run-id RUN_ID]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Extract selected configured target paths. Omit for all. |
| `--run-id RUN_ID` | No | Pin extraction to one path run. Without it, the latest successful run is used. Allowed only with exactly one target. |

Examples:

```bash
orbit path-extract --target O1
orbit path-extract --target O1 --run-id RUN_ID
orbit path-extract --target O1 --target H1
```

Inputs: path records, path preparation records, and path-image
`espresso_scf.pwo` files.

Outputs: `results/path_gaps/TARGET/RUN_ID/gaps.csv` and `extraction.json`.
Incomplete/missing images remain explicit rows. Finish/resubmit calculations
and re-extract before interpreting a partial curve.

Next task: `orbit plot-path`.

### Task: generate the combined heatmap/path viewer — `orbit plot-path`

Purpose: combine sampled gaps, the selected trajectory, calculated path gaps,
structure spheres, periodic copies, and interactive controls into standalone
HTML.

```text
orbit plot-path [--root PROJECT_DIRECTORY] [--target SITE_ID ...] \
  [--run-id RUN_ID] [--output HTML_PATH] [--representatives-only]
```

| Option | Required? | Explanation |
| --- | --- | --- |
| `--root PROJECT_DIRECTORY` | No | Project directory. |
| `--target SITE_ID` | No, repeatable | Plot selected configured targets. Omit for all. Multiple targets become tabs. |
| `--run-id RUN_ID` | No | Pin one target to a specific path run. Without it, the latest successful run is used. Allowed only with exactly one target. |
| `--output HTML_PATH` | No | Select the HTML output path. Relative paths are resolved from the project root. |
| `--representatives-only` | No | Hide symmetry-equivalent sampled heatmap points while retaining calculated representatives. |

Examples:

```bash
orbit plot-path --target O1
orbit plot-path --target O1 --run-id RUN_ID
orbit plot-path --target O1 --representatives-only
orbit plot-path --target O1 --target H1 --target H2
orbit plot-path --target O1 --output plots/O1_final.html
```

Inputs: sampled-gap extraction, selected path run, path-gap extraction, and
structure/site manifests.

Outputs: a standalone HTML viewer and same-basename JSON manifest. It starts in
light HEATMAP mode with path animation stopped; the user can select PATH, drag
the slider, play/pause, show/hide sampled points, and switch LIGHT/DARK themes.

### At-a-glance command summary

| Command | Purpose | Writes files? | Can submit jobs? |
| --- | --- | --- | --- |
| `orbit --help`, `--version` | Show interface or version information. | No | No |
| `orbit init` | Create project directories and starter configuration. | Yes | No |
| `orbit inspect` | Inspect CIF, symmetry, and stable sites. | Yes | No |
| `orbit targets list` | Display sites and current selection. | No | No |
| `orbit targets set` | Replace the explicit target selection. | Yes | No |
| `orbit sample --check-only` | Validate and count displacement samples. | No | No |
| `orbit sample` | Write symmetry-reduced samples and CIFs. | Yes | No |
| `orbit scf --check-only` | Validate sample SCF resources and inputs. | No | No |
| `orbit scf` | Prepare sample QE inputs and Slurm scripts. | Yes | No |
| `orbit scf --submit` | Prepare and submit unfinished sample SCFs. | Yes | Yes |
| `orbit extract` | Extract auditable sampled gaps. | Yes | No |
| `orbit plot` | Write sampled-gap HTML heatmaps. | Yes | No |
| `orbit path` | Construct and record winding paths. | Yes | No |
| `orbit path-scf --check-only` | Validate path-image SCF inputs. | No | No |
| `orbit path-scf` | Prepare path-image QE inputs and scripts. | Yes | No |
| `orbit path-scf --submit` | Prepare and submit unfinished path SCFs. | Yes | Yes |
| `orbit path-extract` | Extract calculated path-image gaps. | Yes | No |
| `orbit plot-path` | Write combined heatmap/path HTML viewers. | Yes | No |

## Project layout

```text
project/
├── crystal.cif
├── orbit.toml
├── manifests/
│   ├── structure.json
│   ├── sites.csv
│   └── targets.csv
├── samples/TARGET/
│   ├── sampling.json
│   ├── samples.csv
│   └── structures/*.cif
├── calculations/
│   ├── samples/TARGET/
│   │   ├── preparation.json
│   │   ├── calculations.csv
│   │   ├── valid_calculations.txt
│   │   ├── run_array.sh
│   │   ├── submit.sh
│   │   └── sample_*/
│   └── paths/TARGET/RUN_ID/
│       ├── preparation.json
│       ├── calculations.csv
│       ├── valid_calculations.txt
│       ├── run_array.sh
│       ├── submit.sh
│       └── image_*/
├── results/
│   ├── gaps/TARGET/
│   │   ├── gaps.csv
│   │   └── extraction.json
│   └── path_gaps/TARGET/RUN_ID/
│       ├── gaps.csv
│       └── extraction.json
├── paths/TARGET/
│   ├── index.csv
│   └── runs/RUN_ID/
│       ├── run.json
│       ├── candidates.csv
│       ├── pathfinding.log
│       ├── path.csv
│       ├── path.json
│       └── cifs/*.cif
├── plots/
└── logs/
```

## Restart, overwrite, and provenance rules

- Re-running an identical completed preparation is skipped.
- `--submit` submits only unfinished valid calculations.
- `--overwrite` is required for conflicting sampling or QE preparation.
- Changed QE inputs preserve old outputs and temporary directories as stale
  timestamped copies.
- Every pathfinding attempt is immutable and receives a new run ID.
- `--run-id` pins path preparation, extraction, or plotting to a historical
  run and prevents accidental selection of a newer path.
- Extracted CSVs retain non-complete statuses instead of hiding missing data.
- Structure and source hashes prevent downstream reuse of stale artifacts.

## Common problems

### `No targets selected`

Run:

```bash
orbit inspect
orbit targets set --all-inequivalent
orbit targets list
```

### `Structure source changed after inspection`

The CIF changed. Rerun inspection and review the stable site IDs before
reselecting targets:

```bash
orbit inspect
orbit targets list
```

### Multiple pseudopotentials found

Add an explicit filename under `[pseudopotentials.files]` in `orbit.toml`.

### Preparation conflict

First inspect why the configuration or source changed. If replacement is
intended, use the relevant `--overwrite` command.

### Some jobs are missing from the Slurm array

This is expected when calculations are invalid or already contain `JOB DONE`.
Inspect `calculations.csv`, `valid_calculations.txt`, and the submission log.

### Pathfinding is infeasible

Inspect `candidates.csv` and `pathfinding.log`. Confirm that sampling SCFs are
complete and extracted, then determine whether missing-data walls, overlap
walls, the grid resolution, or `theta_min_eV` caused the failure. Do not lower
the threshold without examining the physical meaning of the resulting path.

## Current scope

ORBIT currently supports neutral, non-spin-polarized, fixed-occupation SCF
workflows and one displaced target atom at a time. The path search uses exact
sampled representatives and their target-site-stabilizer images; it does not
interpolate uncalculated gaps or automatically introduce helper-atom motion.

## Helper-assisted path refinement

ORBIT can iteratively add one neighboring helper atom at a time to a
calculated target-displacement path. The default workflow is intentionally
human-gated: ORBIT never starts the next helper iteration automatically.

Configure the default helper search with:

```toml
[helper]
target_gap_eV = 0.30
neighbor_cutoff_angstrom = 3.50
scan_axes = "xyz"
scan_radius_angstrom = 0.30
scan_steps_per_axis = 5
minimum_helper_distance_angstrom = 1.20
taper_half_width_images = 8.0
improvement_tolerance_eV = 1.0e-6
```

For a completed and extracted target path, one iteration is:

```bash
orbit helper scan --target Bi1 --run-id RUN_ID --check-only
orbit helper scan --target Bi1 --run-id RUN_ID --submit

# after the individual helper scans finish
orbit helper analyze --target Bi1 --run-id RUN_ID
orbit helper path-scf --target Bi1 --run-id RUN_ID --submit

# after the helper path finishes
orbit helper extract --target Bi1 --run-id RUN_ID
orbit helper plot --target Bi1 --run-id RUN_ID
orbit helper status --target Bi1 --run-id RUN_ID
```

`helper scan` chooses the global minimum calculated gap on the current path,
finds neighboring atoms by periodic minimum-image distance, and scans each
candidate independently on a Cartesian grid. `helper analyze` chooses exactly
one helper atom: the individual scan point with the largest calculated gap,
with smaller displacement used as the tie-breaker. The winner is stored as an
anchor relative to the immutable target-only reference path. Existing anchors
are preserved; if a later iteration returns to the same bottleneck image, that
anchor is updated so additional helper atoms can accumulate there.

Between anchors, the full accumulated helper-displacement field is joined with
cubic smoothstep interpolation. Outside the outermost anchors it returns to
zero with a raised-cosine taper. The target atom remains fixed to the original
target-only path.

After `helper extract`, an iteration is `CONVERGED` when every calculated path
gap is at least `target_gap_eV`. Otherwise it becomes `NEEDS_REVIEW`.
`helper plot` marks a nonconverged iteration `READY_FOR_NEXT`; only then will
the default `helper scan` start the next iteration. This makes plot/review a
required manual checkpoint. `--force` on `helper scan` bypasses that gate for
intentional expert use.

`minimum_helper_distance_angstrom` is a geometry-safety floor. ORBIT never
requires an already-short baseline contact to become longer just to start an
iteration; instead it forbids the helper correction from making that contact
still shorter. Set the value to `0.0` to disable this additional floor while
retaining the ordinary exact-overlap validation.

### Helper run-ID shorthand

Normal helper commands no longer require `--run-id`. ORBIT first reuses the
newest existing helper campaign for the requested target; if none exists, it
uses the newest constructed path run. The resolved run is printed for
provenance. `--run-id` remains available to pin a historical run explicitly.

```bash
orbit helper status --target Bi1
orbit helper plot --target Bi1
orbit helper scan --target Bi1 --check-only
orbit helper scan --target Bi1 --submit
orbit helper analyze --target Bi1
orbit helper path-scf --target Bi1 --submit
orbit helper extract --target Bi1
```



## Path PH and Berry-polarization calculations

After every path-image SCF is complete, prepare Gamma-point `ph.x` Born-charge
and dielectric calculations plus Berry-polarization NSCF jobs for each enabled
direction:

```bash
orbit path-response --target Bi2 --run-id RUN_ID --source path --check-only
orbit path-response --target Bi2 --run-id RUN_ID --source path --submit
```

For a helper-refined path, select the latest ready iteration or specify one:

```bash
orbit path-response --target Bi2 --run-id RUN_ID --source helper --check-only
orbit path-response --target Bi2 --run-id RUN_ID --source helper --iteration 6 --submit
```

The command reuses each image's completed SCF `tmp/PREFIX.save` directory.
It refuses to submit if any SCF output or saved-state directory is missing.
Configure inputs under `[qe.ph]`, `[qe.ph.inputph]`,
`[qe.polarization]`, and its optional `control`, `system`, and `electrons`
subtables. Use `--kind ph` or `--kind polarization` to submit only one family
from the backward-compatible combined preparation.

For fully independent preparation and submission, use `path-ph` and
`path-polarization`. Each command validates and writes only its own input
family and keeps its lists, manifests, scripts, and Slurm logs under
`response/ph` or `response/polarization`:

```bash
orbit path-ph --target Bi1 --run-id RUN_ID \
    --source helper --iteration 10 --check-only
orbit path-ph --target Bi1 --run-id RUN_ID \
    --source helper --iteration 10 --overwrite --submit

orbit path-polarization --target Bi1 --run-id RUN_ID \
    --source helper --iteration 10 --check-only
orbit path-polarization --target Bi1 --run-id RUN_ID \
    --source helper --iteration 10 --overwrite --submit
```

The polarization command has no PH dependency: it reuses the ground-state SCF
saved state directly. Both submitters skip completed outputs, and the
polarization runner resumes missing directions without rerunning completed
directions.


## Polarization and transported-charge analysis

After every PH and three-direction Berry calculation has completed, analyze a
constructed path or helper-refined path with `orbit path-response-plot`.
ORBIT parses arbitrary-size ASR Born tensors, uses all atoms in
`dP = (e/Omega) sum Z* du`, selects the nearest Berry branch, derives the
transport vector from `path.json`, and writes standalone Plotly HTML files plus
CSV audit tables under `response/analysis`.

```bash
orbit path-response-plot --target Bi1 --run-id RUN_ID \
    --source helper --iteration 10
```

The polarization plot contains the Berry representatives, selected continuous
branch, and the cumulative one-step Z* dot displacement estimate. It does not
plot a second-best branch. The transported charge is
`N = (Omega/e) (Delta P dot R) / |R|^2`, with `R` taken from the saved winding
translation rather than a hard-coded lattice axis.
