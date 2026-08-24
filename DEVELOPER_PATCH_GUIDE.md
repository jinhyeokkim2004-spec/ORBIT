# ORBIT Developer Patch Guide

This guide explains the package architecture, what every Python module does,
which files each stage reads and writes, and how to make small patches without
breaking provenance or restart behavior.

The user-facing command sequence belongs in [`README.md`](README.md). Keep that
README synchronized whenever a patch changes a command, configuration key,
output path, or visible plotting behavior.

## Source layout

ORBIT uses a `src/` package layout:

```text
orbit/
├── pyproject.toml
├── README.md
├── DEVELOPER_PATCH_GUIDE.md
├── src/orbit/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── project.py
│   ├── structure.py
│   ├── targets.py
│   ├── periodic.py
│   ├── symmetry.py
│   ├── sampling.py
│   ├── gaps.py
│   ├── plotting.py
│   ├── pathfinding.py
│   ├── path_gaps.py
│   ├── path_plotting.py
│   ├── helpers.py
│   ├── helper_plotting.py
│   ├── machine_profiles/
│   │   ├── generic_slurm.toml
│   │   ├── perlmutter.toml
│   │   └── whoville3.toml
│   ├── qe/
│   │   ├── __init__.py
│   │   ├── resources.py
│   │   ├── input.py
│   │   ├── calculations.py
│   │   ├── path_calculations.py
│   │   ├── path_response.py
│   │   ├── path_response_analysis.py
│   │   └── path_response_plot.py
│   ├── records/
│   │   ├── __init__.py
│   │   └── pathfinding.py
│   └── scheduler/
│       ├── __init__.py
│       └── slurm.py
└── tests/
```

## End-to-end data flow

| Stage | Command | Primary implementation | Reads | Writes |
| --- | --- | --- | --- | --- |
| Initialize | `orbit init` | `project.py`, `cli.py` | Structure path argument | Directories, `orbit.toml` |
| Machine check | `orbit machine show`, `orbit doctor` | `cli.py`, `config.py` | `orbit.toml`, machine profile data | Console diagnostics only |
| Inspect | `orbit inspect` | `structure.py` | CIF, configuration | `structure.json`, `sites.csv` |
| Select | `orbit targets set` | `targets.py`, `config.py` | `sites.csv` | `targets.csv`, `[targets]` config |
| Sample | `orbit sample` | `sampling.py` | CIF, site/target manifests | `sampling.json`, `samples.csv`, CIFs |
| Sample SCF | `orbit scf` | `qe/resources.py`, `qe/input.py`, `qe/calculations.py`, `scheduler/slurm.py` | Samples, UPFs, cutoffs, config | QE inputs, preparation records, Slurm scripts |
| Sample extraction | `orbit extract` | `gaps.py` | Sample preparation and QE outputs | Sample `gaps.csv`, `extraction.json` |
| Heatmap | `orbit plot` | `plotting.py` | Extracted sampled gaps, structure | Standalone HTML and JSON manifest |
| Path search | `orbit path` | `pathfinding.py`, `records/pathfinding.py` | Sampling topology and sampled gaps | Immutable path run and CIFs |
| Path SCF | `orbit path-scf` | `qe/path_calculations.py`, shared QE/scheduler modules | Selected path run, UPFs, config | Path QE inputs and Slurm scripts |
| Path extraction | `orbit path-extract` | `path_gaps.py` | Path preparation and QE outputs | Path `gaps.csv`, `extraction.json` |
| Combined viewer | `orbit plot-path` | `path_plotting.py` | Sample gaps, path gaps, path run, structure | Standalone HTML and JSON manifest |
| Helper refinement | `orbit helper ...` | `helpers.py`, `helper_plotting.py` | Existing path run, helper scan outputs | Helper campaigns, helper results, plots |
| Path response | `orbit path-response`, `path-ph`, `path-polarization` | `qe/path_response.py`, `qe/path_response_analysis.py` | Path run, SCF save state, configuration | PH/polarization input sets, scripts, analysis CSV/plots |

## Python module catalog

### Package entry points

| Module | Responsibility |
| --- | --- |
| `src/orbit/__init__.py` | Defines the package version and exports durable path-record classes. Bump `__version__` with every release. |
| `src/orbit/__main__.py` | Enables `python -m orbit` by forwarding to `cli.main()`. It should contain no workflow logic. |
| `src/orbit/cli.py` | Defines every command and option, validates incompatible flags, calls the stage APIs, prints summaries, invokes `submit.sh` only for explicit `--submit`, and converts expected errors to exit status 2. |

### Configuration and project identity

| Module | Responsibility |
| --- | --- |
| `src/orbit/config.py` | Parses and validates `orbit.toml` into frozen configuration dataclasses. Resolves paths relative to the project root, merges built-in machine defaults with project overrides, and accepts both `orbit.toml` and legacy `gapflow.toml` for compatibility. |
| `src/orbit/project.py` | Defines the stable directory layout, path helpers, starter TOML template, and nondestructive project initialization. Add new top-level output directories to `PROJECT_DIRECTORIES`. |
| `src/orbit/structure.py` | Reads the structure with ASE, obtains symmetry information, associates CIF labels with physical sites, assigns stable site IDs, writes structure/site manifests, and validates the source CIF hash downstream. |
| `src/orbit/targets.py` | Implements explicit site, element, and all-inequivalent target selection and writes `targets.csv`. It does not modify TOML directly; `config.py` owns that update. |
| `src/orbit/machine_profiles/*.toml` | Stores built-in scheduler defaults for `generic_slurm`, `perlmutter`, and `whoville3`. Each profile is configuration data, not Python branching. |

### Geometry, periodicity, and symmetry

| Module | Responsibility |
| --- | --- |
| `src/orbit/periodic.py` | Shared fractional wrapping, fractional-to-Cartesian conversion, and minimum-image displacement/distance operations. Use these helpers instead of component-wise periodic arithmetic for skew cells. |
| `src/orbit/symmetry.py` | Wraps spglib symmetry operations and extracts the subgroup that fixes a target site modulo periodic boundaries. This target-site stabilizer defines the valid sampling equivalence relation. |
| `src/orbit/sampling.py` | Builds the deterministic full-cell target grid, folds it by target-site-stabilizer orbits, detects direct atomic overlaps, writes valid CIFs, and records both valid representatives and overlap walls. It also computes signatures used for idempotent output checks. |

### Sample gaps and visualization

| Module | Responsibility |
| --- | --- |
| `src/orbit/gaps.py` | Parses QE output text, requires `JOB DONE` for `COMPLETE`, calculates HOMO-LUMO gaps, preserves incomplete/missing/invalid rows, writes extracted sample tables, and provides shared target-selection helpers. |
| `src/orbit/plotting.py` | Builds standalone interactive 3D sampled-gap heatmaps, including symmetry images, structure atoms, equilibrium target, shared color ranges, tabbed multi-target HTML, and plot manifests. |
| `src/orbit/helper_plotting.py` | Renders helper-iteration review plots, convergence views, and iteration-by-iteration gap comparisons for the human review gate. |

### Path construction, helper refinement, and records

| Module | Responsibility |
| --- | --- |
| `src/orbit/pathfinding.py` | Reconstructs the exact periodic sampled grid, treats absent/invalid/overlap nodes as walls, evaluates seven positive winding destinations, solves the widest-path problem, selects the best feasible path, interpolates the configured number of images, and writes path records/CIFs. |
| `src/orbit/records/__init__.py` | Re-exports public path-record types. |
| `src/orbit/records/pathfinding.py` | Owns immutable run IDs, candidate rows, transcript logging, `run.json`, `index.csv`, success/failure status transitions, and exception-safe failure recording. |
| `src/orbit/path_gaps.py` | Joins a constructed path with its prepared image calculations, parses each QE output through the shared gap parser, writes one auditable row per image, and signs the resulting path-gap table. |
| `src/orbit/path_plotting.py` | Builds the combined reference-style HEATMAP/PATH Plotly figure, sphere meshes and periodic boundary copies, calculated-gap graph, animation frames, paused manual slider, light/dark themes, active-button styling, tabs, and plot manifest. |
| `src/orbit/helpers.py` | Implements helper-atom scan/analyze/path-scf/extract/plot/status workflows, resolves the active helper campaign, enforces the manual review gate, and maintains helper provenance. |

### Quantum ESPRESSO support

| Module | Responsibility |
| --- | --- |
| `src/orbit/qe/__init__.py` | Re-exports the QE resource API. |
| `src/orbit/qe/resources.py` | Discovers or resolves one UPF per element, reads `cutoffs.json`, hashes resources, parses UPF valence counts, chooses global cutoffs, derives or validates `nbnd`, and rejects unsupported odd-electron automatic setups. |
| `src/orbit/qe/input.py` | Validates exact geometry, assigns species order, and renders one QE SCF input using the resolved cell, coordinates, cutoffs, pseudopotentials, bands, k-points, and electronic settings. |
| `src/orbit/qe/calculations.py` | Converts sampled CIFs into auditable QE calculation directories, preparation manifests, calculation tables, valid lists, and Slurm scripts. It compares signatures and preserves stale results when overwrite is explicit. |
| `src/orbit/qe/path_calculations.py` | Resolves the latest or explicit path run, validates its manifest and CIF inventory, prepares isolated QE inputs for every path image, writes restart-aware lists/scripts, and applies the same conflict/stale-output policy as sampled calculations. |
| `src/orbit/qe/path_response.py` | Prepares PH and Berry-polarization calculations for a selected path or helper iteration: validates the saved-state source, updates namelists, renders per-direction inputs, and writes the response job manifest and scripts. |
| `src/orbit/qe/path_response_analysis.py` | Parses completed PH and polarization outputs, reconstructs Born tensors and Berry branches, computes the transported charge, and writes the analysis CSV and Plotly plots. |

### Scheduler support

| Module | Responsibility |
| --- | --- |
| `src/orbit/scheduler/__init__.py` | Re-exports Slurm rendering functions. |
| `src/orbit/scheduler/slurm.py` | Renders `run_array.sh` and `submit.sh`. Supports `mpirun` and `srun`, validates the calculation list, skips completed outputs, limits concurrency, captures Slurm logs, and requires `JOB DONE` after `pw.x`. |

## Generated scripts

These shell scripts are products of `orbit scf` and `orbit path-scf`; they
are not maintained as standalone source files.

### `run_array.sh`

- Runs one calculation selected by `SLURM_ARRAY_TASK_ID`.
- Receives the current calculation list through `GAPFLOW_CALC_LIST`.
- Loads the configured modules.
- Executes `pw.x` with `mpirun` or `srun`.
- Writes `espresso_scf.pwo`.
- Fails when QE exits without `JOB DONE`.

Patch its template only in `src/orbit/scheduler/slurm.py`.

### `submit.sh`

- Reads `valid_calculations.txt`.
- Verifies every prepared directory and input.
- Excludes calculations whose output already contains `JOB DONE`.
- Writes a timestamped remaining-work list under `slurm_logs/`.
- Calls `sbatch` with the configured concurrency cap.

Patch its template only in `src/orbit/scheduler/slurm.py`.

## Output contracts and invariants

### Stable site IDs, not array indices

User-facing and downstream identities are `site_id` values from `sites.csv`.
ASE indices are provenance only. Never add a workflow feature that stores a
bare reader index as the persistent target identity.

### Structure source validation

Any stage consuming structure-derived manifests must call
`validate_manifest_source()`. A changed CIF must stop the workflow until the
user reruns `orbit inspect` and reviews the target selection.

### Direct overlaps are data, not discarded rows

Sampling-stage direct overlaps stay in `samples.csv`, have no generated CIF,
and become pathfinding walls. Do not filter them out of the topology table.

### Complete means `JOB DONE`

A printed HOMO/LUMO pair from an interrupted QE output is not an accepted
result. Preserve the shared `parse_qe_output()` completion rule for sampled and
path extraction.

### Paths are immutable attempts

Every `orbit path` invocation creates a new run ID. Never reuse a run
directory or overwrite the record of a failed/infeasible attempt.

### Writes are restart-aware

Identical complete outputs are skipped. Conflicts require `--overwrite`.
Changed calculation inputs preserve previous output/tmp data as stale
timestamped copies. New patches must preserve this behavior.

### Manifests and signatures move together

If a patch changes an artifact's scientific meaning or serialized fields:

1. update the corresponding schema version constant;
2. update the signature inputs;
3. update the writer;
4. update every reader/validator;
5. add migration or clear conflict behavior; and
6. add a regression test.

## Standard patch workflow

Work from the package root:

```bash
cd /path/to/orbit
```

### 1. Identify the owning layer

Patch the source module that owns the behavior. Do not directly edit generated
HTML, QE inputs, Slurm scripts, CSVs, or manifests because the next command
will regenerate them.

### 2. Make the smallest coherent change

Preserve unrelated user configuration and existing artifacts. Prefer shared
helpers over duplicate geometry, symmetry, hashing, or parsing code.

### 3. Update tests and documentation

- Add a focused unit test for pure logic.
- Add or extend an integration test for an artifact/command change.
- Update `README.md` for user-visible changes.
- Update this guide for module ownership or patch-procedure changes.

### 4. Bump the version

Keep these two values identical:

- `pyproject.toml`: `[project].version`
- `src/orbit/__init__.py`: `__version__`

### 5. Validate syntax and tests

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

When changing rendered shell scripts, also run `bash -n` on generated
`run_array.sh` and `submit.sh` fixtures. When changing embedded JavaScript,
perform a JavaScript syntax check and regenerate a real HTML viewer.

### 6. Reinstall only when needed

With editable installation, Python source edits take effect immediately:

```bash
python -m pip install -e .
orbit --version
```

Reinstallation is useful after dependency, entry-point, packaging, or version
metadata changes.

### 7. Regenerate only affected outputs

Examples:

```bash
# Plot-only patch
cd /path/to/project
orbit plot-path --target O1

# Slurm-template patch
orbit scf --target O1 --overwrite

# Extractor patch
orbit extract --target O1
orbit path-extract --target O1
```

Do not use `--overwrite` by habit. Confirm that the changed inputs are intended
and that stale-output preservation covers the affected stage.

## Common patch recipes

### Add or change a CLI option

1. Add the argument in `build_parser()` in `cli.py`.
2. Validate incompatible or invalid values in the corresponding `_command()`
   handler.
3. Pass the value into the owning module's public API.
4. Keep scientific/business logic out of `cli.py`.
5. Add parser/handler and behavior tests.
6. Update the command examples in `README.md`.

### Add a configuration key

1. Add it to the appropriate frozen dataclass in `config.py`.
2. Parse and validate it in `load_project_config()`.
3. Add its default to `project.initial_config()`.
4. Decide whether it changes a sampling, preparation, extraction, or plotting
   signature.
5. Add valid, missing-default, and invalid-value tests.
6. Document units and meaning in `README.md`.

### Change sampling or symmetry

Patch `sampling.py`, `periodic.py`, or `symmetry.py` according to ownership.
Regression-test raw grid count, stabilizer order, representative count, orbit
sum, direct-overlap rows, and skew-cell minimum-image behavior. A change to the
equivalence relation requires particularly careful scientific review.

### Change QE input physics

Patch configuration/resource resolution separately from rendering:

- configuration semantics: `config.py`;
- UPFs, cutoffs, valence, bands: `qe/resources.py`;
- QE text and geometry validation: `qe/input.py`;
- sampled/path directory preparation: the corresponding calculations module.

Update preparation signatures so existing incompatible inputs are detected as
conflicts rather than silently reused.

### Change Slurm behavior

Patch `scheduler/slurm.py`, then test both generated scripts with `bash -n`.
Preserve the explicit `--submit` boundary: source code may prepare scripts, but
must not call `sbatch` unless the user requested submission.

### Change QE gap parsing

Patch the shared parser in `gaps.py`; `path_gaps.py` intentionally reuses it.
Test multiple HOMO/LUMO records, Fortran exponents, absent `JOB DONE`, missing
files, and unreadable/incomplete outputs.

### Change pathfinding

Patch `pathfinding.py` for graph/search/interpolation behavior and
`records/pathfinding.py` only for durable run records. Preserve walls for
missing data and overlaps, record every destination candidate, and ensure
exceptions finalize the run as failed.

### Change plotting or HTML controls

- Standalone sampled heatmaps: `plotting.py`.
- Combined heatmap/path viewer: `path_plotting.py`.

Both writers generate complete standalone HTML; never hand-edit the output
HTML. `path_plotting.py` embeds CSS and JavaScript inside a Python f-string, so
literal braces must be doubled (`{{` and `}}`). Keep `auto_play=False` so the
path starts paused, and preserve slider/frame trace indices when inserting a
new trace.

The dark-theme selected-button fix targets Plotly's fixed active fill:

```css
:root[data-theme="dark"] .updatemenu-item-rect[style*="rgb(244, 250, 255)"]{{fill:#1570EF!important;}}
```

If the bundled Plotly version changes, visually verify this selector because
it depends on Plotly's generated SVG style.

## Test suite map

| Test file | Coverage |
| --- | --- |
| `test_config.py` | TOML loading and minimal target-section updates |
| `test_project.py` | Nondestructive initialization |
| `test_structure_integration.py` | Stable labels when CIF row order changes |
| `test_targets.py` | Element/site/orbit selection and target manifests |
| `test_periodic.py` | Half-open wrapping and skew-cell minimum images |
| `test_sampling_integration.py` | H2O counts, manifests/CIFs, overlap walls |
| `test_qe_integration.py` | UPF parsing, automatic bands, QE rendering, invalid geometry, stale preservation, shell syntax |
| `test_gap_extraction.py` | QE completion/gap parsing and auditable rows |
| `test_plotting_integration.py` | Sample extraction to standalone heatmap |
| `test_pathfinding_records.py` | Candidate/run/transcript persistence and failure recording |
| `test_pathfinding_integration.py` | Widest path, 50-image run, path SCF preparation, extraction, and combined viewer |

## Review checklist before release

- [ ] Package and source versions match.
- [ ] `python -m compileall -q src tests` passes.
- [ ] Full unit/integration test suite passes in an environment with ASE,
      NumPy, Plotly, and spglib.
- [ ] Generated Slurm scripts pass `bash -n`.
- [ ] Scientific changes include regression fixtures and stated units.
- [ ] Structure hashes, signatures, schema versions, and stale-output behavior
      still protect incompatible data.
- [ ] No command submits without explicit `--submit`.
- [ ] README command sequence matches `orbit --help`.
- [ ] Plot changes are regenerated and visually inspected in both light and
      dark modes.
- [ ] The release archive excludes `__pycache__`, `.pyc`, and temporary files.

## Release packaging

Create a clean checkpoint archive from the directory containing `orbit/`:

```bash
zip -r orbit_step8_3.zip orbit \
  -x '*/__pycache__/*' '*.pyc' '*.pyo'
unzip -t orbit_step8_3.zip
```

After unpacking on the cluster:

```bash
cd /path/to/orbit
python -m pip install -e .
orbit --version
python -m unittest discover -s tests -v
```
