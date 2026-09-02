# ORBIT

**ORBIT** (Oxidation-state Retrieval Band-gap Insulating path Tracer) is a
reproducible workflow for constructing insulating atomic-displacement paths in
periodic crystals. It combines symmetry-reduced structural sampling, Quantum
ESPRESSO calculations, graph-based pathfinding, optional helper-atom
refinement, and Berry-phase polarization analysis to support topological
oxidation-state calculations.

## Key capabilities

- **Symmetry-aware sampling.** ORBIT samples target-atom displacements across a
  periodic cell and uses the target site's stabilizer subgroup to avoid
  calculating symmetry-equivalent configurations.
- **Geometry filtering.** Invalid or unsafe structures are excluded before
  electronic-structure calculations and remain unavailable to the pathfinding
  graph.
- **First-principles gap evaluation.** Quantum ESPRESSO self-consistent-field
  calculations provide the electronic gaps used by the path search. Missing,
  failed, or non-converged calculations are never assigned artificial values.
- **Maximum-bottleneck pathfinding.** A periodic displacement graph is searched
  for a winding path that maximizes the minimum sampled band gap along the
  route between periodically equivalent target sites.
- **Helper-assisted refinement.** When target-only motion does not remain
  sufficiently insulating, ORBIT can iteratively introduce smooth,
  geometry-safe displacements of nearby atoms at the path bottleneck.
- **Polarization and transported-charge analysis.** Once an insulating path is
  established, ORBIT can prepare Berry-phase polarization and optional
  Born-effective-charge calculations and use them to determine the charge
  transported by the target ion.
- **Cluster execution and restartability.** ORBIT prepares Slurm job arrays,
  resumes incomplete calculation sets, and preserves immutable path runs and
  configuration provenance.
- **Inspectable results.** Sampling, path, gap, helper, and polarization results
  are stored in machine-readable manifests and CSV tables, with interactive
  visualizations for structural and electronic inspection.

## Workflow

For each symmetry-inequivalent target site, ORBIT:

1. inspects the crystal structure and identifies stable site labels;
2. builds and symmetry-reduces a periodic displacement grid;
3. evaluates valid sampled structures with Quantum ESPRESSO;
4. constructs a maximum-bottleneck winding path from the sampled gap field;
5. calculates and verifies every interpolated path image;
6. adds helper-atom motion if the verified path falls below the chosen gap
   threshold; and
7. evaluates Berry-phase polarization along the accepted insulating path.

The sampled graph proposes a path, but the final path classification is based
on explicit calculations of every path image. ORBIT does not interpolate
uncalculated band gaps.

## Reproducibility and safety

ORBIT separates preparation from job submission and does not submit cluster
jobs implicitly. It records structure and configuration hashes, retains failed
and incomplete calculation statuses, assigns every pathfinding attempt a
unique run ID, and prevents stale results from being silently reused after
inputs change.

Helper refinement is also reviewable: candidate atoms are scanned
independently, geometry constraints are enforced, and each selected helper
displacement is recorded relative to the original target-only path.

## Requirements

- Python 3.11 or newer
- Quantum ESPRESSO, including `pw.x` and the optional response executables used
  by the selected analysis
- A Slurm cluster using `srun` or `mpirun`
- One configured UPF pseudopotential per element
- Wavefunction and charge-density cutoff data for the selected
  pseudopotentials

The Python package installs its required Python dependencies. Pseudopotential
files are not distributed with ORBIT. Development and benchmark calculations
have primarily used PBE pseudopotentials from the Standard Solid-State
Pseudopotentials library, but ORBIT is not restricted to that set.

## Current scope

ORBIT currently treats one displaced target atom at a time and is designed for
neutral, non-spin-polarized, fixed-occupation workflows. Its topological
transport analysis requires an insulating adiabatic path: a path that crosses
a metallic region does not support the same continuous Berry-polarization
branch construction.

The insulating threshold is a numerical workflow criterion rather than a
universal physical constant. Results remain subject to the exchange-correlation
functional, pseudopotentials, Brillouin-zone sampling, basis cutoffs, and other
approximations of the underlying electronic-structure calculation.

## Documentation

- **ORBIT User Manual:** installation, project setup, configuration, complete
  command reference, warnings, edge cases, troubleshooting, recovery procedures,
  and a worked test example. *(PDF forthcoming.)*
- [`DEVELOPER_PATCH_GUIDE.md`](DEVELOPER_PATCH_GUIDE.md): guidance for modifying
  or extending ORBIT's implementation.

