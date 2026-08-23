"""Project layout creation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIRECTORIES = (
    "manifests",
    "samples",
    "calculations/samples",
    "calculations/paths",
    "calculations/helpers",
    "results/gaps",
    "results/helper_gaps",
    "paths",
    "helpers",
    "plots",
    "plots/helpers",
    "logs",
)


@dataclass(frozen=True)
class ProjectLayout:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "orbit.toml"

    @property
    def paths(self) -> Path:
        return self.root / "paths"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def structure_manifest(self) -> Path:
        return self.manifests / "structure.json"

    @property
    def sites_manifest(self) -> Path:
        return self.manifests / "sites.csv"

    @property
    def targets_manifest(self) -> Path:
        return self.manifests / "targets.csv"

    def target_paths(self, target_id: str) -> Path:
        return self.paths / target_id


def initial_config(structure: Path, *, machine: str | None = None) -> str:
    profile_name = machine if machine is not None else "generic_slurm"
    return f'''schema_version = 1

[project]
structure = "{structure.as_posix()}"

[machine]
profile = "{profile_name}"

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

[helper]
target_gap_eV = 0.30
neighbor_cutoff_angstrom = 3.50
scan_axes = "xyz"
scan_radius_angstrom = 0.30
scan_steps_per_axis = 5
minimum_helper_distance_angstrom = 1.20
taper_half_width_images = 8.0
improvement_tolerance_eV = 1.0e-6

[qe]
kpoints = [6, 6, 6]
offsets = [0, 0, 0]
conv_thr = 1.0e-10
mixing_beta = 0.30
occupations = "fixed"
# Optional explicit override; otherwise ORBIT derives nbnd from UPF valence.
# nbnd = 16
exact_overlap_tolerance_angstrom = 1.0e-5

# Settings for `orbit path-ph`, `orbit path-polarization`, and `path-response`.
[qe.ph]
qpoint = [0.0, 0.0, 0.0]

[qe.ph.inputph]
tr2_ph = 1.0e-14
epsil = true
zeu = true
trans = false

[qe.polarization]
gdirs = [1, 2, 3]
# Fallback grid when a direction-specific grid is omitted.
kpoints = [6, 6, 6]
kpoints_gdir1 = [12, 6, 6]
kpoints_gdir2 = [6, 12, 6]
kpoints_gdir3 = [6, 6, 12]
offsets = [0, 0, 0]
nppstr = [12, 12, 12]

[qe.polarization.control]
restart_mode = "from_scratch"
verbosity = "high"

[qe.polarization.electrons]
conv_thr = 1.0e-10

[response]
ph_command = "ph.x"
pw_command = "pw.x"

[response.analysis]
# Derive each physical quantum e|a_i|/Omega from the cell. Set a positive
# number only to reproduce printed-QE-modulo division explicitly.
polarization_quantum_divisor = "auto"
branch_range = 4
initial_branch = [0, 0, 0]

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
    "intel/2023.2.1",
    "impi/2021.10.0",
    "mkl/2023.2.0",
    "espresso/7.1.0-intel-mkl",
]
launcher = "mpirun"
pw_command = "pw.x"
'''


def initialize_project(
    root: Path,
    structure: Path,
    *,
    force_config: bool = False,
    machine: str | None = None,
) -> tuple[ProjectLayout, bool]:
    """Create the stable layout and, if absent, a starter configuration."""
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    layout = ProjectLayout(root)
    for relative in PROJECT_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)

    config_written = force_config or not layout.config.exists()
    if config_written:
        layout.config.write_text(initial_config(structure, machine=machine), encoding="utf-8")
    return layout, config_written
