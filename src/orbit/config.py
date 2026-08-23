"""Load and minimally update project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import math
import re
import tomllib

_MACHINE_PROFILE_DIR = Path(__file__).resolve().parent / "machine_profiles"
_DEFAULT_MACHINE_PROFILE = "generic_slurm"


@dataclass(frozen=True)
class HubbardUConfig:
    element: str
    manifold: str
    value_eV: float


@dataclass(frozen=True)
class HubbardConfig:
    projector: str
    u: tuple[HubbardUConfig, ...]


@dataclass(frozen=True)
class QEConfig:
    kpoints: tuple[int, int, int]
    offsets: tuple[int, int, int]
    conv_thr: float
    mixing_beta: float
    occupations: str
    nbnd: int | None
    exact_overlap_tolerance_angstrom: float
    hubbard: HubbardConfig | None


def qe_hubbard_payload(qe: QEConfig) -> dict[str, object] | None:
    if qe.hubbard is None:
        return None
    return {
        "projector": qe.hubbard.projector,
        "u": [
            {
                "element": entry.element,
                "manifold": entry.manifold,
                "value_eV": entry.value_eV,
            }
            for entry in qe.hubbard.u
        ],
    }


@dataclass(frozen=True)
class PseudopotentialConfig:
    root: Path
    files: dict[str, str]


@dataclass(frozen=True)
class SlurmConfig:
    account: str
    partition: str
    qos: str
    constraint: str
    nodes: int
    tasks_per_node: int
    cpus_per_task: int
    time: str
    max_concurrent: int
    modules: tuple[str, ...]
    module_purge: bool
    setup_commands: tuple[str, ...]
    launcher: str
    launcher_args: tuple[str, ...]
    cpu_bind: str
    environment: dict[str, str]
    pw_command: str
    ph_command: str
    pp_command: str

    @property
    def max_running(self) -> int:
        return self.max_concurrent

    @property
    def worker_nodes(self) -> int:
        return self.max_concurrent


@dataclass(frozen=True)
class PathfindingConfig:
    theta_min_eV: float
    n_images: int
    winding_padding: int


@dataclass(frozen=True)
class HelperConfig:
    target_gap_eV: float
    neighbor_cutoff_angstrom: float
    scan_axes: str
    scan_radius_angstrom: float
    scan_steps_per_axis: int
    minimum_helper_distance_angstrom: float
    taper_half_width_images: float
    improvement_tolerance_eV: float


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    path: Path
    structure: Path
    machine_profile: str
    sampling_spacing_angstrom: float
    sampling_overlap_tolerance_angstrom: float
    fold_by_symmetry: bool
    symprec_angstrom: float
    target_site_ids: tuple[str, ...]
    pathfinding: PathfindingConfig
    helper: HelperConfig
    qe: QEConfig
    pseudopotentials: PseudopotentialConfig
    slurm: SlurmConfig
    raw: dict


def _integer_triplet(value: object, name: str, *, allow_zero: bool) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain exactly three integers")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in value):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} entries must be {qualifier}")
    return tuple(value)


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _merge_mapping(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_machine_profile(profile_name: str | None) -> dict[str, object]:
    requested = profile_name or _DEFAULT_MACHINE_PROFILE
    profile_path = _MACHINE_PROFILE_DIR / f"{requested}.toml"
    if not profile_path.is_file():
        raise ValueError(f"Unknown ORBIT machine profile: {requested!r}")
    with profile_path.open("rb") as handle:
        return tomllib.load(handle)


def load_project_config(
    root: Path,
    *,
    machine_profile: str | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> ProjectConfig:
    root = root.expanduser().resolve()
    path = None
    for candidate in (root / "orbit.toml", root / "gapflow.toml"):
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            f"ORBIT configuration not found in {root}; expected orbit.toml or gapflow.toml; run orbit init first"
        )
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc

    built_in_profile = _load_machine_profile(_DEFAULT_MACHINE_PROFILE)
    project_machine = raw.get("machine", {}) if isinstance(raw.get("machine"), Mapping) else {}
    selected_profile = machine_profile or project_machine.get("profile") or _DEFAULT_MACHINE_PROFILE
    selected_profile_data = _load_machine_profile(selected_profile)
    merged_profile = _merge_mapping(built_in_profile, selected_profile_data)
    if isinstance(project_machine.get("overrides"), Mapping):
        merged_profile = _merge_mapping(merged_profile, project_machine["overrides"])

    raw_slurm = raw.get("slurm", {})
    if raw_slurm is not None and not isinstance(raw_slurm, Mapping):
        raise ValueError("[slurm] must be a TOML table")
    merged_slurm = _merge_mapping(merged_profile.get("slurm", {}), raw_slurm)
    if cli_overrides is not None:
        merged_slurm = _merge_mapping(merged_slurm, cli_overrides)

    raw_environment = raw.get("environment", {})
    if raw_environment is not None and not isinstance(raw_environment, Mapping):
        raise ValueError("[environment] must be a TOML table")
    merged_environment = _merge_mapping(merged_profile.get("environment", {}), raw_environment)
    if isinstance(merged_slurm.get("environment"), Mapping):
        merged_environment = _merge_mapping(merged_environment, merged_slurm["environment"])

    project = raw.get("project", {})
    structure_value = project.get("structure")
    if not isinstance(structure_value, str) or not structure_value.strip():
        raise ValueError(f"{path} requires a nonempty [project].structure")
    structure = Path(structure_value).expanduser()
    if not structure.is_absolute():
        structure = root / structure
    structure = structure.resolve()
    if not structure.is_file():
        raise FileNotFoundError(f"Configured structure not found: {structure}")

    sampling = raw.get("sampling", {})
    spacing = float(sampling.get("spacing_angstrom", 0.4))
    if spacing <= 0:
        raise ValueError("[sampling].spacing_angstrom must be positive")
    fold_by_symmetry = sampling.get("fold_by_symmetry", True)
    if not isinstance(fold_by_symmetry, bool):
        raise ValueError("[sampling].fold_by_symmetry must be true or false")
    symprec = float(sampling.get("symprec_angstrom", 1.0e-3))
    if symprec <= 0:
        raise ValueError("[sampling].symprec_angstrom must be positive")
    sampling_overlap_tolerance = float(
        sampling.get("overlap_tolerance_angstrom", 1.0e-5)
    )
    if sampling_overlap_tolerance <= 0:
        raise ValueError("[sampling].overlap_tolerance_angstrom must be positive")

    target_values = raw.get("targets", {}).get("site_ids", [])
    if not isinstance(target_values, list) or not all(
        isinstance(value, str) and value for value in target_values
    ):
        raise ValueError("[targets].site_ids must be an array of nonempty strings")
    if len(target_values) != len(set(target_values)):
        raise ValueError("[targets].site_ids contains duplicates")

    pathfinding_raw = raw.get("pathfinding", {})
    theta_min_eV = float(pathfinding_raw.get("theta_min_eV", 0.3))
    if theta_min_eV < 0:
        raise ValueError("[pathfinding].theta_min_eV must be nonnegative")
    n_images = _positive_integer(
        pathfinding_raw.get("n_images", 50), "[pathfinding].n_images"
    )
    if n_images < 2:
        raise ValueError("[pathfinding].n_images must be at least 2")
    winding_padding = pathfinding_raw.get("winding_padding", 1)
    if (
        not isinstance(winding_padding, int)
        or isinstance(winding_padding, bool)
        or winding_padding < 0
    ):
        raise ValueError("[pathfinding].winding_padding must be a nonnegative integer")

    helper_raw = raw.get("helper", {})
    if not isinstance(helper_raw, dict):
        raise ValueError("[helper] must be a TOML table")

    target_gap_eV = float(helper_raw.get("target_gap_eV", 0.30))
    if not math.isfinite(target_gap_eV) or target_gap_eV < 0:
        raise ValueError("[helper].target_gap_eV must be finite and nonnegative")

    neighbor_cutoff_angstrom = float(
        helper_raw.get("neighbor_cutoff_angstrom", 3.50)
    )
    if not math.isfinite(neighbor_cutoff_angstrom) or neighbor_cutoff_angstrom <= 0:
        raise ValueError("[helper].neighbor_cutoff_angstrom must be positive")

    scan_axes = _nonempty_string(
        helper_raw.get("scan_axes", "xyz"), "[helper].scan_axes"
    ).lower()
    if any(axis not in "xyz" for axis in scan_axes) or len(set(scan_axes)) != len(scan_axes):
        raise ValueError(
            "[helper].scan_axes must be a nonempty subset of xyz without repeats"
        )

    scan_radius_angstrom = float(
        helper_raw.get("scan_radius_angstrom", 0.30)
    )
    if not math.isfinite(scan_radius_angstrom) or scan_radius_angstrom < 0:
        raise ValueError("[helper].scan_radius_angstrom must be finite and nonnegative")

    scan_steps_per_axis = _positive_integer(
        helper_raw.get("scan_steps_per_axis", 5),
        "[helper].scan_steps_per_axis",
    )
    if scan_steps_per_axis % 2 == 0:
        raise ValueError("[helper].scan_steps_per_axis must be odd so zero is included")

    minimum_helper_distance_angstrom = float(
        helper_raw.get("minimum_helper_distance_angstrom", 1.20)
    )
    if (
        not math.isfinite(minimum_helper_distance_angstrom)
        or minimum_helper_distance_angstrom < 0
    ):
        raise ValueError(
            "[helper].minimum_helper_distance_angstrom must be finite and nonnegative"
        )

    taper_half_width_images = float(
        helper_raw.get("taper_half_width_images", 8.0)
    )
    if not math.isfinite(taper_half_width_images) or taper_half_width_images <= 0:
        raise ValueError("[helper].taper_half_width_images must be positive")

    improvement_tolerance_eV = float(
        helper_raw.get("improvement_tolerance_eV", 1.0e-6)
    )
    if not math.isfinite(improvement_tolerance_eV) or improvement_tolerance_eV < 0:
        raise ValueError(
            "[helper].improvement_tolerance_eV must be finite and nonnegative"
        )

    qe_raw = raw.get("qe", {})
    kpoints = _integer_triplet(qe_raw.get("kpoints", [6, 6, 6]), "[qe].kpoints", allow_zero=False)
    offsets = _integer_triplet(qe_raw.get("offsets", [0, 0, 0]), "[qe].offsets", allow_zero=True)
    if any(value not in (0, 1) for value in offsets):
        raise ValueError("[qe].offsets entries must be 0 or 1")
    conv_thr = float(qe_raw.get("conv_thr", 1.0e-10))
    if conv_thr <= 0:
        raise ValueError("[qe].conv_thr must be positive")
    mixing_beta = float(qe_raw.get("mixing_beta", 0.30))
    if not 0 < mixing_beta <= 1:
        raise ValueError("[qe].mixing_beta must lie in (0, 1]")
    occupations = _nonempty_string(
        qe_raw.get("occupations", "fixed"), "[qe].occupations"
    ).lower()
    if occupations != "fixed":
        raise ValueError(
            "ORBIT 0.4 supports only [qe].occupations = \"fixed\""
        )
    nbnd_value = qe_raw.get("nbnd")
    nbnd = None if nbnd_value is None else _positive_integer(nbnd_value, "[qe].nbnd")
    overlap_tolerance = float(
        qe_raw.get("exact_overlap_tolerance_angstrom", 1.0e-5)
    )
    if overlap_tolerance <= 0:
        raise ValueError("[qe].exact_overlap_tolerance_angstrom must be positive")

    hubbard_raw = qe_raw.get("hubbard")
    hubbard: HubbardConfig | None = None
    if hubbard_raw is not None:
        if not isinstance(hubbard_raw, dict):
            raise ValueError("[qe.hubbard] must be a TOML table")

        projector = _nonempty_string(
            hubbard_raw.get("projector", "atomic"),
            "[qe.hubbard].projector",
        ).lower()
        allowed_projectors = {"atomic", "ortho-atomic", "norm-atomic", "wf", "pseudo"}
        if projector not in allowed_projectors:
            choices = ", ".join(sorted(allowed_projectors))
            raise ValueError(f"[qe.hubbard].projector must be one of: {choices}")

        entries_raw = hubbard_raw.get("u")
        if not isinstance(entries_raw, list) or not entries_raw:
            raise ValueError("[qe.hubbard] requires at least one [[qe.hubbard.u]] entry")

        entries: list[HubbardUConfig] = []
        seen: set[tuple[str, str]] = set()
        for index, entry_raw in enumerate(entries_raw):
            label = f"[[qe.hubbard.u]] entry {index + 1}"
            if not isinstance(entry_raw, dict):
                raise ValueError(f"{label} must be a TOML table")

            element = _nonempty_string(entry_raw.get("element"), f"{label}.element")
            if re.fullmatch(r"[A-Z][a-z]?", element) is None:
                raise ValueError(f"{label}.element must be a canonical element symbol")

            manifold = _nonempty_string(entry_raw.get("manifold"), f"{label}.manifold")
            if any(ch.isspace() for ch in manifold):
                raise ValueError(f"{label}.manifold may not contain whitespace")

            value_raw = entry_raw.get("value_eV")
            if isinstance(value_raw, bool) or not isinstance(value_raw, (int, float)):
                raise ValueError(f"{label}.value_eV must be numeric")
            value_eV = float(value_raw)
            if not math.isfinite(value_eV) or value_eV < 0:
                raise ValueError(f"{label}.value_eV must be finite and nonnegative")

            key = (element, manifold)
            if key in seen:
                raise ValueError(f"Duplicate Hubbard U entry for {element}-{manifold}")
            seen.add(key)
            entries.append(HubbardUConfig(element, manifold, value_eV))

        hubbard = HubbardConfig(projector=projector, u=tuple(entries))

    pseudo_raw = raw.get("pseudopotentials", {})
    pseudo_root_value = _nonempty_string(
        pseudo_raw.get("root", "../pseudo"), "[pseudopotentials].root"
    )
    pseudo_root = Path(pseudo_root_value).expanduser()
    if not pseudo_root.is_absolute():
        pseudo_root = root / pseudo_root
    files_raw = pseudo_raw.get("files", {})
    if not isinstance(files_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in files_raw.items()
    ):
        raise ValueError("[pseudopotentials.files] must map elements to filenames")

    modules_raw = merged_slurm.get(
        "modules",
        [
            "intel/2023.2.1",
            "impi/2021.10.0",
            "mkl/2023.2.0",
            "espresso/7.1.0-intel-mkl",
        ],
    )
    if not isinstance(modules_raw, list) or not all(
        isinstance(module, str) and module.strip() for module in modules_raw
    ):
        raise ValueError("[slurm].modules must be an array of nonempty strings")

    setup_commands_raw = merged_slurm.get("setup_commands", [])
    if not isinstance(setup_commands_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in setup_commands_raw
    ):
        raise ValueError("[slurm].setup_commands must be an array of nonempty strings")

    launch_args_raw = merged_slurm.get("launcher_args", [])
    if not isinstance(launch_args_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in launch_args_raw
    ):
        raise ValueError("[slurm].launcher_args must be an array of nonempty strings")

    environment_raw = merged_environment
    if not isinstance(environment_raw, Mapping):
        raise ValueError("[environment] must be a TOML table")
    environment = {str(key): str(value) for key, value in environment_raw.items()}

    account_value = merged_slurm.get("account", "")
    if account_value is None:
        account_value = ""
    if not isinstance(account_value, str):
        raise ValueError("[slurm].account must be a string")
    partition = _nonempty_string(merged_slurm.get("partition", "regular"), "[slurm].partition")
    constraint_value = merged_slurm.get("constraint", "")
    qos_value = merged_slurm.get("qos", "")
    constraint = "" if constraint_value is None else str(constraint_value).strip()
    qos = "" if qos_value is None else str(qos_value).strip()
    max_running_raw = merged_slurm.get(
        "max_running",
        merged_slurm.get("worker_nodes", merged_slurm.get("max_concurrent", 6)),
    )
    if isinstance(max_running_raw, bool) or not isinstance(max_running_raw, int) or max_running_raw <= 0:
        raise ValueError("[slurm].max_running must be a positive integer")

    return ProjectConfig(
        root=root,
        path=path,
        structure=structure,
        machine_profile=selected_profile,
        sampling_spacing_angstrom=spacing,
        sampling_overlap_tolerance_angstrom=sampling_overlap_tolerance,
        fold_by_symmetry=fold_by_symmetry,
        symprec_angstrom=symprec,
        target_site_ids=tuple(target_values),
        pathfinding=PathfindingConfig(
            theta_min_eV=theta_min_eV,
            n_images=n_images,
            winding_padding=winding_padding,
        ),
        helper=HelperConfig(
            target_gap_eV=target_gap_eV,
            neighbor_cutoff_angstrom=neighbor_cutoff_angstrom,
            scan_axes=scan_axes,
            scan_radius_angstrom=scan_radius_angstrom,
            scan_steps_per_axis=scan_steps_per_axis,
            minimum_helper_distance_angstrom=minimum_helper_distance_angstrom,
            taper_half_width_images=taper_half_width_images,
            improvement_tolerance_eV=improvement_tolerance_eV,
        ),
        qe=QEConfig(
            kpoints=kpoints,
            offsets=offsets,
            conv_thr=conv_thr,
            mixing_beta=mixing_beta,
            occupations=occupations,
            nbnd=nbnd,
            exact_overlap_tolerance_angstrom=overlap_tolerance,
            hubbard=hubbard,
        ),
        pseudopotentials=PseudopotentialConfig(
            root=pseudo_root.resolve(),
            files={key: value.strip() for key, value in files_raw.items()},
        ),
        slurm=SlurmConfig(
            account=account_value,
            partition=partition,
            qos=qos,
            constraint=constraint,
            nodes=_positive_integer(merged_slurm.get("nodes", 1), "[slurm].nodes"),
            tasks_per_node=_positive_integer(
                merged_slurm.get("tasks_per_node", 48), "[slurm].tasks_per_node"
            ),
            cpus_per_task=_positive_integer(
                merged_slurm.get("cpus_per_task", 1), "[slurm].cpus_per_task"
            ),
            time=_nonempty_string(merged_slurm.get("time", "02:00:00"), "[slurm].time"),
            max_concurrent=max_running_raw,
            modules=tuple(module.strip() for module in modules_raw),
            module_purge=bool(merged_slurm.get("module_purge", False)),
            setup_commands=tuple(item.strip() for item in setup_commands_raw),
            launcher=_nonempty_string(
                merged_slurm.get("launcher", "mpirun"), "[slurm].launcher"
            ),
            launcher_args=tuple(item.strip() for item in launch_args_raw),
            cpu_bind=str(merged_slurm.get("cpu_bind", "")).strip(),
            environment=environment,
            pw_command=_nonempty_string(
                merged_slurm.get("pw_command", merged_slurm.get("pw", "pw.x")), "[slurm].pw_command"
            ),
            ph_command=_nonempty_string(
                merged_slurm.get("ph_command", merged_slurm.get("ph", "ph.x")), "[slurm].ph_command"
            ),
            pp_command=_nonempty_string(
                merged_slurm.get("pp_command", merged_slurm.get("pp", "pp.x")), "[slurm].pp_command"
            ),
        ),
        raw=raw,
    )


def set_target_site_ids(config_path: Path, site_ids: list[str]) -> None:
    """Replace only the `[targets]` section while preserving the rest verbatim."""
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("target site IDs must be unique")
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    header_pattern = re.compile(r"^\s*\[[^\]]+\]\s*(?:#.*)?$")
    target_pattern = re.compile(r"^\s*\[targets\]\s*(?:#.*)?$")

    start = next(
        (index for index, line in enumerate(lines) if target_pattern.match(line.rstrip("\n"))),
        None,
    )
    end = None
    if start is not None:
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if header_pattern.match(lines[index].rstrip("\n"))
            ),
            len(lines),
        )

    encoded = ", ".join(f'"{value}"' for value in site_ids)
    replacement = ["[targets]\n", f"site_ids = [{encoded}]\n", "\n"]
    if start is None:
        if text and not text.endswith("\n"):
            lines.append("\n")
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend(replacement)
    else:
        lines[start:end] = replacement
    config_path.write_text("".join(lines), encoding="utf-8")
