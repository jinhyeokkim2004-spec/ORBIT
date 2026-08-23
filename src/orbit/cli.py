"""ORBIT command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from . import __version__
from .config import load_project_config, set_target_site_ids
from .gaps import extract_targets
from .helpers import (
    analyze_helper_scan,
    extract_helper_path,
    helper_path_scf,
    helper_status,
    prepare_helper_scan,
    resolve_helper_run_id,
    submit_helper_scan,
    write_helper_plot,
)
from .pathfinding import construct_straight_paths, find_paths
from .path_gaps import extract_path_targets
from .path_plotting import write_path_viewer
from .plotting import write_gap_heatmap
from .project import ProjectLayout, initialize_project
from .sampling import (
    build_sampling_plans,
    sampling_output_state,
    write_sampling_plan,
)
from .qe.calculations import (
    build_prepared_targets,
    calculation_output_state,
    write_prepared_target,
)
from .qe.resources import resolve_qe_resources
from .qe.path_calculations import (
    build_prepared_path_target,
    path_calculation_output_state,
    write_prepared_path_target,
)
from .qe.path_response import prepare_path_response
from .qe.path_response_analysis import analyze_path_response
from .scheduler.slurm import render_array_script, render_submit_script
from .structure import (
    SiteRecord,
    inspect_structure,
    load_site_records,
    validate_manifest_source,
    write_structure_manifests,
)
from .targets import select_targets, write_targets_manifest


def _root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project directory (default: current directory)",
    )
    parser.add_argument(
        "--machine-profile",
        default=None,
        help="override the built-in scheduler profile for this run (e.g. generic_slurm, perlmutter, whoville3)",
    )


def _add_path_response_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_kind: bool,
) -> None:
    _root_argument(parser)
    parser.add_argument("--target", required=True, metavar="SITE_ID")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source", choices=("path", "helper"), default="path",
        help="use a constructed path or a helper-refined path (default: path)",
    )
    parser.add_argument(
        "--iteration", type=int,
        help="helper iteration; default: latest (only with --source helper)",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--submit", action="store_true")
    if include_kind:
        parser.add_argument(
            "--kind", choices=("all", "ph", "polarization"), default="all",
            help="calculation family to submit (default: all)",
        )
    parser.add_argument("--max-concurrent", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbit",
        description="ORBIT: Crystal sampling and insulating-path workflow",
    )
    parser.add_argument("--version", action="version", version=f"orbit {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    machine_parser = commands.add_parser(
        "machine",
        help="show the effective machine profile and scheduler settings",
    )
    machine_commands = machine_parser.add_subparsers(dest="machine_command", required=True)
    machine_show = machine_commands.add_parser("show", help="show resolved machine profile")
    _root_argument(machine_show)

    doctor_parser = commands.add_parser(
        "doctor",
        help="show the effective machine profile and scheduler settings",
    )
    _root_argument(doctor_parser)

    init_parser = commands.add_parser(
        "init",
        help="create the project layout without running calculations",
    )
    _root_argument(init_parser)
    init_parser.add_argument(
        "--structure",
        type=Path,
        required=True,
        help="structure path to store in orbit.toml, relative to --root or absolute",
    )
    init_parser.add_argument(
        "--force-config",
        action="store_true",
        help="replace an existing orbit.toml; other files are never removed",
    )
    init_parser.add_argument(
        "--machine",
        choices=("generic_slurm", "perlmutter", "whoville3"),
        default=None,
        help="preset a machine profile in the new project config",
    )

    inspect_parser = commands.add_parser(
        "inspect",
        help="inspect the configured CIF and write stable site identities",
    )
    _root_argument(inspect_parser)

    targets_parser = commands.add_parser(
        "targets",
        help="list or explicitly select displacement targets",
    )
    target_commands = targets_parser.add_subparsers(dest="targets_command", required=True)
    target_list = target_commands.add_parser("list", help="list inspected sites and target status")
    _root_argument(target_list)
    target_set = target_commands.add_parser("set", help="replace the configured target selection")
    _root_argument(target_set)
    selection = target_set.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all-inequivalent",
        action="store_true",
        help="select one representative from every crystallographic site orbit",
    )
    selection.add_argument(
        "--elements",
        nargs="+",
        metavar="ELEMENT",
        help="select all inequivalent sites of the listed elements",
    )
    selection.add_argument(
        "--sites",
        nargs="+",
        metavar="SITE_ID",
        help="select explicit stable site IDs in the given order",
    )

    sample_parser = commands.add_parser(
        "sample",
        help="build symmetry-reduced one-atom displacement samples",
    )
    _root_argument(sample_parser)
    sample_parser.add_argument(
        "--target",
        action="append",
        metavar="SITE_ID",
        help="sample one configured target; repeat to select several (default: all)",
    )
    sample_parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and report counts without writing sample files",
    )
    sample_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing target sampling dataset",
    )

    scf_parser = commands.add_parser(
        "scf",
        help="prepare and optionally submit QE SCFs for sample datasets",
    )
    _root_argument(scf_parser)
    scf_parser.add_argument(
        "--target",
        action="append",
        metavar="SITE_ID",
        help="prepare one configured target; repeat for several (default: all)",
    )
    scf_parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate resources and every input without writing files",
    )
    scf_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="refresh conflicting inputs while preserving stale outputs by rename",
    )
    scf_parser.add_argument(
        "--submit",
        action="store_true",
        help="prepare inputs, then sbatch all unfinished valid calculations",
    )
    scf_parser.add_argument(
        "--max-concurrent",
        type=int,
        help="override [slurm].max_concurrent for these target arrays",
    )

    extract_parser = commands.add_parser(
        "extract",
        help="extract auditable HOMO/LUMO gap tables from QE outputs",
    )
    _root_argument(extract_parser)
    extract_parser.add_argument(
        "--target",
        action="append",
        metavar="SITE_ID",
        help="extract one configured target; repeat for several (default: all)",
    )

    path_parser = commands.add_parser(
        "path",
        help="find maximum-bottleneck winding paths on extracted sample gaps",
    )
    _root_argument(path_parser)
    path_parser.add_argument(
        "--target",
        action="append",
        metavar="SITE_ID",
        help="find a path for one target; repeat for several (default: all)",
    )
    path_parser.add_argument(
        "--mode",
        choices=("maximin", "straight"),
        default="maximin",
        help="path construction mode (default: maximin)",
    )
    path_parser.add_argument(
        "--destination",
        choices=("a", "b", "c", "ab", "bc", "ac", "abc"),
        help="periodic destination for --mode straight",
    )
    path_parser.add_argument(
        "--theta-min",
        type=float,
        metavar="EV",
        help="override [pathfinding].theta_min_eV",
    )
    path_parser.add_argument(
        "--n-images",
        type=int,
        help="override [pathfinding].n_images",
    )
    path_parser.add_argument(
        "--winding-padding",
        type=int,
        help="override [pathfinding].winding_padding",
    )
    path_parser.add_argument(
        "--no-cifs",
        action="store_true",
        help="write path records without path-image CIF structures",
    )

    path_scf_parser = commands.add_parser(
        "path-scf",
        help="prepare and optionally submit QE SCFs for constructed path images",
    )
    _root_argument(path_scf_parser)
    path_scf_parser.add_argument(
        "--target",
        action="append",
        metavar="SITE_ID",
        help="prepare one target path; repeat for several (default: all configured)",
    )
    path_scf_parser.add_argument(
        "--run-id",
        help="explicit path run ID; allowed only with one --target",
    )
    path_scf_parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the path, resources, and inputs without writing files",
    )
    path_scf_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="refresh conflicting inputs while preserving stale outputs by rename",
    )
    path_scf_parser.add_argument(
        "--submit",
        action="store_true",
        help="prepare inputs, then sbatch unfinished valid path images",
    )
    path_scf_parser.add_argument(
        "--max-concurrent",
        type=int,
        help="override [slurm].max_concurrent for these arrays",
    )

    path_response_parser = commands.add_parser(
        "path-response",
        help="prepare and optionally submit PH and Berry-polarization jobs for a final path",
    )
    _add_path_response_arguments(path_response_parser, include_kind=True)
    path_response_parser.set_defaults(separate_response_root=False)

    path_ph_parser = commands.add_parser(
        "path-ph",
        help="prepare and optionally submit only ph.x jobs for a final path",
    )
    _add_path_response_arguments(path_ph_parser, include_kind=False)
    path_ph_parser.set_defaults(
        response_kind="ph",
        separate_response_root=True,
    )

    path_polarization_parser = commands.add_parser(
        "path-polarization",
        help="prepare and optionally submit only Berry-polarization jobs for a final path",
    )
    _add_path_response_arguments(path_polarization_parser, include_kind=False)
    path_polarization_parser.set_defaults(
        response_kind="polarization",
        separate_response_root=True,
    )

    path_response_plot_parser = commands.add_parser(
        "path-response-plot",
        help="extract Berry branches and plot polarization and transported charge N",
    )
    _root_argument(path_response_plot_parser)
    path_response_plot_parser.add_argument("--target", required=True, metavar="SITE_ID")
    path_response_plot_parser.add_argument("--run-id", required=True)
    path_response_plot_parser.add_argument(
        "--source", choices=("path", "helper"), default="path",
        help="analyze a constructed path or helper-refined path (default: path)",
    )
    path_response_plot_parser.add_argument(
        "--iteration", type=int,
        help="helper iteration; default: latest (only with --source helper)",
    )
    path_response_plot_parser.add_argument(
        "--quantum-divisor", type=float,
        help="override [response.analysis].polarization_quantum_divisor",
    )
    path_response_plot_parser.add_argument(
        "--branch-range", type=int,
        help="override [response.analysis].branch_range",
    )

    path_extract_parser = commands.add_parser(
        "path-extract",
        help="extract auditable HOMO/LUMO gaps from path-image QE outputs",
    )
    _root_argument(path_extract_parser)
    path_extract_parser.add_argument(
        "--target", action="append", metavar="SITE_ID",
        help="extract one target path; repeat for several (default: all configured)",
    )
    path_extract_parser.add_argument(
        "--run-id", help="explicit path run ID; allowed only with one --target"
    )

    plot_path_parser = commands.add_parser(
        "plot-path",
        help="build a combined sampled heatmap and calculated-path viewer",
    )
    _root_argument(plot_path_parser)
    plot_path_parser.add_argument(
        "--target", action="append", metavar="SITE_ID",
        help="plot one target; repeat for several (default: all configured)",
    )
    plot_path_parser.add_argument(
        "--run-id", help="explicit path run ID; allowed only with one --target"
    )
    plot_path_parser.add_argument("--output", type=Path, help="HTML output path")
    plot_path_parser.add_argument(
        "--representatives-only", action="store_true",
        help="hide symmetry-equivalent sampled heatmap points",
    )

    helper_parser = commands.add_parser(
        "helper",
        help="iteratively add one best neighboring helper atom to a calculated path",
    )
    helper_commands = helper_parser.add_subparsers(
        dest="helper_command", required=True
    )

    helper_scan = helper_commands.add_parser(
        "scan",
        help="select the current minimum-gap image and prepare individual helper grids",
    )
    _root_argument(helper_scan)
    helper_scan.add_argument("--target", required=True, metavar="SITE_ID")
    helper_scan.add_argument("--run-id", help="optional parent path run ID; default: current helper campaign or newest constructed path run")
    helper_scan.add_argument("--neighbor-cutoff", type=float, metavar="A")
    helper_scan.add_argument("--radius", type=float, metavar="A")
    helper_scan.add_argument("--steps", type=int)
    helper_scan.add_argument("--axes")
    helper_scan.add_argument(
        "--helper-site",
        action="append",
        metavar="SITE_ID",
        help="scan an explicit helper site; repeat for several (default: cutoff neighbors)",
    )
    helper_scan.add_argument(
        "--check-only",
        action="store_true",
        help="report bottleneck, candidates, and nominal scan size without writing inputs",
    )
    helper_scan.add_argument(
        "--submit",
        action="store_true",
        help="prepare the scan and submit all surviving nonzero scan points",
    )
    helper_scan.add_argument(
        "--max-concurrent",
        type=int,
        help="override [slurm].max_concurrent for this helper scan",
    )
    helper_scan.add_argument(
        "--force",
        action="store_true",
        help="bypass the convergence/manual-review gate intentionally",
    )

    helper_analyze = helper_commands.add_parser(
        "analyze",
        help="rank completed individual scans, choose one helper, and build the helper path",
    )
    _root_argument(helper_analyze)
    helper_analyze.add_argument("--target", required=True, metavar="SITE_ID")
    helper_analyze.add_argument("--run-id", help="optional parent path run ID; default: current helper campaign or newest constructed path run")
    helper_analyze.add_argument(
        "--allow-nonimproving",
        action="store_true",
        help="permit a non-improving helper winner intentionally",
    )
    helper_analyze.add_argument(
        "--max-concurrent",
        type=int,
        help="override [slurm].max_concurrent for the generated helper path",
    )

    helper_path = helper_commands.add_parser(
        "path-scf",
        help="validate and optionally submit the latest generated helper path",
    )
    _root_argument(helper_path)
    helper_path.add_argument("--target", required=True, metavar="SITE_ID")
    helper_path.add_argument("--run-id", help="optional parent path run ID; default: current helper campaign or newest constructed path run")
    helper_path.add_argument("--submit", action="store_true")

    helper_extract = helper_commands.add_parser(
        "extract",
        help="extract the latest helper-path gaps and test the configured threshold",
    )
    _root_argument(helper_extract)
    helper_extract.add_argument("--target", required=True, metavar="SITE_ID")
    helper_extract.add_argument("--run-id", help="optional parent path run ID; default: current helper campaign or newest constructed path run")

    helper_plot = helper_commands.add_parser(
        "plot",
        help="plot target-only, previous, and current helper-path calculated gaps",
    )
    _root_argument(helper_plot)
    helper_plot.add_argument("--target", required=True, metavar="SITE_ID")
    helper_plot.add_argument("--run-id", help="optional parent path run ID; default: current helper campaign or newest constructed path run")
    helper_plot.add_argument("--output", type=Path)

    helper_status_parser = helper_commands.add_parser(
        "status",
        help="show helper iterations, convergence, review state, and next action",
    )
    _root_argument(helper_status_parser)
    helper_status_parser.add_argument("--target", required=True, metavar="SITE_ID")
    helper_status_parser.add_argument("--run-id", help="optional parent path run ID; default: current helper campaign or newest constructed path run")

    plot_parser = commands.add_parser(
        "plot",
        help="build a standalone interactive sampled-gap heatmap",
    )
    _root_argument(plot_parser)
    plot_parser.add_argument(
        "--target",
        action="append",
        metavar="SITE_ID",
        help="plot one extracted target; repeat for several (default: all)",
    )
    plot_parser.add_argument(
        "--output",
        type=Path,
        help="HTML output path, relative to the project unless absolute",
    )
    plot_parser.add_argument(
        "--representatives-only",
        action="store_true",
        help="hide symmetry-equivalent images of calculated representatives",
    )
    return parser


def _machine(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    print(f"Machine: {config.machine_profile}")
    print("Scheduler: slurm")
    print(f"Account: {config.slurm.account or '(unset)'}")
    print(f"Constraint: {config.slurm.constraint or '(unset)'}")
    print(f"QoS: {config.slurm.qos or '(unset)'}")
    print(f"Nodes/calculation: {config.slurm.nodes}")
    print(f"MPI tasks/node: {config.slurm.tasks_per_node}")
    print(f"CPUs/task: {config.slurm.cpus_per_task}")
    print(f"Launcher: {config.slurm.launcher}")
    print(f"CPU bind: {config.slurm.cpu_bind or '(unset)'}")
    print(f"QE pw: {config.slurm.pw_command}")
    print(f"QE ph: {config.slurm.ph_command}")
    print(f"QE pp: {config.slurm.pp_command}")
    print(f"Module purge: {'enabled' if config.slurm.module_purge else 'disabled'}")
    print(f"Max simultaneous jobs: {config.slurm.max_concurrent}")
    return 0


def _print_sites(sites: list[SiteRecord], selected: set[str]) -> None:
    headers = ("TARGET", "SITE_ID", "EL", "CIF_LABEL", "ASE_INDEX", "ORBIT", "REP", "FRAC")
    rows = []
    for site in sites:
        rows.append(
            (
                "yes" if site.site_id in selected else "",
                site.site_id,
                site.element,
                site.cif_label or "-",
                str(site.ase_index_zero_based),
                f"{site.symmetry_orbit_id}({site.orbit_size})",
                "yes" if site.is_orbit_representative else "",
                f"({site.frac_x:.6f}, {site.frac_y:.6f}, {site.frac_z:.6f})",
            )
        )
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _inspect(root: Path, *, machine_profile: str | None = None) -> int:
    config = load_project_config(root, machine_profile=machine_profile)
    layout = ProjectLayout(config.root)
    inspection = inspect_structure(config.structure, config.symprec_angstrom)
    structure_path, sites_path = write_structure_manifests(
        inspection, layout.manifests, config.root
    )
    print(
        f"[structure] {inspection.formula}; {len(inspection.sites)} atoms; "
        f"volume={inspection.volume_A3:.6f} A^3"
    )
    print(
        f"[symmetry] #{inspection.space_group_number} "
        f"{inspection.space_group_symbol}; "
        f"{inspection.symmetry_operation_count} operations at "
        f"symprec={inspection.symprec_angstrom:g} A"
    )
    _print_sites(list(inspection.sites), set(config.target_site_ids))
    print(f"[done] wrote {structure_path}")
    print(f"[done] wrote {sites_path}")
    if not config.target_site_ids:
        print("[next] no targets selected; run: orbit targets set --all-inequivalent")
    return 0


def _load_current_sites(root: Path, *, machine_profile: str | None = None):
    config = load_project_config(root, machine_profile=machine_profile)
    layout = ProjectLayout(config.root)
    validate_manifest_source(layout.structure_manifest, config.structure)
    return config, layout, load_site_records(layout.sites_manifest)


def _list_targets(root: Path, *, machine_profile: str | None = None) -> int:
    config, _layout, sites = _load_current_sites(root, machine_profile=machine_profile)
    known = {site.site_id for site in sites}
    unknown = [site_id for site_id in config.target_site_ids if site_id not in known]
    if unknown:
        raise ValueError(
            "Configured targets are absent from the current site manifest: "
            + ", ".join(unknown)
        )
    _print_sites(sites, set(config.target_site_ids))
    print(
        "[targets] "
        + (", ".join(config.target_site_ids) if config.target_site_ids else "none selected")
    )
    return 0


def _set_targets(args: argparse.Namespace) -> int:
    config, layout, sites = _load_current_sites(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    chosen, mode = select_targets(
        sites,
        all_inequivalent=args.all_inequivalent,
        elements=args.elements,
        site_ids=args.sites,
    )
    if not chosen:
        raise ValueError("Target selection is empty")
    chosen_ids = [site.site_id for site in chosen]
    write_targets_manifest(layout.targets_manifest, chosen, mode)
    set_target_site_ids(config.path, chosen_ids)
    for site_id in chosen_ids:
        (layout.paths / site_id / "runs").mkdir(parents=True, exist_ok=True)
        (config.root / "samples" / site_id).mkdir(parents=True, exist_ok=True)
        (config.root / "calculations" / "samples" / site_id).mkdir(parents=True, exist_ok=True)
    print(f"[targets] selected {len(chosen_ids)}: {', '.join(chosen_ids)}")
    print(f"[done] updated {config.path}")
    print(f"[done] wrote {layout.targets_manifest}")
    return 0


def _sample(args: argparse.Namespace) -> int:
    if args.check_only and args.overwrite:
        raise ValueError("--check-only and --overwrite cannot be combined")
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    layout = ProjectLayout(config.root)
    atoms, plans = build_sampling_plans(config, layout, args.target)
    print(
        f"[sampling] spacing={config.sampling_spacing_angstrom:g} A; "
        f"symmetry_folding={'on' if config.fold_by_symmetry else 'off'}; "
        f"direct_overlap_tolerance={config.sampling_overlap_tolerance_angstrom:g} A"
    )
    for plan in plans:
        print(
            f"[target {plan.target.site_id}] grid={list(plan.divisions)}; "
            f"raw={plan.raw_grid_count}; site_stabilizer={plan.site_stabilizer_order}; "
            f"inequivalent={plan.representative_count}; "
            f"valid={plan.valid_count}; direct_overlap={plan.invalid_count}; "
            f"reduction={plan.reduction_factor:.3f}"
        )
    if args.check_only:
        print(
            f"[check only] validated {len(plans)} targets and "
            f"{sum(plan.valid_count for plan in plans)} valid structures and "
            f"{sum(plan.invalid_count for plan in plans)} overlap walls; "
            "wrote nothing"
        )
        return 0

    states = {
        plan.target.site_id: sampling_output_state(config, layout, plan)
        for plan in plans
    }
    conflicts = [
        site_id for site_id, state in states.items()
        if state == "conflict" and not args.overwrite
    ]
    if conflicts:
        raise FileExistsError(
            "Conflicting sample output exists for "
            + ", ".join(conflicts)
            + "; use --overwrite to replace it"
        )
    for plan in plans:
        output, action = write_sampling_plan(
            config, layout, atoms, plan, overwrite=args.overwrite
        )
        if action == "skipped_identical":
            print(f"[skip] {plan.target.site_id}: identical complete dataset at {output}")
        else:
            print(
                f"[done] {plan.target.site_id}: wrote "
                f"{plan.valid_count} structures, {plan.representative_count} "
                f"manifest rows, and {plan.invalid_count} overlap walls to {output}"
            )
    return 0


def _scf(args: argparse.Namespace) -> int:
    if args.check_only and args.submit:
        raise ValueError("--check-only and --submit cannot be combined")
    if args.check_only and args.overwrite:
        raise ValueError("--check-only and --overwrite cannot be combined")
    if args.max_concurrent is not None and args.max_concurrent <= 0:
        raise ValueError("--max-concurrent must be positive")
    config = load_project_config(args.root)
    layout = ProjectLayout(config.root)
    atoms, plans = build_sampling_plans(config, layout, args.target)
    resources = resolve_qe_resources(config, list(atoms.get_chemical_symbols()))
    max_concurrent = (
        config.slurm.max_concurrent
        if args.max_concurrent is None
        else args.max_concurrent
    )
    prepared_targets = build_prepared_targets(
        config, layout, plans, resources, max_concurrent
    )
    for prepared in prepared_targets:
        render_array_script(prepared.plan.target.site_id, config.slurm)
        render_submit_script(prepared.plan.target.site_id, max_concurrent)

    print(f"[pseudos] directory={resources.pseudo_dir}")
    for item in resources.elements:
        print(
            f"[pseudo {item.element}] {item.pseudopotential}; "
            f"z_valence={item.z_valence:g}; wfc={item.cutoff_wfc_Ry:g} Ry; "
            f"rho={item.cutoff_rho_Ry:g} Ry"
        )
    print(
        f"[qe] ecutwfc={resources.ecutwfc_Ry:g} Ry; "
        f"ecutrho={resources.ecutrho_Ry:g} Ry; nbnd={resources.nbnd} "
        f"({resources.nbnd_source}); kpoints={list(config.qe.kpoints)}; "
        f"offsets={list(config.qe.offsets)}"
    )
    for prepared in prepared_targets:
        print(
            f"[target {prepared.plan.target.site_id}] "
            f"valid={prepared.valid_count}; "
            f"invalid_geometry={prepared.invalid_count}; "
            f"max_concurrent={prepared.max_concurrent}"
        )
    if args.check_only:
        print(
            f"[check only] validated {len(prepared_targets)} targets and "
            f"{sum(item.valid_count for item in prepared_targets)} valid QE inputs; "
            "wrote nothing"
        )
        return 0

    states = {
        item.plan.target.site_id: calculation_output_state(layout, item)
        for item in prepared_targets
    }
    conflicts = [
        target_id
        for target_id, state in states.items()
        if state == "conflict" and not args.overwrite
    ]
    if conflicts:
        raise FileExistsError(
            "Conflicting QE preparation exists for "
            + ", ".join(conflicts)
            + "; use --overwrite to refresh inputs"
        )

    output_roots: list[Path] = []
    for prepared in prepared_targets:
        output, action = write_prepared_target(
            config,
            layout,
            resources,
            prepared,
            overwrite=args.overwrite,
        )
        output_roots.append(output)
        if action == "skipped_identical":
            print(
                f"[skip] {prepared.plan.target.site_id}: identical complete QE "
                f"preparation at {output}"
            )
        else:
            print(
                f"[done] {prepared.plan.target.site_id}: prepared "
                f"{prepared.valid_count} QE calculations at {output}"
            )

    if args.submit:
        for prepared, output in zip(prepared_targets, output_roots):
            print(f"[submit] {prepared.plan.target.site_id}: bash {output / 'submit.sh'}")
            result = subprocess.run(
                ["bash", str(output / "submit.sh")],
                cwd=output,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Submission failed for {prepared.plan.target.site_id} "
                    f"with exit status {result.returncode}"
                )
    return 0


def _extract(args: argparse.Namespace) -> int:
    config = load_project_config(args.root)
    layout = ProjectLayout(config.root)
    results = extract_targets(config, layout, args.target)
    for result in results:
        summary = ", ".join(
            f"{status}={count}" for status, count in result.status_counts.items()
        )
        print(f"[target {result.target_id}] total={result.total_count}; {summary}")
        print(f"[done] wrote {result.gaps_csv}")
        print(f"[done] wrote {result.manifest}")
    return 0


def _plot(args: argparse.Namespace) -> int:
    config = load_project_config(args.root)
    layout = ProjectLayout(config.root)
    output, manifest, payload = write_gap_heatmap(
        config,
        layout,
        args.target,
        args.output,
        representatives_only=args.representatives_only,
    )
    print(
        f"[plot] targets={','.join(item['target_id'] for item in payload['targets'])}; "
        f"color_range_eV={payload['shared_color_range_eV']}"
    )
    print(f"[done] wrote {output}")
    print(f"[done] wrote {manifest}")
    return 0


def _path(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    layout = ProjectLayout(config.root)

    if args.mode == "straight":
        if args.destination is None:
            raise ValueError("--mode straight requires --destination")
        if args.theta_min is not None:
            raise ValueError("--theta-min is not used with --mode straight")
        if args.winding_padding is not None:
            raise ValueError("--winding-padding is not used with --mode straight")
        results = construct_straight_paths(
            config,
            layout,
            args.target,
            destination_id=args.destination,
            n_images=args.n_images,
            write_cifs=not args.no_cifs,
        )
    else:
        if args.destination is not None:
            raise ValueError("--destination is only valid with --mode straight")
        results = find_paths(
            config,
            layout,
            args.target,
            theta_min_eV=args.theta_min,
            n_images=args.n_images,
            winding_padding=args.winding_padding,
            write_cifs=not args.no_cifs,
        )

    for result in results:
        theta = "none" if result.theta_star_eV is None else f"{result.theta_star_eV:.6f} eV"
        print(
            f"[path {result.target_id}] status={result.status}; "
            f"destination={result.selected_destination or 'none'}; theta*={theta}"
        )
        print(f"[run directory] {result.run_directory}")
    return 0


def _path_scf(args: argparse.Namespace) -> int:
    if args.check_only and args.submit:
        raise ValueError("--check-only and --submit cannot be combined")
    if args.check_only and args.overwrite:
        raise ValueError("--check-only and --overwrite cannot be combined")
    if args.max_concurrent is not None and args.max_concurrent <= 0:
        raise ValueError("--max-concurrent must be positive")
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    layout = ProjectLayout(config.root)
    targets = list(config.target_site_ids if args.target is None else args.target)
    if not targets:
        raise ValueError("No targets selected; run orbit targets set first")
    if len(targets) != len(set(targets)):
        raise ValueError("Path-SCF target selection contains duplicates")
    unknown = [target for target in targets if target not in config.target_site_ids]
    if unknown:
        raise ValueError("Requested targets are not configured: " + ", ".join(unknown))
    if args.run_id is not None and len(targets) != 1:
        raise ValueError("--run-id requires exactly one --target")
    from ase.io import read as ase_read

    symbols = list(ase_read(config.structure).get_chemical_symbols())
    resources = resolve_qe_resources(config, symbols)
    max_concurrent = (
        config.slurm.max_concurrent
        if args.max_concurrent is None else args.max_concurrent
    )
    prepared = [
        build_prepared_path_target(
            config,
            layout,
            resources,
            target,
            run_id=args.run_id if len(targets) == 1 else None,
            max_concurrent=max_concurrent,
        )
        for target in targets
    ]
    print(f"[pseudos] directory={resources.pseudo_dir}")
    print(
        f"[qe] ecutwfc={resources.ecutwfc_Ry:g} Ry; "
        f"ecutrho={resources.ecutrho_Ry:g} Ry; nbnd={resources.nbnd}; "
        f"kpoints={list(config.qe.kpoints)}; offsets={list(config.qe.offsets)}"
    )
    for item in prepared:
        print(
            f"[path {item.target_id}] run={item.path_run_id}; "
            f"destination={item.destination_id}; images={len(item.calculations)}; "
            f"valid={item.valid_count}; invalid_geometry={item.invalid_count}; "
            f"max_concurrent={item.max_concurrent}"
        )
    if args.check_only:
        print(
            f"[check only] validated {len(prepared)} paths and "
            f"{sum(item.valid_count for item in prepared)} valid QE inputs; wrote nothing"
        )
        return 0

    conflicts = [
        item.target_id for item in prepared
        if path_calculation_output_state(item) == "conflict" and not args.overwrite
    ]
    if conflicts:
        raise FileExistsError(
            "Conflicting path QE preparation exists for "
            + ", ".join(conflicts)
            + "; use --overwrite to refresh inputs"
        )
    outputs = []
    for item in prepared:
        output, action = write_prepared_path_target(
            config, resources, item, overwrite=args.overwrite
        )
        outputs.append(output)
        if action == "skipped_identical":
            print(
                f"[skip] {item.target_id}: identical path QE preparation at {output}"
            )
        else:
            print(
                f"[done] {item.target_id}: prepared {item.valid_count} path-image "
                f"QE calculations at {output}"
            )
    if args.submit:
        for item, output in zip(prepared, outputs):
            print(f"[submit] {item.target_id}: bash {output / 'submit.sh'}")
            result = subprocess.run(["bash", str(output / "submit.sh")], cwd=output)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Path submission failed for {item.target_id} with "
                    f"exit status {result.returncode}"
                )
    return 0


def _path_response(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    kind = getattr(args, "response_kind", None) or getattr(args, "kind", "all")
    separate = bool(getattr(args, "separate_response_root", False))
    prepared = prepare_path_response(
        config,
        args.target,
        args.run_id,
        source=args.source,
        iteration=args.iteration,
        max_concurrent=args.max_concurrent,
        check_only=args.check_only,
        overwrite=args.overwrite,
        submit=args.submit,
        kind=kind,
        separate=separate,
    )
    iteration = "" if prepared.iteration is None else f"; iteration={prepared.iteration}"
    print(
        f"[response {prepared.target_id}] run={prepared.run_id}; "
        f"source={prepared.source}{iteration}; images={len(prepared.image_directories)}; "
        f"ph={prepared.ph_count}; polarization={prepared.polarization_count}; "
        f"gdirs={list(prepared.gdirs)}; max_concurrent={prepared.max_concurrent}"
    )
    if args.check_only:
        print(f"[check only] validated SCF reuse and {kind} inputs; wrote nothing")
    else:
        print(f"[done] {kind} inputs and submission scripts: {prepared.response_root}")
        if args.submit:
            print(f"[submit] requested kind={kind}")
    return 0


def _path_response_plot(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    result = analyze_path_response(
        config,
        args.target,
        args.run_id,
        source=args.source,
        iteration=args.iteration,
        quantum_divisor=args.quantum_divisor,
        branch_range=args.branch_range,
    )
    iteration = "" if result.iteration is None else f"; iteration={result.iteration}"
    print(
        f"[response plot {result.target_id}] run={result.run_id}; "
        f"source={result.source}{iteration}; images={result.image_count}; "
        f"winding={list(result.requested_displacement_frac)}"
    )
    print(
        f"[transported charge] final N={result.final_n:.8f}; "
        f"nearest integer={result.nearest_integer_n}"
    )
    print(f"[done] polarization CSV: {result.selected_csv}")
    print(f"[done] polarization plot: {result.polarization_plot}")
    print(f"[done] transported-charge CSV: {result.transported_charge_csv}")
    print(f"[done] transported-charge plot: {result.transported_charge_plot}")
    return 0


def _path_extract(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    layout = ProjectLayout(config.root)
    results = extract_path_targets(
        config, layout, args.target, run_id=args.run_id
    )
    for result in results:
        summary = ", ".join(
            f"{status}={count}" for status, count in result.status_counts.items()
        )
        print(
            f"[path {result.target_id}] run={result.path_run_id}; "
            f"destination={result.destination_id}; total={result.total_count}; {summary}"
        )
        print(f"[done] wrote {result.gaps_csv}")
        print(f"[done] wrote {result.manifest}")
    return 0


def _plot_path(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    layout = ProjectLayout(config.root)
    output, manifest, payload = write_path_viewer(
        config, layout, args.target, args.output, run_id=args.run_id,
        representatives_only=args.representatives_only,
    )
    for item in payload["targets"]:
        print(
            f"[path {item['target_id']}] run={item['path_run_id']}; "
            f"destination={item['destination_id']}; calculated="
            f"{item['complete_count']}/{item['image_count']}; "
            f"minimum={item['minimum_gap_eV']} eV at image {item['minimum_gap_image']}"
        )
    print(f"[done] wrote {output}")
    print(f"[done] wrote {manifest}")
    return 0


def _helper(args: argparse.Namespace) -> int:
    config = load_project_config(
        args.root,
        machine_profile=getattr(args, "machine_profile", None),
    )
    layout = ProjectLayout(config.root)

    requested_run_id = args.run_id
    run_id = resolve_helper_run_id(config, args.target, requested_run_id)
    if requested_run_id is None:
        print(f"[helper] resolved run={run_id}")

    if args.helper_command == "scan":
        if args.check_only and args.submit:
            raise ValueError("--check-only and --submit cannot be combined")

        # If this iteration was already prepared, --submit means submit/resume
        # that exact scan rather than trying to create the next iteration.
        if args.submit:
            existing = helper_status(config, args.target, run_id)
            if existing["iterations"]:
                latest = existing["iterations"][-1]
                if latest["status"] == "SCAN_PREPARED":
                    iteration = int(latest["iteration"])
                    scan_root = (
                        config.root
                        / "calculations"
                        / "helpers"
                        / args.target
                        / run_id
                        / f"iteration_{iteration:03d}"
                        / "scan"
                    )
                    print(
                        f"[helper {args.target}] iteration={iteration} already "
                        f"prepared; submitting/resuming {scan_root}"
                    )
                    submit_helper_scan(
                        config, args.target, run_id, iteration
                    )
                    return 0

        summary = prepare_helper_scan(
            config,
            layout,
            args.target,
            run_id,
            neighbor_cutoff=args.neighbor_cutoff,
            radius=args.radius,
            steps=args.steps,
            axes=args.axes,
            helper_site_ids=args.helper_site,
            max_concurrent=args.max_concurrent,
            check_only=args.check_only,
            force=args.force,
        )
        print(
            f"[helper {summary['target_id']}] iteration={summary['iteration']}; "
            f"bottleneck=image_{summary['center_image']:03d}; "
            f"gap={summary['baseline_gap_eV']:.8f} eV; "
            f"target={summary['target_gap_eV']:.8f} eV"
        )
        print(
            f"[neighbors] selected={len(summary['candidate_helpers'])}; "
            f"rule={summary['selection_rule']}"
        )
        for row in summary["candidate_helpers"]:
            print(
                f"  {row['helper_site_id']:<8} index={row['helper_index']:>3} "
                f"{row['helper_element']:<2} distance={row['target_distance_A']:.6f} A"
            )
        print(
            f"[grid] axes={summary['scan_axes']}; radius={summary['scan_radius_A']:g} A; "
            f"steps={summary['scan_steps_per_axis']}; "
            f"nominal={summary['nominal_points_per_helper']} points/helper"
        )
        if args.check_only:
            print(
                "[check only] nominal new SCFs before geometry veto = "
                f"{summary['nominal_new_scf_count_before_geometry_veto']}; wrote nothing"
            )
            return 0
        print(
            f"[scan] valid new SCFs={summary['valid_scan_count']}; "
            f"vetoed={summary['vetoed_count']}; root={summary['scan_root']}"
        )
        if args.submit:
            print(f"[submit] helper scan iteration {summary['iteration']}")
            submit_helper_scan(
                config, args.target, run_id, int(summary["iteration"])
            )
        else:
            print(f"[next] submit the prepared scan with: bash {summary['scan_root']}/submit.sh")
            print(
                "[next] after the scan jobs finish: orbit helper analyze "
                f"--target {args.target} --run-id {run_id}"
            )
        return 0

    if args.helper_command == "analyze":
        result = analyze_helper_scan(
            config,
            layout,
            args.target,
            run_id,
            allow_nonimproving=args.allow_nonimproving,
            max_concurrent=args.max_concurrent,
        )
        skipped = int(result.get("skipped_incomplete_scan_count", 0))
        if skipped:
            plural = "s" if skipped != 1 else ""
            print(
                f"[warning] skipped {skipped} incomplete helper scan "
                f"calculation{plural}; ranking uses completed points only"
            )
        rejected = int(result.get("rejected_higher_gap_path_candidates", 0))
        if rejected:
            plural = "s" if rejected != 1 else ""
            print(
                f"[geometry] rejected {rejected} higher-gap scan candidate{plural} " +
                "because the interpolated path violated geometry safety"
            )
            print(
                f"[geometry] screening: {result['path_candidate_screening_file']}"
            )
        print("[ranking] best individual helper responses:")
        for row in result["rankings"]:
            print(
                f"  {int(row['rank']):>2}. {row['helper_site_id']:<8} "
                f"best={float(row['best_gap_eV']):.8f} eV  "
                f"improvement={float(row['gap_improvement_eV']):+.8f} eV  "
                f"disp=({float(row['dx_A']):+.4f}, "
                f"{float(row['dy_A']):+.4f}, {float(row['dz_A']):+.4f}) A"
            )
        print(
            f"[selected] {result['best_helper_site_id']} at image "
            f"{result['center_image']}: {result['baseline_gap_eV']:.8f} -> "
            f"{result['best_gap_eV']:.8f} eV "
            f"({result['gap_improvement_eV']:+.8f} eV)"
        )
        if result["winner_on_scan_boundaries"]:
            print(
                "[warning] selected point lies on scan boundary: "
                + ", ".join(result["winner_on_scan_boundaries"])
            )
        print(f"[path] prepared helper path at {result['path_root']}")
        print(
            "[next] orbit helper path-scf "
            f"--target {args.target} --run-id {run_id} --submit"
        )
        return 0

    if args.helper_command == "path-scf":
        result = helper_path_scf(
            config, args.target, run_id, submit=args.submit
        )
        print(
            f"[helper path] iteration={result['iteration']}; "
            f"images={result['image_count']}; root={result['path_root']}"
        )
        if result["submitted"]:
            print("[submit] helper path array submitted")
        else:
            print(
                "[next] add --submit to launch this prepared helper path, or after "
                "completed outputs run helper extract"
            )
        return 0

    if args.helper_command == "extract":
        result = extract_helper_path(config, args.target, run_id)
        summary = ", ".join(
            f"{status}={count}" for status, count in result["status_counts"].items()
        )
        print(
            f"[helper extract] iteration={result['iteration']}; "
            f"complete={result['complete_count']}/{result['total_count']}; {summary}"
        )
        if result["minimum_gap_eV"] is None:
            print("[wait] helper path is incomplete; finish jobs and rerun helper extract")
        else:
            print(
                f"[minimum] {result['minimum_gap_eV']:.8f} eV at image "
                f"{result['minimum_gap_image']}; target={result['target_gap_eV']:.8f} eV"
            )
            print(
                "[status] "
                + ("CONVERGED" if result["converged"] else "NEEDS_REVIEW")
            )
            print(
                "[next] orbit helper plot "
                f"--target {args.target} --run-id {run_id}"
            )
        return 0

    if args.helper_command == "plot":
        output, manifest = write_helper_plot(
            config,
            args.target,
            run_id,
            output=args.output,
        )
        print(
            f"[plot] iteration={manifest['iteration']}; "
            f"minimum={manifest['minimum_gap_eV']:.8f} eV at image "
            f"{manifest['minimum_gap_image']}; target={manifest['target_gap_eV']:.8f} eV"
        )
        print(f"[done] wrote {output}")
        state = helper_status(config, args.target, run_id)
        if state["next_action"] == "helper scan":
            print(
                "[review gate] iteration reviewed; start the next iteration only "
                "when you are satisfied with this path:"
            )
            print(
                "  orbit helper scan "
                f"--target {args.target} --check-only"
            )
        elif state["next_action"] == "done":
            print("[converged] no further helper iteration is required")
        return 0

    if args.helper_command == "status":
        state = helper_status(config, args.target, run_id)
        print(
            f"[helper campaign] target={state['target_id']}; run={state['path_run_id']}; "
            f"target_gap={state['target_gap_eV']:.6f} eV"
        )
        if not state["iterations"]:
            print("[iterations] none")
        for row in state["iterations"]:
            print(
                f"  iter {row['iteration']:>3}: {row['status']:<14} "
                f"center={row['center_image']} baseline={row['baseline_gap_eV']} "
                f"helper={row['best_helper_site_id']} min={row['minimum_gap_eV']} "
                f"reviewed={'yes' if row['reviewed'] else 'no'}"
            )
        print(f"[next] {state['next_action']}")
        return 0

    raise ValueError(f"Unsupported helper subcommand: {args.helper_command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "machine":
            return _machine(args)
        if args.command == "doctor":
            args.root = Path.cwd() if getattr(args, "root", None) is None else args.root
            return _machine(args)
        if args.command == "init":
            layout, written = initialize_project(
                args.root,
                args.structure,
                force_config=args.force_config,
                machine=getattr(args, "machine", None),
            )
            print(f"[done] initialized ORBIT layout in {layout.root}")
            if written:
                print(f"[done] wrote {layout.config}")
            else:
                print(f"[skip] preserved existing {layout.config}")
            return 0
        if args.command == "inspect":
            return _inspect(args.root, machine_profile=args.machine_profile)
        if args.command == "targets" and args.targets_command == "list":
            return _list_targets(args.root, machine_profile=args.machine_profile)
        if args.command == "targets" and args.targets_command == "set":
            return _set_targets(args)
        if args.command == "sample":
            return _sample(args)
        if args.command == "scf":
            return _scf(args)
        if args.command == "extract":
            return _extract(args)
        if args.command == "path":
            return _path(args)
        if args.command == "path-scf":
            return _path_scf(args)
        if args.command in {"path-response", "path-ph", "path-polarization"}:
            return _path_response(args)
        if args.command == "path-response-plot":
            return _path_response_plot(args)
        if args.command == "path-extract":
            return _path_extract(args)
        if args.command == "plot-path":
            return _plot_path(args)
        if args.command == "helper":
            return _helper(args)
        if args.command == "plot":
            return _plot(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
