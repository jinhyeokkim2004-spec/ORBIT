import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import ase  # noqa: F401
    import spglib  # noqa: F401
except ImportError:
    HAS_DEPS = False
else:
    HAS_DEPS = True

from orbit.config import load_project_config, set_target_site_ids
from orbit.project import ProjectLayout, initialize_project
from orbit.qe.calculations import (
    build_prepared_targets,
    calculation_output_state,
    write_prepared_target,
)
from orbit.qe.input import (
    InvalidGeometryError,
    render_scf_input,
    validate_scf_geometry,
)
from orbit.qe.resources import parse_upf_z_valence, resolve_qe_resources
from orbit.sampling import build_sampling_plans, write_sampling_plan
from orbit.structure import inspect_structure, write_structure_manifests


FIXTURE = Path(__file__).parent / "fixtures" / "H2O.cif"


def write_pseudos(base: Path):
    library = base / "pseudo" / "library"
    library.mkdir(parents=True)
    (library / "O.test.upf").write_text(
        '<UPF version="2.0.1"><PP_HEADER z_valence="6.0"/></UPF>\n',
        encoding="utf-8",
    )
    (library / "H.test.upf").write_text(
        "1.000000 Z valence\n",
        encoding="utf-8",
    )
    (base / "pseudo" / "cutoffs.json").write_text(
        json.dumps(
            {
                "O": {"cutoff_wfc": 45, "cutoff_rho": 360},
                "H": {"cutoff_wfc": 30, "cutoff_rho": 240},
            }
        ),
        encoding="utf-8",
    )


def prepared_project(base: Path):
    project = base / "H2O"
    project.mkdir()
    shutil.copy2(FIXTURE, project / "H2O.cif")
    write_pseudos(base)
    layout, _ = initialize_project(project, Path("H2O.cif"))
    text = layout.config.read_text(encoding="utf-8")
    layout.config.write_text(
        text.replace("spacing_angstrom = 0.4", "spacing_angstrom = 2.0"),
        encoding="utf-8",
    )
    config = load_project_config(project)
    inspection = inspect_structure(config.structure, config.symprec_angstrom)
    write_structure_manifests(inspection, layout.manifests, project)
    set_target_site_ids(layout.config, ["O1"])
    config = load_project_config(project)
    atoms, plans = build_sampling_plans(config, layout)
    write_sampling_plan(config, layout, atoms, plans[0])
    return config, layout, atoms, plans


@unittest.skipUnless(HAS_DEPS, "ASE and spglib are not installed")
class QEIntegrationTests(unittest.TestCase):
    def test_upf_formats_and_automatic_nbnd(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, _layout, atoms, _plans = prepared_project(base)
            resources = resolve_qe_resources(config, atoms.get_chemical_symbols())
            self.assertEqual(resources.total_valence_electrons, 8)
            self.assertEqual(resources.occupied_bands, 4)
            self.assertEqual(resources.nbnd, 16)
            self.assertEqual(resources.nbnd_source, "auto_from_upf_valence")
            self.assertEqual(resources.ecutwfc_Ry, 45)
            self.assertEqual(resources.ecutrho_Ry, 360)
            self.assertEqual(
                parse_upf_z_valence(base / "pseudo/library/H.test.upf"), 1.0
            )

    def test_input_preparation_manifests_and_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, layout, atoms, plans = prepared_project(base)
            resources = resolve_qe_resources(config, atoms.get_chemical_symbols())
            prepared = build_prepared_targets(
                config, layout, plans, resources, max_concurrent=3
            )[0]
            self.assertEqual(prepared.valid_count, plans[0].representative_count)
            self.assertEqual(prepared.invalid_count, 0)
            output, action = write_prepared_target(
                config, layout, resources, prepared
            )
            self.assertEqual(action, "written")
            self.assertEqual(calculation_output_state(layout, prepared), "identical")
            first_input = output / "sample_00000/espresso_scf.pwi"
            text = first_input.read_text(encoding="utf-8")
            self.assertIn("ecutwfc          = 45", text)
            self.assertIn("ecutrho          = 360", text)
            self.assertIn("nbnd             = 16", text)
            self.assertIn("6 6 6 0 0 0", text)
            self.assertNotIn("HUBBARD", text)
            self.assertEqual(
                subprocess.run(
                    ["bash", "-n", str(output / "run_array.sh")], check=False
                ).returncode,
                0,
            )
            self.assertEqual(
                subprocess.run(
                    ["bash", "-n", str(output / "submit.sh")], check=False
                ).returncode,
                0,
            )
            same_output, second_action = write_prepared_target(
                config, layout, resources, prepared
            )
            self.assertEqual(same_output, output)
            self.assertEqual(second_action, "skipped_identical")

    def test_renderer_uses_species_order_and_absolute_pseudo_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, _layout, atoms, _plans = prepared_project(base)
            resources = resolve_qe_resources(config, atoms.get_chemical_symbols())
            text = render_scf_input(atoms, "O1", "sample_00000", config, resources)
            self.assertIn(f"pseudo_dir       = '{resources.pseudo_dir}'", text)
            species_section = text.split("ATOMIC_SPECIES\n", 1)[1].split(
                "\n\nK_POINTS", 1
            )[0]
            self.assertEqual(species_section.splitlines()[0].split()[0], "O")
            self.assertEqual(species_section.splitlines()[1].split()[0], "H")

    def test_exact_overlap_is_an_explicit_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, _layout, atoms, _plans = prepared_project(base)
            resources = resolve_qe_resources(config, atoms.get_chemical_symbols())
            overlapping = atoms.copy()
            positions = overlapping.get_positions()
            positions[1] = positions[0]
            overlapping.set_positions(positions)
            with self.assertRaisesRegex(InvalidGeometryError, "minimum nuclear distance"):
                validate_scf_geometry(
                    overlapping,
                    resources,
                    config.qe.exact_overlap_tolerance_angstrom,
                )

    def test_overwrite_preserves_old_output_as_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, layout, atoms, plans = prepared_project(base)
            resources = resolve_qe_resources(config, atoms.get_chemical_symbols())
            prepared = build_prepared_targets(config, layout, plans, resources, 3)[0]
            output, _ = write_prepared_target(config, layout, resources, prepared)
            old_output = output / "sample_00000/espresso_scf.pwo"
            old_output.write_text("JOB DONE.\n", encoding="utf-8")

            config_text = layout.config.read_text(encoding="utf-8")
            layout.config.write_text(
                config_text.replace("kpoints = [6, 6, 6]", "kpoints = [4, 4, 4]"),
                encoding="utf-8",
            )
            changed_config = load_project_config(config.root)
            changed_resources = resolve_qe_resources(
                changed_config, atoms.get_chemical_symbols()
            )
            changed_prepared = build_prepared_targets(
                changed_config, layout, plans, changed_resources, 3
            )[0]
            with self.assertRaises(FileExistsError):
                write_prepared_target(
                    changed_config, layout, changed_resources, changed_prepared
                )
            write_prepared_target(
                changed_config,
                layout,
                changed_resources,
                changed_prepared,
                overwrite=True,
            )
            self.assertFalse(old_output.exists())
            self.assertEqual(
                len(list(old_output.parent.glob("espresso_scf.pwo.stale_*"))), 1
            )


    def test_hubbard_card_changes_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, layout, atoms, plans = prepared_project(base)
            resources = resolve_qe_resources(config, atoms.get_chemical_symbols())
            baseline = build_prepared_targets(config, layout, plans, resources, 3)[0]

            with layout.config.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n[qe.hubbard]\n"
                    'projector = "atomic"\n'
                    "\n[[qe.hubbard.u]]\n"
                    'element = "O"\n'
                    'manifold = "2p"\n'
                    "value_eV = 6.0\n"
                )

            changed = load_project_config(config.root)
            changed_resources = resolve_qe_resources(changed, atoms.get_chemical_symbols())
            prepared = build_prepared_targets(changed, layout, plans, changed_resources, 3)[0]
            self.assertNotEqual(baseline.preparation_signature, prepared.preparation_signature)

            valid = next(x for x in prepared.calculations if x.preparation_status == "VALID")
            assert valid.input_text is not None
            self.assertIn("HUBBARD {atomic}\nU O-2p 6.0", valid.input_text)


if __name__ == "__main__":
    unittest.main()
