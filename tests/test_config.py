from pathlib import Path
import tempfile
import unittest

from orbit.config import SlurmConfig, load_project_config, set_target_site_ids
from orbit.project import initialize_project
from orbit.qe.path_response import _render_all_array
from orbit.scheduler.slurm import render_array_script, render_submit_script


class ConfigTests(unittest.TestCase):
    def test_step1_configuration_can_gain_targets_without_other_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            layout, _ = initialize_project(root, Path("H2O.cif"))
            original = layout.config.read_text(encoding="utf-8")
            set_target_site_ids(layout.config, ["O1", "H1", "H2"])
            updated = layout.config.read_text(encoding="utf-8")
            self.assertIn('site_ids = ["O1", "H1", "H2"]', updated)
            self.assertIn("spacing_angstrom = 0.4", updated)
            self.assertIn("kpoints = [6, 6, 6]", updated)
            self.assertNotEqual(original, updated)
            config = load_project_config(root)
            self.assertEqual(config.target_site_ids, ("O1", "H1", "H2"))

    def test_existing_step1_config_without_targets_loads_as_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "orbit.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n[sampling]\nsymprec_angstrom = 0.001\n',
                encoding="utf-8",
            )
            self.assertEqual(load_project_config(root).target_site_ids, ())

    def test_machine_profile_is_selected_with_project_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "orbit.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n'
                '[machine]\nprofile = "whoville3"\n\n'
                '[slurm]\npartition = "debug"\nmax_concurrent = 9\n',
                encoding="utf-8",
            )
            config = load_project_config(root)
            self.assertEqual(config.machine_profile, "whoville3")
            self.assertEqual(config.slurm.partition, "debug")
            self.assertEqual(config.slurm.max_concurrent, 9)

            cli_config = load_project_config(root, machine_profile="perlmutter")
            self.assertEqual(cli_config.machine_profile, "perlmutter")
            self.assertEqual(cli_config.slurm.partition, "debug")
            self.assertEqual(cli_config.slurm.max_concurrent, 9)

    def test_hubbard_configuration_is_optional_and_parses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            layout, _ = initialize_project(root, Path("H2O.cif"))
            self.assertIsNone(load_project_config(root).qe.hubbard)

            with layout.config.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n[qe.hubbard]\n"
                    'projector = "atomic"\n'
                    "\n[[qe.hubbard.u]]\n"
                    'element = "O"\n'
                    'manifold = "2p"\n'
                    "value_eV = 6.0\n"
                )

            config = load_project_config(root)
            self.assertIsNotNone(config.qe.hubbard)
            assert config.qe.hubbard is not None
            self.assertEqual(config.qe.hubbard.projector, "atomic")
            self.assertEqual(config.qe.hubbard.u[0].element, "O")
            self.assertEqual(config.qe.hubbard.u[0].manifold, "2p")
            self.assertEqual(config.qe.hubbard.u[0].value_eV, 6.0)

    def test_gapflow_toml_is_accepted_as_backward_compatible_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "gapflow.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n'
                '[slurm]\naccount = "m4013"\nmax_concurrent = 7\n',
                encoding="utf-8",
            )
            config = load_project_config(root)
            self.assertEqual(config.slurm.account, "m4013")
            self.assertEqual(config.slurm.max_running, 7)

    def test_perlmutter_profile_sets_cpu_and_omp_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "orbit.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n'
                '[machine]\nprofile = "perlmutter"\n\n'
                '[slurm]\naccount = "m4013"\n',
                encoding="utf-8",
            )
            config = load_project_config(root)
            self.assertEqual(config.machine_profile, "perlmutter")
            self.assertEqual(config.slurm.launcher, "srun")
            self.assertEqual(config.slurm.cpus_per_task, 2)
            self.assertEqual(config.slurm.cpu_bind, "cores")
            self.assertFalse(config.slurm.module_purge)
            self.assertEqual(config.slurm.environment["OMP_NUM_THREADS"], "1")

    def test_rendered_perlmutter_script_has_expected_slurm_and_omp_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "orbit.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n'
                '[machine]\nprofile = "perlmutter"\n\n'
                '[slurm]\naccount = "m4013"\n',
                encoding="utf-8",
            )
            config = load_project_config(root)
            script = render_array_script("target1", config.slurm)
            self.assertIn("#SBATCH --account=m4013", script)
            self.assertIn("#SBATCH --constraint=cpu", script)
            self.assertIn("#SBATCH --qos=regular", script)
            self.assertIn("#SBATCH --ntasks-per-node=64", script)
            self.assertIn("#SBATCH --cpus-per-task=2", script)
            self.assertIn("module load espresso/7.5-libxc-7.0.0-cpu", script)
            self.assertIn("export OMP_NUM_THREADS=1", script)
            self.assertIn("export OMP_PROC_BIND=spread", script)
            self.assertIn("export OMP_PLACES=threads", script)
            self.assertIn("srun --cpu-bind=cores pw.x -in espresso_scf.pwi > espresso_scf.pwo", script)
            self.assertNotIn("module purge", script)

    def test_rendered_whoville_script_keeps_mpirun_and_module_purge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "orbit.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n'
                '[machine]\nprofile = "whoville3"\n',
                encoding="utf-8",
            )
            config = load_project_config(root)
            script = render_array_script("target2", config.slurm)
            self.assertIn("module purge", script)
            self.assertIn("mpirun -np \"${SLURM_NTASKS}\" pw.x", script)
            self.assertIn("module load intel/2023.2.1", script)

    def test_submit_script_uses_array_limit_and_single_runner_selection(self):
        script = render_submit_script("target3", 7)
        self.assertIn('--array="0-$((COUNT - 1))%7"', script)
        self.assertIn("Maximum simultaneous tasks: 7", script)
        config = SlurmConfig(
            account="m4013",
            partition="regular",
            qos="regular",
            constraint="cpu",
            nodes=1,
            tasks_per_node=64,
            cpus_per_task=2,
            time="02:00:00",
            max_concurrent=7,
            modules=("espresso/7.5-libxc-7.0.0-cpu",),
            module_purge=False,
            setup_commands=(),
            launcher="srun",
            launcher_args=("--cpu-bind=cores",),
            cpu_bind="cores",
            environment={
                "OMP_NUM_THREADS": "1",
                "OMP_PROC_BIND": "spread",
                "OMP_PLACES": "threads",
            },
            pw_command="pw.x",
            ph_command="ph.x",
            pp_command="pp.x",
        )
        array_script = render_array_script("target3", config)
        self.assertIn('CALC_DIR="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GAPFLOW_CALC_LIST}")"', array_script)
        self.assertNotIn("for CALC_DIR", array_script)

    def test_response_script_runs_all_gdirs_sequentially_for_one_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "H2O.cif").write_text("placeholder", encoding="utf-8")
            (root / "orbit.toml").write_text(
                '[project]\nstructure = "H2O.cif"\n\n'
                '[machine]\nprofile = "generic_slurm"\n',
                encoding="utf-8",
            )
            config = load_project_config(root)
            script = _render_all_array(
                config,
                "example",
                config.slurm.ph_command,
                config.slurm.pw_command,
                (1, 2, 3),
            )
            self.assertIn("for GDIR in 1 2 3; do", script)
            self.assertIn('INPUT="espresso_pol_gdir${GDIR}.pwi"', script)
            self.assertIn('OUTPUT="espresso_pol_gdir${GDIR}.pwo"', script)
            self.assertLess(script.index("for GDIR in 1 2 3; do"), script.index("done"))
            self.assertEqual(script.count("for GDIR in 1 2 3; do"), 1)


if __name__ == "__main__":
    unittest.main()

