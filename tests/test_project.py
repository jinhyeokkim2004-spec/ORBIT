from pathlib import Path
import tempfile
import unittest

from orbit.project import PROJECT_DIRECTORIES, initialize_project


class ProjectTests(unittest.TestCase):
    def test_init_is_nondestructive_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout, written = initialize_project(root, Path("H2O.cif"))
            self.assertTrue(written)
            for relative in PROJECT_DIRECTORIES:
                self.assertTrue((root / relative).is_dir())

            layout.config.write_text("user configuration\n", encoding="utf-8")
            _, written_again = initialize_project(root, Path("other.cif"))
            self.assertFalse(written_again)
            self.assertEqual(
                layout.config.read_text(encoding="utf-8"), "user configuration\n"
            )


if __name__ == "__main__":
    unittest.main()

