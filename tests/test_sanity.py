import sys
import unittest
from pathlib import Path

# Add project root to sys.path so src can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestSanity(unittest.TestCase):
    """Sanity checks for directory structure, package imports, and basic config."""

    def test_directory_structure_exists(self):
        """Verify that all essential directories are present in the repository."""
        expected_dirs = [
            "configs",
            "data/raw",
            "data/processed",
            "data/samples",
            "data/golden",
            "src/data",
            "src/preprocessing",
            "src/intents",
            "src/retrieval",
            "src/generation",
            "src/escalation",
            "src/pipeline",
            "src/evaluation",
            "scripts",
            "tests",
            "experiments",
            "reports",
        ]

        for dir_path in expected_dirs:
            full_path = PROJECT_ROOT / dir_path
            self.assertTrue(
                full_path.exists() and full_path.is_dir(),
                f"Expected directory missing: {dir_path}"
            )

    def test_package_imports(self):
        """Verify that src and its subpackages can be imported cleanly."""
        import src
        import src.data
        import src.preprocessing
        import src.intents
        import src.retrieval
        import src.generation
        import src.escalation
        import src.pipeline
        import src.evaluation

        self.assertEqual(src.__version__, "0.1.0")

    def test_default_config_file_exists(self):
        """Verify that the default configuration file exists and contains expected sections."""
        config_path = PROJECT_ROOT / "configs" / "default_config.yaml"
        self.assertTrue(config_path.exists(), "configs/default_config.yaml does not exist")
        
        content = config_path.read_text(encoding="utf-8")
        expected_keys = ["brand:", "data:", "embeddings:", "llm:", "evaluation:"]
        for key in expected_keys:
            self.assertIn(key, content, f"Expected key '{key}' missing from default_config.yaml")


if __name__ == "__main__":
    unittest.main()

