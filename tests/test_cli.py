from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_cli_accepts_model_and_horizon(self) -> None:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_model.py"),
            "--commodity",
            "brent",
            "--model",
            "naive",
            "--horizon-days",
            "2",
            "--features",
            "base",
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("Model: naive", result.stdout)
        self.assertIn("Forecast horizon (days): 2", result.stdout)
        self.assertIn("Multi-day forecast:", result.stdout)


if __name__ == "__main__":
    unittest.main()
