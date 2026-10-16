import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from tools import run_forecasting_suite


class ForecastingSuiteRunnerTests(unittest.TestCase):
    def test_gpu_index_cli_override_is_written_to_resolved_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.yaml"
            suite_path = root / "suite.yaml"
            output_path = root / "output"
            base_path.write_text("gpu_index: 1\nmodels: {}\n", encoding="utf-8")
            suite_path.write_text(
                yaml.safe_dump(
                    {
                        "base_config": "base.yaml",
                        "gpu_index": 1,
                        "experiments": {"E1": {"input_n": 30, "output_n": 10}},
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "run_forecasting_suite.py",
                "--suite-config",
                str(suite_path),
                "--output-root",
                str(output_path),
                "--experiments",
                "E1",
                "--gpu-index",
                "0",
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                return_code = run_forecasting_suite.main()

            self.assertEqual(return_code, 0)
            resolved = yaml.safe_load((output_path / "E1" / "resolved_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(resolved["gpu_index"], 0)

    def test_later_experiments_run_after_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.yaml"
            suite_path = root / "suite.yaml"
            output_path = root / "output"
            base_path.write_text("models: {}\n", encoding="utf-8")
            suite_path.write_text(
                yaml.safe_dump(
                    {
                        "base_config": "base.yaml",
                        "experiments": {
                            "E1": {"input_n": 30, "output_n": 10},
                            "E2": {"input_n": 30, "output_n": 30},
                        },
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "run_forecasting_suite.py",
                "--suite-config",
                str(suite_path),
                "--output-root",
                str(output_path),
                "--experiments",
                "E1",
                "E2",
            ]
            results = [
                subprocess.CompletedProcess([], 1),
                subprocess.CompletedProcess([], 0),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run_forecasting_suite.subprocess,
                "run",
                side_effect=results,
            ) as run:
                return_code = run_forecasting_suite.main()

            self.assertEqual(return_code, 1)
            self.assertEqual(run.call_count, 2)
            self.assertTrue((output_path / "E1" / "resolved_config.yaml").exists())
            self.assertTrue((output_path / "E2" / "resolved_config.yaml").exists())


if __name__ == "__main__":
    unittest.main()
