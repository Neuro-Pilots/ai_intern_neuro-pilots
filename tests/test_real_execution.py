"""Fast regression tests for the real-execution boundary contracts."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from execution.real_execution import RealExecutionService
from execution.hpo import SimpleHPO
from execution.nlp_modernbert import AmazonAutomotiveModernBERTExecutor
from diagnostics.comparison_engine import ComparisonEngine
from ingestion.dataset_versioning import DatasetVersioner
from utils.models import ExperimentResult


class DatasetVersioningTests(unittest.TestCase):
    def test_fingerprint_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "reviews.jsonl"
            path.write_text('{"text": "good", "rating": 5}\n')
            first = DatasetVersioner().identify(path, "Automotive")
            path.write_text('{"text": "poor", "rating": 1}\n')
            second = DatasetVersioner().identify(path, "Automotive")
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.file_count, 1)


class RealExecutionTests(unittest.TestCase):
    def test_cv_rejects_missing_yolo_yaml_without_fabricating_metrics(self) -> None:
        result = RealExecutionService().execute("computer_vision", {"data_yaml": "/missing/idd.yaml"})
        self.assertFalse(result.success)
        self.assertEqual(result.metrics, {})
        self.assertIn("data_yaml", result.error or "")

    def test_nlp_augmentation_is_deterministic_and_preserves_short_text(self) -> None:
        import random

        executor = AmazonAutomotiveModernBERTExecutor()
        self.assertEqual(executor._augment_text("short text", random.Random(1), 1.0, 0.5), "short text")
        self.assertEqual(
            executor._augment_text("one two three four", random.Random(7), 1.0, 0.25),
            executor._augment_text("one two three four", random.Random(7), 1.0, 0.25),
        )

    def test_comparison_identifies_metric_improvement_and_regression(self) -> None:
        baseline = ExperimentResult(experiment_id="baseline", metrics={"accuracy": 0.7, "loss": 0.6})
        candidate = ExperimentResult(experiment_id="candidate", metrics={"accuracy": 0.8, "loss": 0.7})
        comparison = ComparisonEngine().compare(baseline, candidate)
        self.assertIn("accuracy", comparison.improvements)
        self.assertIn("loss", comparison.regressions)

    def test_hpo_runs_only_the_configured_trial_bound(self) -> None:
        results = [
            ExperimentResult(experiment_id="one", success=True, metrics={"accuracy": 0.7}),
            ExperimentResult(experiment_id="two", success=True, metrics={"accuracy": 0.8}),
        ]
        with patch.object(RealExecutionService, "execute", side_effect=results) as execute:
            outcome = SimpleHPO().run("nlp", {}, {"learning_rate": [1e-5, 2e-5, 3e-5]}, "accuracy", max_trials=2)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(outcome.best_experiment_id, "two")

    def test_nlp_rejects_missing_local_dataset_without_fabricating_metrics(self) -> None:
        result = RealExecutionService().execute("nlp", {"dataset_path": "/missing/automotive.csv"})
        self.assertFalse(result.success)
        self.assertEqual(result.metrics, {})


if __name__ == "__main__":
    unittest.main()
