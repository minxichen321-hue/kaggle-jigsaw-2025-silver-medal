"""Checks for row alignment and invalid-input protection in the blending utility."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location(
    "blend_predictions", Path(__file__).resolve().parents[1] / "scripts/blend_predictions.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BlendPredictionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, ids, scores):
        path = self.directory / name
        pd.DataFrame({"row_id": ids, "rule_violation": scores}).to_csv(path, index=False)
        return path

    def test_blends_shuffled_rows_and_tied_scores(self):
        deberta = self.write("deberta.csv", [30, 10, 20], [0.8, 0.1, 0.8])
        gte = self.write("gte.csv", [20, 30, 10], [4, 1, 2])
        llm = self.write("llm.csv", [10, 20, 30], [0.2, 0.5, 0.9])
        result = module.blend_predictions(deberta, gte, llm)
        self.assertEqual(result["row_id"].tolist(), [10, 20, 30])
        np.testing.assert_allclose(result["rule_violation"], [0.45, 0.8875, 0.9125])

    def test_rejects_missing_prediction_rows(self):
        full = self.write("full.csv", [10, 20], [0.1, 0.9])
        missing = self.write("missing.csv", [10], [0.2])
        with self.assertRaisesRegex(ValueError, "row_id values do not match"):
            module.blend_predictions(full, missing, full)

    def test_rejects_duplicate_ids(self):
        duplicate = self.write("duplicate.csv", [10, 10], [0.1, 0.9])
        with self.assertRaisesRegex(ValueError, "unique"):
            module.read_predictions(duplicate)

    def test_rejects_non_finite_scores(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                path = self.write("invalid.csv", [10], [value])
                with self.assertRaisesRegex(ValueError, "finite"):
                    module.read_predictions(path)

    def test_rejects_empty_predictions_and_missing_columns(self):
        empty = self.write("empty.csv", [], [])
        with self.assertRaisesRegex(ValueError, "empty"):
            module.read_predictions(empty)
        invalid = self.directory / "columns.csv"
        pd.DataFrame({"row_id": [10]}).to_csv(invalid, index=False)
        with self.assertRaisesRegex(ValueError, "expected columns"):
            module.read_predictions(invalid)


if __name__ == "__main__":
    unittest.main()
