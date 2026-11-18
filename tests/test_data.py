"""Lightweight data validation tests (run: python -m unittest discover -s tests)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import KM_OPTIONS, load_time_series


class LoadTimeSeriesTests(unittest.TestCase):
    def test_demo_has_expected_columns_and_valid_dates(self):
        path = Path(__file__).resolve().parents[1] / "data" / "sample_traffic.csv"
        result = load_time_series(path, KM_OPTIONS[0])
        self.assertGreaterEqual(len(result), 10)
        self.assertEqual(list(result.columns), ["ds", "y"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["ds"]))

    def test_missing_segment_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            pd.DataFrame({"Дата": ["2024-01-01"] * 10, "abc": [1] * 10}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Не найден столбец"):
                load_time_series(path, KM_OPTIONS[0])


if __name__ == "__main__":
    unittest.main()
