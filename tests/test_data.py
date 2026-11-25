"""Tests for data generation/loading (credit_fraud.data)."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from credit_fraud.data import (
    generate_applications,
    load_applications,
    write_applications_csv,
    write_quarantine_csv,
)
from credit_fraud.exceptions import DataQualityError
from credit_fraud.validation import REQUIRED_COLUMNS


class TestGenerateApplications(unittest.TestCase):
    def test_row_count_and_schema(self):
        rows = generate_applications(100, seed=11)
        self.assertEqual(len(rows), 100)
        for row in rows:
            self.assertGreaterEqual(row.income, 25_000)
            self.assertLessEqual(row.income, 180_000)
            self.assertIn(row.delinquencies, (0, 1, 2, 3))
            self.assertGreaterEqual(row.utilisation, 0.0)
            self.assertLessEqual(row.utilisation, 1.0)
            self.assertIn(row.fraud, (0, 1))

    def test_same_seed_same_data(self):
        first = generate_applications(200, seed=42)
        second = generate_applications(200, seed=42)
        self.assertEqual(first, second)

    def test_different_seed_different_data(self):
        self.assertNotEqual(
            generate_applications(200, seed=1),
            generate_applications(200, seed=2),
        )

    def test_non_positive_rows_rejected(self):
        with self.assertRaises(ValueError):
            generate_applications(0, seed=11)

    def test_generated_labels_have_signal(self):
        # High-risk rows should be labelled fraud more often than low-risk
        # ones; this guards against the generator silently going flat.
        rows = generate_applications(2000, seed=11)
        high = [r for r in rows if r.delinquencies >= 2]
        low = [r for r in rows if r.delinquencies == 0 and r.utilisation < 0.3]
        self.assertGreater(len(high), 0)
        self.assertGreater(len(low), 0)
        high_rate = sum(r.fraud for r in high) / len(high)
        low_rate = sum(r.fraud for r in low) / len(low)
        self.assertGreater(high_rate, low_rate)


class TestLoadApplications(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_csv(self, name, header, rows):
        path = self.dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_roundtrip_write_then_load(self):
        generated = generate_applications(50, seed=7)
        path = write_applications_csv(self.dir / "apps.csv", generated)
        loaded = load_applications(path)
        self.assertEqual(loaded.rows, generated)
        self.assertEqual(loaded.rows_read, 50)
        self.assertEqual(loaded.rows_quarantined, 0)

    def test_bad_rows_quarantined_not_dropped_silently(self):
        path = self._write_csv(
            "mixed.csv",
            list(REQUIRED_COLUMNS),
            [
                [72000, 0, 0.31, 0],          # valid
                [72000, 0, 0.31, 0],          # duplicate of line 2
                ["", 0, 0.31, 0],             # missing income
                [50000, "many", 0.2, 0],      # unparseable delinquencies
                [50000, 0, 1.5, 0],           # utilisation out of range
                [50000, 0, 0.2, 2],           # fraud label not 0/1
                [90000, 1, 0.9, 1],           # valid
            ],
        )
        loaded = load_applications(path)
        self.assertEqual(loaded.rows_read, 7)
        self.assertEqual(loaded.rows_valid, 2)
        self.assertEqual(loaded.rows_quarantined, 5)
        reasons = [q.reason for q in loaded.quarantined]
        self.assertTrue(any("duplicate" in r for r in reasons))
        self.assertTrue(any("income" in r for r in reasons))
        self.assertTrue(any("delinquencies" in r for r in reasons))
        self.assertTrue(any("utilisation" in r for r in reasons))
        self.assertTrue(any("fraud" in r for r in reasons))
        self.assertEqual(
            [q.line_no for q in loaded.quarantined], [3, 4, 5, 6, 7]
        )

    def test_quarantine_file_written(self):
        path = self._write_csv(
            "bad.csv", list(REQUIRED_COLUMNS), [["", 0, 0.1, 0], [80000, 0, 0.2, 0]]
        )
        loaded = load_applications(path)
        out = write_quarantine_csv(self.dir / "q.csv", loaded.quarantined)
        self.assertTrue(out.is_file())
        with out.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["line_no"], "2")

    def test_missing_file_raises(self):
        with self.assertRaises(DataQualityError):
            load_applications(self.dir / "nope.csv")

    def test_missing_column_raises(self):
        path = self._write_csv(
            "nocols.csv", ["income", "delinquencies"], [[72000, 0]]
        )
        with self.assertRaises(DataQualityError) as ctx:
            load_applications(path)
        self.assertIn("utilisation", str(ctx.exception))

    def test_all_rows_bad_raises(self):
        path = self._write_csv(
            "allbad.csv", list(REQUIRED_COLUMNS), [["", "", "", ""]]
        )
        with self.assertRaises(DataQualityError):
            load_applications(path)


if __name__ == "__main__":
    unittest.main()
