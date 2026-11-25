"""Tests for input and CSV-row validation (credit_fraud.validation)."""

import math
import unittest

from credit_fraud.exceptions import RowValidationError, ValidationError
from credit_fraud.validation import parse_application_row, validate_application


class TestValidateApplication(unittest.TestCase):
    def test_valid_inputs_pass(self):
        result = validate_application(72_000, 0, 0.31)
        self.assertEqual((result.income, result.delinquencies, result.utilisation),
                         (72_000, 0, 0.31))

    def test_boundary_values_pass(self):
        validate_application(0, 0, 0.0)
        validate_application(10_000_000, 50, 1.0)

    def test_negative_income_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application(-1, 0, 0.1)

    def test_income_above_max_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application(10_000_001, 0, 0.1)

    def test_fractional_income_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application(72_000.5, 0, 0.1)

    def test_bool_is_not_a_number(self):
        with self.assertRaises(ValidationError):
            validate_application(True, 0, 0.1)
        with self.assertRaises(ValidationError):
            validate_application(72_000, False, 0.1)

    def test_nan_and_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValidationError, msg=f"income={bad}"):
                validate_application(bad, 0, 0.1)
            with self.assertRaises(ValidationError, msg=f"utilisation={bad}"):
                validate_application(72_000, 0, bad)

    def test_negative_delinquencies_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application(72_000, -1, 0.1)

    def test_delinquencies_above_max_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application(72_000, 51, 0.1)

    def test_utilisation_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application(72_000, 0, -0.01)
        with self.assertRaises(ValidationError):
            validate_application(72_000, 0, 1.01)

    def test_string_inputs_rejected(self):
        with self.assertRaises(ValidationError):
            validate_application("72000", 0, 0.1)


class TestParseApplicationRow(unittest.TestCase):
    def _row(self, **overrides):
        row = {"income": "72000", "delinquencies": "1",
               "utilisation": "0.5", "fraud": "0"}
        row.update(overrides)
        return row

    def test_valid_row_parses(self):
        record = parse_application_row(self._row(), line_no=2)
        self.assertEqual(
            (record.income, record.delinquencies, record.utilisation, record.fraud),
            (72_000, 1, 0.5, 0),
        )

    def test_missing_value_quarantined_with_line_number(self):
        with self.assertRaises(RowValidationError) as ctx:
            parse_application_row(self._row(income=""), line_no=7)
        self.assertEqual(ctx.exception.line_no, 7)
        self.assertIn("income", str(ctx.exception))

    def test_unparseable_value_reports_reason(self):
        with self.assertRaises(RowValidationError) as ctx:
            parse_application_row(self._row(delinquencies="many"), line_no=3)
        self.assertIn("delinquencies", ctx.exception.reason)

    def test_out_of_range_value_rejected(self):
        with self.assertRaises(RowValidationError):
            parse_application_row(self._row(utilisation="1.5"), line_no=4)

    def test_fraud_label_must_be_zero_or_one(self):
        with self.assertRaises(RowValidationError):
            parse_application_row(self._row(fraud="2"), line_no=5)

    def test_missing_column_is_an_error(self):
        row = {"income": "72000", "delinquencies": "0", "utilisation": "0.1"}
        with self.assertRaises(RowValidationError):
            parse_application_row(row, line_no=2)


if __name__ == "__main__":
    unittest.main()
