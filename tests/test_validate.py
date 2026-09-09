"""
Tests for skim.validate.

These are ordinary synthetic tests and that is appropriate here: unlike
corner detection, arithmetic is not defined by real-world mess. A
hand-built receipt with a known one-cent error proves the checker catches
one-cent errors, and does it without spending an API call.

The real receipts still matter, but they answer a different question --
"does the model read receipts correctly" -- and they answer it through
scripts/validate_extracted.py, not here.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.extract import LineItem, Receipt
from skim.validate import (
    CheckStatus,
    check_items_sum_to_subtotal,
    check_line_arithmetic,
    check_subtotal_plus_tax_is_total,
    to_cents,
    validate,
)


def line(number, description, total, quantity=None, unit_price=None, voided=False):
    return LineItem(
        line_number=number,
        raw_description=description,
        line_total=total,
        quantity=quantity,
        unit_price=unit_price,
        is_voided=voided,
    )


def receipt(items, subtotal=None, tax=None, total=None):
    return Receipt(
        merchant_name="Test Store",
        store_number=None,
        address=None,
        phone=None,
        purchase_date="2026-08-25",
        purchase_time="12:00",
        line_items=items,
        subtotal=subtotal,
        tax=tax,
        total=total,
    )


class ToCentsTests(unittest.TestCase):
    def test_dollars_become_whole_cents(self):
        self.assertEqual(to_cents(41.80), 4180)
        self.assertEqual(to_cents(0.05), 5)
        self.assertEqual(to_cents(0), 0)

    def test_none_stays_none(self):
        self.assertIsNone(to_cents(None))

    def test_float_representation_error_does_not_leak_through(self):
        # 0.1 + 0.2 == 0.30000000000000004 in binary floating point.
        # Converting to cents first is what makes the comparison exact,
        # and this whole module depends on that being true.
        self.assertEqual(to_cents(0.1) + to_cents(0.2), to_cents(0.3))


class ItemsSumToSubtotalTests(unittest.TestCase):
    def test_matching_sum_passes(self):
        r = receipt([line(1, "A", 3.43), line(2, "B", 9.98)], subtotal=13.41)
        self.assertIs(check_items_sum_to_subtotal(r).status, CheckStatus.PASS)

    def test_one_cent_discrepancy_fails(self):
        # No tolerance between two printed numbers: a single misread
        # digit is exactly what this check exists to catch.
        r = receipt([line(1, "A", 3.43), line(2, "B", 9.98)], subtotal=13.42)
        result = check_items_sum_to_subtotal(r)
        self.assertIs(result.status, CheckStatus.FAIL)
        self.assertEqual(result.delta_cents, -1)

    def test_voided_line_is_excluded_from_the_sum(self):
        # The Walmart case. Including the voided line would break a
        # correctly-read receipt.
        r = receipt(
            [line(1, "A", 3.43), line(2, "VOID", None, voided=True), line(3, "B", 9.98)],
            subtotal=13.41,
        )
        self.assertIs(check_items_sum_to_subtotal(r).status, CheckStatus.PASS)

    def test_missing_subtotal_is_uncheckable_not_failed(self):
        r = receipt([line(1, "A", 3.43)], subtotal=None)
        self.assertIs(check_items_sum_to_subtotal(r).status, CheckStatus.UNCHECKABLE)

    def test_charged_line_without_an_amount_is_uncheckable(self):
        # We detected missing data, not a discrepancy. Reporting FAIL
        # would claim we found something we did not.
        r = receipt([line(1, "A", 3.43), line(2, "B", None)], subtotal=13.41)
        result = check_items_sum_to_subtotal(r)
        self.assertIs(result.status, CheckStatus.UNCHECKABLE)
        self.assertIn("2", result.detail)


class SubtotalPlusTaxTests(unittest.TestCase):
    def test_matching_total_passes(self):
        r = receipt([], subtotal=41.80, tax=1.97, total=43.77)
        self.assertIs(check_subtotal_plus_tax_is_total(r).status, CheckStatus.PASS)

    def test_wrong_total_fails(self):
        r = receipt([], subtotal=41.80, tax=1.97, total=43.78)
        self.assertIs(check_subtotal_plus_tax_is_total(r).status, CheckStatus.FAIL)

    def test_absent_tax_counts_as_zero_and_still_proves_something(self):
        # India Market: groceries are untaxed in Connecticut, so no tax
        # line is printed. 63.07 + 0 = 63.07 is a real check on real
        # evidence, not a missing one.
        r = receipt([], subtotal=63.07, tax=None, total=63.07)
        self.assertIs(check_subtotal_plus_tax_is_total(r).status, CheckStatus.PASS)

    def test_missing_subtotal_is_uncheckable(self):
        r = receipt([], subtotal=None, tax=1.97, total=43.77)
        self.assertIs(
            check_subtotal_plus_tax_is_total(r).status, CheckStatus.UNCHECKABLE
        )


class LineArithmeticTests(unittest.TestCase):
    def test_weighted_item_rounding_is_tolerated(self):
        # 0.52 lb x $2.49 = $1.2948, printed as $1.29. That penny is the
        # register rounding, not a transcription error.
        r = receipt([line(1, "DESI OKRA", 1.29, quantity=0.52, unit_price=2.49)])
        results = check_line_arithmetic(r)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, CheckStatus.PASS)

    def test_error_beyond_a_cent_fails_and_names_the_line(self):
        r = receipt([
            line(1, "A", 2.49, quantity=1, unit_price=2.49),
            line(2, "WRONG", 9.99, quantity=2, unit_price=2.49),
        ])
        failures = [r_ for r_ in check_line_arithmetic(r) if r_.status is CheckStatus.FAIL]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].line_number, 2)

    def test_lines_without_a_unit_price_are_skipped_not_failed(self):
        # The Walmart receipt prints amounts only. Nothing is wrong;
        # there is simply nothing here to check.
        r = receipt([line(1, "RAT TRAP", 3.43)])
        self.assertEqual(check_line_arithmetic(r), [])

    def test_voided_lines_are_skipped(self):
        r = receipt([line(1, "VOID", None, quantity=1, unit_price=1.0, voided=True)])
        self.assertEqual(check_line_arithmetic(r), [])


class ValidationReportTests(unittest.TestCase):
    def test_a_clean_receipt_is_trustworthy(self):
        r = receipt(
            [line(1, "A", 3.43), line(2, "B", 9.98)],
            subtotal=13.41, tax=0.85, total=14.26,
        )
        report = validate(r)
        self.assertTrue(report.is_trustworthy)
        self.assertEqual(len(report.failed), 0)

    def test_any_single_failure_sends_it_to_review(self):
        # Deliberately strict: a false alarm costs seconds of looking at
        # a photo, a missed error corrupts the price index permanently.
        r = receipt([line(1, "A", 3.43)], subtotal=3.43, tax=0.20, total=9.99)
        report = validate(r)
        self.assertFalse(report.is_trustworthy)

    def test_uncheckable_results_are_excluded_from_the_denominator(self):
        # A receipt that prints less proves less about itself. Counting
        # what could not run as if it had passed would flatter the rate.
        r = receipt([line(1, "A", 3.43)], subtotal=None, total=None)
        report = validate(r)
        self.assertEqual(report.evidence_count, 0)
        self.assertGreater(len(report.uncheckable), 0)

    def test_receipt_proving_nothing_says_so(self):
        r = receipt([line(1, "A", 3.43)], subtotal=None, total=None)
        self.assertIn("proves nothing", validate(r).summary())

    def test_validate_does_not_modify_the_receipt(self):
        # A validator that repaired what it found would destroy the
        # evidence of what the model actually returned.
        r = receipt([line(1, "A", 3.43)], subtotal=99.99, tax=0.0, total=99.99)
        before = (r.subtotal, r.total, r.line_items[0].line_total)
        validate(r)
        self.assertEqual((r.subtotal, r.total, r.line_items[0].line_total), before)


if __name__ == "__main__":
    unittest.main()
