"""
Tests for skim.analysis.

Most of these guard restraint rather than arithmetic. The maths here is
trivial -- percentage change and a few sums. What's easy to break is the
refusal to answer: it is always tempting to make an empty page look
fuller, and every one of those temptations produces a number the data
does not support.

So the tests that matter say what the module must NOT claim.
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import analysis, storage


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = storage.connect(Path(self.tmp.name) / "skim.db")
        self.addCleanup(self.conn.close)
        self.conn.execute(
            "INSERT INTO shoppers (shopper_id, name_key, display_name, pin_hash, "
            "pin_salt, created_at) VALUES (1,'me','Me','h','s','2026-01-01')")
        self.conn.commit()
        # Every receipt needs its own source_file -- that UNIQUE column is
        # what makes re-ingestion safe in production, and an earlier
        # version of this helper derived the name from id(date), which
        # collides because Python interns identical strings.
        self._n = 0

    def receipt(self, date, store, total=10.0):
        self._n += 1
        cursor = self.conn.execute(
            "INSERT INTO receipts (source_file, merchant_name, purchase_date, "
            "total_cents, ingested_at, shopper_id) VALUES (?,?,?,?,?,1)",
            (f"{store}-{date}-{self._n}.jpg", store, date,
             int(total * 100), "2026-01-01"))
        return cursor.lastrowid

    def product(self, product_id, name, category="produce"):
        storage.save_product(self.conn, product_id, name, category=category)

    def line(self, receipt_id, product_id, price, unit="lb", total=5.0):
        self.conn.execute(
            "INSERT INTO line_items (receipt_id, raw_description, product_id, "
            "line_total_cents, price_per_display, display_unit, is_voided) "
            "VALUES (?,?,?,?,?,?,0)",
            (receipt_id, "RAW", product_id, int(total * 100), price, unit))
        self.conn.commit()

    def histories(self):
        return analysis.price_histories(self.conn, 1)

    # -- the refusals ------------------------------------------------

    def test_one_purchase_reports_no_change(self):
        # Not 0%. Zero would read as "the price held steady", which is a
        # claim a single reading cannot support.
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.49)
        self.assertIsNone(self.histories()[0].change)

    def test_two_purchases_on_the_SAME_day_report_no_change(self):
        """The one that caught a real bug.

        Uploading the same receipt twice produced two price points with
        zero elapsed time between them, and the page showed a whole
        table of "0.0% change" rows -- implying a stability nothing had
        measured. A price cannot move over zero days.
        """
        self.product("p", "onion")
        first = self.receipt("2026-01-01", "A")
        second = self.receipt("2026-01-01", "A")
        self.line(first, "p", 2.49)
        self.line(second, "p", 2.49)

        history = self.histories()[0]
        self.assertEqual(history.times_bought, 2)
        self.assertEqual(history.distinct_dates, 1)
        self.assertIsNone(history.change)

    def test_two_purchases_on_different_days_do_report_a_change(self):
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.00)
        self.line(self.receipt("2026-02-01", "A"), "p", 2.50)
        self.assertAlmostEqual(self.histories()[0].change, 25.0)

    def test_a_price_drop_is_negative(self):
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.00)
        self.line(self.receipt("2026-02-01", "A"), "p", 1.50)
        self.assertAlmostEqual(self.histories()[0].change, -25.0)

    def test_products_with_no_change_are_left_out_of_the_changes_list(self):
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.49)
        self.assertEqual(analysis.price_changes(self.histories()), [])

    def test_the_summary_averages_nothing_when_there_is_nothing(self):
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.49)
        self.assertIsNone(analysis.summary(self.conn, 1)["average_change"])

    # -- store comparison --------------------------------------------

    def test_one_store_is_not_a_comparison(self):
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.00)
        self.line(self.receipt("2026-02-01", "A"), "p", 2.50)
        self.assertEqual(analysis.store_comparison(self.histories()), [])

    def test_the_same_item_at_two_stores_is_compared(self):
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "Cheap Mart"), "p", 2.00)
        self.line(self.receipt("2026-01-02", "Posh Foods"), "p", 3.00)

        result = analysis.store_comparison(self.histories())[0]
        self.assertEqual(result["cheapest_store"], "Cheap Mart")
        self.assertEqual(result["dearest_store"], "Posh Foods")
        self.assertAlmostEqual(result["gap_percent"], 50.0)

    # -- things that work from a single receipt ----------------------

    def test_category_spend_works_with_one_receipt(self):
        self.product("a", "onion", category="produce")
        self.product("b", "mop", category="household")
        receipt = self.receipt("2026-01-01", "A")
        self.line(receipt, "a", 2.0, total=6.0)
        self.line(receipt, "b", 1.5, total=4.0)

        rows = analysis.spend_by_category(self.conn, 1)
        self.assertEqual(rows[0]["category"], "produce")
        self.assertAlmostEqual(rows[0]["spent"], 6.0)
        self.assertAlmostEqual(rows[0]["share"], 60.0)

    def test_spend_over_time_is_oldest_first(self):
        self.receipt("2026-03-01", "B", total=20)
        self.receipt("2026-01-01", "A", total=10)
        rows = analysis.spend_over_time(self.conn, 1)
        self.assertEqual([r["date"] for r in rows], ["2026-01-01", "2026-03-01"])

    # -- one person's analysis is their own --------------------------

    def test_another_shopper_is_not_included(self):
        self.conn.execute(
            "INSERT INTO shoppers (shopper_id, name_key, display_name, pin_hash, "
            "pin_salt, created_at) VALUES (2,'you','You','h','s','2026-01-01')")
        self.product("p", "onion")
        cursor = self.conn.execute(
            "INSERT INTO receipts (source_file, merchant_name, purchase_date, "
            "total_cents, ingested_at, shopper_id) VALUES "
            "('theirs.jpg','A','2026-01-01',999,'2026-01-01',2)")
        self.line(cursor.lastrowid, "p", 9.99)

        self.assertEqual(self.histories(), [])
        self.assertEqual(analysis.summary(self.conn, 1)["shops"], 0)

    # -- telling someone what is missing -----------------------------

    def test_an_empty_account_is_told_what_it_needs(self):
        unlocks = analysis.what_unlocks_next(self.conn, 1)
        self.assertTrue(unlocks)
        self.assertTrue(all(u["what"] and u["needs"] and u["why"] for u in unlocks))

    def test_the_price_trend_hint_disappears_once_earned(self):
        # The hint has to stop once the thing it asks for has happened,
        # or the page keeps nagging for something already done.
        self.product("p", "onion")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.00)
        self.line(self.receipt("2026-02-01", "A"), "p", 2.50)

        wants = [u["what"] for u in analysis.what_unlocks_next(self.conn, 1)]
        self.assertNotIn("Whether prices are going up or down", wants)

    def test_the_hint_names_something_already_bought(self):
        # "Buy onion again" is actionable; "buy something again" is not.
        self.product("p", "onion yellow 10 lb")
        self.line(self.receipt("2026-01-01", "A"), "p", 2.00)
        hint = next(u for u in analysis.what_unlocks_next(self.conn, 1)
                    if u["what"] == "Whether prices are going up or down")
        self.assertIn("onion", hint["needs"])


if __name__ == "__main__":
    unittest.main()
