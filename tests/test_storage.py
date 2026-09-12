"""
Tests for skim.storage.

Every test runs against a temporary database file rather than an
in-memory one, because two of the behaviours that matter most --
foreign key cascades and re-ingest replacing rows -- are exactly the
kind that behave differently once real persistence is involved.
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import storage
from skim.extract import LineItem, Receipt
from skim.units import normalize_line
from skim.validate import validate


def receipt(**overrides):
    base = dict(
        merchant_name="Walmart", store_number="03803", address=None, phone=None,
        purchase_date="2026-08-25", purchase_time="12:29",
        line_items=[], subtotal=41.80, tax=1.97, total=43.77,
        tax_rate_percent=6.35, amount_paid=60.00, change_given=16.25,
        rounding_adjustment=0.02, item_count_printed=9,
    )
    base.update(overrides)
    return Receipt(**base)


def line(number=1, description="RAT TRAP", total=3.43, **kwargs):
    return LineItem(line_number=number, raw_description=description,
                    line_total=total, **kwargs)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "skim.db"

    def test_connecting_creates_the_schema(self):
        conn = storage.connect(self.path)
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for expected in ("receipts", "line_items", "products",
                         "product_aliases", "validation_results"):
            self.assertIn(expected, tables)

    def test_foreign_keys_are_actually_enabled(self):
        # SQLite ignores foreign keys unless asked, per connection. If
        # this is off, every ON DELETE CASCADE in the schema is
        # decoration and deleting a receipt orphans its line items into
        # permanently invisible rows.
        conn = storage.connect(self.path)
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_reopening_an_existing_database_is_fine(self):
        storage.connect(self.path).close()
        conn = storage.connect(self.path)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0], 1)

    def test_an_unknown_schema_version_fails_loudly(self):
        conn = storage.connect(self.path)
        conn.execute("UPDATE schema_version SET version = 99")
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError):
            storage.connect(self.path)


class ReceiptWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = storage.connect(Path(self.tmp.name) / "skim.db")
        self.addCleanup(self.conn.close)

    def test_money_is_stored_as_integer_cents(self):
        storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        row = self.conn.execute("SELECT * FROM receipts").fetchone()
        self.assertEqual(row["total_cents"], 4377)
        self.assertIsInstance(row["total_cents"], int)

    def test_change_is_stored_as_a_magnitude(self):
        # Dollar Tree prints change as "-2.77". Step 4 established that
        # the sign is a register quirk and the magnitude is the fact.
        storage.save_receipt(self.conn, "IMG_1.jpg",
                             receipt(change_given=-2.77, total=12.23))
        row = self.conn.execute("SELECT change_cents FROM receipts").fetchone()
        self.assertEqual(row["change_cents"], 277)

    def test_re_ingesting_replaces_rather_than_duplicates(self):
        # The single most important property here. Re-running is the
        # normal case -- every prompt change means reprocessing the same
        # photos -- and without this, every price in the index doubles.
        storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        second = storage.save_receipt(self.conn, "IMG_1.jpg", receipt(total=99.99))

        rows = self.conn.execute("SELECT * FROM receipts").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total_cents"], 9999)
        # The returned id must address the row that now exists -- SQLite
        # may reuse the old rowid after the delete, which is harmless
        # because the cascade removed everything referencing it.
        self.assertEqual(rows[0]["receipt_id"], second)

    def test_replacing_a_receipt_takes_its_line_items_with_it(self):
        receipt_id = storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        storage.save_line_item(self.conn, receipt_id, line())
        self.conn.commit()

        storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        remaining = self.conn.execute("SELECT COUNT(*) c FROM line_items").fetchone()
        # Stale line items surviving a re-ingest would attach a previous
        # extraction's prices to the new receipt id -- or to nothing.
        self.assertEqual(remaining["c"], 0)

    def test_different_photos_are_different_receipts(self):
        storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        storage.save_receipt(self.conn, "IMG_2.jpg", receipt())
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) c FROM receipts").fetchone()["c"], 2)


class ProductAndEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = storage.connect(Path(self.tmp.name) / "skim.db")
        self.addCleanup(self.conn.close)

    def test_embeddings_round_trip_through_a_blob(self):
        vector = np.array([0.1, -0.25, 0.5], dtype=np.float32)
        storage.save_product(self.conn, "p1", "guava", embedding=vector)
        self.conn.commit()

        blob = self.conn.execute("SELECT embedding FROM products").fetchone()[0]
        restored = storage.blob_to_embedding(blob)
        np.testing.assert_allclose(restored, vector)

    def test_saving_a_product_twice_updates_it(self):
        storage.save_product(self.conn, "p1", "guava", category="produce")
        storage.save_product(self.conn, "p1", "guava", category="fruit")
        self.conn.commit()
        rows = self.conn.execute("SELECT * FROM products").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "fruit")

    def test_an_alias_links_a_receipt_string_to_a_product(self):
        storage.save_product(self.conn, "p1", "guava")
        storage.save_alias(self.conn, "India Market", "GUAVA - 1", "p1")
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM product_aliases").fetchone()
        self.assertEqual(row["product_id"], "p1")
        self.assertEqual(row["store"], "INDIA MARKET")

    def test_unidentified_strings_are_kept_as_a_worklist(self):
        # BLUE BANDED. Storing it with needs_review makes it queryable;
        # dropping it would mean re-parsing to rediscover the same
        # unknown.
        storage.save_alias(self.conn, "Walmart", "BLUE BANDED", None,
                           needs_review=True)
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM product_aliases WHERE needs_review = 1").fetchone()
        self.assertEqual(row["raw_description"], "BLUE BANDED")
        self.assertIsNone(row["product_id"])


class ValidationAndNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = storage.connect(Path(self.tmp.name) / "skim.db")
        self.addCleanup(self.conn.close)

    def test_validation_verdicts_are_kept_not_printed(self):
        # The pass rate over time is the project's accuracy metric, and
        # it cannot be computed from output that only reached a terminal.
        items = [line(1, "A", 3.43), line(2, "B", 9.98)]
        r = receipt(line_items=items, subtotal=13.41, tax=0.85, total=14.26,
                    tax_rate_percent=None, amount_paid=None, change_given=None,
                    item_count_printed=None)
        receipt_id = storage.save_receipt(self.conn, "IMG_1.jpg", r)
        storage.save_validation(self.conn, receipt_id, validate(r))
        self.conn.commit()

        statuses = {row["status"] for row in self.conn.execute(
            "SELECT status FROM validation_results")}
        self.assertIn("pass", statuses)

    def test_normalized_price_and_its_dimension_are_stored_together(self):
        from skim.normalize import ParsedProduct

        parsed = ParsedProduct("ONION 10LB YELLOW", None, "onion", None,
                               10, "lb", None, "produce", "onion 10 lb", False)
        normalized = normalize_line(parsed, quantity=1, unit_price=6.99,
                                    line_total=6.99)
        receipt_id = storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        storage.save_line_item(self.conn, receipt_id,
                               line(description="ONION 10LB YELLOW", total=6.99),
                               product_id=None, normalized=normalized)
        self.conn.commit()

        row = self.conn.execute("SELECT * FROM line_items").fetchone()
        # The dimension must travel with the price, or a later query can
        # average price-per-gram against price-per-item.
        self.assertEqual(row["dimension"], "weight")
        self.assertEqual(row["base_unit"], "g")
        self.assertAlmostEqual(row["price_per_base"] * 453.59237, 0.699, places=4)

    def test_an_assumed_unit_is_recorded_on_the_row(self):
        from skim.normalize import ParsedProduct

        parsed = ParsedProduct("DESI OKRA", None, "okra", None, None, None,
                               None, "produce", "okra", False)
        normalized = normalize_line(parsed, quantity=0.52, unit_price=2.49,
                                    line_total=1.29)
        receipt_id = storage.save_receipt(self.conn, "IMG_1.jpg", receipt())
        storage.save_line_item(self.conn, receipt_id, line(), None, normalized)
        self.conn.commit()

        row = self.conn.execute("SELECT inferred_unit FROM line_items").fetchone()
        # Every price resting on an assumption stays findable.
        self.assertEqual(row["inferred_unit"], "lb")


if __name__ == "__main__":
    unittest.main()
