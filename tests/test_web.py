"""
Tests for the web app's access boundaries.

Not the HTML -- the rules. Every one of these is a way someone could see
data that isn't theirs, which makes them the tests most worth having:
a broken layout is embarrassing, a broken boundary is a breach.

No API calls. Uploading is not exercised here (it would cost money and
need a network); what is exercised is who can reach what.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class WebAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # Point the app at a throwaway database and fix the admin name
        # before it is imported -- both are read at import time.
        os.environ["SKIM_DATA_DIR"] = cls.tmp.name
        os.environ["SKIM_ADMIN"] = "chethan"
        os.environ["SKIM_SECRET_KEY"] = "test-key-not-a-real-secret"

        from fastapi.testclient import TestClient
        from web import app as webapp

        cls.webapp = webapp
        cls.client_factory = lambda self=None: TestClient(webapp.app, follow_redirects=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def signed_in(self, name, pin="4821"):
        client = self.client_factory()
        client.post("/signin", data={"name": name, "pin": pin})
        return client

    # -- signed out --------------------------------------------------

    def test_the_dashboard_needs_a_session(self):
        response = self.client_factory().get("/me")
        self.assertEqual(response.status_code, 303)

    def test_a_receipt_needs_a_session(self):
        response = self.client_factory().get("/receipt/1")
        self.assertEqual(response.status_code, 303)

    def test_the_health_page_needs_a_session(self):
        response = self.client_factory().get("/admin")
        self.assertEqual(response.status_code, 303)

    # -- sessions ----------------------------------------------------

    def test_signing_in_creates_a_session(self):
        client = self.signed_in("Chethan")
        self.assertEqual(client.get("/me").status_code, 200)

    def test_a_forged_cookie_is_refused(self):
        # A cookie holding just "5" would let anyone become shopper 5.
        # The HMAC is what stops that, and this is the test that says so.
        client = self.client_factory()
        client.cookies.set(self.webapp.COOKIE_NAME, "1.deadbeefdeadbeef")
        self.assertEqual(client.get("/me").status_code, 303)

    def test_a_cookie_without_a_signature_is_refused(self):
        client = self.client_factory()
        client.cookies.set(self.webapp.COOKIE_NAME, "1")
        self.assertEqual(client.get("/me").status_code, 303)

    def test_signing_out_ends_the_session(self):
        client = self.signed_in("Phoebe", "7777")
        self.assertEqual(client.get("/me").status_code, 200)
        client.get("/logout")
        self.assertEqual(client.get("/me").status_code, 303)

    # -- the admin boundary ------------------------------------------

    def test_the_admin_can_see_the_health_page(self):
        client = self.signed_in("Chethan")
        self.assertEqual(client.get("/admin").status_code, 200)

    def test_a_friend_cannot_see_the_health_page(self):
        # The whole point of the page being aggregate is undone if
        # everyone can open it.
        client = self.signed_in("Monica", "1234")
        self.assertEqual(client.get("/admin").status_code, 303)

    def test_admin_is_matched_case_insensitively(self):
        client = self.signed_in("CHETHAN")
        self.assertEqual(client.get("/admin").status_code, 200)

    def test_the_health_page_shows_no_item_names(self):
        """The page reports how receipts were READ, never what was bought.

        Asserting on the wording would be brittle -- an earlier version of
        this test broke on a line wrap. So it plants a product with an
        unmistakable name and checks that name never reaches the page.
        If this fails, the page has started leaking shopping.
        """
        from skim import storage

        connection = storage.connect(Path(self.tmp.name) / "skim.db")
        try:
            storage.save_product(connection, "leak-check",
                                 "UNMISTAKABLE PRODUCT NAME", product="canary")
            connection.commit()
        finally:
            connection.close()

        body = self.signed_in("Chethan").get("/admin").text
        self.assertNotIn("UNMISTAKABLE PRODUCT NAME", body)
        self.assertNotIn("canary", body)

    # -- one person's receipts are not another's ---------------------

    def test_a_receipt_belonging_to_someone_else_is_refused(self):
        """The most important test in the file.

        Without `AND shopper_id = ?` in the lookup, changing the number
        in the URL walks straight through everybody's receipts. That is
        the single most common way an app like this leaks, and it costs
        one clause to prevent.
        """
        from skim import storage

        connection = storage.connect(Path(self.tmp.name) / "skim.db")
        try:
            owner = connection.execute(
                "SELECT shopper_id FROM shoppers WHERE name_key = 'monica'"
            ).fetchone()
            if owner is None:
                self.signed_in("Monica", "1234")
                owner = connection.execute(
                    "SELECT shopper_id FROM shoppers WHERE name_key = 'monica'"
                ).fetchone()

            cursor = connection.execute(
                "INSERT INTO receipts (source_file, merchant_name, ingested_at, "
                "shopper_id) VALUES (?,?,?,?)",
                ("monicas_receipt.jpg", "Store", "2026-01-01", owner["shopper_id"]),
            )
            receipt_id = cursor.lastrowid
            connection.commit()
        finally:
            connection.close()

        # Monica can see her own.
        self.assertEqual(
            self.signed_in("Monica", "1234").get(f"/receipt/{receipt_id}").status_code, 200)
        # Chethan cannot -- not even as the admin.
        self.assertEqual(
            self.signed_in("Chethan").get(f"/receipt/{receipt_id}").status_code, 303)

    def test_deleting_someone_elses_receipt_does_nothing(self):
        from skim import storage

        connection = storage.connect(Path(self.tmp.name) / "skim.db")
        try:
            self.signed_in("Rachel", "2222")
            owner = connection.execute(
                "SELECT shopper_id FROM shoppers WHERE name_key = 'rachel'").fetchone()
            cursor = connection.execute(
                "INSERT INTO receipts (source_file, merchant_name, ingested_at, "
                "shopper_id) VALUES (?,?,?,?)",
                ("rachels.jpg", "Store", "2026-01-01", owner["shopper_id"]))
            receipt_id = cursor.lastrowid
            connection.commit()
        finally:
            connection.close()

        self.signed_in("Chethan").post(f"/receipt/{receipt_id}/delete")

        connection = storage.connect(Path(self.tmp.name) / "skim.db")
        try:
            still_there = connection.execute(
                "SELECT 1 FROM receipts WHERE receipt_id = ?", (receipt_id,)).fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(still_there, "someone else's receipt was deleted")


if __name__ == "__main__":
    unittest.main()
