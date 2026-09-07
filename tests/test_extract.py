"""
Tests for skim.extract.

None of these call the API. That is deliberate: a test that costs money
and needs a network is a test you stop running, and a test whose result
depends on what a model felt like saying today can't tell you whether
*your code* broke.

So the model is stubbed out and what gets tested is everything we
actually wrote -- the retry policy, the JSON-to-dataclass conversion, the
failure modes. Whether the model reads receipts correctly is a different
question, answered by running real photos and checking the arithmetic
(Step 4), not by unit tests.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

from google.genai import errors as genai_errors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import extract as ex


def server_error(code):
    return genai_errors.ServerError(code, {"error": {"message": "overloaded"}})


def client_error(code):
    return genai_errors.ClientError(code, {"error": {"message": "bad request"}})


class FakeModels:
    """Stands in for client.models, recording how often it was called."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


class RetryPolicyTests(unittest.TestCase):
    def setUp(self):
        # Patch out the backoff sleep -- we're testing the decision to
        # retry, not our ability to wait around for six seconds.
        patcher = mock.patch.object(ex.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def _call(self, client):
        return ex._generate_with_retry(client, "test-model", contents=[], config=None)

    def test_transient_overload_is_retried_until_it_succeeds(self):
        client = FakeClient([server_error(503), server_error(503), "the response"])
        self.assertEqual(self._call(client), "the response")
        self.assertEqual(client.models.calls, 3)

    def test_rate_limiting_is_retried(self):
        client = FakeClient([client_error(429), "the response"])
        self.assertEqual(self._call(client), "the response")
        self.assertEqual(client.models.calls, 2)

    def test_permanent_failure_is_not_retried(self):
        # A 404 will fail identically on the third attempt as the first.
        # Retrying it wastes time and buries a real bug behind a delay.
        client = FakeClient([client_error(404)])
        with self.assertRaises(genai_errors.ClientError):
            self._call(client)
        self.assertEqual(client.models.calls, 1)

    def test_sustained_overload_raises_a_readable_error(self):
        client = FakeClient([server_error(503)] * ex.MAX_ATTEMPTS)
        with self.assertRaises(ex.ModelUnavailableError):
            self._call(client)
        self.assertEqual(client.models.calls, ex.MAX_ATTEMPTS)

    def test_backoff_grows_between_attempts(self):
        client = FakeClient([server_error(503)] * ex.MAX_ATTEMPTS)
        with self.assertRaises(ex.ModelUnavailableError):
            self._call(client)
        waits = [call.args[0] for call in self.sleep.call_args_list]
        self.assertEqual(waits, [2.0, 4.0])


class ParseReceiptTests(unittest.TestCase):
    def payload(self, **overrides):
        base = {
            "merchant_name": "Walmart",
            "store_number": "03803",
            "purchase_date": "2026-08-25",
            "subtotal": 41.80,
            "tax": 1.97,
            "total": 43.77,
            "line_items": [
                {
                    "line_number": 1,
                    "raw_description": "BNLS CK BRST",
                    "line_total": 9.83,
                    "quantity": 1,
                    "unit": "each",
                    "tax_flag": "N",
                    "is_voided": False,
                },
                {
                    "line_number": 2,
                    "raw_description": "1PK INC AUTO",
                    "line_total": None,
                    "is_voided": True,
                },
            ],
        }
        base.update(overrides)
        return base

    def test_payload_becomes_typed_objects(self):
        receipt = ex._parse_receipt(self.payload())
        self.assertEqual(receipt.merchant_name, "Walmart")
        self.assertEqual(receipt.subtotal, 41.80)
        self.assertEqual(len(receipt.line_items), 2)

        first = receipt.line_items[0]
        self.assertEqual(first.raw_description, "BNLS CK BRST")
        self.assertEqual(first.line_total, 9.83)
        self.assertFalse(first.is_voided)

    def test_voided_line_is_kept_not_dropped(self):
        # Dropping it would look tidier and would be wrong: the receipt
        # printed it, and Step 4 needs to know why the item count and the
        # charged count disagree.
        receipt = ex._parse_receipt(self.payload())
        voided = receipt.line_items[1]
        self.assertTrue(voided.is_voided)
        self.assertIsNone(voided.line_total)

    def test_missing_optional_fields_become_none(self):
        receipt = ex._parse_receipt(self.payload())
        self.assertIsNone(receipt.line_items[1].tax_flag)
        self.assertIsNone(receipt.line_items[1].quantity)
        self.assertIsNone(receipt.phone)

    def test_currency_defaults_to_usd_when_absent(self):
        self.assertEqual(ex._parse_receipt(self.payload()).currency, "USD")

    def test_currency_defaults_when_model_sends_null(self):
        self.assertEqual(
            ex._parse_receipt(self.payload(currency=None)).currency, "USD"
        )

    def test_line_item_missing_required_field_fails_loudly(self):
        # Better to fail at this boundary than to surface as a TypeError
        # inside the analysis layer five stages later.
        broken = self.payload(line_items=[{"line_number": 1, "is_voided": False}])
        with self.assertRaises(ex.MalformedResponseError):
            ex._parse_receipt(broken)

    def test_payload_with_no_line_items_key_fails_loudly(self):
        with self.assertRaises(ex.MalformedResponseError):
            ex._parse_receipt({"merchant_name": "Walmart"})


class ApiKeyTests(unittest.TestCase):
    def test_missing_key_explains_how_to_fix_it(self):
        # Patch load_dotenv out, or it would helpfully load the real .env
        # and the test would pass only on machines with no key.
        with mock.patch.object(ex, "load_dotenv"), \
                mock.patch.dict(ex.os.environ, {"GEMINI_API_KEY": ""}, clear=True):
            with self.assertRaises(ex.MissingAPIKeyError) as caught:
                ex._load_client()
        self.assertIn("aistudio.google.com", str(caught.exception))

    def test_placeholder_key_is_treated_as_missing(self):
        # .env.example ships with this value; copying it without editing
        # is the obvious mistake, so name it rather than sending a
        # doomed request to the API.
        with mock.patch.object(ex, "load_dotenv"), \
                mock.patch.dict(
                    ex.os.environ, {"GEMINI_API_KEY": "your-key-here"}, clear=True):
            with self.assertRaises(ex.MissingAPIKeyError):
                ex._load_client()


class SchemaTests(unittest.TestCase):
    def test_raw_description_is_required_of_every_line_item(self):
        # The whole project depends on the verbatim string surviving to
        # Step 5. If it were optional, the model could omit it and we'd
        # store a price with nothing to attach it to.
        item_schema = ex.RECEIPT_SCHEMA["properties"]["line_items"]["items"]
        self.assertIn("raw_description", item_schema["required"])
        self.assertIn("line_number", item_schema["required"])

    def test_totals_are_nullable(self):
        # A receipt with an unreadable subtotal must still extract. If
        # these were required, one smudged number would fail the whole
        # receipt instead of costing us one field.
        for field_name in ("subtotal", "tax", "total"):
            self.assertTrue(ex.RECEIPT_SCHEMA["properties"][field_name]["nullable"])


if __name__ == "__main__":
    unittest.main()
