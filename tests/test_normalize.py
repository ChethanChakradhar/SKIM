"""
Tests for skim.normalize.

No API calls. What is tested is the code we wrote -- the coercion rules
that decide which model output we are willing to stand behind, and the
retry policy that decides when a second call is worth paying for.

Whether the model decodes `MS SC HK SET` correctly is a different
question, answered by running scripts/parse_products.py against real
receipt strings and reading the output.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import normalize as nz


def payload(**overrides):
    base = {
        "brand": "Great Value",
        "product": "milk",
        "variant": "2%",
        "size_value": 1,
        "size_unit": "gal",
        "pack_count": None,
        "category": "dairy",
        "canonical_text": "Great Value milk 2% 1 gallon",
        "needs_review": False,
        "parse_note": None,
    }
    base.update(overrides)
    return base


class CoerceTests(unittest.TestCase):
    def test_well_formed_payload_survives_intact(self):
        p = nz._coerce(payload(), "GV MLK 2% 1GAL", attempts=1)
        self.assertEqual(p.brand, "Great Value")
        self.assertEqual(p.size_value, 1)
        self.assertEqual(p.size_unit, "gal")
        self.assertTrue(p.is_identified)

    def test_unknown_unit_is_discarded(self):
        # Step 6 computes price-per-ounce from this field and cannot
        # itself detect a bad unit, so a unit we can't convert is worth
        # less than no unit at all.
        p = nz._coerce(payload(size_unit="gallons-ish"), "X", attempts=1)
        self.assertIsNone(p.size_unit)

    def test_size_without_a_usable_unit_is_dropped_too(self):
        # A bare 310 invites a later stage to assume a unit for it.
        p = nz._coerce(payload(size_value=310, size_unit="furlong"), "X", attempts=1)
        self.assertIsNone(p.size_value)
        self.assertIsNone(p.size_unit)

    def test_an_unseen_category_is_kept_not_flattened(self):
        # A headset is electronics. Forcing it to "other" -- which an
        # earlier fixed list did -- throws away real information about
        # what gets bought, and is how the field became useless for
        # Step 8's "which parts of my basket are inflating".
        p = nz._coerce(payload(category="Electronics"), "HEADSET", attempts=1)
        self.assertEqual(p.category, "electronics")

    def test_known_synonyms_fold_together(self):
        # The open-vocabulary risk is four names for one idea across
        # four receipts, discovered months later when a chart is wrong.
        for variant in ("Dairy", "dairy products", "REFRIGERATED DAIRY"):
            p = nz._coerce(payload(category=variant), "X", attempts=1)
            self.assertEqual(p.category, "dairy")

    def test_category_casing_and_spacing_are_normalized(self):
        p = nz._coerce(payload(category="Personal Care"), "X", attempts=1)
        self.assertEqual(p.category, "personal_care")

    def test_missing_product_forces_review_even_if_model_says_otherwise(self):
        # The model claiming success with nothing to match on is not
        # success.
        p = nz._coerce(payload(product=None, needs_review=False), "BLUE BANDED", attempts=1)
        self.assertTrue(p.needs_review)
        self.assertFalse(p.is_identified)

    def test_null_brand_and_size_are_not_a_review_trigger(self):
        # Loose produce has no brand and no printed size. That is the
        # correct answer, not a failure -- and this is what stops the
        # retry path from firing on almost every item.
        p = nz._coerce(
            payload(brand=None, size_value=None, size_unit=None,
                    product="okra", category="produce"),
            "DESI OKRA", attempts=1,
        )
        self.assertTrue(p.is_identified)
        self.assertFalse(p.needs_review)

    def test_blank_strings_become_none(self):
        p = nz._coerce(payload(brand="   ", variant=""), "X", attempts=1)
        self.assertIsNone(p.brand)
        self.assertIsNone(p.variant)


class RetryPolicyTests(unittest.TestCase):
    """The policy: retry only when unidentified, only with new context,
    and only keep the retry if it actually resolved something."""

    def _run(self, responses, **kwargs):
        calls = []

        def fake_ask(client, prompt, model):
            calls.append(prompt)
            return responses[len(calls) - 1]

        with mock.patch.object(nz, "_ask", fake_ask):
            result = nz.parse_product("BLUE BANDED", client=object(), **kwargs)
        return result, calls

    def test_identified_first_time_makes_one_call(self):
        result, calls = self._run([payload()], store="Walmart",
                                  sibling_descriptions=["IODIZED SALT"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.attempts, 1)

    def test_unidentified_with_context_retries(self):
        unknown = payload(product=None, needs_review=True, parse_note="unclear")
        result, calls = self._run(
            [unknown, payload(product="tilapia", canonical_text="blue banded tilapia")],
            store="Walmart", sibling_descriptions=["BNLS CK BRST", "IODIZED SALT"],
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.product, "tilapia")
        self.assertEqual(result.attempts, 2)

    def test_the_second_prompt_actually_carries_the_new_context(self):
        # A retry with an identical prompt at temperature 0 would just
        # re-buy the same answer. The point of the retry is the context.
        unknown = payload(product=None, needs_review=True)
        _, calls = self._run(
            [unknown, unknown],
            store="Walmart", sibling_descriptions=["BNLS CK BRST", "IODIZED SALT"],
        )
        self.assertNotIn("IODIZED SALT", calls[0])
        self.assertIn("IODIZED SALT", calls[1])
        self.assertIn("Walmart", calls[1])

    def test_no_context_means_no_retry(self):
        # Nothing new to say, so the second call would cost money to
        # receive the same answer.
        unknown = payload(product=None, needs_review=True)
        result, calls = self._run([unknown])
        self.assertEqual(len(calls), 1)
        self.assertFalse(result.is_identified)

    def test_a_failed_retry_keeps_the_first_answer_but_records_the_cost(self):
        first = payload(product=None, needs_review=True, parse_note="first note")
        second = payload(product=None, needs_review=True, parse_note="second note")
        result, calls = self._run(
            [first, second], store="Walmart", sibling_descriptions=["IODIZED SALT"],
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.parse_note, "first note")
        # attempts is the cost record, not which answer we kept.
        self.assertEqual(result.attempts, 2)

    def test_the_item_itself_is_not_listed_among_its_own_siblings(self):
        unknown = payload(product=None, needs_review=True)
        _, calls = self._run(
            [unknown, unknown],
            store="Walmart",
            sibling_descriptions=["BLUE BANDED", "IODIZED SALT"],
        )
        siblings_block = calls[1].split("Other items on the same receipt:")[1]
        self.assertNotIn("- BLUE BANDED", siblings_block)


class VocabularyTests(unittest.TestCase):
    def test_units_step_six_needs_are_present(self):
        # Step 6 converts to a common unit; these are the ones the real
        # receipts actually printed.
        for unit in ("lb", "oz", "g", "gal", "each", "ct"):
            self.assertIn(unit, nz.KNOWN_UNITS)

    def test_a_new_category_is_remembered(self):
        vocab = nz.CategoryVocabulary()
        self.assertTrue(vocab.is_new("electronics"))
        self.assertEqual(vocab.add("Electronics"), "electronics")
        self.assertFalse(vocab.is_new("ELECTRONICS"))

    def test_the_vocabulary_is_offered_back_to_the_model(self):
        # This is what keeps an open vocabulary stable: the second
        # phone charger joins "electronics" instead of founding
        # "consumer electronics". Without it, growth means fragmentation.
        vocab = nz.CategoryVocabulary()
        vocab.add("electronics")
        listed = vocab.as_prompt_list()
        self.assertIn("electronics", listed)
        self.assertIn("produce", listed)

    def test_vocabulary_survives_a_restart(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            nz.CategoryVocabulary(path).add("electronics")
            # A cache restored without its vocabulary would start
            # re-inventing names for products it had already grouped.
            self.assertFalse(nz.CategoryVocabulary(path).is_new("electronics"))

    def test_seed_categories_are_a_starting_point_not_a_ceiling(self):
        vocab = nz.CategoryVocabulary()
        before = len(vocab.categories)
        vocab.add("stationery")
        vocab.add("clothing")
        self.assertEqual(len(vocab.categories), before + 2)


if __name__ == "__main__":
    unittest.main()
