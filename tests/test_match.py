"""
Tests for skim.match.

The decision layer is the part worth testing hard, because it is the
part that can silently corrupt the price index. A wrong merge is
permanent and invisible: two products become one, their prices average
together, and nothing downstream ever flags it.

Embeddings and adjudication are stubbed. What is tested is the rules.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import match as mt
from skim.normalize import ParsedProduct


def parsed(product, brand=None, variant=None, size_value=None, size_unit=None,
           raw="RAW", canonical=None):
    return ParsedProduct(
        raw_description=raw,
        brand=brand,
        product=product,
        variant=variant,
        size_value=size_value,
        size_unit=size_unit,
        pack_count=None,
        category="produce",
        canonical_text=canonical or product,
        needs_review=False,
    )


def catalog_entry(product, brand=None, variant=None, canonical=None, pid="p1"):
    return mt.CatalogProduct(
        product_id=pid,
        canonical_text=canonical or product,
        product=product,
        brand=brand,
        variant=variant,
        category="produce",
        embedding=[1.0, 0.0],
    )


class CompareAttributesTests(unittest.TestCase):
    def test_the_guava_case_merges(self):
        # The bug this layer exists to fix. Same product noun, and the
        # only difference is a size the register invented. Similarity
        # alone could not settle this: 0.839 for the guavas vs 0.828 for
        # two genuinely different chicken products.
        verdict, _ = mt.compare_attributes(
            parsed("guava", size_value=1, size_unit="each", canonical="guava 1 each"),
            catalog_entry("guava"),
        )
        self.assertEqual(verdict, mt.SAME)

    def test_different_package_sizes_are_the_same_product(self):
        # Counter-intuitive but correct: Step 6 has already converted
        # both to a price per millilitre, so a gallon and a quart are
        # one product bought in two amounts. Splitting them would
        # defeat the purpose of normalizing units at all.
        verdict, _ = mt.compare_attributes(
            parsed("milk", brand="Great Value", variant="2%",
                   size_value=1, size_unit="gal"),
            catalog_entry("milk", brand="Great Value", variant="2%"),
        )
        self.assertEqual(verdict, mt.SAME)

    def test_the_chicken_case_is_not_auto_merged(self):
        # whole chicken vs chicken breast: 0.828 similarity, genuinely
        # different products. Different nouns go to the adjudicator
        # rather than being guessed either way.
        verdict, _ = mt.compare_attributes(
            parsed("chicken breast", variant="boneless"),
            catalog_entry("whole chicken", variant="skinless"),
        )
        self.assertIn(verdict, (mt.UNSURE, mt.DIFFERENT))

    def test_different_brands_never_merge(self):
        verdict, why = mt.compare_attributes(
            parsed("milk", brand="Lactaid"),
            catalog_entry("milk", brand="Great Value"),
        )
        self.assertEqual(verdict, mt.DIFFERENT)
        self.assertIn("brand", why)

    def test_a_missing_brand_is_not_a_conflict(self):
        # Receipts routinely omit a brand that another receipt names.
        # Treating absence as disagreement would split products
        # constantly.
        verdict, _ = mt.compare_attributes(
            parsed("paneer", brand=None),
            catalog_entry("paneer", brand="Swad"),
        )
        self.assertEqual(verdict, mt.SAME)

    def test_different_variants_never_merge(self):
        verdict, why = mt.compare_attributes(
            parsed("milk", variant="whole"),
            catalog_entry("milk", variant="2%"),
        )
        self.assertEqual(verdict, mt.DIFFERENT)
        self.assertIn("variant", why)

    def test_comparison_ignores_case_and_spacing(self):
        verdict, _ = mt.compare_attributes(
            parsed("  Plum Tomato "), catalog_entry("plum tomato"))
        self.assertEqual(verdict, mt.SAME)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.catalog = mt.ProductCatalog(Path(self.tmp.name) / "catalog.json")

    def _add(self, name, vector):
        v = np.array(vector, dtype=np.float32)
        v /= np.linalg.norm(v)
        return self.catalog.add(parsed(name), v, f"STORE||{name}")

    def test_an_empty_catalog_returns_no_candidates(self):
        self.assertEqual(self.catalog.find_candidates(np.array([1.0, 0.0])), [])

    def test_candidates_come_back_ranked(self):
        self._add("near", [1.0, 0.1])
        self._add("far", [0.2, 1.0])
        results = self.catalog.find_candidates(np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(results[0][1].product, "near")

    def test_the_floor_excludes_unrelated_products(self):
        # A retrieval floor, not a match threshold -- nothing is ever
        # merged for clearing it.
        self._add("orthogonal", [0.0, 1.0])
        self.assertEqual(
            self.catalog.find_candidates(np.array([1.0, 0.0], dtype=np.float32)), []
        )

    def test_catalog_survives_a_restart(self):
        created = self._add("guava", [1.0, 0.0])
        reloaded = mt.ProductCatalog(self.catalog.path)
        self.assertIn(created.product_id, reloaded.products)
        self.assertEqual(reloaded.products[created.product_id].product, "guava")


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.catalog = mt.ProductCatalog(Path(self.tmp.name) / "catalog.json")
        self.embed = mock.patch.object(
            mt, "embed_texts",
            side_effect=lambda texts, client=None: np.array(
                [[1.0, 0.0]] * len(texts), dtype=np.float32),
        ).start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(mt, "_load_client", return_value=object()).start()

    def test_a_known_alias_costs_no_api_call(self):
        # The common case once a catalog has settled, and the reason
        # per-receipt cost trends toward zero.
        p = parsed("guava", raw="GUAVA")
        mt.resolve(p, self.catalog, store="India Market")
        self.embed.reset_mock()

        decision = mt.resolve(p, self.catalog, store="India Market")
        self.assertEqual(decision.outcome, mt.MATCHED)
        self.embed.assert_not_called()

    def test_a_new_product_is_created(self):
        decision = mt.resolve(parsed("guava", raw="GUAVA"), self.catalog,
                              store="India Market")
        self.assertEqual(decision.outcome, mt.CREATED)
        self.assertEqual(len(self.catalog.products), 1)

    def test_a_second_string_for_the_same_product_becomes_an_alias(self):
        mt.resolve(parsed("guava", raw="GUAVA"), self.catalog, store="India Market")
        decision = mt.resolve(
            parsed("guava", size_value=1, size_unit="each",
                   raw="GUAVA - 1", canonical="guava 1 each"),
            self.catalog, store="India Market",
        )
        self.assertEqual(decision.outcome, mt.MATCHED)
        # One product, two receipt strings -- the entire point.
        self.assertEqual(len(self.catalog.products), 1)
        product = list(self.catalog.products.values())[0]
        self.assertEqual(len(product.aliases), 2)

    def test_a_conflicting_attribute_creates_a_separate_product(self):
        mt.resolve(parsed("milk", brand="Great Value", raw="GV MLK"),
                   self.catalog, store="Walmart")
        mt.resolve(parsed("milk", brand="Lactaid", raw="LACTAID MLK"),
                   self.catalog, store="Walmart")
        self.assertEqual(len(self.catalog.products), 2)

    def test_an_unsure_adjudication_goes_to_review_and_does_not_merge(self):
        # The most dangerous moment in the pipeline: two products that
        # look alike, where a wrong merge would never be noticed.
        mt.resolve(parsed("whole chicken", raw="WHOLE CHICKEN"), self.catalog,
                   store="Walmart")
        with mock.patch.object(mt, "adjudicate", return_value=(mt.UNSURE, "too terse")):
            decision = mt.resolve(parsed("chicken breast", raw="BNLS CK BRST"),
                                  self.catalog, store="Walmart")
        self.assertEqual(decision.outcome, mt.REVIEW)
        self.assertEqual(len(self.catalog.products), 1)  # nothing created, nothing merged

    def test_adjudication_can_confirm_a_match(self):
        mt.resolve(parsed("chili", raw="GREEN CHILI"), self.catalog, store="X")
        with mock.patch.object(mt, "adjudicate", return_value=(mt.SAME, "spelling")):
            decision = mt.resolve(parsed("chilli", raw="GREEN CHILLI"),
                                  self.catalog, store="X")
        self.assertEqual(decision.outcome, mt.MATCHED)
        self.assertTrue(decision.adjudicated)

    def test_adjudication_can_reject_a_match(self):
        mt.resolve(parsed("whole chicken", raw="WHOLE CHICKEN"), self.catalog,
                   store="Walmart")
        with mock.patch.object(mt, "adjudicate",
                               return_value=(mt.DIFFERENT, "different cuts")):
            decision = mt.resolve(parsed("chicken breast", raw="BNLS CK BRST"),
                                  self.catalog, store="Walmart")
        self.assertEqual(decision.outcome, mt.CREATED)
        self.assertEqual(len(self.catalog.products), 2)

    def test_an_unidentified_parse_never_reaches_the_catalog(self):
        # BLUE BANDED. Nothing to match on, so nothing should be
        # guessed -- and it must not pollute the catalog either.
        unknown = ParsedProduct("BLUE BANDED", None, None, None, None, None,
                                None, None, None, needs_review=True,
                                parse_note="lacks the item noun")
        decision = mt.resolve(unknown, self.catalog, store="Walmart")
        self.assertEqual(decision.outcome, mt.REVIEW)
        self.assertEqual(len(self.catalog.products), 0)


if __name__ == "__main__":
    unittest.main()
