"""
Tests for skim.units.

Pure arithmetic, so synthetic inputs are appropriate -- but the cases
are taken from the real receipts rather than invented, because the
failure modes that matter here are the ones real registers produce.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.normalize import ParsedProduct
from skim.units import (
    COUNT,
    VOLUME,
    WEIGHT,
    dimension_of,
    normalize_line,
    to_base,
)


def product(size_value=None, size_unit=None, pack_count=None, product_name="thing"):
    return ParsedProduct(
        raw_description="RAW",
        brand=None,
        product=product_name,
        variant=None,
        size_value=size_value,
        size_unit=size_unit,
        pack_count=pack_count,
        category="pantry",
        canonical_text=product_name,
        needs_review=False,
    )


class ConversionTests(unittest.TestCase):
    def test_weight_units_convert_to_grams(self):
        self.assertAlmostEqual(to_base(1, "lb"), 453.59237)
        self.assertAlmostEqual(to_base(1, "kg"), 1000.0)
        self.assertAlmostEqual(to_base(16, "oz"), 453.59237, places=4)

    def test_volume_units_convert_to_millilitres(self):
        self.assertAlmostEqual(to_base(1, "gal"), 3785.411784)
        self.assertAlmostEqual(to_base(128, "fl_oz"), 3785.411784, places=4)

    def test_dimension_only_units_are_refused(self):
        # A 15x25 towel has a size, but price-per-inch is not a number
        # anyone wants. Better to decline than to invent.
        self.assertIsNone(dimension_of("in"))
        self.assertIsNone(to_base(15, "in"))

    def test_unknown_units_are_refused(self):
        self.assertIsNone(dimension_of("furlong"))


class WeighedItemTests(unittest.TestCase):
    def test_desi_okra(self):
        # 0.52 @ 2.49 = 1.29. The unit price is ALREADY per pound, so
        # price per pound is 2.49 -- not 1.29/0.52 of anything.
        result = normalize_line(product(), quantity=0.52, unit_price=2.49,
                                line_total=1.29)
        self.assertEqual(result.basis, "weighed")
        self.assertEqual(result.dimension, WEIGHT)
        self.assertAlmostEqual(result.price_per_base * 453.59237, 2.49, places=6)
        self.assertAlmostEqual(result.base_quantity, 0.52 * 453.59237, places=4)

    def test_assumed_pound_is_recorded_not_hidden(self):
        # Step 3 refused to guess the unit. The guess lives here, where
        # it travels with the result and can be corrected.
        result = normalize_line(product(), quantity=3.93, unit_price=1.29,
                                line_total=5.07)
        self.assertEqual(result.inferred_unit, "lb")

    def test_a_printed_unit_is_preferred_over_the_assumption(self):
        result = normalize_line(product(), quantity=0.5, unit_price=4.00,
                                line_total=2.00, printed_unit="kg")
        self.assertIsNone(result.inferred_unit)
        self.assertAlmostEqual(result.price_per_base * 1000, 4.00, places=6)

    def test_missing_unit_price_falls_back_to_the_line_total(self):
        result = normalize_line(product(), quantity=0.52, unit_price=None,
                                line_total=1.29)
        self.assertAlmostEqual(result.price_per_base * 453.59237, 1.29 / 0.52,
                               places=6)


class UnitItemTests(unittest.TestCase):
    def test_onion_sack_uses_the_size_from_the_description(self):
        # 1 @ 6.99 of a 10 lb sack is $0.699/lb. Treating this as a
        # weighed line would report $6.99/lb -- onions priced like
        # saffron, and the error is silent.
        result = normalize_line(product(10, "lb"), quantity=1,
                                unit_price=6.99, line_total=6.99)
        self.assertEqual(result.basis, "unit")
        self.assertAlmostEqual(result.price_per_base * 453.59237, 0.699, places=4)

    def test_two_tubs_of_paneer_double_the_stuff(self):
        # 2 @ 5.99 = 11.98 of 14oz paneer is 28oz, not 14. Missing this
        # makes every multipack wrong by the pack size.
        result = normalize_line(product(14, "oz"), quantity=2,
                                unit_price=5.99, line_total=11.98)
        self.assertAlmostEqual(result.base_quantity, 28 * 28.349523125, places=4)
        self.assertAlmostEqual(result.price_per_base * 28.349523125,
                               11.98 / 28, places=6)

    def test_pack_count_multiplies_on_top_of_quantity(self):
        result = normalize_line(product(12, "fl_oz", pack_count=6), quantity=2,
                                unit_price=5.99, line_total=11.98)
        self.assertEqual(result.dimension, VOLUME)
        # 2 purchases x 6 cans x 12 fl oz
        self.assertAlmostEqual(result.base_quantity, 144 * 29.5735295625, places=3)

    def test_no_size_still_gives_a_price_per_item(self):
        # A rat trap is a rat trap. Price per item is a real comparable
        # number even with no package size.
        result = normalize_line(product(), quantity=1, unit_price=3.43,
                                line_total=3.43)
        self.assertEqual(result.dimension, COUNT)
        self.assertAlmostEqual(result.price_per_base, 3.43)

    def test_a_towel_sized_in_inches_is_declined(self):
        result = normalize_line(product(15, "in"), quantity=1,
                                unit_price=1.50, line_total=1.50)
        self.assertFalse(result.is_comparable)
        self.assertIn("not an amount", result.note)


class DisplayHelperTests(unittest.TestCase):
    def test_price_per_ounce_for_weight(self):
        result = normalize_line(product(10, "lb"), quantity=1,
                                unit_price=6.99, line_total=6.99)
        # 6.99 for 160 oz
        self.assertAlmostEqual(result.price_per_oz, 6.99 / 160, places=5)

    def test_price_per_ounce_is_fluid_ounces_for_volume(self):
        result = normalize_line(product(1, "gal"), quantity=1,
                                unit_price=3.20, line_total=3.20)
        self.assertAlmostEqual(result.price_per_oz, 3.20 / 128, places=5)

    def test_count_items_have_no_per_ounce_price(self):
        result = normalize_line(product(), quantity=1, unit_price=3.43,
                                line_total=3.43)
        self.assertIsNone(result.price_per_oz)


class AmbiguityTests(unittest.TestCase):
    def test_quantity_of_one_is_treated_as_a_unit_item(self):
        # Genuinely ambiguous: one package, or one pound weighed? The
        # cost of being wrong is asymmetric -- a unit item with a known
        # size normalizes correctly, while the weighed reading would
        # report a whole package price as a per-pound price.
        result = normalize_line(product(10, "lb"), quantity=1,
                                unit_price=6.99, line_total=6.99)
        self.assertEqual(result.basis, "unit")

    def test_whole_number_quantities_are_counts(self):
        result = normalize_line(product(), quantity=2, unit_price=0.50,
                                line_total=1.00)
        self.assertEqual(result.basis, "unit")
        self.assertAlmostEqual(result.price_per_base, 0.50)

    def test_a_dimension_is_always_attached_when_comparable(self):
        # Step 8 must never average price-per-gram against
        # price-per-item. A bare number with no dimension would invite
        # exactly that.
        for result in (
            normalize_line(product(10, "lb"), 1, 6.99, 6.99),
            normalize_line(product(), 0.52, 2.49, 1.29),
            normalize_line(product(), 1, 3.43, 3.43),
        ):
            self.assertTrue(result.is_comparable)
            self.assertIsNotNone(result.dimension)


if __name__ == "__main__":
    unittest.main()
