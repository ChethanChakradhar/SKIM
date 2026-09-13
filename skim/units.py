"""
Step 6 of the Skim pipeline: Unit normalization.

Responsibility of this module: make prices comparable.

`$6.99` for onions and `$3.99` for onions tells you nothing until you
know one was a 10 lb sack and the other a 3 lb bag. A personal price
index needs price *per unit of stuff*, not price per package, or every
question it exists to answer is unanswerable:

    "What did I pay for milk in March vs August?"
        -- meaningless if March was a quart and August was a gallon.
    "Is Store A cheaper than Store B for the things I buy?"
        -- meaningless if they sell different package sizes.

Two rules shape everything here.

FIRST: comparison only happens within a dimension.

Price per gram and price per millilitre are not comparable numbers, and
price per gram against price per item is nonsense. So every normalized
price carries the dimension it belongs to, and Step 8 must never compare
across them. A single "normalized price" column with no dimension
attached would silently invite exactly that mistake.

SECOND: a receipt line is one of two completely different things, and
the arithmetic is different for each.

    UNIT item:     ONION 10LB YELLOW    1 @ 6.99 = 6.99
                   You bought one package. The size lives in the product
                   description (10 lb), not in the quantity column.
                   Price per lb = 6.99 / 10 = 0.699

    WEIGHED item:  DESI OKRA            0.52 @ 2.49 = 1.29
                   You bought 0.52 lb. The quantity column IS the
                   weight, and the unit price is ALREADY per pound.
                   Price per lb = 2.49

Run the unit-item arithmetic on a weighed line and you get 1.29 / 0.52
per "package" -- a number with no meaning. Telling them apart is the
core job of this module, and it is done on a heuristic, because the
receipt never says which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from skim.normalize import ParsedProduct

# Everything convertible reduces to one base per dimension. Grams and
# millilitres are chosen over ounces because they are exact integers in
# the metric definitions and because most conversion error comes from
# chaining conversions -- with a single base, every value is one
# multiplication away from every other.
WEIGHT_TO_GRAMS: Dict[str, float] = {
    "g": 1.0,
    "kg": 1000.0,
    "oz": 28.349523125,   # international avoirdupois ounce, exact
    "lb": 453.59237,      # exact by definition since 1959
}

VOLUME_TO_ML: Dict[str, float] = {
    "ml": 1.0,
    "l": 1000.0,
    "fl_oz": 29.5735295625,  # US fluid ounce, exact
    "cup": 236.5882365,      # US legal cup = 8 US fl oz
    "pt": 473.176473,        # US liquid pint = 16 fl oz
    "qt": 946.352946,        # US liquid quart = 32 fl oz
    "gal": 3785.411784,      # US liquid gallon = 128 fl oz
}

# Countable things. "sheet" belongs here rather than in a dimension of
# its own: 75 lint-roller sheets and 80 sketchbook sheets are both just
# counts, and price-per-sheet is the comparable number.
COUNT_TO_EACH: Dict[str, float] = {
    "each": 1.0,
    "ct": 1.0,
    "sheet": 1.0,
}

# Units that describe a shape rather than an amount. A 15x25 kitchen
# towel has a size, but "price per inch" is not a thing anyone wants to
# know, and pretending otherwise would produce confident nonsense. These
# are recognized precisely so they can be declined.
DIMENSION_ONLY_UNITS = {"in"}

WEIGHT, VOLUME, COUNT = "weight", "volume", "count"

BASE_UNITS = {WEIGHT: "g", VOLUME: "ml", COUNT: "each"}


def dimension_of(unit: Optional[str]) -> Optional[str]:
    """Which family of units this belongs to, or None if not comparable."""
    if not unit:
        return None
    if unit in WEIGHT_TO_GRAMS:
        return WEIGHT
    if unit in VOLUME_TO_ML:
        return VOLUME
    if unit in COUNT_TO_EACH:
        return COUNT
    return None


def to_base(value: float, unit: str) -> Optional[float]:
    """Convert an amount into its dimension's base unit (g, ml, or each).

    Returns None for units that describe a shape rather than an amount,
    and for anything unrecognized. A None here propagates as "not
    comparable", which is the honest answer -- far better than a number
    Step 8 would happily average.
    """
    for table in (WEIGHT_TO_GRAMS, VOLUME_TO_ML, COUNT_TO_EACH):
        if unit in table:
            return value * table[unit]
    return None


# When a receipt prints "0.52 @ 2.49" with no unit label, the quantity
# is a weight -- nobody buys 0.52 of an item. On a US grocery receipt
# that weight is in pounds. This is an inference, not a reading, so it
# is recorded on the result rather than applied silently: Step 3
# deliberately returned `unit: null` there instead of guessing, and the
# guess belongs here where it is visible and correctable.
ASSUMED_WEIGHED_UNIT = "lb"

# Below this, a quantity is a weight rather than a count. Anything with
# a fractional part is decisive -- you cannot buy 0.52 onions.
WEIGHED_QUANTITY_EPSILON = 0.001


# Comparison happens in base units, but PEOPLE do not think in base
# units. "$0.549 per 100g of okra" is unreadable to an American shopper
# and, worse, throws away the number actually printed on the receipt --
# which was $2.49 a pound.
#
# So every price carries a second form: the same value expressed in the
# unit the item was really sold in. Pounds for produce weighed at a US
# register, per-100g for something labelled in grams, per item for
# things you just buy one of. The base unit stays underneath for
# comparing; this is purely what gets shown.
DISPLAY_RULES = {
    # origin unit -> (unit to display, how many base units it contains)
    "lb":    ("lb",    WEIGHT_TO_GRAMS["lb"]),
    "oz":    ("lb",    WEIGHT_TO_GRAMS["lb"]),     # US shelf tags price meat by the pound
    "g":     ("100g",  100.0),
    "kg":    ("kg",    1000.0),
    "fl_oz": ("fl oz", VOLUME_TO_ML["fl_oz"]),
    "cup":   ("fl oz", VOLUME_TO_ML["fl_oz"]),
    "pt":    ("fl oz", VOLUME_TO_ML["fl_oz"]),
    "qt":    ("qt",    VOLUME_TO_ML["qt"]),
    "gal":   ("gal",   VOLUME_TO_ML["gal"]),
    "ml":    ("100ml", 100.0),
    "l":     ("l",     1000.0),
}


@dataclass
class NormalizedPrice:
    """A price made comparable, plus the reasoning that got there."""

    dimension: Optional[str]  # weight / volume / count -- NEVER compare across
    base_unit: Optional[str]  # g / ml / each
    base_quantity: Optional[float]  # how much stuff was bought, in base units
    price_per_base: Optional[float]  # dollars per gram / ml / each
    basis: str  # "weighed" or "unit" -- which arithmetic was used
    inferred_unit: Optional[str] = None  # set when we assumed pounds
    note: Optional[str] = None  # why it could not be normalized
    display_unit: Optional[str] = None  # "lb", "100g", "each" -- for humans
    price_per_display: Optional[float] = None  # the same price, in that unit

    @property
    def is_comparable(self) -> bool:
        return self.price_per_base is not None

    @property
    def price_per_100g(self) -> Optional[float]:
        if self.dimension != WEIGHT or self.price_per_base is None:
            return None
        return self.price_per_base * 100

    @property
    def price_per_oz(self) -> Optional[float]:
        """Familiar units for display. US shelf tags use ounces."""
        if self.price_per_base is None:
            return None
        if self.dimension == WEIGHT:
            return self.price_per_base * WEIGHT_TO_GRAMS["oz"]
        if self.dimension == VOLUME:
            return self.price_per_base * VOLUME_TO_ML["fl_oz"]
        return None


def _with_display(price: NormalizedPrice, origin_unit: Optional[str]) -> NormalizedPrice:
    """Attach the human-facing price, in the unit the item was sold in."""
    if price.price_per_base is None:
        return price
    if price.dimension == COUNT:
        price.display_unit, price.price_per_display = "each", price.price_per_base
        return price
    label, base_per_display = DISPLAY_RULES.get(
        (origin_unit or "").lower(),
        ("100g", 100.0) if price.dimension == WEIGHT else ("100ml", 100.0),
    )
    price.display_unit = label
    price.price_per_display = price.price_per_base * base_per_display
    return price


def _looks_weighed(quantity: Optional[float]) -> bool:
    """Is this quantity a weight rather than a count of packages?

    The receipt never says. The signal is that a count is a whole
    number: `2 @ 5.99` is two tubs of paneer, `0.52 @ 2.49` is just
    over half a pound of okra. Nobody buys 0.52 onions.

    A quantity of exactly 1 is genuinely ambiguous -- one package, or
    one pound of something weighed? It is treated as a unit item,
    because the cost of being wrong is asymmetric: a unit item with a
    known package size normalizes correctly, while treating it as
    weighed would report the whole package price as a per-pound price.
    """
    if quantity is None:
        return False
    return abs(quantity - round(quantity)) > WEIGHED_QUANTITY_EPSILON


def normalize_line(
    parsed: ParsedProduct,
    quantity: Optional[float],
    unit_price: Optional[float],
    line_total: Optional[float],
    printed_unit: Optional[str] = None,
) -> NormalizedPrice:
    """Turn one receipt line into a price per unit of stuff.

    `parsed` supplies the package size, which for a unit item lives in
    the product description rather than anywhere in the numbers.
    """
    if _looks_weighed(quantity):
        return _normalize_weighed(quantity, unit_price, line_total, printed_unit)
    return _normalize_unit_item(parsed, quantity, line_total)


def _normalize_weighed(
    quantity: float,
    unit_price: Optional[float],
    line_total: Optional[float],
    printed_unit: Optional[str],
) -> NormalizedPrice:
    """`0.52 @ 2.49 = 1.29` -- the quantity is the amount bought.

    The unit price here is already a price per unit of weight, which is
    why this path is so much simpler than the unit-item one: the
    receipt has done the normalization for us. We only have to know
    what the weight is measured in.
    """
    unit = printed_unit or ASSUMED_WEIGHED_UNIT
    inferred = None if printed_unit else ASSUMED_WEIGHED_UNIT

    dimension = dimension_of(unit)
    if dimension is None:
        return NormalizedPrice(
            None, None, None, None, "weighed",
            note=f"weighed in '{unit}', which is not a convertible unit",
        )

    base_quantity = to_base(quantity, unit)
    # Prefer the printed unit price; fall back to dividing the line
    # total, which gives the same answer when the receipt is consistent
    # and is the only option when no unit price was printed.
    if unit_price is not None:
        per_unit = unit_price
    elif line_total is not None and quantity:
        per_unit = line_total / quantity
    else:
        return NormalizedPrice(
            dimension, BASE_UNITS[dimension], base_quantity, None, "weighed",
            inferred_unit=inferred, note="no unit price and no line total",
        )

    # per_unit is dollars per `unit` (e.g. per lb); divide by how many
    # base units are in one of those to get dollars per base unit.
    price_per_base = per_unit / to_base(1.0, unit)
    return _with_display(NormalizedPrice(
        dimension, BASE_UNITS[dimension], base_quantity, price_per_base,
        "weighed", inferred_unit=inferred,
    ), unit)


def _normalize_unit_item(
    parsed: ParsedProduct,
    quantity: Optional[float],
    line_total: Optional[float],
) -> NormalizedPrice:
    """`ONION 10LB YELLOW  1 @ 6.99` -- the size is in the description.

    This is the path that needs Step 5 to have worked. Without a parsed
    size there is nothing to divide by, and the honest result is "not
    comparable" rather than a price per package dressed up as a
    normalized number.
    """
    count = quantity if quantity else 1.0

    # A size that describes the object is not an amount you bought. A
    # 12.25 oz tumbler is one glass, not 347 grams of glassware; a 10
    # gallon tote is one tote, not ten gallons of tote. Dividing the
    # price by that capacity produces a confident number meaning nothing
    # -- and nothing downstream could ever detect it, because the
    # arithmetic is perfectly valid. Only the premise is wrong.
    #
    # These are priced per item, which is the honest comparable: one
    # storage tote against another storage tote.
    if getattr(parsed, "size_is_capacity", False):
        if line_total is not None:
            return _with_display(NormalizedPrice(
                COUNT, "each", count, line_total / count, "unit",
                note=f"{parsed.size_value:g}{parsed.size_unit} is the size of "
                     "the thing, not how much you got -- priced per item",
            ), "each")
        return NormalizedPrice(None, None, None, None, "unit",
                               note="size describes the object; no line total")

    if parsed.size_unit is None or parsed.size_value is None:
        # Countable goods with no printed size are still comparable as
        # a price per item -- a rat trap is a rat trap.
        if line_total is not None:
            return _with_display(NormalizedPrice(
                COUNT, "each", count, line_total / count, "unit",
                note="no package size; comparable per item only",
            ), "each")
        return NormalizedPrice(None, None, None, None, "unit",
                               note="no package size and no line total")

    if parsed.size_unit in DIMENSION_ONLY_UNITS:
        return NormalizedPrice(
            None, None, None, None, "unit",
            note=f"size is a measurement ('{parsed.size_unit}'), not an amount",
        )

    dimension = dimension_of(parsed.size_unit)
    if dimension is None:
        return NormalizedPrice(None, None, None, None, "unit",
                               note=f"unconvertible unit '{parsed.size_unit}'")

    # Total stuff bought = packages x size per package x packs per unit.
    # `2 @ 5.99` of 14oz paneer is 28oz, not 14.
    packs = parsed.pack_count or 1
    base_quantity = to_base(parsed.size_value, parsed.size_unit) * count * packs

    if line_total is None or not base_quantity:
        return NormalizedPrice(dimension, BASE_UNITS[dimension], base_quantity,
                               None, "unit", note="no line total to divide")

    return _with_display(NormalizedPrice(
        dimension, BASE_UNITS[dimension], base_quantity,
        line_total / base_quantity, "unit",
    ), parsed.size_unit)
