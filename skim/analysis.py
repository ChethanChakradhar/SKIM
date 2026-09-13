"""
Step 8 of the Skim pipeline: Analysis.

Responsibility of this module: turn stored prices into answers.

Everything before this made data trustworthy. This is where it finally
earns its keep -- "what did I pay for milk in March vs August" stops
being a slogan and becomes a query.

THE HONEST CONSTRAINT, stated up front because it shapes every function
here: most of these questions need data that only TIME produces.

    price change for an item   2 purchases of it, on different dates
    cheaper store for an item  the same item bought at 2 stores
    personal inflation rate    a stable basket at 2+ points in time
    which prices spiked        ~5 observations per product for a baseline
    forecast next month        20-30 observations, realistically months

More people do not fix this. A friend's shopping adds breadth -- new
products, new stores, new ways to break the parser -- but a trend needs
the SAME product at DIFFERENT times, and only repeat shopping by one
person produces that.

So every function here degrades honestly. With one receipt it answers
what it can and says nothing it cannot support. `what_unlocks_next`
exists so the app can tell someone what is missing rather than showing
them an empty chart, or worse, a confident line drawn through two points.

A wrong number here is the worst kind this project can produce: every
earlier stage refuses to guess, and it would be perverse to spend that
carefulness on a forecast fitted to four receipts.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# A price series needs readings on at least two separate DAYS before "it
# went up" means anything. Not two readings -- two days. The same receipt
# uploaded twice produces two points and zero elapsed time, and a table
# of "0.0% change" rows built from those would claim a stability that was
# never observed.
MIN_POINTS_FOR_CHANGE = 2

# Below this many observations, the spread of a product's price is not a
# baseline, it is noise. Calling a third purchase "unusually expensive"
# on the strength of two prior ones would be a confident lie.
MIN_POINTS_FOR_ANOMALY = 5

# A basket index compares the cost of the same set of goods at two
# times. Fewer than this and one item's move swamps the number.
MIN_BASKET_SIZE = 5


@dataclass
class PricePoint:
    date: Optional[str]
    store: Optional[str]
    price: float          # in the product's display unit
    unit: Optional[str]   # "lb", "100g", "each"


@dataclass
class ProductHistory:
    """Everything known about one product's price over time."""

    product_id: str
    name: str
    category: Optional[str]
    points: List[PricePoint] = field(default_factory=list)

    @property
    def times_bought(self) -> int:
        return len(self.points)

    @property
    def distinct_dates(self) -> int:
        """How many SEPARATE days this was bought on.

        Not the same as times_bought. Two purchases on one day -- the
        same receipt uploaded twice, or two trips in an afternoon -- give
        two price points with no time between them. A price cannot have
        moved over zero elapsed days, and reporting "0.0% change" for a
        row like that implies a stability nothing here measured.
        """
        return len({p.date for p in self.points if p.date})

    @property
    def unit(self) -> Optional[str]:
        return self.points[0].unit if self.points else None

    @property
    def latest(self) -> Optional[PricePoint]:
        return self.points[-1] if self.points else None

    @property
    def cheapest(self) -> Optional[PricePoint]:
        return min(self.points, key=lambda p: p.price) if self.points else None

    @property
    def dearest(self) -> Optional[PricePoint]:
        return max(self.points, key=lambda p: p.price) if self.points else None

    @property
    def change(self) -> Optional[float]:
        """Price movement from first purchase to most recent, as a percent.

        None when there is only one purchase -- deliberately, rather than
        returning 0%. Zero would read as "the price held steady", which
        is a claim this data cannot make.
        """
        if self.distinct_dates < MIN_POINTS_FOR_CHANGE:
            return None
        first, last = self.points[0].price, self.points[-1].price
        if not first:
            return None
        return (last - first) / first * 100

    @property
    def stores(self) -> List[str]:
        return sorted({p.store for p in self.points if p.store})


def _rows(connection: sqlite3.Connection, sql: str, *args) -> List[sqlite3.Row]:
    return connection.execute(sql, args).fetchall()


def price_histories(connection: sqlite3.Connection,
                    shopper_id: int) -> List[ProductHistory]:
    """Every product this person has bought, with its price each time.

    Ordered oldest first, because the direction of change is the point.
    """
    rows = _rows(connection, """
        SELECT p.product_id, p.canonical_text, p.category,
               r.purchase_date, r.merchant_name,
               li.price_per_display, li.display_unit
        FROM line_items li
        JOIN receipts r ON r.receipt_id = li.receipt_id
        JOIN products  p ON p.product_id = li.product_id
        WHERE r.shopper_id = ?
          AND li.is_voided = 0
          AND li.price_per_display IS NOT NULL
        ORDER BY p.product_id, r.purchase_date, r.receipt_id
    """, shopper_id)

    histories: Dict[str, ProductHistory] = {}
    for row in rows:
        history = histories.setdefault(row["product_id"], ProductHistory(
            product_id=row["product_id"],
            name=row["canonical_text"],
            category=row["category"],
        ))
        history.points.append(PricePoint(
            date=row["purchase_date"],
            store=row["merchant_name"],
            price=row["price_per_display"],
            unit=row["display_unit"],
        ))
    return sorted(histories.values(), key=lambda h: (-h.times_bought, h.name))


def price_changes(histories: List[ProductHistory]) -> List[ProductHistory]:
    """Only the products that have actually been bought more than once.

    Everything else has no change to report, and padding the list with
    them would imply otherwise.
    """
    return [h for h in histories if h.change is not None]


def store_comparison(histories: List[ProductHistory]) -> List[Dict[str, Any]]:
    """Products bought at more than one store, and where they were cheaper.

    This is the "is Store A actually cheaper FOR THE THINGS I BUY"
    question, and it is deliberately per-product. Comparing total spend
    between stores would mostly measure what you happened to buy there.
    """
    results = []
    for history in histories:
        if len(history.stores) < 2:
            continue
        by_store: Dict[str, List[float]] = {}
        for point in history.points:
            if point.store:
                by_store.setdefault(point.store, []).append(point.price)
        averages = {s: sum(v) / len(v) for s, v in by_store.items()}
        cheapest = min(averages, key=averages.get)
        dearest = max(averages, key=averages.get)
        if averages[dearest] == 0:
            continue
        results.append({
            "name": history.name,
            "unit": history.unit,
            "cheapest_store": cheapest,
            "cheapest_price": averages[cheapest],
            "dearest_store": dearest,
            "dearest_price": averages[dearest],
            "gap_percent": (averages[dearest] - averages[cheapest])
                           / averages[cheapest] * 100,
        })
    return sorted(results, key=lambda r: -r["gap_percent"])


def spend_by_category(connection: sqlite3.Connection,
                      shopper_id: int) -> List[Dict[str, Any]]:
    """Where the money went. Works from a single receipt."""
    rows = _rows(connection, """
        SELECT COALESCE(p.category, 'unsorted') category,
               SUM(li.line_total_cents) cents,
               COUNT(*) items
        FROM line_items li
        JOIN receipts r ON r.receipt_id = li.receipt_id
        LEFT JOIN products p ON p.product_id = li.product_id
        WHERE r.shopper_id = ? AND li.is_voided = 0
              AND li.line_total_cents IS NOT NULL
        GROUP BY category ORDER BY cents DESC
    """, shopper_id)
    total = sum(r["cents"] for r in rows) or 1
    return [{
        "category": r["category"],
        "spent": r["cents"] / 100,
        "items": r["items"],
        "share": r["cents"] / total * 100,
    } for r in rows]


def spend_over_time(connection: sqlite3.Connection,
                    shopper_id: int) -> List[Dict[str, Any]]:
    """Every shop, oldest first."""
    rows = _rows(connection, """
        SELECT r.purchase_date date, r.merchant_name store,
               r.total_cents cents, COUNT(li.line_item_id) items
        FROM receipts r
        LEFT JOIN line_items li ON li.receipt_id = r.receipt_id AND li.is_voided = 0
        WHERE r.shopper_id = ?
        GROUP BY r.receipt_id
        ORDER BY r.purchase_date, r.receipt_id
    """, shopper_id)
    return [{
        "date": r["date"], "store": r["store"],
        "spent": (r["cents"] or 0) / 100, "items": r["items"],
    } for r in rows]


def what_unlocks_next(connection: sqlite3.Connection,
                      shopper_id: int) -> List[Dict[str, Any]]:
    """What this person still needs before each answer becomes possible.

    The alternative to this is an empty chart, which tells someone their
    data is boring rather than that it is early. Saying "buy onions once
    more and I can tell you whether they got dearer" is both true and a
    reason to come back.
    """
    histories = price_histories(connection, shopper_id)
    receipts = len(spend_over_time(connection, shopper_id))
    repeat = len(price_changes(histories))
    multi_store = len(store_comparison(histories))
    dates = {p.date for h in histories for p in h.points if p.date}

    unlocks = []

    if repeat == 0:
        # Prefer suggesting something already bought -- it is the likeliest
        # thing to be bought again.
        candidates = sorted(histories, key=lambda h: -h.times_bought)
        example = candidates[0].name if candidates else "something"
        unlocks.append({
            "what": "Whether prices are going up or down",
            "needs": f"Buy {example} again on a different day",
            "why": "A price needs readings on two separate days before it "
                   "can be said to have moved.",
        })

    if multi_store == 0 and receipts:
        unlocks.append({
            "what": "Which store is cheaper for what you actually buy",
            "needs": "Buy the same item at a second store",
            "why": "Comparing whole baskets mostly measures what you bought, "
                   "not what things cost.",
        })

    if len(dates) < 2:
        unlocks.append({
            "what": "Your spending trend",
            "needs": "Buy something on a different day",
            "why": "Everything so far is from a single date.",
        })

    if len(histories) < MIN_BASKET_SIZE or len(dates) < 2:
        unlocks.append({
            "what": "Your own inflation rate",
            "needs": f"About {MIN_BASKET_SIZE} regular items priced on two "
                     "separate store visits",
            "why": "An index compares the same basket at two times. Fewer "
                   "items and one price swing swamps the number.",
        })

    deepest = max((h.distinct_dates for h in histories), default=0)
    if deepest < MIN_POINTS_FOR_ANOMALY:
        unlocks.append({
            "what": "Spotting a price that spiked",
            "needs": f"{MIN_POINTS_FOR_ANOMALY} purchases of the same item "
                     f"(best so far: {deepest})",
            "why": "Calling something unusual needs enough history to know "
                   "what usual looks like.",
        })

    return unlocks


def summary(connection: sqlite3.Connection, shopper_id: int) -> Dict[str, Any]:
    """The handful of numbers worth putting at the top of the page."""
    shops = spend_over_time(connection, shopper_id)
    histories = price_histories(connection, shopper_id)
    changes = price_changes(histories)

    total = sum(s["spent"] for s in shops)
    dates = sorted({s["date"] for s in shops if s["date"]})

    return {
        "shops": len(shops),
        "total_spent": total,
        "average_shop": total / len(shops) if shops else 0,
        "products": len(histories),
        "repeat_products": len(changes),
        "days_covered": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        # Deliberately None, not zero, until there is something to average.
        "average_change": (sum(h.change for h in changes) / len(changes)
                           if changes else None),
    }
