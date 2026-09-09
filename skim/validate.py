"""
Step 4 of the Skim pipeline: Validation.

Responsibility of this module: decide how much to trust an extracted
receipt, without anyone having labeled anything.

The idea it rests on: a receipt is a *redundant* document. It states the
same facts more than once, in different forms. The line items imply the
subtotal. The subtotal and tax imply the total. The tax flags and the
printed tax rate imply the tax. Cash tendered minus the total implies the
change. When the model misreads a digit, those statements stop agreeing.

That gives us an accuracy signal that costs nothing to produce and needs
no ground truth -- which is the only kind of accuracy signal a one-person
project is ever going to have.

What this module does NOT do, and it matters:

It checks arithmetic consistency, not transcription fidelity. If the
model read "MILK" as "MILT", every check here still passes. And if the
price column were read shifted by one row against the descriptions, the
items would still sum to the subtotal and each line's arithmetic would
still agree internally -- the description-to-price *binding* is the one
thing none of these checks can verify.

So a 95% pass rate means "95% internally consistent", not "95% correct".
Worth saying out loud, because it is easy to quote the number as if it
meant the second thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from skim.extract import Receipt

# Per-line checks recompute quantity x unit_price and compare against the
# printed line total, and the receipt itself rounds that product to the
# cent: 0.52 lb x $2.49 = $1.2948, printed as $1.29. So the recomputed
# value is allowed to differ by a cent. Comparisons between two printed
# numbers (items vs subtotal, subtotal + tax vs total) get no tolerance
# at all -- those are exact sums of exact printed values, and "close
# enough" would quietly hide the single-digit misreads we're hunting.
LINE_TOTAL_TOLERANCE_CENTS = 1
EXACT = 0


class CheckStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    # Not every receipt prints every number. India Market prints no tax
    # line at all. That is an absence of evidence, not evidence of an
    # error, and collapsing the two would let a receipt that proved
    # nothing score the same as one that proved everything.
    UNCHECKABLE = "uncheckable"


@dataclass
class CheckResult:
    """The verdict from one check, and enough context to act on it."""

    name: str
    status: CheckStatus
    detail: str  # human-readable, for the review queue
    expected_cents: Optional[int] = None
    actual_cents: Optional[int] = None
    line_number: Optional[int] = None  # set when a check localizes to a line

    @property
    def delta_cents(self) -> Optional[int]:
        if self.expected_cents is None or self.actual_cents is None:
            return None
        return self.actual_cents - self.expected_cents


def to_cents(amount: Optional[float]) -> Optional[int]:
    """Convert dollars to whole cents.

    Every comparison in this module happens in integer cents, never in
    floats. Binary floating point cannot represent most decimal
    fractions exactly -- 0.1 + 0.2 evaluates to 0.30000000000000004 --
    and this module's entire job is comparing money for equality. In
    cents, 10 + 20 == 30 always, and a one-cent discrepancy is the
    integer 1 rather than a number we have to decide how to round.
    """
    if amount is None:
        return None
    return int(round(amount * 100))


def format_cents(cents: Optional[int]) -> str:
    """Render cents back to dollars for messages a human will read."""
    if cents is None:
        return "--"
    return f"{cents / 100:.2f}"


def check_items_sum_to_subtotal(receipt: Receipt) -> CheckResult:
    """Do the charged line items add up to the printed subtotal?

    Voided lines are excluded: the register printed them, but the
    customer was never charged for them. Including them is the single
    most likely way to fail this check on a correctly-read receipt --
    it is exactly what the Walmart `** VOIDED ENTRY **` would do.

    Note this check is blind to ordering. If every price were read one
    row out of step with its description, the sum would be unchanged and
    this would still pass. It validates the multiset of prices, not the
    mapping from item to price.
    """
    name = "items_sum_to_subtotal"

    if receipt.subtotal is None:
        return CheckResult(name, CheckStatus.UNCHECKABLE, "no subtotal printed")

    charged = [i for i in receipt.line_items if not i.is_voided]
    missing = [i.line_number for i in charged if i.line_total is None]
    if missing:
        return CheckResult(
            name,
            CheckStatus.UNCHECKABLE,
            f"line(s) {missing} have no amount, so no sum is possible",
        )

    total_cents = sum(to_cents(i.line_total) for i in charged)
    subtotal_cents = to_cents(receipt.subtotal)
    difference = abs(total_cents - subtotal_cents)

    if difference <= EXACT:
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"{len(charged)} items sum to {format_cents(total_cents)}",
            expected_cents=subtotal_cents,
            actual_cents=total_cents,
        )

    return CheckResult(
        name,
        CheckStatus.FAIL,
        f"{len(charged)} items sum to {format_cents(total_cents)} but the "
        f"printed subtotal is {format_cents(subtotal_cents)} "
        f"(off by {format_cents(difference)})",
        expected_cents=subtotal_cents,
        actual_cents=total_cents,
    )


def check_subtotal_plus_tax_is_total(receipt: Receipt) -> CheckResult:
    """Does subtotal + tax equal the printed total?

    Independent of the check above: it uses the printed subtotal rather
    than the line items, so a receipt can pass one and fail the other.
    That independence is the point -- two checks that share an input
    would fail together and tell us one thing, not two.
    """
    name = "subtotal_plus_tax_is_total"

    if receipt.subtotal is None or receipt.total is None:
        return CheckResult(name, CheckStatus.UNCHECKABLE, "subtotal or total not printed")

    # A receipt with no tax line is normal (unprepared groceries are
    # untaxed in Connecticut), and means tax contributes zero here --
    # unlike a missing subtotal, which makes the check impossible.
    tax_cents = to_cents(receipt.tax) or 0
    subtotal_cents = to_cents(receipt.subtotal)
    total_cents = to_cents(receipt.total)
    expected = subtotal_cents + tax_cents
    difference = abs(expected - total_cents)

    if difference <= EXACT:
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"{format_cents(subtotal_cents)} + {format_cents(tax_cents)} "
            f"= {format_cents(total_cents)}",
            expected_cents=expected,
            actual_cents=total_cents,
        )

    return CheckResult(
        name,
        CheckStatus.FAIL,
        f"{format_cents(subtotal_cents)} + {format_cents(tax_cents)} "
        f"= {format_cents(expected)} but the printed total is "
        f"{format_cents(total_cents)}",
        expected_cents=expected,
        actual_cents=total_cents,
    )


def check_line_arithmetic(receipt: Receipt) -> List[CheckResult]:
    """Does quantity x unit_price match each printed line total?

    Returns one result per checkable line rather than a single verdict,
    because this is the only check that can point at a specific item.
    The sum check can tell you a receipt is wrong; this one can tell you
    it is line 7 that is wrong, which is the difference between a review
    queue that is useful and one that is a pile.
    """
    results: List[CheckResult] = []

    for item in receipt.line_items:
        if item.is_voided:
            continue
        if item.quantity is None or item.unit_price is None or item.line_total is None:
            # Plenty of receipts print only an amount -- the Walmart one
            # prints no unit price at all. Nothing is wrong; there is
            # simply nothing here to check.
            continue

        expected = int(round(item.quantity * item.unit_price * 100))
        actual = to_cents(item.line_total)
        difference = abs(expected - actual)
        summary = (
            f"line {item.line_number} ({item.raw_description[:24]}): "
            f"{item.quantity:g} x {item.unit_price:.2f}"
        )

        if difference <= LINE_TOTAL_TOLERANCE_CENTS:
            results.append(CheckResult(
                "line_arithmetic",
                CheckStatus.PASS,
                f"{summary} = {format_cents(actual)}",
                expected_cents=expected,
                actual_cents=actual,
                line_number=item.line_number,
            ))
        else:
            results.append(CheckResult(
                "line_arithmetic",
                CheckStatus.FAIL,
                f"{summary} = {format_cents(expected)} but the line total "
                f"printed is {format_cents(actual)}",
                expected_cents=expected,
                actual_cents=actual,
                line_number=item.line_number,
            ))

    return results


def check_tax_rate(receipt: Receipt) -> CheckResult:
    """Does the printed tax rate, applied to the taxable items, give the
    printed tax?

    This is the strongest check available, because it is the only one
    that tests the per-line tax flags. Nothing else in the pipeline can
    tell whether "N" was read where "X" was printed.

    The subtlety is that the rate does not apply to the whole subtotal.
    Connecticut does not tax unprepared groceries, so the Walmart receipt
    charges 6.35% on $31.03 of taxable goods, not on its $41.80 subtotal
    -- 1.97, not 2.65. Getting this wrong would fail every mixed-basket
    receipt in the country.
    """
    name = "tax_rate"

    if receipt.tax_rate_percent is None:
        return CheckResult(name, CheckStatus.UNCHECKABLE, "no tax rate printed")
    if receipt.tax is None:
        return CheckResult(name, CheckStatus.UNCHECKABLE, "no tax amount printed")

    # Flags vary by chain, so treat the ones that mean "not taxed" as the
    # known set and assume anything else is taxable. Guessing wrong in
    # this direction produces a visible failed check rather than a silent
    # pass, which is the safer way to be wrong.
    NON_TAXABLE_FLAGS = {"N", "F", "O"}
    taxable = [
        i for i in receipt.line_items
        if not i.is_voided
        and i.line_total is not None
        and (i.tax_flag or "").strip().upper() not in NON_TAXABLE_FLAGS
    ]
    if not taxable:
        return CheckResult(
            name, CheckStatus.UNCHECKABLE, "no line items are flagged taxable"
        )
    if any(i.tax_flag is None for i in receipt.line_items if not i.is_voided):
        return CheckResult(
            name, CheckStatus.UNCHECKABLE, "some lines have no tax flag printed"
        )

    taxable_cents = sum(to_cents(i.line_total) for i in taxable)
    expected = int(round(taxable_cents * receipt.tax_rate_percent / 100))
    actual = to_cents(receipt.tax)
    difference = abs(expected - actual)
    summary = (
        f"{len(taxable)} taxable items = {format_cents(taxable_cents)} "
        f"x {receipt.tax_rate_percent:g}%"
    )

    # A cent of tolerance: the register rounds the tax it computes, and
    # on a mixed basket some chains round per line rather than on the
    # taxable total.
    if difference <= LINE_TOTAL_TOLERANCE_CENTS:
        return CheckResult(
            name, CheckStatus.PASS, f"{summary} = {format_cents(actual)}",
            expected_cents=expected, actual_cents=actual,
        )

    return CheckResult(
        name, CheckStatus.FAIL,
        f"{summary} = {format_cents(expected)} but the printed tax is "
        f"{format_cents(actual)} -- a tax flag or an amount is misread",
        expected_cents=expected, actual_cents=actual,
    )


def check_payment_reconciles(receipt: Receipt) -> CheckResult:
    """Does amount paid minus the total equal the change given?

    Independent of every other check: it uses the total and two numbers
    nothing else touches. A receipt can have perfect item arithmetic and
    still fail here if the total was misread.

    Change is compared by magnitude because registers disagree about the
    sign -- Dollar Tree prints "$-2.77" for change handed back, others
    print "2.77". Extraction copies the sign as printed; deciding it
    doesn't matter is this layer's call to make, in the open.
    """
    name = "payment_reconciles"

    if receipt.amount_paid is None or receipt.change_given is None:
        return CheckResult(
            name, CheckStatus.UNCHECKABLE, "payment or change not printed"
        )
    if receipt.total is None:
        return CheckResult(name, CheckStatus.UNCHECKABLE, "no total printed")

    # Cash registers may round the amount due to the nearest nickel and
    # print the adjustment. Without it, the Walmart receipt looks two
    # cents out: 60.00 - 43.77 = 16.23, but the printed change is 16.25.
    rounding_cents = to_cents(receipt.rounding_adjustment) or 0
    expected = to_cents(receipt.amount_paid) - to_cents(receipt.total) + rounding_cents
    actual = abs(to_cents(receipt.change_given))
    difference = abs(expected - actual)

    detail_rounding = (
        f" (+{format_cents(rounding_cents)} rounding)" if rounding_cents else ""
    )
    summary = (
        f"paid {receipt.amount_paid:.2f} - total {receipt.total:.2f}"
        f"{detail_rounding}"
    )

    if difference <= EXACT:
        return CheckResult(
            name, CheckStatus.PASS, f"{summary} = change {format_cents(actual)}",
            expected_cents=expected, actual_cents=actual,
        )

    return CheckResult(
        name, CheckStatus.FAIL,
        f"{summary} = {format_cents(expected)} but the printed change is "
        f"{format_cents(actual)}",
        expected_cents=expected, actual_cents=actual,
    )


def check_item_count(receipt: Receipt) -> CheckResult:
    """Does the number of items we extracted match the count the receipt
    printed?

    The only check that can catch a *dropped* line. Every arithmetic
    check above is computed from the items we have, so a receipt missing
    an item entirely can still be perfectly self-consistent -- the sum
    would just be short, and would disagree with the subtotal. But if the
    model dropped a line AND misread the subtotal to match, only a
    printed count would notice.

    Voided lines are excluded: the register's count is of items sold.
    """
    name = "item_count"

    if receipt.item_count_printed is None:
        return CheckResult(name, CheckStatus.UNCHECKABLE, "no item count printed")

    extracted = len([i for i in receipt.line_items if not i.is_voided])
    if extracted == receipt.item_count_printed:
        return CheckResult(
            name, CheckStatus.PASS,
            f"extracted {extracted} items, receipt says {receipt.item_count_printed}",
        )

    return CheckResult(
        name, CheckStatus.FAIL,
        f"extracted {extracted} items but the receipt says "
        f"{receipt.item_count_printed} -- a line was dropped or invented",
    )


@dataclass
class ValidationReport:
    """Everything we know about how much to trust one extracted receipt."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def passed(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.PASS]

    @property
    def uncheckable(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.UNCHECKABLE]

    @property
    def is_trustworthy(self) -> bool:
        """One failed check is enough to route a receipt to review.

        Deliberately strict. A receipt that fails any arithmetic test has
        something wrong in it, and the cost of the two outcomes is wildly
        asymmetric: a false alarm costs a few seconds of looking at a
        photo, while a wrong price accepted silently corrupts the price
        index permanently and invisibly.
        """
        return not self.failed

    @property
    def evidence_count(self) -> int:
        """How many checks actually ran.

        The denominator for any accuracy rate. Receipts from stores that
        print less prove less about themselves, and averaging them in as
        if they had passed everything would flatter the numbers.
        """
        return len(self.passed) + len(self.failed)

    def summary(self) -> str:
        if not self.evidence_count:
            return "no checks could run -- this receipt proves nothing about itself"
        verdict = "TRUSTWORTHY" if self.is_trustworthy else "NEEDS REVIEW"
        return (
            f"{verdict}: {len(self.passed)}/{self.evidence_count} checks passed"
            + (f", {len(self.uncheckable)} not printed" if self.uncheckable else "")
        )


def validate(receipt: Receipt) -> ValidationReport:
    """Run every check we can against a receipt.

    Pure: it reads the receipt and reports on it, never modifies it. A
    validator that repaired what it found would destroy the evidence of
    what the model actually returned, and that evidence is what tells us
    whether a prompt change or a model swap helped.
    """
    checks: List[CheckResult] = [
        check_items_sum_to_subtotal(receipt),
        check_subtotal_plus_tax_is_total(receipt),
        check_tax_rate(receipt),
        check_payment_reconciles(receipt),
        check_item_count(receipt),
    ]
    checks.extend(check_line_arithmetic(receipt))
    return ValidationReport(checks=checks)
