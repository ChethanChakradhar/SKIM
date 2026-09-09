"""
Step 5 of the Skim pipeline: Product normalization.

Responsibility of this module: turn a receipt's cryptic printed string
into a structured product, so that `GV MLK 2% 1GAL` and
`GREAT VAL MILK 2% GALLON` can be recognized as the same thing.

This is the heart of the project. Everything downstream depends on it:
a price index is only meaningful if "milk" means the same milk in March
and August.

Why this can't be string matching, measured rather than assumed
(experiment run Sept 2026 on real strings from the first three receipts):

    same-product pairs      character fuzzy: 0.63   raw embedding: 0.61
    DIFFERENT-product pairs                          raw embedding: 0.66

The distributions overlap. `PLUM TOMATO` and `GUAVA` score 0.66 against
each other -- higher than `BNLS CK BRST` scores against
`BONELESS CHICKEN BREAST` (0.66) or `STERLT-10G` against
`STERILITE 10 GAL TOTE` (0.51). No threshold separates them.

The reason is worth understanding: embeddings measure semantic
*relatedness*, and entity resolution needs *identity*. Two different
fruits are genuinely related; an abbreviation and its expansion are
genuinely dissimilar as text. The signal points the wrong way.

Parsing first fixes it. Embedding the parsed form instead of the raw
string moves same-product similarity from 0.61 to 0.96 while
different-product similarity stays at 0.66 -- a clean margin where a
threshold works. So: parse, then embed. This module is the parse half.

The rule inherited from Step 3 applies with more force here: an
unidentifiable string gets flagged, never guessed. A wrong price fails
an arithmetic check and gets caught; a wrong product identity fails
nothing and quietly corrupts the index forever.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from skim.extract import (
    GEMINI_MODEL,
    ExtractionError,
    _generate_with_retry,
    _load_client,
)

# Units Step 6 knows how to convert. The model is free to return
# anything, but a unit outside this list is discarded rather than
# trusted: Step 6 computes price-per-ounce from these, and a
# hallucinated unit would silently corrupt every comparison built on it.
# Better a null Step 6 can skip than a number it cannot question.
KNOWN_UNITS = {
    "each", "ct",           # countable things
    "lb", "oz", "g", "kg",  # weight
    "fl_oz", "ml", "l", "gal", "qt", "pt", "cup",  # volume
    "sheet", "in",          # the long tail: paper goods, linens
}
# "cup" was added after a real run: MEASURING CUP PLASTIC 4-CUP lost its
# size because the vocabulary lacked the unit, and _coerce correctly
# refused to trust what it could not recognize. The coercion worked; the
# list was short. Expect this list to grow the same way -- from real
# receipts, not from imagination.

# Kept deliberately coarse. These exist for Step 8 ("what is my produce
# inflation vs household inflation"), not to build a taxonomy. A short
# fixed list also stops the model inventing a new category per item,
# which would make grouping useless.
KNOWN_CATEGORIES = {
    "produce", "dairy", "meat", "seafood", "bakery", "pantry", "frozen",
    "beverage", "snack", "household", "personal_care", "pet", "baby", "other",
}


class ParseError(ExtractionError):
    """Raised when the model's parse can't be turned into a ParsedProduct."""


@dataclass
class ParsedProduct:
    """One receipt string, decomposed.

    Every field here earns its place by serving a later stage rather
    than by being interesting on its own:

      brand, product, variant  -> matching (this step)
      size_value, size_unit    -> price per ounce (Step 6)
      pack_count               -> 2 @ 5.99 of 14oz paneer is 28oz total
      category                 -> basket grouping (Step 8)
      canonical_text           -> the string we embed, per the experiment
    """

    raw_description: str
    brand: Optional[str]
    product: Optional[str]  # the generic noun: "milk", "okra", "storage tote"
    variant: Optional[str]  # "2%", "boneless", "black/taupe/blue"
    size_value: Optional[float]
    size_unit: Optional[str]
    pack_count: Optional[int]
    category: Optional[str]
    canonical_text: Optional[str]
    needs_review: bool  # True when the model could not identify the product
    parse_note: Optional[str] = None  # why, when it couldn't
    attempts: int = 1  # 2 means the retry-with-context path was used

    @property
    def is_identified(self) -> bool:
        """The one question that decides whether this can enter the catalog."""
        return bool(self.product) and not self.needs_review


PRODUCT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "brand": {"type": "string", "nullable": True},
        "product": {"type": "string", "nullable": True},
        "variant": {"type": "string", "nullable": True},
        "size_value": {"type": "number", "nullable": True},
        "size_unit": {"type": "string", "nullable": True},
        "pack_count": {"type": "integer", "nullable": True},
        "category": {"type": "string", "nullable": True},
        "canonical_text": {"type": "string", "nullable": True},
        "needs_review": {"type": "boolean"},
        "parse_note": {"type": "string", "nullable": True},
    },
    "required": ["needs_review"],
}

PROMPT = """\
You are decoding the abbreviated product descriptions printed on US
retail receipts into structured products.

Registers abbreviate aggressively and inconsistently. Expand what you
recognize, using your knowledge of US retailers and their house brands.

  "BNLS CK BRST"   -> brand null, product "chicken breast", variant "boneless"
  "STERLT-10G"     -> brand "Sterilite", product "storage tote", 10 gal
  "MS SC HK SET"   -> brand "Mainstays", product "shower curtain hook set"
  "GV MLK 2% 1GAL" -> brand "Great Value", product "milk", variant "2%", 1 gal

FIELDS
- `brand`: the maker or store house brand, expanded to its real name.
  Null for unbranded goods -- loose produce, meat sold by weight. A null
  brand is a correct and common answer, not a failure.
- `product`: the generic noun a person would say out loud. "milk",
  "okra", "storage tote", "kitchen towel". Not the marketing name.
- `variant`: what distinguishes this from others of the same product --
  "2%", "boneless", "black/taupe/blue". Null if nothing distinguishes it.
- `size_value` and `size_unit`: the package size, if printed.
  Units must be one of: each, ct, lb, oz, g, kg, fl_oz, ml, l, gal, qt,
  pt, sheet, in. Null both if no size is printed. Do NOT convert between
  units -- report what is printed. "310 GM" is 310 g, not 10.9 oz.
- `pack_count`: how many packages, if the description says so
  ("2-PACK", "4-CUP" is a size not a pack). Null otherwise.
- `category`: exactly one of: produce, dairy, meat, seafood, bakery,
  pantry, frozen, beverage, snack, household, personal_care, pet, baby,
  other.
- `canonical_text`: a short natural phrase combining brand, product,
  variant and size, in that order, lowercase except brand names:
  "Great Value milk 2% 1 gallon", "boneless chicken breast".
  This string is what gets compared against other products, so two
  descriptions of the same item must produce the same phrase. Do not add
  words that are not implied by the receipt string.

WHEN YOU CANNOT IDENTIFY IT
- Set `needs_review` true, leave `product` null, and say what is
  ambiguous in `parse_note`. Some receipt strings genuinely cannot be
  decoded without seeing the shelf.
- Never invent a plausible product. A wrong identity is worse than an
  admitted unknown: it enters the price index and is never questioned
  again, whereas an unknown gets looked at by a human once.
- A null brand or a null size is NOT a reason to set `needs_review`.
  Only set it when you cannot say what the product itself is.
"""

# Appended to the prompt on the second attempt only.
CONTEXT_PROMPT = """\

ADDITIONAL CONTEXT
A first attempt could not identify this product. Here is what surrounds
it, which may make it recognizable:

  Store: {store}
  Other items on the same receipt:
{siblings}

Use this to narrow it down -- a store's product mix and the company an
item keeps are both evidence. If it is still not identifiable, say so
again. A second guess made under pressure is worth less than an honest
unknown.
"""


def _coerce(payload: Dict[str, Any], raw_description: str, attempts: int) -> ParsedProduct:
    """Turn the model's JSON into a ParsedProduct, discarding values we
    can't stand behind.

    The schema constrains shape, not content: the model can return a
    well-formed string in `size_unit` that is nonsense. Anything outside
    the known vocabularies is dropped here rather than carried forward,
    because these two fields feed arithmetic (Step 6) and grouping
    (Step 8) that cannot themselves detect a bad value.
    """
    unit = (payload.get("size_unit") or "").strip().lower().replace(" ", "_") or None
    if unit and unit not in KNOWN_UNITS:
        unit = None

    category = (payload.get("category") or "").strip().lower() or None
    if category and category not in KNOWN_CATEGORIES:
        category = "other"

    size = payload.get("size_value")
    # A size without a usable unit is not a size. Keeping the number
    # alone would invite a later stage to assume a unit for it.
    if unit is None:
        size = None

    product = (payload.get("product") or "").strip() or None

    return ParsedProduct(
        raw_description=raw_description,
        brand=(payload.get("brand") or "").strip() or None,
        product=product,
        variant=(payload.get("variant") or "").strip() or None,
        size_value=size,
        size_unit=unit,
        pack_count=payload.get("pack_count"),
        category=category,
        canonical_text=(payload.get("canonical_text") or "").strip() or None,
        # Trust the model's own flag, but override it when the parse is
        # empty regardless of what the model claimed: no product name
        # means nothing to match on, whatever the flag says.
        needs_review=bool(payload.get("needs_review")) or product is None,
        parse_note=(payload.get("parse_note") or "").strip() or None,
        attempts=attempts,
    )


def _ask(client: genai.Client, prompt: str, model: str) -> Dict[str, Any]:
    import json

    response = _generate_with_retry(
        client,
        model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PRODUCT_SCHEMA,
            temperature=0.0,
        ),
    )
    if not response.text:
        raise ParseError("Model returned no text when parsing a product description")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ParseError(f"Parse response wasn't valid JSON: {e}") from e


def parse_product(
    raw_description: str,
    store: Optional[str] = None,
    sibling_descriptions: Optional[List[str]] = None,
    model: str = GEMINI_MODEL,
    client: Optional[genai.Client] = None,
) -> ParsedProduct:
    """Decode one receipt string into a structured product.

    Makes a second attempt when the first cannot identify the product --
    but only then, and only with *more information* than the first had.

    Both halves of that matter. Retrying on any null field would retry
    almost everything: a null brand is the correct answer for loose
    produce and a null size is correct for a rat trap. And retrying with
    an identical prompt at temperature 0 would mostly re-buy the same
    answer, so the second attempt adds the store and the other items on
    the receipt. `BLUE BANDED` alone is a mystery; `BLUE BANDED` sitting
    between `BNLS CK BRST` and `IODIZED SALT` at Walmart is groceries.

    If the second attempt still can't identify it, that is the answer.
    We flag it for review rather than accepting a guess made under
    pressure.
    """
    client = client or _load_client()
    base_prompt = f'{PROMPT}\nReceipt description to decode: "{raw_description}"'

    parsed = _coerce(_ask(client, base_prompt, model), raw_description, attempts=1)
    if parsed.is_identified:
        return parsed

    # Nothing more to tell it on a second pass -- a bare retry at
    # temperature 0 would mostly return the same answer for the same
    # money, so don't spend it.
    if not store and not sibling_descriptions:
        return parsed

    others = [d for d in (sibling_descriptions or []) if d != raw_description]
    context = CONTEXT_PROMPT.format(
        store=store or "unknown",
        siblings="\n".join(f"    - {d}" for d in others[:20]) or "    (none)",
    )
    retried = _coerce(
        _ask(client, base_prompt + context, model), raw_description, attempts=2
    )

    # Only take the retry if it actually resolved something. A second
    # unidentified answer is not better than the first, and the first
    # may carry a more useful parse_note.
    if retried.is_identified:
        return retried

    # Keep the first parse, but record that two calls were paid for --
    # `attempts` is the cost record, not a description of which answer
    # we kept. Understating it would hide the price of the retry policy
    # from any later measurement of whether it earns its keep.
    parsed.attempts = 2
    return parsed


class ProductCache:
    """Remembers every string we have already decoded.

    A store prints the same string for the same product every time:
    `GV MLK 2% 1GAL` will be identical on every Walmart receipt you ever
    photograph. So this is not really a matching problem, it is a
    remembering problem -- parse each distinct string once and the cost
    trends to zero as the catalog fills.

    Keyed by (store, description) rather than description alone, because
    the abbreviations are a store's private dialect. `GV` means Great
    Value at Walmart and nothing in particular anywhere else.

    Writes after every new entry rather than at the end. That looks
    wasteful for a file this small, and it is the whole point: a run
    that dies partway -- which the free tier's five-requests-per-minute
    quota makes routine -- keeps everything it had already paid for.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            self.entries = json.loads(path.read_text())

    @staticmethod
    def key(store: Optional[str], raw_description: str) -> str:
        return f"{(store or 'unknown').strip().upper()}||{raw_description.strip()}"

    def get(self, store: Optional[str], raw_description: str) -> Optional[ParsedProduct]:
        entry = self.entries.get(self.key(store, raw_description))
        return ParsedProduct(**entry) if entry else None

    def put(self, store: Optional[str], parsed: ParsedProduct) -> None:
        self.entries[self.key(store, parsed.raw_description)] = asdict(parsed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=True))

    def parse(
        self,
        raw_description: str,
        store: Optional[str] = None,
        sibling_descriptions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ParsedProduct:
        """Return the remembered parse, or make one and remember it.

        Entries needing review are cached too. Re-parsing them would buy
        the same unknown again -- what they need is a human, not another
        call. Delete the entry to force a fresh attempt.
        """
        cached = self.get(store, raw_description)
        if cached is not None:
            return cached

        parsed = parse_product(
            raw_description, store=store,
            sibling_descriptions=sibling_descriptions, **kwargs
        )
        self.put(store, parsed)
        return parsed
