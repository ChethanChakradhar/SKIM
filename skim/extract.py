"""
Step 3 of the Skim pipeline: Extraction.

Responsibility of this module: send the preprocessed photo to a
vision-language model and get back a structured receipt -- every line
item, the printed totals, the merchant, the date -- as typed Python
objects rather than a wall of text.

The one idea this module is built around: the model transcribes, it does
not interpret.

That sounds like a small distinction and it is not. Three consequences,
each of which shapes the code below:

1. `raw_description` is verbatim. If the model helpfully rewrites
   "GV MLK 2% 1GAL" as "Great Value Milk 2% Gallon", it has destroyed the
   exact string that Step 5 (product normalization) exists to resolve.
   Worse, it has done the hard part of this project invisibly, inside a
   black box, where we can neither audit it nor measure it. Normalization
   is a separate stage precisely so it can be inspected and improved.

2. The model never computes a value that isn't printed. If it derives the
   subtotal by adding up the line items it just read, then Step 4's
   check -- do the line items sum to the subtotal? -- is comparing the
   model against itself, and passes even when every price is wrong. That
   check is our only accuracy signal that costs nothing to produce. A
   model that does arithmetic for us destroys it.

3. Unreadable means null, never a guess. A null is a flag we can act on.
   A plausible invented price is silently wrong forever, and a personal
   price index built on it is worse than no index at all.

We ask for structured output (`response_schema`), so the API returns JSON
conforming to the shape we specified rather than prose that happens to
contain JSON. No markdown fences to strip, no regex, no "sometimes the
model adds a friendly preamble" failure mode.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from skim.preprocess import PreprocessResult

# Flash is the cheapest frontier vision model per image and is accurate
# enough for receipts; a few hundred receipts costs pennies. Kept as a
# constant so swapping it is a one-line experiment, which is the point --
# comparing models on the same receipts is part of the project.
#
# Pinned to an explicit version, not the `gemini-flash-latest` alias: an
# alias moves under you, so an accuracy benchmark run today wouldn't
# reproduce next month, and a silent model change would look like a bug
# in our own pipeline.
GEMINI_MODEL = "gemini-3.6-flash"

# JPEG quality for the image we put on the wire. 90 is visually
# lossless for text at this scale; going higher inflates the payload
# without giving the model anything more to read.
WIRE_JPEG_QUALITY = 90

# Deliberately NOT downscaling before sending. Image tokens scale with
# resolution, so there is a real cost/accuracy tradeoff here -- but we
# have no accuracy measurement yet (that arrives in Step 4), and guessing
# at a cap risks quietly destroying faded thermal text to save a fraction
# of a cent. We record the payload size and token usage on every call so
# the tradeoff can be measured later instead of assumed now.

# Popular models get temporarily overloaded; the very first call this
# module ever made came back 503. That is a normal operating condition,
# not an error worth surfacing to a user, so we wait and try again.
#
# We retry ONLY transient failures: 503 (overloaded) and 429 (rate
# limited). A 400 (bad request) or 404 (no such model) will fail exactly
# the same way on the third attempt as the first -- retrying those just
# burns time and hides a real bug behind a delay.
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2.0
RETRYABLE_STATUS_CODES = {429, 503}

# A 429 quota error carries the answer with it: the response says how
# long until the window resets ("Please retry in 39.4s", and a
# RetryInfo.retryDelay field). Our own exponential guess of 2s then 4s
# is wrong by an order of magnitude against a per-minute quota, so when
# the server states a delay we honor it instead of guessing.
#
# Capped, because we are waiting inside a synchronous call and a server
# asking us to sleep for ten minutes is a signal to stop and tell the
# user, not to hang.
MAX_SERVER_REQUESTED_WAIT_SECONDS = 75.0
_RETRY_DELAY_PATTERN = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


class ExtractionError(Exception):
    """Base class for every reason extraction can fail."""


class MissingAPIKeyError(ExtractionError):
    pass


class EmptyResponseError(ExtractionError):
    pass


class MalformedResponseError(ExtractionError):
    pass


class ModelUnavailableError(ExtractionError):
    """The model stayed overloaded or rate-limited across every attempt."""


@dataclass
class LineItem:
    """One printed line on the receipt.

    `line_total` is the only field we insist on: it is the number that
    makes the arithmetic check in Step 4 possible. Everything else is
    optional because real receipts genuinely omit it -- plenty of stores
    print no unit price, no product code and no tax flag.
    """

    line_number: int
    raw_description: str  # EXACTLY as printed. Never normalized here.
    line_total: Optional[float]
    product_code: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None  # "each", "lb", "kg", "oz"
    unit_price: Optional[float] = None
    discount: Optional[float] = None
    tax_flag: Optional[str] = None  # "T", "N", "X", "F" -- as printed
    is_voided: bool = False


@dataclass
class Receipt:
    merchant_name: Optional[str]
    store_number: Optional[str]
    address: Optional[str]
    phone: Optional[str]
    purchase_date: Optional[str]  # YYYY-MM-DD
    purchase_time: Optional[str]  # HH:MM, 24-hour
    line_items: List[LineItem]
    subtotal: Optional[float]  # as PRINTED, never summed by the model
    tax: Optional[float]
    total: Optional[float]
    # The fields below exist for one reason: Step 4 can check them
    # against each other. They were added after validation was written,
    # because the checks are what decide which numbers are worth
    # capturing -- a receipt prints plenty we have no use for.
    tax_rate_percent: Optional[float] = None  # "TAX1 6.3500 %" -> 6.35
    amount_paid: Optional[float] = None  # cash tendered or card charged
    change_given: Optional[float] = None  # sign varies by store; as printed
    rounding_adjustment: Optional[float] = None  # "ROUNDING 0.02"
    item_count_printed: Optional[int] = None  # "# ITEMS SOLD 9"
    currency: str = "USD"
    unreadable_notes: Optional[str] = None


@dataclass
class ExtractionResult:
    """What Step 4 (validation) receives."""

    receipt: Receipt
    model: str
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    wire_bytes: int  # size of the JPEG we uploaded
    raw_json: Dict[str, Any] = field(repr=False, default_factory=dict)


# The contract we hand the API. Writing it out as an explicit dict rather
# than deriving it from type annotations keeps it readable: this is the
# exact shape the model is required to return, and it is worth being able
# to see it at a glance when an extraction comes back wrong.
RECEIPT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "merchant_name": {"type": "string", "nullable": True},
        "store_number": {"type": "string", "nullable": True},
        "address": {"type": "string", "nullable": True},
        "phone": {"type": "string", "nullable": True},
        "purchase_date": {"type": "string", "nullable": True},
        "purchase_time": {"type": "string", "nullable": True},
        "currency": {"type": "string", "nullable": True},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_number": {"type": "integer"},
                    "raw_description": {"type": "string"},
                    "product_code": {"type": "string", "nullable": True},
                    "quantity": {"type": "number", "nullable": True},
                    "unit": {"type": "string", "nullable": True},
                    "unit_price": {"type": "number", "nullable": True},
                    "line_total": {"type": "number", "nullable": True},
                    "discount": {"type": "number", "nullable": True},
                    "tax_flag": {"type": "string", "nullable": True},
                    "is_voided": {"type": "boolean"},
                },
                "required": ["line_number", "raw_description", "is_voided"],
            },
        },
        "subtotal": {"type": "number", "nullable": True},
        "tax": {"type": "number", "nullable": True},
        "total": {"type": "number", "nullable": True},
        "tax_rate_percent": {"type": "number", "nullable": True},
        "amount_paid": {"type": "number", "nullable": True},
        "change_given": {"type": "number", "nullable": True},
        "rounding_adjustment": {"type": "number", "nullable": True},
        "item_count_printed": {"type": "integer", "nullable": True},
        "unreadable_notes": {"type": "string", "nullable": True},
    },
    "required": ["line_items"],
}

PROMPT = """\
You are transcribing a photograph of a retail receipt into structured data.

Transcribe what is printed. Do not interpret, normalize, or compute.

ITEM DESCRIPTIONS
- Copy `raw_description` character for character as printed, including
  abbreviations, misspellings and cryptic codes. "GV MLK 2% 1GAL" stays
  "GV MLK 2% 1GAL". Never expand it to "Great Value Milk". A later stage
  does that work and needs the original string.
- `raw_description` is the product description ONLY. Some receipts print
  a sequence number in front of each item ("1 ONION 10LB YELLOW",
  "2 MUSHROOMS"). That number is a separate column, not part of the
  product's name: put it in `line_number` and start `raw_description` at
  the product ("ONION 10LB YELLOW"). Likewise drop a trailing ":" or
  other punctuation that only separates columns.
  This matters more than it looks: a later stage decides whether two
  descriptions refer to the same product, and a position number baked
  into the name would make the same item bought twice look like two
  different products.

TOTALS
- Report subtotal, tax and total ONLY as printed on the receipt. Never
  add up the line items yourself. If the subtotal is not printed or is
  unreadable, return null. A computed total is worse than a missing one,
  because a later stage checks the printed totals against the items to
  detect transcription errors, and your arithmetic would hide them.
- `tax_rate_percent`: the tax percentage if one is printed, as a number.
  "TAX1 6.3500 %" is 6.35. Do not derive it by dividing tax by subtotal.
- `amount_paid`: cash tendered or the amount charged to a card.
- `change_given`: the change line exactly as printed, keeping its sign.
  Some registers print change as a negative number. Copy what is there;
  do not correct it.
- `rounding_adjustment`: a printed rounding line ("ROUNDING 0.02"), if any.
- `item_count_printed`: the item count if the receipt states one
  ("# ITEMS SOLD 9"), as an integer. Do not count the lines yourself --
  a later stage compares your line count against this printed number,
  and counting would make that comparison meaningless.

EVERY LINE
- Include every purchased line in printed order, numbered from 1.
- Voided lines: include them with `is_voided` true and `line_total` null
  if no amount is printed. Do not silently drop them.
- Weighted items printed like "0.87 LB @ $3.99": quantity 0.87, unit
  "lb", unit_price 3.99, and `line_total` as the printed amount.
- Ordinary items: quantity is the printed count, unit "each".
- Per-line discounts ("You Saved: 0.38") go in `discount` on that line.
- `tax_flag` is the single letter printed beside the amount (T, N, X, F),
  copied as-is. Not your judgment about whether tax applied.

WHAT TO IGNORE
- Anything that is not the receipt: fingers, table, other objects.
- Text showing through from the reverse side of the paper, which appears
  mirrored or upside down. Thermal receipts often print on the back, and
  it bleeds through. Transcribe only right-reading text on the front.
- Store marketing, survey invitations, return policies and barcodes.

WHEN YOU CANNOT READ SOMETHING
- Use null. Never guess a price, a date, or a description.
- Briefly say what was unreadable in `unreadable_notes` (or null if all
  of it was legible).

Dates as YYYY-MM-DD; a two-digit year like 8/25/26 means 2026. Times as
HH:MM on a 24-hour clock. Amounts as plain numbers: 12.23, not "$12.23".
"""


def _load_client() -> genai.Client:
    """Build an API client, or explain exactly what's missing.

    `load_dotenv` reads the gitignored .env file into environment
    variables, so the key lives in a file git will never commit rather
    than hardcoded in source.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not set. Get a key at "
            "https://aistudio.google.com/apikey, then copy .env.example to "
            ".env and put the real key in it."
        )
    return genai.Client(api_key=api_key)


def _image_to_jpeg_bytes(result: PreprocessResult) -> bytes:
    buffer = io.BytesIO()
    result.image.save(buffer, "JPEG", quality=WIRE_JPEG_QUALITY)
    return buffer.getvalue()


def _server_requested_wait(error: Exception) -> Optional[float]:
    """How long the server asked us to wait, if it said.

    A quota error states its own reset window, which is strictly better
    information than any backoff curve we could invent. Read from the
    structured RetryInfo detail when present, and fall back to the
    human-readable message, which carries the same number.
    """
    details = getattr(error, "details", None) or {}
    if isinstance(details, dict):
        for detail in details.get("error", {}).get("details", []) or []:
            delay = detail.get("retryDelay") if isinstance(detail, dict) else None
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    pass

    match = _RETRY_DELAY_PATTERN.search(str(error))
    return float(match.group(1)) if match else None


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: List[Any],
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """Call the API, retrying transient failures.

    Two different waits, because they are two different problems. A 503
    means the model is momentarily overloaded and nobody knows for how
    long, so we back off exponentially (2s, then 4s) rather than
    hammering a queue we are already contributing to. A 429 means we hit
    a quota with a known reset window, and the response says exactly how
    long it is -- guessing 2 seconds against a per-minute quota is wrong
    by an order of magnitude, so we do what we are told.
    """
    last_error: Optional[Exception] = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except httpx.TransportError as e:
            # The connection failed before any HTTP status existed --
            # reset by peer, DNS hiccup, read timeout. These never reach
            # the API layer, so the status-code check below cannot see
            # them, and an earlier version of this function let them
            # through untouched: a single connection reset killed a
            # 35-item batch on its first call.
            #
            # httpx is not a dependency we chose; google-genai requires
            # it and makes its calls through it, so it is guaranteed
            # present wherever this module runs.
            last_error = e
            if attempt >= MAX_ATTEMPTS - 1:
                break
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            if getattr(e, "code", None) not in RETRYABLE_STATUS_CODES:
                raise  # permanent -- fail now, with the real reason
            last_error = e
            if attempt >= MAX_ATTEMPTS - 1:
                break

            requested = _server_requested_wait(e)
            if requested is not None:
                # Add a second of slack: sleeping for exactly the stated
                # window tends to land right on the boundary and fail again.
                wait = min(requested + 1.0, MAX_SERVER_REQUESTED_WAIT_SECONDS)
            else:
                wait = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            time.sleep(wait)

    raise ModelUnavailableError(
        f"{model} was unavailable after {MAX_ATTEMPTS} attempts: {last_error}. "
        "Usually temporary -- try again shortly, or switch GEMINI_MODEL to "
        "another Flash version."
    ) from last_error


def _parse_receipt(payload: Dict[str, Any]) -> Receipt:
    """Turn the API's JSON into typed objects.

    The schema makes the shape overwhelmingly likely to be right, but
    "overwhelmingly likely" is not "guaranteed" -- so we convert here and
    let a bad payload fail loudly at the boundary, rather than surfacing
    as a TypeError five stages downstream in the analysis layer.
    """
    try:
        items = [
            LineItem(
                line_number=int(raw["line_number"]),
                raw_description=str(raw["raw_description"]),
                line_total=raw.get("line_total"),
                product_code=raw.get("product_code"),
                quantity=raw.get("quantity"),
                unit=raw.get("unit"),
                unit_price=raw.get("unit_price"),
                discount=raw.get("discount"),
                tax_flag=raw.get("tax_flag"),
                is_voided=bool(raw.get("is_voided", False)),
            )
            for raw in payload["line_items"]
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise MalformedResponseError(
            f"Model returned line items in an unexpected shape: {e}"
        ) from e

    return Receipt(
        merchant_name=payload.get("merchant_name"),
        store_number=payload.get("store_number"),
        address=payload.get("address"),
        phone=payload.get("phone"),
        purchase_date=payload.get("purchase_date"),
        purchase_time=payload.get("purchase_time"),
        line_items=items,
        subtotal=payload.get("subtotal"),
        tax=payload.get("tax"),
        total=payload.get("total"),
        tax_rate_percent=payload.get("tax_rate_percent"),
        amount_paid=payload.get("amount_paid"),
        change_given=payload.get("change_given"),
        rounding_adjustment=payload.get("rounding_adjustment"),
        item_count_printed=payload.get("item_count_printed"),
        currency=payload.get("currency") or "USD",
        unreadable_notes=payload.get("unreadable_notes"),
    )


def extract(
    preprocessed: PreprocessResult,
    model: str = GEMINI_MODEL,
    client: Optional[genai.Client] = None,
) -> ExtractionResult:
    """Read a preprocessed receipt photo into a structured Receipt.

    Costs money on every call, which is why Step 1 rejects unusable
    photos before they ever reach here.
    """
    client = client or _load_client()
    jpeg = _image_to_jpeg_bytes(preprocessed)

    response = _generate_with_retry(
        client,
        model,
        contents=[
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RECEIPT_SCHEMA,
            # Zero temperature: transcription has one right answer, and
            # we would rather the same photo give the same result every
            # time than have the model be creative about a price.
            temperature=0.0,
        ),
    )

    if not response.text:
        raise EmptyResponseError(
            "Model returned no text. Usually a safety filter or a truncated "
            "response -- check response.candidates for the finish reason."
        )

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise MalformedResponseError(
            f"Model response wasn't valid JSON despite the schema: {e}"
        ) from e

    usage = response.usage_metadata
    return ExtractionResult(
        receipt=_parse_receipt(payload),
        model=model,
        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
        wire_bytes=len(jpeg),
        raw_json=payload,
    )
