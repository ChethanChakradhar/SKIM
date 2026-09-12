"""
Step 5, second half: matching parsed products to a canonical catalog.

`skim/normalize.py` decodes one receipt string into attributes. This
module decides whether that product is one we have seen before.

The architecture is set by a measurement, not a preference. Embedding
the 33 real products from the first three receipts and ranking all 465
pairs by cosine similarity gives:

    0.839   guava                  <-> guava 1 each              SAME
    0.828   skinless whole chicken <-> boneless chicken breast   DIFFERENT
    0.718   kitchen towel sent/pet <-> kitchen towel solid blk   DIFFERENT

The true match and the nearest false match are 0.011 apart. No threshold
separates them: at 0.83 whole chickens merge with chicken breasts, at
0.84 the two guavas stay split forever.

So similarity is used ONLY to retrieve candidates, never to decide.
Narrowing 465 pairs to a handful is what it is good at -- the cheap
"blocking" stage that entity resolution has always split out. The
decision is made afterwards, on the structured fields.

And that is the real argument for parsing first. Not that embeddings
score better on parsed text, but that parsing produces fields you can
compare *deterministically*: the two chickens differ in `product`
("whole chicken" vs "chicken breast") while the two guavas share it
exactly, differing only in a size the register invented. Embeddings
cannot see that distinction. Two string comparisons can.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import types

from skim.extract import _load_client
from skim.normalize import ParsedProduct

EMBED_MODEL = "gemini-embedding-2"

# How many catalog neighbours to consider. Retrieval only has to put the
# right answer somewhere in a short list; the decision layer does the
# rest, so a generous k costs a few string comparisons and buys
# robustness against the ranking being slightly off.
RETRIEVAL_TOP_K = 5

# Below this, not even worth considering. Taken from the measured
# distribution: the 99th percentile of genuinely-different pairs sits at
# 0.669, so 0.60 keeps every plausible candidate while discarding the
# long tail of unrelated products. This is a RETRIEVAL floor, not a
# match threshold -- nothing is ever merged because it cleared it.
RETRIEVAL_FLOOR = 0.60


@dataclass
class CatalogProduct:
    """One canonical product, and every receipt string that means it."""

    product_id: str
    canonical_text: str
    product: Optional[str]
    brand: Optional[str]
    variant: Optional[str]
    category: Optional[str]
    embedding: List[float] = field(default_factory=list, repr=False)
    aliases: List[str] = field(default_factory=list)  # "STORE||RAW DESCRIPTION"


def embed_texts(texts: List[str], client: Optional[genai.Client] = None) -> np.ndarray:
    """Embed and L2-normalize, so cosine similarity is a dot product.

    Normalizing once here means every later comparison is a plain
    matrix multiply -- no repeated norm computation, and no chance of
    comparing a normalized vector against a raw one.
    """
    client = client or _load_client()
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class ProductCatalog:
    """The canonical products seen so far, with their embeddings.

    Deliberately a flat file and a numpy array rather than a vector
    database. At a few thousand products the entire index is a few tens
    of megabytes and retrieval is one matrix-vector product -- roughly a
    millisecond. Approximate nearest-neighbour indexes (FAISS, HNSW)
    exist to make billion-scale search tractable; here they would add a
    dependency, a build step and an approximation, to speed up an
    operation that is already instant.

    Storage moves to SQLite in Step 7. This shape is intentionally easy
    to migrate: rows plus a blob.
    """

    def __init__(self, path: Path):
        self.path = path
        self.products: Dict[str, CatalogProduct] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self.products = {
                pid: CatalogProduct(**entry) for pid, entry in raw.items()
            }

    # -- persistence ------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {pid: asdict(p) for pid, p in self.products.items()},
            indent=2,
        ))

    # -- retrieval --------------------------------------------------

    def _matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Stack the catalog's embeddings into one array for comparison."""
        ids = [pid for pid, p in self.products.items() if p.embedding]
        if not ids:
            return np.empty((0, 0), dtype=np.float32), []
        return (
            np.array([self.products[pid].embedding for pid in ids], dtype=np.float32),
            ids,
        )

    def find_candidates(
        self, embedding: np.ndarray, top_k: int = RETRIEVAL_TOP_K
    ) -> List[Tuple[float, CatalogProduct]]:
        """The handful of catalog entries worth actually comparing.

        This is the whole contribution of embeddings to the pipeline:
        turning "compare against every product ever seen" into "compare
        against these five". It narrows; it never concludes.
        """
        matrix, ids = self._matrix()
        if not ids:
            return []

        scores = matrix @ embedding
        order = np.argsort(scores)[::-1][:top_k]
        return [
            (float(scores[i]), self.products[ids[i]])
            for i in order
            if scores[i] >= RETRIEVAL_FLOOR
        ]

    # -- mutation ---------------------------------------------------

    def add(
        self, parsed: ParsedProduct, embedding: np.ndarray, alias_key: str
    ) -> CatalogProduct:
        product = CatalogProduct(
            product_id=str(uuid.uuid4())[:8],
            canonical_text=parsed.canonical_text or parsed.raw_description,
            product=parsed.product,
            brand=parsed.brand,
            variant=parsed.variant,
            category=parsed.category,
            embedding=[round(float(x), 5) for x in embedding],
            aliases=[alias_key],
        )
        self.products[product.product_id] = product
        self.save()
        return product

    def add_alias(self, product_id: str, alias_key: str) -> None:
        """Record that another receipt string means an existing product.

        This is where the value accumulates. Every alias learned is a
        string that will never need an API call again, and the reason
        the per-receipt cost of this stage trends toward zero.
        """
        product = self.products[product_id]
        if alias_key not in product.aliases:
            product.aliases.append(alias_key)
            self.save()

    def find_by_alias(self, alias_key: str) -> Optional[CatalogProduct]:
        for product in self.products.values():
            if alias_key in product.aliases:
                return product
        return None


# --- the decision layer ---------------------------------------------

SAME, DIFFERENT, UNSURE = "same", "different", "unsure"


def _norm(value: Optional[str]) -> Optional[str]:
    return value.strip().lower() if value else None


def compare_attributes(a: ParsedProduct, b: CatalogProduct) -> Tuple[str, str]:
    """Decide identity from the structured fields. Returns (verdict, why).

    The rules, and the reasoning behind each:

    A DIFFERENT BRAND means a different product. Great Value milk and
    Lactaid milk are not interchangeable at any price. A brand present
    on one side and absent on the other is NOT a conflict -- plenty of
    receipts omit the brand for an item another receipt names.

    A DIFFERENT VARIANT means a different product. 2% and whole milk
    have different prices and tracking them as one would average away
    the thing being measured.

    A DIFFERENT SIZE DOES NOT. This is the rule that fixes the guava
    split, and it is counter-intuitive enough to be worth stating: a
    gallon of milk and a quart of milk are the SAME product bought in
    different amounts. Step 6 has already converted both to a price per
    millilitre, so they are directly comparable -- and splitting them
    would defeat the purpose of having normalized units at all. The
    size stays on the line item, where it belongs; it is not part of
    identity.

    THE PRODUCT NOUN DECIDES THE REST. Matching nouns with compatible
    brand and variant is a match. Differing nouns are not automatically
    a mismatch, because "chili" and "chilli" and "chicken breast" and
    "chicken breasts" are the same thing spelled differently -- that is
    handed to the adjudicator rather than guessed at.
    """
    if _norm(a.brand) and _norm(b.brand) and _norm(a.brand) != _norm(b.brand):
        return DIFFERENT, f"different brands: {a.brand} vs {b.brand}"

    if _norm(a.variant) and _norm(b.variant) and _norm(a.variant) != _norm(b.variant):
        return DIFFERENT, f"different variants: {a.variant} vs {b.variant}"

    if _norm(a.product) and _norm(a.product) == _norm(b.product):
        return SAME, f"same product noun '{a.product}', no conflicting attributes"

    return UNSURE, f"product nouns differ: '{a.product}' vs '{b.product}'"


ADJUDICATION_PROMPT = """\
Two product descriptions from grocery receipts. Decide whether they
refer to the SAME product for the purpose of tracking its price over
time.

  A: {a}
  B: {b}

Same product means: a shopper would consider these interchangeable
purchases of one item. Spelling differences, pluralization, word order
and register abbreviations do not make products different.

DIFFERENT PACKAGE SIZES ARE STILL THE SAME PRODUCT -- a gallon of milk
and a quart of milk are one product bought in two amounts, and prices
are already normalized per unit before comparison.

Genuinely different products include: different cuts or forms of the
same animal or plant (whole chicken vs chicken breast, ground beef vs
steak), different varieties (gala apple vs granny smith), different
flavours or scents, and different brands.

Answer `same`, `different`, or `unsure`. Use `unsure` when the
descriptions are too abbreviated to tell -- a wrong merge is permanent
and silent, while an `unsure` costs one human glance.
"""

ADJUDICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [SAME, DIFFERENT, UNSURE]},
        "reason": {"type": "string"},
    },
    "required": ["verdict"],
}


def adjudicate(
    a: ParsedProduct,
    b: CatalogProduct,
    client: Optional[genai.Client] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Ask a model whether two near-neighbours are the same product.

    Runs only on pairs that survived retrieval AND that the structured
    comparison could not settle, which is a small fraction of a small
    number. This is the expensive, accurate stage that the cheap
    retrieval stage exists to protect.
    """
    from skim.extract import GEMINI_MODEL, _generate_with_retry

    client = client or _load_client()
    response = _generate_with_retry(
        client,
        model or GEMINI_MODEL,
        contents=[ADJUDICATION_PROMPT.format(
            a=a.canonical_text or a.raw_description, b=b.canonical_text
        )],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ADJUDICATION_SCHEMA,
            temperature=0.0,
        ),
    )
    payload = json.loads(response.text or "{}")
    verdict = payload.get("verdict", UNSURE)
    return (
        verdict if verdict in (SAME, DIFFERENT, UNSURE) else UNSURE,
        payload.get("reason") or "",
    )


@dataclass
class MatchDecision:
    """What happened to one parsed product, and why."""

    outcome: str  # "matched" | "created" | "review"
    product_id: Optional[str]
    similarity: Optional[float]
    reason: str
    adjudicated: bool = False


MATCHED, CREATED, REVIEW = "matched", "created", "review"


def resolve(
    parsed: ParsedProduct,
    catalog: ProductCatalog,
    store: Optional[str] = None,
    client: Optional[genai.Client] = None,
    allow_adjudication: bool = True,
) -> MatchDecision:
    """Find this product in the catalog, or add it.

    The order of the stages is the design:

      0. Alias hit          free, and the common case once settled
      1. Retrieve           embeddings narrow the field; they decide nothing
      2. Compare fields     deterministic, auditable, settles most pairs
      3. Adjudicate         a model call, only for what survives step 2
      4. Review             when even that is unsure

    Each stage is more expensive and more capable than the last, and
    each only sees what the cheaper ones could not settle.
    """
    client = client or _load_client()
    alias_key = f"{(store or 'unknown').strip().upper()}||{parsed.raw_description.strip()}"

    existing = catalog.find_by_alias(alias_key)
    if existing is not None:
        return MatchDecision(MATCHED, existing.product_id, None,
                             "already a known alias of this product")

    if not parsed.is_identified:
        return MatchDecision(REVIEW, None, None,
                             parsed.parse_note or "product could not be identified")

    text = parsed.canonical_text or parsed.raw_description
    embedding = embed_texts([text], client=client)[0]
    candidates = catalog.find_candidates(embedding)

    unsure_best: Optional[Tuple[float, CatalogProduct, str]] = None

    for similarity, candidate in candidates:
        verdict, why = compare_attributes(parsed, candidate)

        if verdict == DIFFERENT:
            continue  # a conflicting attribute is decisive; look at the next one

        if verdict == SAME:
            catalog.add_alias(candidate.product_id, alias_key)
            return MatchDecision(MATCHED, candidate.product_id, similarity, why)

        if allow_adjudication:
            model_verdict, model_reason = adjudicate(parsed, candidate, client=client)
            if model_verdict == SAME:
                catalog.add_alias(candidate.product_id, alias_key)
                return MatchDecision(MATCHED, candidate.product_id, similarity,
                                     model_reason or why, adjudicated=True)
            if model_verdict == UNSURE and unsure_best is None:
                unsure_best = (similarity, candidate, model_reason or why)
        elif unsure_best is None:
            unsure_best = (similarity, candidate, why)

    if unsure_best is not None:
        similarity, candidate, why = unsure_best
        # Deliberately NOT merged. An unresolved near-match is the exact
        # situation where guessing is most tempting and most costly: the
        # two products look alike, so a wrong merge would go unnoticed.
        return MatchDecision(REVIEW, candidate.product_id, similarity,
                             f"possible match, not confirmed: {why}",
                             adjudicated=allow_adjudication)

    created = catalog.add(parsed, embedding, alias_key)
    top = f", nearest was {candidates[0][0]:.2f}" if candidates else ""
    return MatchDecision(CREATED, created.product_id,
                         candidates[0][0] if candidates else None,
                         f"no catalog product matched{top}")

