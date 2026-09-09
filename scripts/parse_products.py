"""
Parse every product string from the saved extraction JSON.

    python3 scripts/parse_products.py

Reads data/processed/*_extracted.json, so it costs no extraction calls --
only one parse call per unique description (two for the ones that need
the retry).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.extract import _load_client
from skim.normalize import ProductCache

PROCESSED = ROOT / "data" / "processed"
CACHE_PATH = PROCESSED / "product_cache.json"


def main() -> None:
    client = _load_client()
    cache = ProductCache(CACHE_PATH)
    already = len(cache.entries)
    seen = {}
    retried = []
    unidentified = []

    for path in sorted(PROCESSED.glob("*_extracted.json")):
        payload = json.loads(path.read_text())
        store = payload.get("merchant_name")
        descriptions = [
            i["raw_description"] for i in payload["line_items"] if not i.get("is_voided")
        ]

        print(f"\n=== {store} ({path.stem.replace('_extracted', '')}) ===", flush=True)
        for description in descriptions:
            if description in seen:
                print(f"  [repeat] {description}", flush=True)
                continue

            was_cached = cache.get(store, description) is not None
            parsed = cache.parse(
                description, store=store, sibling_descriptions=descriptions,
                client=client,
            )
            seen[description] = parsed
            if parsed.attempts > 1:
                retried.append(parsed)
            if not parsed.is_identified:
                unidentified.append(parsed)
            if was_cached:
                print(f"  [cached] {description:<32} -> {parsed.canonical_text or '???'}",
                      flush=True)
                continue

            size = (
                f"{parsed.size_value:g}{parsed.size_unit}"
                if parsed.size_value is not None else
                (parsed.size_unit or "")
            )
            flag = "  <-- REVIEW" if not parsed.is_identified else ""
            mark = "*" if parsed.attempts > 1 else " "
            print(f" {mark}{description:<34} -> {parsed.canonical_text or '???':<38}"
                  f" [{parsed.category or '?':<13}] {size}{flag}", flush=True)
            if parsed.parse_note:
                print(f"      note: {parsed.parse_note}", flush=True)

    print(f"\n{len(seen)} unique descriptions across all receipts")
    print(f"{len(cache.entries) - already} newly parsed, {already} served from cache")
    print(f"{len(retried)} needed the retry-with-context path")
    print(f"{len(unidentified)} unidentified -> human review")
    for p in unidentified:
        print(f"    {p.raw_description}: {p.parse_note or 'no reason given'}")


if __name__ == "__main__":
    main()
