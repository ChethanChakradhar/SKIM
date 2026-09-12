"""
Resolve every parsed product against the canonical catalog.

    python3 scripts/build_catalog.py

Reads the parse cache (no parse calls), embeds each product once, and
either matches it to an existing catalog entry or creates a new one.
Safe to re-run: known aliases short-circuit before any API call.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.extract import _load_client
from skim.match import CREATED, MATCHED, REVIEW, ProductCatalog, resolve
from skim.normalize import ProductCache

PROCESSED = ROOT / "data" / "processed"
MARK = {MATCHED: "match ", CREATED: "new   ", REVIEW: "REVIEW"}


def main() -> None:
    client = _load_client()
    cache = ProductCache(PROCESSED / "product_cache.json")
    catalog = ProductCatalog(PROCESSED / "product_catalog.json")

    counts = {MATCHED: 0, CREATED: 0, REVIEW: 0}

    for path in sorted(PROCESSED.glob("*_extracted.json")):
        payload = json.loads(path.read_text())
        store = payload.get("merchant_name")
        print(f"\n=== {store} ===", flush=True)

        for item in payload["line_items"]:
            if item.get("is_voided"):
                continue
            parsed = cache.get(store, item["raw_description"])
            if parsed is None:
                print(f"  (not parsed yet) {item['raw_description']}", flush=True)
                continue

            decision = resolve(parsed, catalog, store=store, client=client)
            counts[decision.outcome] += 1
            sim = f"{decision.similarity:.3f}" if decision.similarity else "  -  "
            adj = " [model]" if decision.adjudicated else ""
            print(f"  {MARK[decision.outcome]} {sim}  "
                  f"{parsed.raw_description:<30} {decision.reason[:60]}{adj}", flush=True)

    print(f"\ncatalog holds {len(catalog.products)} canonical products")
    print(f"  {counts[CREATED]} created, {counts[MATCHED]} matched, "
          f"{counts[REVIEW]} need review")

    multi = [p for p in catalog.products.values() if len(p.aliases) > 1]
    if multi:
        print("\nproducts with more than one receipt string -- the point of all this:")
        for p in multi:
            print(f"  {p.canonical_text}")
            for a in p.aliases:
                print(f"      {a}")


if __name__ == "__main__":
    main()
