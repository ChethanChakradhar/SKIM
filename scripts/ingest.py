"""
Run everything into the database: extraction JSON -> parse cache ->
catalog -> validation -> units -> SQLite.

    python3 scripts/ingest.py

Reads the saved extraction JSON rather than re-calling the VLM, so this
is free to re-run. Re-ingesting a receipt replaces its rows.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim import storage
from skim.extract import _parse_receipt
from skim.match import ProductCatalog
from skim.normalize import ProductCache
from skim.units import normalize_line
from skim.validate import validate

PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "skim.db"


def main() -> None:
    connection = storage.connect(DB_PATH)
    cache = ProductCache(PROCESSED / "product_cache.json")
    catalog = ProductCatalog(PROCESSED / "product_catalog.json")

    # Products first: line items reference them.
    for product in catalog.products.values():
        storage.save_product(
            connection, product.product_id, product.canonical_text,
            product.product, product.brand, product.variant, product.category,
            embedding=product.embedding or None,
        )
    connection.commit()

    alias_to_product = {
        alias: p.product_id
        for p in catalog.products.values() for alias in p.aliases
    }

    receipts = 0
    for path in sorted(PROCESSED.glob("*_extracted.json")):
        payload = json.loads(path.read_text())
        receipt = _parse_receipt(payload)
        source = f"{path.stem.replace('_extracted', '')}.jpg"

        receipt_id = storage.save_receipt(connection, source, receipt)
        storage.save_validation(connection, receipt_id, validate(receipt))

        store = receipt.merchant_name
        for item in receipt.line_items:
            alias_key = f"{(store or 'unknown').strip().upper()}||{item.raw_description.strip()}"
            product_id = alias_to_product.get(alias_key)
            parsed = cache.get(store, item.raw_description)

            normalized = None
            if parsed is not None:
                normalized = normalize_line(
                    parsed, item.quantity, item.unit_price, item.line_total
                )
                storage.save_alias(connection, store, item.raw_description,
                                   product_id, parsed, parsed.needs_review)

            storage.save_line_item(connection, receipt_id, item,
                                   product_id, normalized)

        connection.commit()
        receipts += 1
        print(f"  {source:<16} {len(receipt.line_items)} lines  -> receipt {receipt_id}")

    counts = {
        table: connection.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        for table in ("receipts", "line_items", "products", "product_aliases",
                      "validation_results")
    }
    print(f"\n{DB_PATH.relative_to(ROOT)}")
    for table, count in counts.items():
        print(f"  {table:<20} {count}")

    connection.close()


if __name__ == "__main__":
    main()
