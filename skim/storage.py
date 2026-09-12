"""
Step 7 of the Skim pipeline: Storage.

Responsibility of this module: keep what the pipeline produces, so that
questions can be asked of it months later.

Everything upstream currently evaporates when a script ends. The price
index this project exists to build is a *time series* -- it only becomes
possible once prices from March and prices from August are in the same
place.

SQLite, from the standard library. No new dependency, one file on disk,
no server to run, and it comfortably handles a lifetime of one person's
grocery receipts. It is the right answer until there are concurrent
writers or a hosted dashboard, and neither is real yet.

THE SCHEMA IS DESIGNED BACKWARDS FROM THE QUESTIONS.

That is the lesson Step 4 taught, when validation turned out to dictate
what extraction had to capture. Every question in the README reduces to
the same shape:

    "milk in March vs August"        -> product, date, price per unit
    "is Store A cheaper for me"      -> product, store, price per unit
    "my personal inflation rate"     -> product, date, price per unit, weight
    "which prices spiked"            -> product, date, price per unit
    "what am I due to buy next"      -> product, date

So the central fact is (product, date, price-per-unit, store), and the
schema exists to make that join cheap. Everything else is support.

ONE NOTE ON MONEY, because it differs from Step 4 on purpose.

Amounts charged are stored as INTEGER cents, for the reason validation
established: SQLite has no decimal type, REAL is binary floating point,
and these numbers get compared for equality.

But `price_per_base` is stored as REAL, and that is not an inconsistency.
It is a derived *ratio* -- dollars per gram, often 0.00154 -- not a
ledger entry. Rounding it to cents would destroy it, and nothing ever
compares two of them for exact equality. The rule was never "floats are
bad"; it was "amounts that must reconcile exactly cannot be floats".
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 4

# How to bring an EXISTING database up to each version. Keyed by the
# version being migrated TO.
#
# A fresh database never runs these: `SCHEMA` below already describes the
# current shape, so a new file is stamped at SCHEMA_VERSION directly.
# These statements exist only for databases created by older code --
# which, the moment there is real data in one, is the only kind that
# matters. `CREATE TABLE IF NOT EXISTS` silently does nothing to a table
# that already exists, so without this an added column would simply never
# appear and every write to it would fail on a database that had been in
# use.
# Each version lists the columns it introduced, as (table, column, type).
#
# Expressed as data rather than raw ALTER statements so they can be
# applied IDEMPOTENTLY -- which turns out to be essential, not tidy.
# `CREATE TABLE IF NOT EXISTS` builds any *missing* table at the CURRENT
# shape, columns and all. So a database old enough to predate a table
# gets that table complete, and a migration that then adds a column to it
# fails with "duplicate column name". Checking first is what makes
# "create what is missing, then patch what is old" safe to combine.
MIGRATIONS = {
    # Multi-user, minimally: a display name on each receipt.
    2: [("receipts", "uploaded_by", "TEXT")],
    # A name alone cannot separate one person's receipts from another's --
    # anyone could type any name. A shopper owns their receipts.
    3: [("receipts", "shopper_id", "INTEGER REFERENCES shoppers(shopper_id)")],
    # People do not read prices in grams. Keep the same price in the unit
    # the item was actually sold in, so the page never has to guess.
    4: [("line_items", "display_unit", "TEXT"),
        ("line_items", "price_per_display", "REAL")],
}


def _add_column_if_missing(connection: sqlite3.Connection, table: str,
                           column: str, declaration: str) -> None:
    existing = {row["name"] for row in
                connection.execute("PRAGMA table_info(" + table + ")")}
    if column not in existing:
        connection.execute(
            "ALTER TABLE " + table + " ADD COLUMN " + column + " " + declaration)


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- One photographed receipt.
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id          INTEGER PRIMARY KEY,
    -- Unique so that re-running the pipeline over data/raw updates a
    -- receipt instead of inserting a second copy of it. Without this,
    -- every re-run would double every price in the index.
    source_file         TEXT    NOT NULL UNIQUE,
    merchant_name       TEXT,
    store_number        TEXT,
    purchase_date       TEXT,   -- ISO 8601 'YYYY-MM-DD'; sorts as text
    purchase_time       TEXT,   -- 'HH:MM'
    subtotal_cents      INTEGER,
    tax_cents           INTEGER,
    total_cents         INTEGER,
    tax_rate_percent    REAL,
    amount_paid_cents   INTEGER,
    change_cents        INTEGER,
    item_count_printed  INTEGER,
    currency            TEXT    DEFAULT 'USD',
    -- Provenance. Which model read this receipt, and what it cost. Kept
    -- so a later accuracy comparison between models has something to
    -- group by, rather than needing the run to be repeated.
    extraction_model    TEXT,
    prompt_tokens       INTEGER,
    output_tokens       INTEGER,
    thinking_tokens     INTEGER,
    was_deskewed        INTEGER,
    ingested_at         TEXT    NOT NULL,
    -- Who uploaded this. A display name and nothing else: no account, no
    -- password, no email. Deliberately the smallest possible step into
    -- multi-user, because it needs no authentication to work and adds no
    -- personal data beyond what a friend types into a box.
    --
    -- Worth being clear-eyed that this field is NOT where the sensitive
    -- data is. The receipt image shows where someone shopped, when, and
    -- what they bought, whatever name sits beside it. That is the reason
    -- receipts are deletable, not the name.
    uploaded_by         TEXT,
    -- Who owns this receipt. Every query a person sees is filtered by
    -- this, which is what makes "only my analysis" true rather than
    -- merely displayed.
    shopper_id          INTEGER REFERENCES shoppers(shopper_id)
);

-- A person who uploads receipts. Deliberately the smallest identity
-- that works: a name they pick and a four-digit PIN. No email, no
-- password reset, no profile.
--
-- Four digits is only 10,000 combinations, so the PIN alone is weak --
-- `failed_attempts` and `locked_until` are what make it usable, by
-- turning "guess in seconds" into "guess over weeks". The hash is
-- stored rather than the PIN so a leaked database does not hand over
-- the PINs directly, even though a four-digit space is brute-forceable
-- offline. It is the correct habit and costs nothing.
CREATE TABLE IF NOT EXISTS shoppers (
    shopper_id      INTEGER PRIMARY KEY,
    name_key        TEXT NOT NULL UNIQUE,  -- lowercased, for lookup
    display_name    TEXT NOT NULL,         -- as they typed it
    pin_hash        TEXT NOT NULL,
    pin_salt        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT
);

-- A canonical product: the thing whose price is being tracked.
CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,
    canonical_text  TEXT NOT NULL,
    product         TEXT,
    brand           TEXT,
    variant         TEXT,
    category        TEXT,
    -- float32 vectors as raw bytes. A 3072-dim embedding is 12 KB as a
    -- blob and roughly 60 KB as JSON text, and the blob round-trips
    -- through numpy without parsing.
    embedding       BLOB,
    created_at      TEXT NOT NULL
);

-- Every receipt string that means a given product, plus the parse that
-- produced it. This is the cache from Step 5, now a table: a string
-- seen before never needs another API call.
CREATE TABLE IF NOT EXISTS product_aliases (
    store           TEXT NOT NULL,
    raw_description TEXT NOT NULL,
    product_id      TEXT REFERENCES products(product_id) ON DELETE SET NULL,
    parsed_json     TEXT,    -- the full ParsedProduct, for audit and re-use
    needs_review    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (store, raw_description)
);

-- One printed line. The fact table.
CREATE TABLE IF NOT EXISTS line_items (
    line_item_id    INTEGER PRIMARY KEY,
    receipt_id      INTEGER NOT NULL REFERENCES receipts(receipt_id) ON DELETE CASCADE,
    line_number     INTEGER,
    raw_description TEXT NOT NULL,
    product_id      TEXT REFERENCES products(product_id) ON DELETE SET NULL,
    quantity          REAL,
    unit_price_cents  INTEGER,
    line_total_cents  INTEGER,
    discount_cents    INTEGER,
    tax_flag          TEXT,
    is_voided         INTEGER NOT NULL DEFAULT 0,
    -- Step 6 output. `dimension` travels with the price so that a query
    -- can never average price-per-gram against price-per-item.
    dimension         TEXT,
    base_unit         TEXT,
    base_quantity     REAL,
    price_per_base    REAL,   -- a ratio, not an amount -- see module docstring
    inferred_unit     TEXT,   -- set when 'lb' was assumed rather than read
    normalization_note TEXT,
    -- The same price, in the unit a person recognises: "lb" for produce
    -- weighed at a US register, "100g" for something labelled in grams,
    -- "each" for things you just buy one of.
    display_unit      TEXT,
    price_per_display REAL
);

-- Step 4's verdicts, kept rather than printed. The pass rate over time
-- is the project's accuracy metric, and it cannot be computed from
-- results that were only ever written to a terminal.
CREATE TABLE IF NOT EXISTS validation_results (
    receipt_id  INTEGER NOT NULL REFERENCES receipts(receipt_id) ON DELETE CASCADE,
    check_name  TEXT NOT NULL,
    status      TEXT NOT NULL,   -- pass / fail / uncheckable
    detail      TEXT,
    line_number INTEGER
);

"""

# Indexes are applied separately, AFTER migrations. They name columns, and
# a migration that adds an indexed column must run before the index that
# uses it -- see _migrate for the full ordering.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_line_items_product ON line_items(product_id);
CREATE INDEX IF NOT EXISTS idx_line_items_receipt ON line_items(receipt_id);
CREATE INDEX IF NOT EXISTS idx_receipts_date      ON receipts(purchase_date);
CREATE INDEX IF NOT EXISTS idx_receipts_merchant  ON receipts(merchant_name);
CREATE INDEX IF NOT EXISTS idx_validation_receipt ON validation_results(receipt_id);
CREATE INDEX IF NOT EXISTS idx_receipts_shopper    ON receipts(shopper_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open the database, creating it if needed.

    Two pragmas matter here:

    `foreign_keys=ON` -- SQLite ignores foreign keys unless asked, per
    connection, for backwards compatibility. Without it the ON DELETE
    CASCADE above is decoration, and deleting a receipt would orphan its
    line items into permanent invisible rows.

    `journal_mode=WAL` -- lets a reader (a dashboard, a notebook) work
    while a writer is ingesting, instead of one blocking the other.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    _migrate(connection)
    return connection


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring the database up to the current schema version.

    Three cases, and the middle one is the whole reason this exists:

    A BRAND NEW FILE gets `SCHEMA` (already the current shape) and is
    stamped at SCHEMA_VERSION. No migrations run -- they would try to add
    columns that are already there.

    AN OLDER DATABASE gets each migration between its version and the
    current one, in order. This is the case that has real data in it, so
    it is the case worth getting right.

    A NEWER DATABASE -- written by code from the future -- is refused.
    Operating on a schema this code does not understand risks writing
    data that the newer code cannot read back.
    """
    known = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()

    if known is None:
        # Brand new file. SCHEMA is already the current shape, so it is
        # created once and stamped; the migrations would only try to add
        # columns that are already there.
        connection.executescript(SCHEMA)
        connection.executescript(INDEXES)
        connection.execute("DELETE FROM schema_version")
        connection.execute("INSERT INTO schema_version (version) VALUES (?)",
                           (SCHEMA_VERSION,))
        connection.commit()
        return

    row = connection.execute("SELECT version FROM schema_version").fetchone()
    current = int(row["version"]) if row else 0

    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database is schema version {current}, but this code only "
            f"understands {SCHEMA_VERSION}. It was written by a newer version "
            "of Skim -- update the code rather than downgrading the database."
        )

    # Three phases, and the order is the whole point.
    #
    # 1. TABLES first. A migration that ALTERs a table needs that table to
    #    exist -- including tables introduced by a later version than the
    #    database currently has.
    # 2. MIGRATIONS next, adding columns to tables that now certainly exist.
    # 3. INDEXES last, because an index names columns, and the column it
    #    names may have arrived in step 2.
    #
    # Both orderings have already bitten this project once each: indexes
    # before columns, then a migration altering a table that had not been
    # created yet. Three phases is what actually holds.
    connection.executescript(SCHEMA)

    for version in range(current + 1, SCHEMA_VERSION + 1):
        for table, column, declaration in MIGRATIONS.get(version, []):
            _add_column_if_missing(connection, table, column, declaration)
        connection.execute("UPDATE schema_version SET version = ?", (version,))

    connection.executescript(INDEXES)
    connection.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_cents(amount: Optional[float]) -> Optional[int]:
    """Same conversion Step 4 uses, applied at the storage boundary."""
    return None if amount is None else int(round(amount * 100))


def embedding_to_blob(vector: Optional[np.ndarray]) -> Optional[bytes]:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def save_receipt(
    connection: sqlite3.Connection,
    source_file: str,
    receipt: Any,          # skim.extract.Receipt
    extraction: Any = None,  # skim.extract.ExtractionResult, for provenance
    was_deskewed: Optional[bool] = None,
    uploaded_by: Optional[str] = None,
) -> int:
    """Insert or update one receipt, returning its id.

    Re-ingesting the same photo REPLACES the previous rows rather than
    adding to them. That is what makes the whole pipeline safe to re-run
    -- and re-running is the normal case here, since every prompt change
    means reprocessing the same photographs.

    Deleting the receipt row cascades to its line items and validation
    results, so a re-ingest cannot leave a half-updated receipt behind.
    """
    existing = connection.execute(
        "SELECT receipt_id FROM receipts WHERE source_file = ?", (source_file,)
    ).fetchone()
    if existing is not None:
        connection.execute("DELETE FROM receipts WHERE receipt_id = ?",
                           (existing["receipt_id"],))

    cursor = connection.execute(
        """
        INSERT INTO receipts (
            source_file, merchant_name, store_number, purchase_date,
            purchase_time, subtotal_cents, tax_cents, total_cents,
            tax_rate_percent, amount_paid_cents, change_cents,
            item_count_printed, currency, extraction_model, prompt_tokens,
            output_tokens, thinking_tokens, was_deskewed, ingested_at,
            uploaded_by
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            source_file, receipt.merchant_name, receipt.store_number,
            receipt.purchase_date, receipt.purchase_time,
            to_cents(receipt.subtotal), to_cents(receipt.tax),
            to_cents(receipt.total), receipt.tax_rate_percent,
            to_cents(receipt.amount_paid),
            # Registers disagree about the sign of change; the magnitude
            # is what means anything, and Step 4 already settled that.
            to_cents(abs(receipt.change_given)) if receipt.change_given is not None else None,
            receipt.item_count_printed, receipt.currency,
            getattr(extraction, "model", None),
            getattr(extraction, "prompt_tokens", None),
            getattr(extraction, "output_tokens", None),
            getattr(extraction, "thinking_tokens", None),
            None if was_deskewed is None else int(was_deskewed),
            _now(),
            (uploaded_by or "").strip() or None,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def save_line_item(
    connection: sqlite3.Connection,
    receipt_id: int,
    item: Any,                   # skim.extract.LineItem
    product_id: Optional[str] = None,
    normalized: Any = None,      # skim.units.NormalizedPrice
) -> None:
    connection.execute(
        """
        INSERT INTO line_items (
            receipt_id, line_number, raw_description, product_id, quantity,
            unit_price_cents, line_total_cents, discount_cents, tax_flag,
            is_voided, dimension, base_unit, base_quantity, price_per_base,
            inferred_unit, normalization_note, display_unit, price_per_display
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            receipt_id, item.line_number, item.raw_description, product_id,
            item.quantity, to_cents(item.unit_price), to_cents(item.line_total),
            to_cents(item.discount), item.tax_flag, int(item.is_voided),
            getattr(normalized, "dimension", None),
            getattr(normalized, "base_unit", None),
            getattr(normalized, "base_quantity", None),
            getattr(normalized, "price_per_base", None),
            getattr(normalized, "inferred_unit", None),
            getattr(normalized, "note", None),
            getattr(normalized, "display_unit", None),
            getattr(normalized, "price_per_display", None),
        ),
    )


def save_product(
    connection: sqlite3.Connection,
    product_id: str,
    canonical_text: str,
    product: Optional[str] = None,
    brand: Optional[str] = None,
    variant: Optional[str] = None,
    category: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
) -> None:
    connection.execute(
        """
        INSERT INTO products (product_id, canonical_text, product, brand,
                              variant, category, embedding, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(product_id) DO UPDATE SET
            canonical_text = excluded.canonical_text,
            product        = excluded.product,
            brand          = excluded.brand,
            variant        = excluded.variant,
            category       = excluded.category,
            embedding      = excluded.embedding
        """,
        (product_id, canonical_text, product, brand, variant, category,
         embedding_to_blob(embedding), _now()),
    )


def save_alias(
    connection: sqlite3.Connection,
    store: Optional[str],
    raw_description: str,
    product_id: Optional[str],
    parsed: Any = None,
    needs_review: bool = False,
) -> None:
    """Remember that a receipt string means a product.

    Aliases for products that could not be identified are stored too,
    with needs_review set. They are a worklist, not a failure to
    discard -- and re-parsing them would buy the same unknown again.
    """
    from dataclasses import asdict, is_dataclass

    payload = json.dumps(asdict(parsed)) if is_dataclass(parsed) else None
    connection.execute(
        """
        INSERT INTO product_aliases (store, raw_description, product_id,
                                     parsed_json, needs_review)
        VALUES (?,?,?,?,?)
        ON CONFLICT(store, raw_description) DO UPDATE SET
            product_id   = excluded.product_id,
            parsed_json  = excluded.parsed_json,
            needs_review = excluded.needs_review
        """,
        ((store or "unknown").strip().upper(), raw_description.strip(),
         product_id, payload, int(needs_review)),
    )


def save_validation(
    connection: sqlite3.Connection, receipt_id: int, report: Any
) -> None:
    """Keep Step 4's verdicts. The pass rate over time is the project's
    accuracy metric, and it cannot be computed from output that only
    ever reached a terminal."""
    for check in report.checks:
        connection.execute(
            """
            INSERT INTO validation_results (receipt_id, check_name, status,
                                            detail, line_number)
            VALUES (?,?,?,?,?)
            """,
            (receipt_id, check.name, check.status.value, check.detail,
             check.line_number),
        )


def delete_receipt(connection: sqlite3.Connection, receipt_id: int) -> bool:
    """Remove a receipt and everything derived from it.

    Built in from the start rather than bolted on later, because this
    app will hold other people's receipt photographs. The name someone
    types in a box is not the sensitive part -- the photograph is, since
    it shows where they shopped, when, and what they bought. Anyone
    whose data is here should be able to take it back out, and that is
    much harder to add convincingly after the fact.

    The cascade removes line items and validation results. Products are
    deliberately left alone: they are shared across receipts, so
    deleting "onion" because one receipt went away would damage
    everybody else's price history.
    """
    cursor = connection.execute(
        "DELETE FROM receipts WHERE receipt_id = ?", (receipt_id,)
    )
    connection.commit()
    return cursor.rowcount > 0
