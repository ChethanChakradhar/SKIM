"""
The Skim web app.

`skim/` is a library: modules that turn a photograph into structured,
validated, normalized prices. This package is the application that puts
that library in front of a person.

Keeping them apart matters. Every pipeline stage is testable without a
browser, and every route here is thin -- it handles HTTP, sessions and
templates, and delegates the actual work. When something breaks it is
obvious which half to look in.

TWO DECISIONS WORTH KNOWING ABOUT.

1. Uploading is split into fast and slow halves.

   Extraction takes about six seconds. Product parsing takes one API
   call per product the catalog has not seen, so a first receipt with
   eighteen new products takes a minute or more -- long enough that a
   browser and most hosts give up on the request entirely.

   So the request does extraction, validation and storage, then returns
   the receipt. Parsing and matching run afterwards in the background,
   and the per-unit prices appear on a later page load. The person sees
   their line items in six seconds instead of staring at a spinner.

2. Sessions are a hand-signed cookie, not a session library.

   A cookie holding "shopper 4" would let anyone edit it to "shopper 5"
   and read someone else's receipts. So the cookie carries the id AND an
   HMAC of that id made with a server-side secret: the server can verify
   it produced the value, and nobody without the secret can forge one.
   That is the whole mechanism a session library would provide here, in
   about fifteen lines and no dependency.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import shutil
import sqlite3
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from skim import storage
from skim.capture import CaptureError, capture
from skim.extract import ExtractionError, extract
from skim.match import ProductCatalog, resolve
from skim.normalize import ProductCache
from skim.preprocess import preprocess
from skim.shoppers import ShopperError, get_shopper, sign_in
from skim.units import normalize_line
from skim.validate import validate

# Railway mounts a persistent volume; everything that must survive a
# deploy lives under it. Locally it falls back to ./data, so the app runs
# the same way in both places.
DATA_DIR = Path(os.getenv("SKIM_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "skim.db"

# Signing key for session cookies. In production this comes from the
# environment and stays stable; a generated fallback keeps local
# development working, at the cost of logging everyone out on restart.
SECRET_KEY = os.getenv("SKIM_SECRET_KEY") or secrets.token_hex(32)

COOKIE_NAME = "skim_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # a month; this is a receipt app, not a bank

MAX_UPLOAD_BYTES = 25_000_000

app = FastAPI(title="Skim")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
          name="static")


# --- sessions --------------------------------------------------------

def _sign(shopper_id: int) -> str:
    mac = hmac.new(SECRET_KEY.encode(), str(shopper_id).encode(), sha256).hexdigest()
    return f"{shopper_id}.{mac}"


def _verify(cookie: Optional[str]) -> Optional[int]:
    """Read a shopper id out of a cookie, or None if it wasn't ours.

    `compare_digest` again, for the same reason as the PIN check: a
    plain `==` on the signature leaks, through timing, how many leading
    characters a forged value got right.
    """
    if not cookie or "." not in cookie:
        return None
    raw_id, _, mac = cookie.partition(".")
    if not raw_id.isdigit():
        return None
    if not hmac.compare_digest(_sign(int(raw_id)), cookie):
        return None
    return int(raw_id)


def current_shopper(request: Request, connection: sqlite3.Connection):
    shopper_id = _verify(request.cookies.get(COOKIE_NAME))
    return get_shopper(connection, shopper_id) if shopper_id else None


def db() -> sqlite3.Connection:
    return storage.connect(DB_PATH)


# --- the slow half of an upload --------------------------------------

def enrich_receipt(receipt_id: int, store: Optional[str]) -> None:
    """Parse and match every product on a receipt, then fill in prices.

    Runs after the response has been sent. One API call per product the
    catalog has never seen, so the first few receipts are slow and later
    ones are nearly free -- a store prints the same strings every time.
    """
    connection = db()
    try:
        cache = ProductCache(DATA_DIR / "processed" / "product_cache.json")
        catalog = ProductCatalog(DATA_DIR / "processed" / "product_catalog.json")

        rows = connection.execute(
            "SELECT line_item_id, raw_description, quantity, unit_price_cents, "
            "line_total_cents, is_voided FROM line_items WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchall()

        for row in rows:
            if row["is_voided"]:
                continue
            description = row["raw_description"]
            parsed = cache.parse(description, store=store,
                                 sibling_descriptions=[r["raw_description"] for r in rows])
            decision = resolve(parsed, catalog, store=store)

            normalized = normalize_line(
                parsed,
                row["quantity"],
                (row["unit_price_cents"] or 0) / 100 if row["unit_price_cents"] else None,
                (row["line_total_cents"] or 0) / 100 if row["line_total_cents"] else None,
            )

            storage.save_product(
                connection, decision.product_id,
                *_catalog_fields(catalog, decision.product_id),
            ) if decision.product_id else None

            connection.execute(
                """
                UPDATE line_items SET product_id = ?, dimension = ?, base_unit = ?,
                    base_quantity = ?, price_per_base = ?, inferred_unit = ?,
                    normalization_note = ?, display_unit = ?, price_per_display = ?
                WHERE line_item_id = ?
                """,
                (decision.product_id, normalized.dimension, normalized.base_unit,
                 normalized.base_quantity, normalized.price_per_base,
                 normalized.inferred_unit, normalized.note,
                 normalized.display_unit, normalized.price_per_display,
                 row["line_item_id"]),
            )
            storage.save_alias(connection, store, description, decision.product_id,
                               parsed, parsed.needs_review)
        connection.commit()
    finally:
        connection.close()


def _catalog_fields(catalog: ProductCatalog, product_id: str):
    product = catalog.products[product_id]
    return (product.canonical_text, product.product, product.brand,
            product.variant, product.category, None)


# --- routes ----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    connection = db()
    try:
        shopper = current_shopper(request, connection)
        if shopper:
            return RedirectResponse("/me", status_code=303)
        total = connection.execute(
            "SELECT COUNT(*) c FROM receipts").fetchone()["c"]
        products = connection.execute(
            "SELECT COUNT(*) c FROM products").fetchone()["c"]
        return templates.TemplateResponse("signin.html", {
            "request": request, "error": None,
            "receipt_count": total, "product_count": products,
        })
    finally:
        connection.close()


@app.post("/signin", response_class=HTMLResponse)
def do_signin(request: Request, name: str = Form(...), pin: str = Form(...)):
    connection = db()
    try:
        try:
            shopper_id = sign_in(connection, name, pin)
        except ShopperError as e:
            return templates.TemplateResponse("signin.html", {
                "request": request, "error": str(e),
                "receipt_count": connection.execute(
                    "SELECT COUNT(*) c FROM receipts").fetchone()["c"],
                "product_count": connection.execute(
                    "SELECT COUNT(*) c FROM products").fetchone()["c"],
            }, status_code=400)

        response = RedirectResponse("/me", status_code=303)
        response.set_cookie(
            COOKIE_NAME, _sign(shopper_id), max_age=COOKIE_MAX_AGE,
            httponly=True,   # JavaScript cannot read it, so an XSS bug cannot steal it
            samesite="lax",  # not sent on cross-site POSTs, which blocks basic CSRF
            secure=bool(os.getenv("SKIM_SECRET_KEY")),  # HTTPS-only in production
        )
        return response
    finally:
        connection.close()


@app.get("/logout")
def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/me", response_class=HTMLResponse)
def dashboard(request: Request):
    connection = db()
    try:
        shopper = current_shopper(request, connection)
        if not shopper:
            return RedirectResponse("/", status_code=303)

        # Every query is filtered by shopper_id. That is what makes
        # "only your receipts" true rather than merely displayed.
        receipts = connection.execute(
            """
            SELECT r.*, COUNT(li.line_item_id) items
            FROM receipts r LEFT JOIN line_items li ON li.receipt_id = r.receipt_id
            WHERE r.shopper_id = ?
            GROUP BY r.receipt_id ORDER BY r.purchase_date DESC, r.receipt_id DESC
            """, (shopper["shopper_id"],)).fetchall()

        prices = connection.execute(
            """
            SELECT p.canonical_text, p.category, r.merchant_name, r.purchase_date,
                   li.price_per_display, li.display_unit, li.inferred_unit
            FROM line_items li
            JOIN receipts r ON r.receipt_id = li.receipt_id
            JOIN products p ON p.product_id = li.product_id
            WHERE r.shopper_id = ? AND li.price_per_display IS NOT NULL
            ORDER BY li.price_per_display DESC
            """, (shopper["shopper_id"],)).fetchall()

        spend = connection.execute(
            "SELECT COALESCE(SUM(total_cents),0) c FROM receipts WHERE shopper_id = ?",
            (shopper["shopper_id"],)).fetchone()["c"]

        return templates.TemplateResponse("dashboard.html", {
            "request": request, "shopper": shopper, "receipts": receipts,
            "prices": prices, "spend_cents": spend,
        })
    finally:
        connection.close()


@app.post("/upload")
async def upload(request: Request, background: BackgroundTasks,
                 photo: UploadFile = File(...)):
    connection = db()
    try:
        shopper = current_shopper(request, connection)
        if not shopper:
            return RedirectResponse("/", status_code=303)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(photo.filename or "receipt.jpg").suffix.lower() or ".jpg"
        stored_name = f"{shopper['shopper_id']}_{uuid.uuid4().hex[:10]}{suffix}"
        target = UPLOAD_DIR / stored_name

        with target.open("wb") as out:
            shutil.copyfileobj(photo.file, out, length=1024 * 1024)

        if target.stat().st_size > MAX_UPLOAD_BYTES:
            target.unlink(missing_ok=True)
            return _error_redirect("That file is larger than 25MB.")

        try:
            captured = capture(target)
            prepped = preprocess(captured)
            result = extract(prepped)
        except (CaptureError, ExtractionError) as e:
            target.unlink(missing_ok=True)
            return _error_redirect(str(e))

        receipt = result.receipt
        receipt_id = storage.save_receipt(
            connection, stored_name, receipt, extraction=result,
            was_deskewed=prepped.deskewed,
            uploaded_by=shopper["display_name"],
        )
        connection.execute("UPDATE receipts SET shopper_id = ? WHERE receipt_id = ?",
                           (shopper["shopper_id"], receipt_id))
        for item in receipt.line_items:
            storage.save_line_item(connection, receipt_id, item)
        storage.save_validation(connection, receipt_id, validate(receipt))
        connection.commit()

        # The slow half: one API call per unseen product. Runs after the
        # response, so the person is looking at their line items instead
        # of a spinner.
        background.add_task(enrich_receipt, receipt_id, receipt.merchant_name)

        return RedirectResponse(f"/receipt/{receipt_id}", status_code=303)
    finally:
        connection.close()


def _error_redirect(message: str) -> RedirectResponse:
    from urllib.parse import quote
    return RedirectResponse(f"/me?error={quote(message)}", status_code=303)


@app.get("/receipt/{receipt_id}", response_class=HTMLResponse)
def receipt_detail(request: Request, receipt_id: int):
    connection = db()
    try:
        shopper = current_shopper(request, connection)
        if not shopper:
            return RedirectResponse("/", status_code=303)

        # The shopper_id in the WHERE clause is the access check. Without
        # it, changing the number in the URL would show someone else's
        # receipt -- the most common way an app like this leaks data.
        receipt = connection.execute(
            "SELECT * FROM receipts WHERE receipt_id = ? AND shopper_id = ?",
            (receipt_id, shopper["shopper_id"])).fetchone()
        if receipt is None:
            return RedirectResponse("/me", status_code=303)

        items = connection.execute(
            """
            SELECT li.*, p.canonical_text, p.category
            FROM line_items li
            LEFT JOIN products p ON p.product_id = li.product_id
            WHERE li.receipt_id = ? ORDER BY li.line_number
            """, (receipt_id,)).fetchall()
        checks = connection.execute(
            "SELECT * FROM validation_results WHERE receipt_id = ?",
            (receipt_id,)).fetchall()

        passed = sum(1 for c in checks if c["status"] == "pass")
        ran = sum(1 for c in checks if c["status"] in ("pass", "fail"))
        pending = sum(1 for i in items
                      if not i["is_voided"] and i["product_id"] is None)

        return templates.TemplateResponse("receipt.html", {
            "request": request, "shopper": shopper, "receipt": receipt,
            "items": items, "checks": checks, "passed": passed, "ran": ran,
            "pending": pending,
        })
    finally:
        connection.close()


@app.post("/receipt/{receipt_id}/delete")
def delete(request: Request, receipt_id: int):
    connection = db()
    try:
        shopper = current_shopper(request, connection)
        if not shopper:
            return RedirectResponse("/", status_code=303)
        owned = connection.execute(
            "SELECT source_file FROM receipts WHERE receipt_id = ? AND shopper_id = ?",
            (receipt_id, shopper["shopper_id"])).fetchone()
        if owned:
            storage.delete_receipt(connection, receipt_id)
            # The photograph goes too. Deleting the row but keeping the
            # image would not be deletion in any sense that matters.
            (UPLOAD_DIR / owned["source_file"]).unlink(missing_ok=True)
        return RedirectResponse("/me", status_code=303)
    finally:
        connection.close()


@app.get("/health")
def health():
    """Railway restarts the container if this stops answering."""
    return {"ok": True}
