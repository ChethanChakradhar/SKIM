"""
Run one receipt photo through the full pipeline so far --
capture -> preprocess -> extract -- and print what came back.

    python3 scripts/extract_receipt.py data/raw/IMG_2653.jpg

Every run costs a real (tiny) API call, so this takes one photo at a
time rather than looping over the folder by default.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.capture import capture
from skim.extract import extract
from skim.preprocess import preprocess


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    path = Path(sys.argv[1])
    captured = capture(path)
    prepped = preprocess(captured)
    result = extract(prepped)
    r = result.receipt

    # Keep the raw response. Every re-read of it is free, whereas every
    # re-extraction costs another call -- and when a later stage disagrees
    # with what we think the model said, this is the record that settles it.
    json_path = ROOT / "data" / "processed" / f"{path.stem}_extracted.json"
    json_path.write_text(json.dumps(result.raw_json, indent=2))

    print(f"\n{path.name} -> {result.model}")
    print(f"  deskewed: {prepped.deskewed}   upload: {result.wire_bytes / 1024:.0f} KB")
    print(f"  tokens: {result.prompt_tokens} in / {result.output_tokens} out "
          f"/ {result.thinking_tokens} thinking")

    print(f"\n  {r.merchant_name or '?'}  store {r.store_number or '?'}"
          f"   {r.purchase_date or '?'} {r.purchase_time or ''}")

    print(f"\n  {'#':>2}  {'description':<34} {'qty':>6} {'unit$':>7} {'total':>8}  flag")
    for item in r.line_items:
        mark = " VOID" if item.is_voided else ""
        qty = f"{item.quantity:g}" if item.quantity is not None else ""
        unit_price = f"{item.unit_price:.2f}" if item.unit_price is not None else ""
        total = f"{item.line_total:.2f}" if item.line_total is not None else "--"
        print(f"  {item.line_number:>2}  {item.raw_description[:34]:<34} "
              f"{qty:>6} {unit_price:>7} {total:>8}  {item.tax_flag or ''}{mark}")

    print(f"\n  printed subtotal: {r.subtotal}   tax: {r.tax}   total: {r.total}")

    # A preview of the accuracy signal Step 4 will formalize. Voided
    # lines are excluded because they were never charged.
    charged = [i.line_total for i in r.line_items
               if not i.is_voided and i.line_total is not None]
    if charged and r.subtotal is not None:
        summed = round(sum(charged), 2)
        verdict = "MATCH" if abs(summed - r.subtotal) < 0.01 else "MISMATCH"
        print(f"  items sum to {summed:.2f} vs printed {r.subtotal:.2f}  -> {verdict}")

    if r.unreadable_notes:
        print(f"\n  model flagged: {r.unreadable_notes}")


if __name__ == "__main__":
    main()
