"""
Validate the extraction JSON already saved in data/processed/, without
calling the API again.

    python3 scripts/validate_extracted.py

Re-reading saved JSON is free; re-extracting costs money. This is why
scripts/extract_receipt.py writes the raw response to disk.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.extract import _parse_receipt
from skim.validate import CheckStatus, validate

PROCESSED = ROOT / "data" / "processed"

ICON = {CheckStatus.PASS: "PASS", CheckStatus.FAIL: "FAIL", CheckStatus.UNCHECKABLE: "n/a "}


def main() -> None:
    files = sorted(PROCESSED.glob("*_extracted.json"))
    if not files:
        print("No extraction JSON found. Run scripts/extract_receipt.py first.")
        raise SystemExit(1)

    for path in files:
        receipt = _parse_receipt(json.loads(path.read_text()))
        report = validate(receipt)

        print(f"\n{path.stem.replace('_extracted', '')}  --  "
              f"{receipt.merchant_name or '?'}")
        print(f"  {report.summary()}")
        for check in report.checks:
            print(f"    [{ICON[check.status]}] {check.name}: {check.detail}")


if __name__ == "__main__":
    main()
