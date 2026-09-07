"""
Run every photo in data/raw/ through capture + preprocess and write the
results to data/processed/, printing what happened to each one.

This is the eyeball test for Step 2. There is no ground truth for
"correctly cropped", so the check is: open the output and look at it.
Run from the project root:

    python3 scripts/preprocess_raw.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skim.capture import SUPPORTED_EXTENSIONS, CaptureError, capture
from skim.preprocess import preprocess

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)

    for path in photos:
        try:
            captured = capture(path)
        except CaptureError as e:
            print(f"{path.name}: REJECTED AT CAPTURE -- {e}")
            continue

        result = preprocess(captured)
        out_path = OUT_DIR / f"{path.stem}_processed.jpg"
        result.image.save(out_path, "JPEG", quality=95)

        before = f"{captured.width}x{captured.height}"
        after = f"{result.image.width}x{result.image.height}"
        if result.deskewed:
            print(f"{path.name}: deskewed + cropped  {before} -> {after}")
        else:
            print(f"{path.name}: PASSED THROUGH {before} -- {result.detection_note}")


if __name__ == "__main__":
    main()
