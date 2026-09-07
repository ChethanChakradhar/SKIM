# Skim

*You skim the receipt, the app skims the prices.*

Photograph a receipt, extract every line item, and over months build a **personal price
index** — what *my* basket actually costs over time, not the national average.

Questions it must eventually answer:
- What did I pay for milk in March vs. August?
- Is Store A actually cheaper than Store B *for the things I buy*?
- What's my personal inflation rate?
- Which prices spiked unusually this month?
- What am I buying most often, and how often — what am I due to buy next?
- What will I spend next month?

Why this project: the data comes to me (a photo I already have), I'm user number one from
day one, and there's a genuinely hard modeling problem in the middle of it.

---

## Who you're working with

Chethan — data scientist (M.S. Data Science, University of New Haven, 2025), currently job
searching for data science / AI engineering / GenAI / analytics roles. **This is portfolio
work.** Its job is to make him stand out in applications and give him something concrete to
discuss in interviews. Prior relevant work: ARIMA forecasting on admissions data with a
Power BI dashboard; Gmail API integration for a personal job-application tracker.

### How to explain things — important

- Explain everything at a **beginner level**. Define every technical term the first time it
  appears. Motivation and concept *before* code.
- **Assume strong data science knowledge.** Do not explain statistics, ML fundamentals,
  pandas, or modeling concepts. He knows forecasting and dashboarding well.
- **Do not assume software engineering or computer vision knowledge.** Explain from the
  ground up: modules, unit tests, exceptions, dataclasses, git, OpenCV operations, how
  images are represented numerically, API mechanics, database design.
- Use concrete analogies for abstract mechanics.
- The bar: he should be able to explain **any pinpoint of this project** in an interview
  without notes. Depth of understanding beats speed of delivery.

### How he wants you to work

- Build **from scratch, step by step.** Explain each step as you go — what we're doing and
  why. Don't hand over a finished outline and move on to the next thing.
- Build a **real, working system he actually uses**, not an academic or tutorial exercise.
- **Push back honestly.** If an approach is wrong, over-engineered, or a dead end, say so
  before he spends a week on it.
- When he's making a technical choice, give the tradeoff **with numbers** where numbers exist.
- **Don't let him scope-creep.** A previous project (CivicWatch, a civic-alerts tool) died
  from being too ambitious. Guard against a repeat: ship a thin working slice before adding
  anything.

---

## Pipeline

```
Photo -> preprocess -> VLM extraction -> validation -> product normalization
      -> unit normalization -> storage -> analysis layer
```

1. **Capture** — phone photo or upload. *(DONE)*
2. **Preprocess** — deskew, crop to receipt boundary, boost contrast. Moves accuracy more
   than swapping models does. *(DONE)*
3. **Extraction** — vision-language model returns strict JSON. *(NEXT)*
4. **Validation** — do line items sum to subtotal? Does subtotal + tax = total? A free
   accuracy signal with no labeling required. Flag failures for review.
5. **Product normalization** — resolve `GV MLK 2% 1GAL` and `GREAT VAL MILK 2% GALLON` to
   one canonical product. **This is the hard part and the heart of the project.**
6. **Unit normalization** — price per ounce, not price per package, or nothing is
   comparable. Watch for weighted items (`0.87 LB @ $3.99`) vs. unit items.
7. **Storage** — SQLite to start.
8. **Analysis** — basket index, anomaly detection, forecasting.

## Decisions already made — do not relitigate without new information

- **Do not train a custom model.** Fine-tuning Donut/LayoutLM is the least interesting part
  of this and general VLMs now beat it.
- **Primary model: Gemini Flash.** Cheapest frontier vision model per image, accurate enough
  for receipts. A few hundred receipts costs pennies.
- **Baseline to beat: a purpose-built receipt API** (Veryfi, Taggun, Tabscanner, or Azure
  Document Intelligence) — for comparison only. If the whole pipeline is one API call,
  there's no project.
- **Stretch (month two, not week one): self-hosted open-weights model** (GLM-OCR,
  PaddleOCR-VL, dots.ocr, Qwen2.5-VL). Far cheaper per page and strong on raw OCR benchmarks.
- **Avoid: Tesseract, Google Cloud Vision, raw AWS Textract.** They return unstructured text
  and leave all the field-mapping work to us.
- **Database: SQLite** to start. Revisit only if concurrent writers or a hosted dashboard
  become real requirements.

---

## Environment

- macOS (Apple Silicon), VS Code with the integrated terminal.
- **Python 3.9.6** — the system Python from Xcode Command Line Tools. This is old.
  **Write 3.9-compatible code:** no `match` statements, no `X | Y` type unions at runtime
  (use `from __future__ import annotations` if you want that syntax in annotations), no
  `dict | dict` merge operator.
- Virtual environment at `.venv/` in the project root. Activate with
  `source .venv/bin/activate` — the prompt shows `(.venv)` when active.
- Installed: numpy, Pillow, opencv-python-headless, pillow-heif, python-dotenv,
  google-generativeai, pytest. Pinned in `requirements.txt`.
- Tests currently use stdlib `unittest` (pytest is installed and can also run them).
  Run with: `python3 -m unittest discover tests -v`
- Secrets go in `.env` (gitignored). `.env.example` shows the shape. `GEMINI_API_KEY` is
  **not yet obtained** — that's a task for Step 3.

## Layout

```
skim/            one module per pipeline stage
  capture.py     Step 1 (done)
  preprocess.py  Step 2 (done)
data/raw/        original receipt photos (gitignored - personal data)
data/processed/  preprocessed images (gitignored)
tests/
scripts/
  preprocess_raw.py   run capture+preprocess over data/raw, write to data/processed
```

---

## Status

**Done — Step 1: Capture** (`skim/capture.py`, 7 passing tests)

Validates and normalizes an incoming photo, or fails loudly with a specific reason.
Checks in order: file exists -> supported extension -> file-size bounds -> decodable by
Pillow -> EXIF orientation applied -> minimum resolution (600px short side). Returns a
`CaptureResult` dataclass. Custom exception hierarchy under `CaptureError` so callers can
react differently per failure type.

Two design principles established here that should carry forward:
- **Fail fast and cheaply.** Everything downstream costs money (paid VLM calls) or human
  attention (manual review). Reject bad input at the earliest, cheapest point.
- **Normalize weirdness at the boundary.** EXIF rotation and color mode are handled once,
  at entry, so no later stage has to think about them.

**Done — Step 2: Preprocess** (`skim/preprocess.py`, 7 passing tests, 14 suite-wide)

Finds the receipt's four corners, warps them onto a rectangle, then boosts local contrast.
The load-bearing idea: **crop and deskew are the same operation.** A receipt shot at an angle
arrives as a trapezoid, not a rotated rectangle, so a perspective transform (homography) is
what fixes it — and that transform crops and straightens in one pass.

Detection is Canny edges -> contours -> `approxPolyDP`, with three gates a shape must pass:
≥10% of the frame, exactly 4 convex points, and ≥0.80 fill of its own `minAreaRect`.
Contrast is CLAHE on the L channel of LAB (color left untouched, so pen marks survive).

Measured on the three real photos: **2 of 3 detected.** Both table shots crop cleanly;
the hand-held one fails because fingers break the paper's outline into fragments. That one
passes through **uncropped, flagged with a reason** — the third design principle, alongside
Step 1's two:

- **A wrong crop is worse than no crop.** Slicing off the TOTAL line corrupts data silently;
  an uncropped photo just costs the model a little more work. Never guess a boundary.

Rejected during the build, with data: brightness/Otsu segmentation performed *identically*
to Canny on all three photos, so it was not worth a second code path. Canny was kept because
Otsu breaks on light backgrounds (white counter) where an edge still exists. If detection
later fails on a light surface, Otsu-as-second-attempt is the first thing to try.

Thresholds are calibrated on a sample of three. `find_receipt_corners` returns its failure
reason so the tally, not intuition, drives the next tuning pass.

**Next — Step 3: Extraction.** Gemini Flash returns strict JSON. Needs `GEMINI_API_KEY` in
`.env` — not yet obtained, that's the first task.

Keep showing before/after images at each stage so Chethan can see each operation doing its
job rather than taking it on trust. Step 2's stage-by-stage review page:
https://claude.ai/code/artifact/fac10e7e-ca2a-4598-b6ac-3207f52a5ae6

Still true and still worth honoring: **do not develop image operations against synthetic
photos.** Pure geometry helpers can have synthetic tests (see `tests/test_preprocess.py`,
which splits on exactly this line); detection cannot. A useful next photo would be a receipt
on a *white* counter — the one background we have no sample of.

---

*Keep this file current as the project progresses — it's what makes a new session
productive immediately.*
