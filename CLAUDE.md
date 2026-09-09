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
- **Ask before changing dependencies or his environment.** Adding, removing or re-pinning a
  package, or installing into `.venv`, is his call — present the tradeoff and wait. Telling
  him while doing it is not asking. (He called this out when the Gemini SDK was swapped
  during Step 3.) Ordinary code he asked for doesn't need this.

---

## Pipeline

```
Photo -> preprocess -> VLM extraction -> validation -> product normalization
      -> unit normalization -> storage -> analysis layer
```

1. **Capture** — phone photo or upload. *(DONE)*
2. **Preprocess** — deskew, crop to receipt boundary, boost contrast. Moves accuracy more
   than swapping models does. *(DONE)*
3. **Extraction** — vision-language model returns strict JSON. *(DONE)*
4. **Validation** — do line items sum to subtotal? Does subtotal + tax = total? A free
   accuracy signal with no labeling required. Flag failures for review. *(NEXT)*
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
  for receipts. **Pinned to `gemini-3.6-flash`** in
  `skim/extract.py`. Two things learned the hard way: `gemini-2.5-flash` still appears in
  `models.list()` but returns 404 "no longer available to new users", and the newest models
  (3.7/3.8-flash) return 503 under load. Always pin an explicit version, never the
  `gemini-flash-latest` alias — an alias moves under you, so a benchmark run today wouldn't
  reproduce next month and a silent model change would look like a bug in our own pipeline.
- **SDK: `google-genai`**, not the deprecated `google-generativeai` it replaced.
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
  google-genai, pytest. Pinned in `requirements.txt`.
- Tests currently use stdlib `unittest` (pytest is installed and can also run them).
  Run with: `python3 -m unittest discover tests -v`
- Secrets go in `.env` (gitignored). `.env.example` shows the shape. `GEMINI_API_KEY` is
  **set and working**. Note `.env` must be created by copying `.env.example` — a fresh
  clone has only the example, which is what makes "there's nowhere to put the key" the
  first confusing moment for a new setup.
- The Google SDK prints a `FutureWarning` on every import because **Python 3.9 is past end
  of life**. Harmless today; upgrading Python is worth doing before this project grows,
  but it is not urgent and should not derail a pipeline step.
- Extraction tests never call the API. A test that costs money and needs a network is one
  you stop running, and a test whose result depends on what the model felt like saying
  can't tell you whether *your code* broke. The model is stubbed; real accuracy is measured
  by running real photos and checking the arithmetic.

## Layout

```
skim/            one module per pipeline stage
  capture.py     Step 1 (done)
  preprocess.py  Step 2 (done)
  extract.py     Step 3 (done)
data/raw/        original receipt photos (gitignored - personal data)
data/processed/  preprocessed images + raw extraction JSON (gitignored)
tests/
scripts/
  preprocess_raw.py   run capture+preprocess over data/raw, write to data/processed
  extract_receipt.py  run one photo end to end, print items, save raw JSON
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

**Done — Step 3: Extraction** (`skim/extract.py`, 16 passing tests, 30 suite-wide)

Sends the preprocessed photo to Gemini with `response_schema` structured output, so the API
returns JSON conforming to `RECEIPT_SCHEMA` rather than prose containing JSON. No markdown
fences to strip.

**Result on all three real receipts: extracted correctly, arithmetic matched on all three.**
Including the hard cases — the Walmart `** VOIDED ENTRY **` was captured with `is_voided`
true and a null total and correctly excluded from the sum, and the faded 18-item India
Market receipt got every weighted item right (`DESI OKRA 0.52 @ 2.49 = 1.29`). The
uncropped hand-held photo extracted just as well as the deskewed ones, which is evidence
that Step 2's crop failure cost nothing.

The module is built around one rule: **the model transcribes, it does not interpret.**

- `raw_description` is verbatim. If the model rewrote `GV MLK 2%` as `Great Value Milk`, it
  would destroy the input Step 5 exists to work on, and do the hard part of this project
  invisibly where it can't be audited.
- **The model never computes what isn't printed.** If it derived the subtotal by summing the
  items it just read, Step 4's check would compare the model against itself and pass even
  when every price is wrong. That would destroy the only free accuracy signal we have.
- Unreadable means null, never a guess.

Two findings worth carrying forward:

- **The sum check is permutation-invariant.** Dollar Tree prints the price column offset
  half a line above its description, so an off-by-one item→price mapping is a live risk. If
  the model shifted every price by one row, the items would still sum to the subtotal and
  Step 4 would still say MATCH. The arithmetic validates the *multiset* of prices, not the
  mapping — and for a personal price index, the mapping is the entire point. Step 4 should
  not be trusted as a complete accuracy measure.
- **Weighted items come back with `unit` null** when the receipt prints `0.52 @ 2.49` with
  no unit label, which is correct behavior (never guess). Step 6 must infer pounds from
  context — a US grocery receipt with fractional produce quantities — and that inference
  belongs in the normalization layer where it's auditable, not smuggled into extraction.

### Measured cost (looked up Sept 2026, verify before quoting)

`gemini-3.6-flash` is $0.75/1M input and $3.75/1M output, promotional through Dec 31 2026,
**doubling to $1.50/$7.50 on Jan 1 2027**. Thinking tokens bill as output.

| Receipt | in | out | thinking | cost | thinking share |
|---|---|---|---|---|---|
| Dollar Tree | 1,695 | 696 | 1,571 | $0.0098 | 60% |
| Walmart | 1,685 | 852 | 2,744 | $0.0147 | 70% |
| India Market | 1,673 | 2,144 | 2,411 | $0.0183 | 49% |

**~$0.014 per receipt**, so 100/month ≈ $1.43, doubling in 2027. The earlier "a few hundred
receipts costs pennies" claim was wrong — 500 receipts is about $7.

**Thinking tokens are half to two-thirds of the bill.** That is the single biggest cost
lever and it is untested: nobody has checked whether receipt transcription actually needs
reasoning tokens. Test `thinking_budget=0` against the arithmetic check once Step 4 exists.

Cheaper models worth A/B-ing at that same point (same token counts, list price):
`gemini-3.1-flash-lite` ≈ $0.0056/receipt (2.5x cheaper), `gemini-2.5-flash-lite` ≈
$0.0016 (9x cheaper). Do not switch on price alone — a lite model that misreads one digit
of a price silently corrupts the index, which costs far more than a cent saved.

### Privacy: use the paid tier, not the free tier

Google's API terms say that on the **unpaid tier, human reviewers may read and annotate API
input and output**, and that content is used for product improvement. The terms state
plainly: *"Do not submit sensitive, confidential, or personal information to the Unpaid
Services."* On the **paid tier** Google does not use prompts or responses (including
uploaded images) to improve its products; content is retained ~30 days only for abuse
detection.

Receipts are personal data — where Chethan shops, when, how often, what he eats, what he
pays with. Enabling billing is the correct call and costs ~$1.43/month at 100 receipts.
This is also a good thing to be able to explain in an interview: knowing *why* the tier
matters for personal data is a data-governance signal, not just a billing detail.

**Status: the key is on the FREE tier** (as of Sept 2026 — no billing account linked, which
is why nothing has been charged). The three test receipts were sent under free-tier terms.
Chethan should link billing before running a real backlog through.

**Do not misread `x-gemini-service-tier` as the billing tier.** That response header reports
the *processing* class — standard / flex / priority / batch — and returns "standard" by
default regardless of whether the account pays. Billing tier is account-level and is only
visible in the console at https://aistudio.google.com/apikey. This mistake was made once in
this project already; the header looks authoritative and isn't.

**Done — Step 4: Validation** (`skim/validate.py`, 34 passing tests, 64 suite-wide)

A receipt is a *redundant document* — it states the same facts more than once, so it
validates itself. That is an accuracy signal costing nothing and needing no labels, which is
the only kind a one-person project will ever have.

Six checks:

| Check | What only it can catch |
|---|---|
| `items_sum_to_subtotal` | a misread or extra line amount |
| `subtotal_plus_tax_is_total` | a misread tax or total (independent of the above) |
| `line_arithmetic` | localizes to a *specific line* — qty × unit_price vs printed total |
| `tax_rate` | **a misread per-line tax flag.** Nothing else in the pipeline can see this |
| `payment_reconciles` | a misread total, via a completely independent path |
| `item_count` | **a dropped line.** All arithmetic is computed from items we have, so a drop plus a compensating subtotal misread is self-consistent |

Design rules established here:

- **Money is compared in integer cents, never floats.** The module's whole job is comparing
  money for equality and `0.1 + 0.2 != 0.3` in binary floating point.
- **Tolerance is asymmetric on purpose.** Zero between two *printed* numbers; one cent where
  we recompute a product the register itself rounded (`0.52 × 2.49 = 1.2948`, printed 1.29).
  Backwards would pass broken receipts and fail good ones.
- **`UNCHECKABLE` is a distinct status from `FAIL`,** and `evidence_count` counts only checks
  that ran. A receipt printing less proves less about itself; scoring absence as a pass
  flatters the numbers, scoring it as failure queues correct receipts for review.
- **`validate()` is pure and never repairs.** The evidence of what the model actually
  returned is what will tell us whether a prompt or model change helped.
- **A check is only as good as the independence of its two sides.** This is why the prompt
  forbids the model from deriving the tax rate or counting its own line items — a check
  whose sides share a source is decorative.
- **One failure sends a receipt to review.** A false alarm costs seconds; a wrong price
  accepted corrupts the index permanently and invisibly. Same asymmetry as Step 2's
  "a wrong crop is worse than no crop".

**Step 4 forced a Step 3 schema change**, which is the lesson worth keeping: the validation
layer decides what extraction must capture. `tax_rate_percent`, `amount_paid`,
`change_given`, `rounding_adjustment` and `item_count_printed` were added to
`RECEIPT_SCHEMA` only because checks needed them. Design the schema backwards from the
checks, not forwards from what receipts happen to print.

Real-world quirks encoded in the checks (each has a test named after it):
- Connecticut does not tax unprepared groceries, so the printed rate applies to the
  **taxable subset**, not the subtotal — Walmart charges 6.35% on $31.03, not $41.80.
- Registers print change with inconsistent signs (Dollar Tree prints `$-2.77`). Extraction
  copies the sign; `check_payment_reconciles` takes the magnitude, in the open.
- Cash rounding is printed on its own line (`ROUNDING 0.02`). Ignore it and Walmart looks
  two cents wrong.

**Still true — validation checks arithmetic consistency, not transcription fidelity.** If
the model read `MILK` as `MILT`, everything passes. The permutation blind spot stands: a
price column read one row out of step against the descriptions sums identically and each
line still agrees internally. A 95% pass rate means 95% *internally consistent*.

### Fidelity was measured once, by hand (Sept 2026)

All three receipts were read by eye and diffed against the extraction. **Zero transcription
errors** — 37 line items across three receipts, every description, quantity, unit price,
amount and tax flag correct, including the Dollar Tree half-line price offset and every
faded digit on the India Market receipt. Only difference: trivial separator normalization
(`CUT : 310 GM` → `CUT: 310 GM`). One nice moment — line 14 `GUAVA` has a genuinely
illegible unit price on the paper, and the arithmetic check confirmed the model's `1.49`
via `1.83 × 1.49 = 2.73`. The check saw what the eye could not.

### KNOWN RISK — representational drift between runs (not yet addressed)

The danger is not that the model misreads. It is that it **represents the same receipt
differently on different runs**. Observed across two runs of identical code on identical
images:

- India Market: `ONION 10LB YELLOW` became `1 ONION 10LB YELLOW` (the receipt's own line
  sequence number leaked into the product name). Fixed in the prompt, and verified fixed.
- Walmart: 10 line items became 11, with the literal `** VOIDED ENTRY **` marker promoted
  to its own "product". Not addressed.

**Both runs passed every check, both times.** Validation is blind to this by construction,
and `temperature=0` reduces drift without eliminating it.

For a project premised on tracking one product's price across months, drift is more
dangerous than a misread digit: a wrong price fails an arithmetic check and gets caught,
whereas `ONION 10LB YELLOW` silently becoming a second distinct product never fails
anything — it just halves the onion price history.

The fix when it becomes worth it is **self-consistency sampling**: extract twice, diff, and
flag disagreements. Costs one extra call per receipt (~1.4¢) and is the only technique that
can see this failure mode. Deliberately deferred — with three receipts there is no way to
measure how often drift actually matters. Revisit at ~30 receipts and measure the drift
rate rather than guessing at it.

**Next — Step 5: Product normalization.** Resolve `GV MLK 2% 1GAL` and
`GREAT VAL MILK 2% GALLON` to one canonical product. The hard part and the heart of the
project.

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
