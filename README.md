# Skim

You skim the receipt, the app skims the prices.

Live app: https://skim-production-9d69.up.railway.app/

I wanted to know what my own groceries actually cost over time, not the national CPI. So Skim takes a photo of a receipt, pulls out every line item, and stores it so I can ask things like: what did I pay for milk in March vs. August? Is Store A really cheaper than Store B for the stuff I buy?

The data comes to me for free (I already have the receipts), I'm the first user, and there's a genuinely hard problem in the middle of it: figuring out that `GV MLK 2% 1GAL` and `GREAT VAL MILK 2% GALLON` are the same product.

## How it works

```
photo -> preprocess -> extract -> validate -> normalize product -> normalize units -> store -> insights
```

**Preprocess** (`skim/preprocess.py`). A receipt shot at an angle shows up as a trapezoid, not a rotated rectangle, so I find the four corners (Canny edges, contours, `approxPolyDP`) and warp them flat with a perspective transform. That one step crops and deskews at the same time. Then CLAHE for contrast. If it can't find a clean outline (fingers in the shot, for example) it passes the photo through uncropped and says why. A wrong crop that cuts off the TOTAL line is worse than no crop.

**Extract** (`skim/extract.py`). Gemini Flash with a response schema, so I get JSON back, not prose. The rule I kept coming back to: the model transcribes, it doesn't interpret. Descriptions stay verbatim, it never calculates anything that isn't printed, and unreadable means null.

**Validate** (`skim/validate.py`). A receipt states the same facts more than once, so it can check itself. Six checks: items sum to subtotal, subtotal + tax = total, per-line math, tax rate, payment/change, and printed item count. Money is compared in integer cents because `0.1 + 0.2 != 0.3`. Any failure sends the receipt to review.

**Normalize products** (`skim/normalize.py`, `skim/match.py`). My first idea was to embed the raw strings and match by similarity. I measured it on all 465 pairs of real products and it doesn't work: the true match and the closest false match were 0.011 apart (two different chickens scored almost as high as two spellings of guava). So now an LLM parses each line into fields (brand, product, variant, size), embeddings only retrieve candidates, and deterministic rules on the fields make the actual decision. Anything still unsure goes to me, never auto-merged.

**Normalize units** (`skim/units.py`). Price per ounce, not per package, or nothing is comparable. Weighed items (`0.52 @ 2.49`) and packaged items (`ONION 10LB 1 @ 6.99`) need different math, and the receipt doesn't tell you which is which.

**Store** (`skim/storage.py`). SQLite. Re-running a receipt replaces it instead of duplicating it, since I reprocess the same photos every time I change a prompt.

**Insights** (`skim/analysis.py`, `web/templates/insights.html`). Total and average spend, spend by category, every store visit, how each product's price has moved, and which store is cheaper for a given item. I compare stores per item, not per basket, because comparing whole baskets mostly measures what you happened to buy that day.

The rule here is that nothing claims more than the data supports. A price change needs readings on two different days, not just two readings. My first version showed a row of 0.0% changes, and it turned out I'd uploaded the same receipt twice: two points, zero days apart. With only one reading, the change shows as blank, never 0%, because 0% would mean "the price held steady," which one reading can't tell you. When something can't be computed yet, the page says what would unlock it (for example, "buy onions again on a different day") instead of showing an empty chart.

## Results so far

Small sample, so take these as a starting point:

- 3 real receipts, 37 line items, checked by eye against the paper: zero transcription errors
- about 1.4 cents per receipt with Gemini Flash
- 33 raw product strings resolved to 30 products, 2 sent to review
- 145+ unit tests (the model is stubbed, so tests never call the API)

## What's not done yet

No forecasting, personal inflation rate, or anomaly detection yet. Each of those needs weeks of receipts, and a forecast fitted to a handful of them would be a number I couldn't defend. The web app is live on Railway so I actually use it day to day, which is what builds up that data.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your GEMINI_API_KEY
python3 -m unittest discover tests -v
python3 scripts/extract_receipt.py path/to/receipt.jpg
```

Or with Docker (this is also how it runs on Railway):

```bash
docker build -t skim .
docker run -p 8000:8000 --env-file .env skim
```
