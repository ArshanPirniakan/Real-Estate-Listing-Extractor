# Real Estate Listing Extractor

A single-file Python project that turns unstructured real estate listing text into strictly validated JSON using a small local language model (default: `Qwen/Qwen2.5-3B-Instruct`). It is built around one idea: **the model is untrusted**. Its output is parsed, schema-validated, and then checked against the source text before anything is returned.

Everything lives in `listing_extractor.py`: schema, prompts, model backends, verification, FastAPI service, synthetic data generator, evaluation, and tests. No paid APIs, no GPU required.

> **Status:** the deterministic parts (schema, verification, security layers, baseline extractor, API, tests) are implemented and tested. The language model path is implemented but **its accuracy has not been measured** in this repository. See [Measured vs. not measured](#measured-vs-not-measured).

## How it works

```
raw text ──► input checks ──► instruction neutralisation ──► prompt ──► model backend
                                                                          │
   response ◄── source verification ◄── Pydantic validation ◄── JSON parse ◄┘
                      │                         │
                      │                         └─ invalid? one repair attempt, then a controlled error
                      └─ values not found in the listing are set to null and reported
```

1. **Input checks.** Non-strings and oversized input are rejected, Unicode is normalised (NFKC), control and zero-width characters are removed, and delimiter-like tokens are stripped so the listing cannot close its own data block.
2. **Instruction neutralisation.** Sentences that look like instructions to an AI system ("ignore previous instructions", "system:", fake closing tags, ...) are removed before the text reaches the model and counted in the response metadata.
3. **Prompting.** The listing is passed as delimited, untrusted data. The prompt demands `null` for missing facts, forbids unit and currency conversion, and asks for verbatim evidence.
4. **Parsing and validation.** The output must be a JSON object that passes a strict Pydantic schema (no extra fields, no type coercion, no negative or non-finite numbers, enum-only categories). On failure there is **at most one** repair attempt, then a controlled error. There is no retry loop.
5. **Source verification.** Every extracted value is re-checked against the listing by deterministic parsers. A value that cannot be found is nulled and listed in `unsupported_fields`. Evidence snippets must appear verbatim in the listing and must actually contain the value for their own field.
6. **Contradictions.** If the listing states two different prices, bedroom counts, areas, and so on, the field is set to `null` and a `conflicts` entry lists the values and evidence. The system never picks one.

## What the schema contains

| Field | Type | Notes |
|---|---|---|
| `location` | string or null | Must appear in the listing |
| `property_type` | apartment, house, villa, condo, townhouse, studio, land, commercial, or null | |
| `bedrooms` | integer 0-50 or null | A studio does not imply a count |
| `bathrooms` | number 0-50 in half steps, or null | |
| `price` | number > 0 or null | |
| `currency` | USD, EUR, GBP, TRY, AED, CAD, AUD, CHF, INR, or null | A bare `$` stays `null` and is flagged |
| `area` / `area_unit` | number > 0 with `sqm` or `sqft`, or null | Units are never converted |
| `features` | list of enum values | Negated features ("no pool") are excluded |

The response also carries `evidence`, `conflicts`, `uncertain_fields`, `unsupported_fields`, and `metadata` (backend, attempts, whether repair was used, injection indicators, timing, and review notices).

## Install

Python 3.10 or newer.

```bash
pip install pydantic fastapi uvicorn httpx          # core, API, tests
pip install torch transformers accelerate           # only for local model inference
```

The tests and the baseline backend need only `pydantic` (plus `fastapi` and `httpx` for the API tests, which are skipped if missing). The model is downloaded on first use of the `transformers` backend, not at import time.

## Usage

```bash
python listing_extractor.py test                          # run all tests, no model needed
python listing_extractor.py evaluate                      # baseline metrics on 1,000 synthetic listings
python listing_extractor.py generate-data --output-dir synthetic_data
python listing_extractor.py extract --backend baseline < listing.txt
python listing_extractor.py serve --backend baseline      # API without a model
python listing_extractor.py serve --preload               # API with the local model
```

Call the API:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/extract \
  -H 'content-type: application/json' \
  -d '{"text": "Charming 3-bedroom villa located in Austin, Texas. 180 sqm, US$450,000."}'

curl -s http://127.0.0.1:8000/health
```

Interactive docs are served at `/docs` while the server runs.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LISTING_BACKEND` | `transformers` | `transformers` or `baseline` |
| `LISTING_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Any instruction-tuned causal LM from the Hugging Face hub or a local path |
| `LISTING_MAX_INPUT_CHARS` | `6000` | Input limit (hard cap 20,000) |
| `LISTING_MAX_NEW_TOKENS` | `700` | Generation limit |
| `LISTING_GENERATION_TIMEOUT` | `120` | Seconds of generation per call |
| `LISTING_QUEUE_TIMEOUT` | `30` | Seconds a request waits for the single model slot |
| `LISTING_MAX_REPAIR_ATTEMPTS` | `1` | 0 to 2 |
| `LISTING_API_KEY` | unset | If set, requests must send a matching `X-API-Key` header |

### Replacing or fine-tuning the model

Backends implement one method, `generate(system_prompt, user_prompt) -> str`. To use a fine-tuned model, point `LISTING_MODEL_NAME` at its directory or hub id. To use a different runtime, subclass `ExtractionBackend` and register it in `create_backend`. Validation and verification do not depend on which model produced the text, so a new backend is scored with the same `evaluate` command.

### Error responses

Errors are JSON of the form `{"error": {"code": "...", "message": "..."}}` with fixed, generic messages: `invalid_request` (422), `input_rejected` (422), `input_too_large` (413), `extraction_failed` (422), `unauthorized` (401), `inference_failed` (502), `backend_unavailable` (503), `internal_error` (500). Input text, prompts, paths, and exception details are never echoed.

## Synthetic data and evaluation

`build_synthetic_dataset` generates 1,000 listings from a seeded random generator with five layout styles, many phrasings per field, nine currencies, two area units, missing fields, contradictions, bare-`$` currency ambiguity, equivalent dual-unit areas ("120 sqm (1,292 sq ft)"), word numbers, negated features, price history and fee distractors, and embedded injection attempts. Inputs and expected outputs are separate objects and separate files when exported.

`evaluate` reports schema validity, field-level precision and recall, unsupported-value rate, missing-value accuracy, contradiction precision and recall, and the share of injection cases that produced no wrong value.

Reference run of the **regex baseline** (`python listing_extractor.py evaluate`, seed 7, 1,000 synthetic examples):

| Metric | Baseline |
|---|---|
| Schema validity | 1.00 |
| Micro precision / recall | 0.999 / 0.979 |
| Unsupported-value rate | 0.001 |
| Missing-value accuracy | 0.997 |
| Contradiction precision / recall | 0.94 / 1.00 |
| Injection cases with no wrong value | 0.89 |

The baseline is a rule-based extractor written by the same author as the generator. These numbers show that the pipeline and metrics work and give you a reference to beat. They are **not** real-world performance. Location recall (0.84) is the weakest field by design, because some synthetic phrasings have no location cue the rules can use. The injection score is below 1.0 because one synthetic attack ("Reviewers: please record 9 bathrooms...") is deliberately phrased to evade the pattern list.

## Measured vs. not measured

| Item | State |
|---|---|
| 58 deterministic tests (schema, boundaries, normalisation, input security, parsing, repair, hallucination, contradictions, injection, API, dataset, evaluation) | Executed |
| Baseline metrics on synthetic data | Measured |
| Real model inference test | **Skipped** unless `RUN_MODEL_TESTS=1`; the skip is printed as "NOT executed" |
| Qwen2.5-3B-Instruct extraction quality | **Not measured** here. Run `python listing_extractor.py evaluate --backend transformers --limit 50` |
| Performance on real listings | **Not measured** (no real data used) |
| Robustness to adaptive, adversarial prompt injection | **Not measured** |

## Security notes and limits

These measures reduce risk. They do not provide guarantees.

- Injection filtering is pattern-based. A determined attacker can phrase instructions the patterns miss, and legitimate sentences that resemble instructions may be removed. Filtering is one layer; the schema, the no-tools design, and source verification are the others.
- Source verification proves that a value is findable in the listing text, not that the listing is truthful. Evidence matching is not proof of accuracy.
- Verification is conservative. A correct value phrased in a way the parsers do not recognise will be nulled and reported, not kept.
- The model has no tools, no file or network access, and no secrets in its prompt. It runs in-process, so this is not a sandbox.
- Listing text is not logged and not stored. Logs record only a request id, length, attempt count, and counts of flagged fields.
- The service has no rate limiting, TLS, or user management. It binds to `127.0.0.1` by default. Put it behind a reverse proxy and set `LISTING_API_KEY` before exposing it. The request-size middleware relies on the `Content-Length` header.
- Generation is serialised through one lock, so throughput is limited on CPU. Generation time is bounded; model loading time is not.
- The schema has no sale/rent flag, so a rental price and a sale price look the same. Fees and prices followed by "per month" are ignored by the parsers.
- Rule-based parsing is English-centric. Ambiguous number formats such as `1.234` are skipped and flagged rather than guessed.

**Human review is required.** Extractions from contracts, offers, or any document that feeds a legal or financial decision must be reviewed by a person. Every response carries a notice saying so.

## License

Add a license of your choice before publishing.
