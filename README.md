# End-to-End Prompt Engineering Build

A modular internship capstone that analyzes unstructured product reviews, produces schema-validated JSON, and measures its behavior against labeled cases. The Streamlit app includes single-review analysis, CSV batches, evaluation metrics, failed-case diagnostics, and JSON/CSV downloads.

## What is included

- `review_analyzer.py` — deterministic local sentiment, feature-evidence, and explicit feature-request extraction.
- `json_validator.py` — strict Pydantic models and standards-compliant JSON parsing.
- `pipeline.py` — connected analyzer/formatter/validator, bounded retries, CSV input, and report helpers.
- `evaluation.py` — labeled test cases and metrics calculated from actual pipeline predictions.
- `app.py` — Streamlit interface.
- `tests/` — automated pytest coverage for analysis, schema rules, retries, CSV handling, exports, and evaluation.
- `sample_reviews.csv` — example batch input.

## Requirements

- Python 3.10 or newer.
- Dependencies in `requirements.txt`.
- No paid API key or internet connection is needed by the analysis pipeline.

The analyzer is a deterministic rules-based baseline rather than an LLM. Its sentiment vocabulary, spelling tolerance, and supported feature aliases are intentionally inspectable and can be expanded using evaluation failures. Confidence is a heuristic score, not a statistically calibrated probability. The baseline does not implement an LLM provider; `.env.example` is included to make clear that no secret is required.

## Install and launch on Windows PowerShell

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run this once in the same PowerShell session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Open the local URL printed by Streamlit. Paste a review and choose **Analyze Review**, upload `sample_reviews.csv` and choose **Analyze uploaded CSV**, or choose **Run Evaluation**. Analysis and evaluation pages provide JSON and CSV downloads.

## Run automated tests

```powershell
pytest -q
```

The tests assert behavior and metric bounds; they do not bake successful accuracy numbers into the app. The dashboard reports measured results and compares them with explicit targets for each run. Feature precision/recall and request precision/recall use the included labels; refusal rate is measured over test cases marked as valid product reviews.

## Evaluation labels and interpretation

The built-in suite covers positive, negative, neutral, mixed-feature, explicit-request, ambiguous, empty, malformed/control-character, unusual-character/slang, spelling-error, and instruction-like review text. A test passes when its expected overall sentiment, feature labels, and explicit request labels all match. Overall sentiment uses only `positive`, `negative`, or `neutral`; mixed feedback is represented through per-feature sentiment and may yield neutral overall sentiment.

The analyzer treats review text as untrusted content. It does not obey instruction-like text in a review, and its summary is a short evidence-based rationale rather than private chain-of-thought. Validation errors are logged without placing review contents in logs. Invalid output is retried a bounded number of times and reported as a pipeline error if it remains invalid.

## Known limitations

- The local baseline uses a small sentiment vocabulary and feature taxonomy; domain-specific phrasing can be missed.
- Spelling tolerance is approximate and can produce false positives. Inspect evaluation failures before broadening it.
- Requests are extracted only when an explicit request cue appears in the same sentence as a recognized feature.
- Suggested improvements are clearly separated from user-stated feature requests and are derived only from negative feature evidence.
- No model-based language understanding, persistence, user accounts, or external-service calls are included.
