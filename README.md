# FinSight

FinSight is a local-first Flask bank statement analyser. It combines document parsing with machine learning to classify transaction descriptions and detect unusual spending patterns.

## What changed in the ML version

### 1. Transaction classification

The original application used a hard-coded keyword map. The updated version keeps high-confidence rules for deterministic entities such as salary and fixed deposits, but adds:

- TF-IDF text vectorisation
- Logistic Regression classification
- confidence scores
- a local seed dataset
- user corrections that become new training examples

The classification pipeline is:

`merchant description → text normalisation → TF-IDF → Logistic Regression → category + confidence`

If scikit-learn is unavailable, FinSight falls back to a small local
token-similarity classifier so the Flask app can still import and basic
analysis can continue. The fallback is a resilience path, not the preferred
model.

### 2. Human-in-the-loop learning

Every transaction has a correction control. When a user selects the correct category, FinSight stores that labelled example in `models/feedback.json` and retrains the classifier locally.

No transaction data is sent to an external service.

Feedback is also stored as entity-level category votes in
`models/entity_profiles.json`, so correcting one UPI recipient can influence
future noisy descriptions of the same recipient.

### 3. Entity-aware classification

FinSight separates short-lived statement behaviour from persisted learning:

- current statement entity frequency is computed fresh for each analysis
- repeated unknown UPI recipients can be treated as peer transfers
- one-off or two-off small unknown UPI debits can be treated as likely local transport
- persisted entity profiles contribute only explicit user-learned category votes

This avoids the earlier failure mode where refreshing the results page inflated
entity frequencies and changed later classifications.

### 4. Anomaly detection

Isolation Forest is applied to debit transactions when there are at least 10 of them.

The model uses:

- transaction amount
- day of month
- day of week
- account balance

A transaction marked `ANOMALY` should be treated as a review signal, not proof of fraud.

### 5. PDF and CSV parsing improvements

The parser retains the original backward-scanning idea but adds:

- multiple date formats
- more tolerant amount parsing
- broader transaction prefixes
- stopping at the previous transaction date while scanning backwards
- duplicate protection
- safer merchant extraction
- tolerant CSV column names
- CSV credit/debit support

## Privacy

The ML inference and training are local Python operations. Uploaded statements are stored temporarily under `uploads/`, and the classifier is stored under `models/`.

The application does use CDN-hosted JavaScript on the results page and external Google Fonts in the original version. The financial data itself is not sent to those services by the application.

For a fully offline deployment, replace Chart.js with a local copy and remove external fonts.

## Requirements

Python 3.11+ is recommended.

Install:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install flask pdfplumber scikit-learn numpy
```

Optional for the existing test suite: no external test runner is required.

## Run

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

## Test

```powershell
python -m unittest discover -s tests
```

## Project structure

```text
FinSight/
├── app.py
├── parser.py
├── categoriser.py
├── templates/
│   ├── index.html
│   └── results.html
├── models/
│   ├── transaction_classifier.pkl
│   └── feedback.json
└── uploads/
```

`models/` and `uploads/` should be added to `.gitignore`.

## Important limitation

The initial classifier is trained on a small curated seed dataset because the project does not ship with a large labelled bank transaction corpus. Therefore, its confidence and generalisation should not be presented as production-grade.

The strongest way to improve it is to collect more labelled transaction descriptions from the user's own statements and corrections.


## License

This project is for educational and personal use.
