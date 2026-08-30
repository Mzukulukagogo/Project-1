from collections import Counter, defaultdict
from datetime import datetime
import json
import os
import pickle
import re

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except Exception:
    np = None
    IsolationForest = None
    TfidfVectorizer = None
    LogisticRegression = None
    Pipeline = None
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# FinSight intelligent transaction categorisation
#
# Classification now combines:
#   1. Known merchant/category rules
#   2. Entity normalisation
#   3. Entity frequency and behavioural profiling
#   4. TF-IDF + Logistic Regression
#   5. User feedback
#
# The important distinction is between a recurring merchant and a recurring
# person. A known merchant such as SWIGGY remains Food even if it appears
# many times. An unfamiliar UPI recipient appearing repeatedly can gradually
# be recognised as a Peer Transfer.
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Food & Dining": [
        "ZOMATO", "DOMINOS", "SWIGGY", "BLINKIT", "SMARTPED", "RELIANCE",
        "FRESHCHO", "LRFOOD", "TRAVELFO", "COFFEE", "IBACO", "KFC", "SWAMI",
        "MCDONALDS", "PIZZA", "RESTAURANT", "CAFE", "DINE", "FOOD",
    ],
    "Travel & Transport": [
        "UBER", "RAPIDO", "APSRTC", "MAKONI", "KERALA", "OLA", "RED BUS",
        "IRCTC", "MAKEMYTRIP", "INDIGO", "AIRBNB", "STAYFL", "HMS",
        "TRAVEL", "INDIAN RAILWAYS", "METRO", "BUS", "TAXI", "CAB",
        "RICKSHAW", "AUTO", "AUTORICKSHAW",
    ],
    "Entertainment": [
        "INOX", "PVR", "WONDERLA", "BOOKMYSHOW", "LIVEYOU", "CONNAU",
        "STEAM", "PLAYSTATION", "SPOTIFY", "MOVIE",
    ],
    "Shopping": [
        "AMAZON", "FLIPKART", "MEESHO", "VISHAL", "ZUDIO", "ROTARY",
        "MYNTRA", "AJIO", "SHOP", "MART", "STORE",
    ],
    "Subscriptions": [
        "NETFLIX", "SPOTIFY", "APPLE.COM/BILL", "OPENAI", "ADOBE",
        "MICROSOFT", "GOOGLE ONE", "SUBSCRIPTION",
    ],
    "Education": [
        "IIT", "ADITYA", "PAX", "COLLEGE", "UNIVERSITY", "COURSE",
        "UDEMY", "COURSERA", "EDUCATION",
    ],
    "Healthcare": [
        "APOLLO", "PHARMACY", "MEDICAL", "HOSPITAL", "CLINIC", "HEALTH",
    ],
    "Debt & Loans": [
        "EMI", "LOAN", "CREDIT CARD", "CC PAYMENT", "CARD PAYMENT",
        "PAYLATER", "BAJAJ FINANCE", "HDFC BANK CARD", "ICICI CARD",
        "SBI CARD", "KOTAK CARD", "AXIS CARD", "LENDING", "FINANCE",
    ],
    "Peer Transfers": [
        "NEFT", "IMPS", "RTGS", "PAYTM", "PHONEPE", "GOOGLEPAY", "GOOGLE PAY",
        "AMAZONPAY", "AMAZON PAY", "BHIM", "CASH WITHDRAWAL", "ATM",
        "TRANSFER", "FRIEND", "FAMILY",
    ],
    "Income": [
        "SALARY", "INCOME", "CASH DEPOSIT", "FD REDEMPTION", "BANK INTEREST",
        "PRAISEND", "DLOCAL", "PRAISE",
    ],
    "Savings": ["FIXED DEPOSIT"],
    "Utilities": [
        "ELECTRICITY", "WATER BILL", "GAS", "AIRTEL", "JIO", "VI ",
        "BROADBAND", "INTERNET", "UTILITY",
    ],
}

CATEGORY_OPTIONS = list(CATEGORIES) + ["Other"]

# Entities that are strongly identifiable as businesses/services.
# These are protected from the "3+ appearances = Peer Transfer" heuristic.
KNOWN_MERCHANTS = set()
for _category, _keywords in CATEGORIES.items():
    if _category not in {"Peer Transfers", "Income", "Savings"}:
        KNOWN_MERCHANTS.update(_keywords)

# Terms that strongly suggest a person-to-person transfer.
PERSON_HINTS = {
    "FRIEND", "FAMILY", "BROTHER", "SISTER", "MOM", "MUM", "DAD",
    "PARENT", "ROOMMATE", "HOSTEL", "COLLEGE MATE",
}

# Generic terms which should not be treated as an entity name by themselves.
GENERIC_ENTITY_TERMS = {
    "UPI", "DR", "CR", "PAYMENT", "TRANSFER", "TO", "FROM", "SELF",
    "ACCOUNT", "BANK", "UNKNOWN", "CASH", "DEBIT", "CREDIT",
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "transaction_classifier.pkl")
FEEDBACK_PATH = os.path.join(MODEL_DIR, "feedback.json")
ENTITY_PATH = os.path.join(MODEL_DIR, "entity_profiles.json")

CLASSIFICATION_SOURCE_LABELS = {
    "rule": "Rule classifier",
    "ml": "ML classifier",
    "ml-fallback": "Fallback classifier",
    "entity-frequency": "Entity frequency",
    "entity-profile": "Entity profile",
    "user-feedback": "User feedback",
}


SEED_DATA = [
    ("swiggy food order", "Food & Dining"),
    ("zomato restaurant", "Food & Dining"),
    ("dominos pizza", "Food & Dining"),
    ("blinkit groceries", "Food & Dining"),
    ("kfc restaurant", "Food & Dining"),
    ("coffee shop", "Food & Dining"),
    ("uber trip", "Travel & Transport"),
    ("ola cab", "Travel & Transport"),
    ("rapido ride", "Travel & Transport"),
    ("auto rickshaw ride", "Travel & Transport"),
    ("rickshaw driver", "Travel & Transport"),
    ("irctc train ticket", "Travel & Transport"),
    ("indigo flight", "Travel & Transport"),
    ("makemytrip booking", "Travel & Transport"),
    ("inox cinema", "Entertainment"),
    ("pvr movies", "Entertainment"),
    ("bookmyshow ticket", "Entertainment"),
    ("wonderla amusement park", "Entertainment"),
    ("amazon purchase", "Shopping"),
    ("flipkart order", "Shopping"),
    ("myntra clothing", "Shopping"),
    ("zudio shopping", "Shopping"),
    ("netflix subscription", "Subscriptions"),
    ("spotify premium", "Subscriptions"),
    ("openai subscription", "Subscriptions"),
    ("apple com bill", "Subscriptions"),
    ("college fee", "Education"),
    ("aditya college", "Education"),
    ("udemy course", "Education"),
    ("apollo pharmacy", "Healthcare"),
    ("hospital payment", "Healthcare"),
    ("medical clinic", "Healthcare"),
    ("loan emi payment", "Debt & Loans"),
    ("credit card payment", "Debt & Loans"),
    ("paylater repayment", "Debt & Loans"),
    ("bajaj finance emi", "Debt & Loans"),
    ("phonepe transfer", "Peer Transfers"),
    ("google pay transfer", "Peer Transfers"),
    ("upi transfer", "Peer Transfers"),
    ("cash withdrawal atm", "Peer Transfers"),
    ("neft transfer", "Peer Transfers"),
    ("monthly salary", "Income"),
    ("cash deposit", "Income"),
    ("bank interest", "Income"),
    ("fd redemption", "Income"),
    ("fixed deposit", "Savings"),
    ("electricity bill", "Utilities"),
    ("airtel bill", "Utilities"),
    ("jio recharge", "Utilities"),
    ("internet bill", "Utilities"),
]


def _normalise_text(text):
    text = str(text or "").upper()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\d{3,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class _SimpleTextClassifier:
    """
    Lightweight fallback used when scikit-learn is unavailable.

    It is not a replacement for the TF-IDF model; it only keeps local
    categorisation usable and testable until the full ML dependency is
    installed.
    """

    def __init__(self, data):
        self.classes_ = sorted({label for _, label in data})
        self._examples = []

        for text, label in data:
            tokens = set(_normalise_text(text).split())
            if tokens:
                self._examples.append((tokens, label))

    def predict_proba(self, texts):
        rows = []

        for text in texts:
            tokens = set(_normalise_text(text).split())
            scores = {category: 0.01 for category in self.classes_}

            if tokens:
                for example_tokens, label in self._examples:
                    overlap = len(tokens & example_tokens)
                    if not overlap:
                        continue

                    precision = overlap / len(tokens)
                    recall = overlap / len(example_tokens)
                    similarity = (2 * precision * recall) / (precision + recall)
                    scores[label] = max(scores[label], 0.01 + 5.0 * similarity)

            total = sum(scores.values()) or 1.0
            rows.append([scores[category] / total for category in self.classes_])

        return rows


def _normalise_entity(text):
    """
    Convert noisy UPI descriptions into a stable entity key.

    Examples:
        UPI/DR/938271/RAJU/...
            -> RAJU

        UPI/DR/1234/RAMESH
            -> RAMESH

        RAMESH
            -> RAMESH

    Known merchants are returned in normalised form so repeated merchant
    payments are grouped consistently.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    # UPI descriptions often contain IDs, transaction references and a name.
    parts = [p.strip() for p in re.split(r"[/|_\-]+", raw) if p.strip()]

    candidates = []
    for part in parts:
        clean = _normalise_text(part)
        if not clean:
            continue
        if clean in GENERIC_ENTITY_TERMS:
            continue
        if re.fullmatch(r"\d+", clean):
            continue
        if len(clean) < 2:
            continue
        candidates.append(clean)

    # Prefer an explicit merchant/keyword.
    for candidate in candidates:
        for keywords in CATEGORIES.values():
            for keyword in keywords:
                k = _normalise_text(keyword)
                if k and k in candidate:
                    return candidate

    # For UPI, the useful recipient is normally one of the middle tokens.
    if len(candidates) >= 2:
        # Ignore common bank/transaction words.
        filtered = [
            c for c in candidates
            if c not in {"YES", "HDFC", "ICICI", "SBI", "AXIS", "BANK"}
        ]
        if filtered:
            return max(filtered, key=len)

    # For a plain description, use the cleaned whole description.
    return _normalise_text(raw)[:80]


def _looks_like_known_merchant(text):
    normalised = _normalise_text(text)
    if not normalised:
        return False

    for keyword in KNOWN_MERCHANTS:
        key = _normalise_text(keyword)
        if key and key in normalised:
            return True

    return False


def _rule_category(merchant, is_upi=False):
    text = _normalise_text(merchant)

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            key = _normalise_text(keyword)
            if key and key in text:
                return category

    # Explicit UPI alone does not mean peer transfer. Unknown UPI recipients
    # are handled by the entity/behaviour layer below.
    return "Other"


def _load_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        return []

    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return [
            (item["text"], item["category"])
            for item in data
            if isinstance(item, dict)
            and item.get("text")
            and item.get("category")
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _save_feedback(text, category):
    os.makedirs(MODEL_DIR, exist_ok=True)

    feedback = [
        {"text": t, "category": c}
        for t, c in _load_feedback()
    ]

    # Avoid adding the exact same correction repeatedly.
    item = {"text": text, "category": category}
    if item not in feedback:
        feedback.append(item)

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as handle:
        json.dump(feedback, handle, indent=2)


def _training_data():
    return SEED_DATA + _load_feedback()


def _build_model():
    data = _training_data()

    texts = [_normalise_text(text) for text, _ in data]
    labels = [label for _, label in data]

    if not SKLEARN_AVAILABLE:
        return _SimpleTextClassifier(data)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_features=3000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(texts, labels)
    return model


def _save_model(model):
    if not SKLEARN_AVAILABLE or isinstance(model, _SimpleTextClassifier):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)


def _load_or_train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if SKLEARN_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as handle:
                return pickle.load(handle)
        except Exception:
            # Corrupt/old model: rebuild it.
            pass

    model = _build_model()
    _save_model(model)
    return model


def _load_entity_profiles():
    if not os.path.exists(ENTITY_PATH):
        return {}

    try:
        with open(ENTITY_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_entity_profiles(profiles):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(ENTITY_PATH, "w", encoding="utf-8") as handle:
        json.dump(profiles, handle, indent=2)


def _new_entity_profile():
    return {
        "frequency": 0,
        "debit_count": 0,
        "credit_count": 0,
        "total_debit": 0.0,
        "total_credit": 0.0,
        "known_categories": {},
        "person_likelihood": 0.0,
        "last_seen": "",
    }


def _best_known_category(profile):
    if not profile:
        return None

    known_categories = profile.get("known_categories", {})
    if not isinstance(known_categories, dict) or not known_categories:
        return None

    valid_categories = set(CATEGORIES) | {"Other"}
    votes = []
    for category, count in known_categories.items():
        if category not in valid_categories:
            continue
        try:
            vote_count = int(count or 0)
        except (TypeError, ValueError):
            continue
        if vote_count > 0:
            votes.append((category, vote_count))

    if not votes:
        return None

    return max(votes, key=lambda item: item[1])[0]


def _merge_known_categories(profile, persisted_profile):
    if not isinstance(persisted_profile, dict):
        return

    persisted_categories = persisted_profile.get("known_categories", {})
    if not isinstance(persisted_categories, dict):
        return

    known_categories = profile.setdefault("known_categories", {})
    for category, count in persisted_categories.items():
        try:
            count = int(count or 0)
        except (TypeError, ValueError):
            continue

        known_categories[category] = int(known_categories.get(category, 0)) + count


def _entity_is_probably_person(entity, merchant):
    text = _normalise_text(merchant)
    entity_text = _normalise_text(entity)

    if any(hint in text for hint in PERSON_HINTS):
        return True

    # A short, alphabetic entity with no business vocabulary is a reasonable
    # person candidate. This is intentionally probabilistic, not definitive.
    words = entity_text.split()
    if 1 <= len(words) <= 3 and all(re.fullmatch(r"[A-Z]+", w) for w in words):
        if not _looks_like_known_merchant(entity_text):
            return True

    return False


def _build_entity_profiles(transactions):
    """
    Build behavioural profiles from the current statement without side effects.

    Current-statement frequency is a classification feature, not training
    state. Persisted entity profiles are used only for learned category votes
    from explicit user feedback, so refreshing /results cannot inflate future
    classifications.
    """
    persisted_profiles = _load_entity_profiles()
    profiles = {}

    for transaction in transactions:
        merchant = transaction.get("merchant", "")
        entity = _normalise_entity(merchant)

        if not entity:
            continue

        key = entity.upper()
        if key not in profiles:
            profiles[key] = _new_entity_profile()
            _merge_known_categories(
                profiles[key],
                persisted_profiles.get(key, {}),
            )

        profile = profiles[key]

        withdrawal = float(transaction.get("withdrawal", 0) or 0)
        deposit = float(transaction.get("deposit", 0) or 0)

        profile["frequency"] += 1
        profile["debit_count"] += int(withdrawal > 0)
        profile["credit_count"] += int(deposit > 0)
        profile["total_debit"] += withdrawal
        profile["total_credit"] += deposit

        if withdrawal > 0:
            profile["last_seen"] = str(transaction.get("date", ""))

        if transaction.get("is_upi") and _entity_is_probably_person(entity, merchant):
            # Repeated observations increase confidence that the entity is a
            # person-like recipient, but it never becomes an absolute fact.
            profile["person_likelihood"] = min(
                1.0,
                float(profile.get("person_likelihood", 0.0)) + 0.2,
            )

        transaction["_entity"] = key

    # Round persisted values to keep the JSON readable.
    for profile in profiles.values():
        profile["total_debit"] = round(float(profile.get("total_debit", 0)), 2)
        profile["total_credit"] = round(float(profile.get("total_credit", 0)), 2)

    return profiles


def _entity_behaviour_category(merchant, is_upi, profile):
    """
    Behaviour-based classification for unknown recipients.

    Core heuristic requested for FinSight:
      - 1–2 appearances + small unknown UPI debit -> Transportation
      - 3+ appearances + recurring person-like recipient -> Peer Transfers

    Known merchants are protected by the rule classifier before this function.
    """
    if not profile:
        return None, 0.0, None

    frequency = int(profile.get("frequency", 0))
    debit_count = int(profile.get("debit_count", 0))
    total_debit = float(profile.get("total_debit", 0))
    person_likelihood = float(profile.get("person_likelihood", 0))

    average_debit = total_debit / debit_count if debit_count else 0.0

    if not is_upi or debit_count == 0:
        return None, 0.0, None

    # Frequent person-like recipients are much more likely to be peer
    # transfers. The threshold is configurable and intentionally transparent.
    if frequency >= 3 and (person_likelihood >= 0.4 or average_debit < 2500):
        return "Peer Transfers", 0.92, "entity-frequency"

    # One-off/two-off unknown small UPI payments are often local transport
    # payments where each driver can have a different name/UPI ID.
    if frequency <= 2 and average_debit <= 1000:
        return "Travel & Transport", 0.70, "entity-frequency"

    # A slightly larger one/two-off payment remains uncertain. Do not force
    # it into peer transfers simply because the recipient is unfamiliar.
    if frequency <= 2:
        return "Travel & Transport", 0.55, "entity-frequency"

    return None, 0.0, None


def _predict_category(
    model,
    merchant,
    is_upi=False,
    profile=None,
    feedback_lookup=None,
):
    text = _normalise_text(merchant)

    # Explicit user feedback is the strongest signal.
    if feedback_lookup:
        entity = _normalise_entity(merchant).upper()
        if entity and entity in feedback_lookup:
            return feedback_lookup[entity], 1.0, "user-feedback"

    profile_category = _best_known_category(profile)
    if profile_category:
        return profile_category, 1.0, "entity-profile"

    # Known merchant/rule classifications take precedence over frequency.
    rule = _rule_category(merchant, is_upi)
    if rule != "Other":
        return rule, 1.0, "rule"

    # Behavioural/entity layer comes before generic NLP for unknown UPI
    # recipients because frequency contains information that TF-IDF cannot see.
    behaviour_category, behaviour_confidence, behaviour_source = _entity_behaviour_category(
        merchant,
        is_upi,
        profile,
    )

    if behaviour_category:
        return (
            behaviour_category,
            behaviour_confidence,
            behaviour_source,
        )

    # Fall back to NLP.
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    confidence = float(probabilities[index])
    source = "ml-fallback" if isinstance(model, _SimpleTextClassifier) else "ml"

    if confidence < 0.45:
        return "Other", confidence, source

    return str(classes[index]), confidence, source


def _build_feedback_lookup():
    """
    Build an entity-level lookup from user corrections.

    A correction for 'RAJU' should influence future occurrences of RAJU rather
    than only the exact original description.
    """
    lookup = {}

    for text, category in _load_feedback():
        entity = _normalise_entity(text)
        if entity:
            lookup[entity.upper()] = category

    return lookup


def record_feedback(merchant, category):
    valid_categories = set(CATEGORY_OPTIONS)

    if category not in valid_categories:
        return False

    raw_text = str(merchant or "").strip()
    text = _normalise_text(raw_text)
    entity = _normalise_entity(raw_text)

    if not text or not entity:
        return False

    # Store the original merchant text for training and entity-level lookup.
    _save_feedback(raw_text, category)

    # Retrain immediately so the correction is reflected in later analyses.
    model = _build_model()
    _save_model(model)

    # Also update the entity profile's known category.
    profiles = _load_entity_profiles()
    key = entity.upper()

    profile = profiles.setdefault(key, _new_entity_profile())

    known_categories = profile.setdefault("known_categories", {})
    known_categories[category] = int(known_categories.get(category, 0)) + 1
    _save_entity_profiles(profiles)

    return True


def categorise(merchant, is_upi=False, model=None, profile=None):
    model = model or _load_or_train_model()
    feedback_lookup = _build_feedback_lookup()

    category, _, _ = _predict_category(
        model,
        merchant,
        is_upi,
        profile,
        feedback_lookup,
    )
    return category


def _anomaly_scores(transactions):
    """
    Isolation Forest on debit transactions.

    Features:
      - transaction amount
      - day of month
      - day of week
      - running balance

    For very small statements, anomaly detection is disabled to avoid
    pretending that a tiny sample provides a meaningful baseline.
    """
    debits = [
        t for t in transactions
        if float(t.get("withdrawal", 0) or 0) > 0
    ]

    if len(debits) < 10 or not SKLEARN_AVAILABLE:
        return {id(t): (False, 0.0) for t in transactions}

    X = np.array([
        [
            float(t.get("withdrawal", 0) or 0),
            float(t.get("parsed_date", datetime.min).day),
            float(t.get("parsed_date", datetime.min).weekday()),
            float(t.get("balance", 0) or 0),
        ]
        for t in debits
    ])

    model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42,
    )

    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    result = {id(t): (False, 0.0) for t in transactions}

    for transaction, prediction, score in zip(debits, predictions, scores):
        result[id(transaction)] = (
            prediction == -1,
            round(float(score), 4),
        )

    return result


def _parse_date(value):
    for fmt in (
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            pass

    return datetime.min


def _clamp(value, low, high):
    return max(low, min(high, value))


def _safe_percentage(numerator, denominator):
    if not denominator:
        return 0.0
    return (float(numerator) / float(denominator)) * 100


def _score_band(score):
    if score >= 800:
        return "Excellent"
    if score >= 740:
        return "Strong"
    if score >= 670:
        return "Fair"
    if score >= 580:
        return "Watch"
    return "High risk"


def _risk_level(risk_score):
    if risk_score <= 25:
        return "Low"
    if risk_score <= 50:
        return "Moderate"
    if risk_score <= 75:
        return "Elevated"
    return "High"


def _build_category_breakdown(summary):
    total = sum(float(amount or 0) for amount in summary.values())
    rows = []

    for category, amount in sorted(summary.items(), key=lambda item: item[1], reverse=True):
        rows.append({
            "category": category,
            "amount": round(float(amount), 2),
            "share": round(_safe_percentage(amount, total), 1),
        })

    return rows


def _top_merchants(transactions, total_out, limit=6):
    totals = defaultdict(lambda: {"amount": 0.0, "count": 0})

    for transaction in transactions:
        amount = float(transaction.get("withdrawal", 0) or 0)
        if amount <= 0:
            continue

        merchant = str(transaction.get("merchant", "Unknown") or "Unknown")
        totals[merchant]["amount"] += amount
        totals[merchant]["count"] += 1

    rows = []
    for merchant, data in totals.items():
        rows.append({
            "merchant": merchant,
            "amount": round(data["amount"], 2),
            "count": data["count"],
            "share": round(_safe_percentage(data["amount"], total_out), 1),
        })

    return sorted(rows, key=lambda row: row["amount"], reverse=True)[:limit]


def _largest_transactions(transactions, limit=5):
    debits = [
        transaction
        for transaction in transactions
        if float(transaction.get("withdrawal", 0) or 0) > 0
    ]

    rows = []
    for transaction in sorted(
        debits,
        key=lambda item: float(item.get("withdrawal", 0) or 0),
        reverse=True,
    )[:limit]:
        rows.append({
            "date": transaction.get("date", ""),
            "merchant": transaction.get("merchant", "Unknown"),
            "category": transaction.get("category", "Other"),
            "amount": round(float(transaction.get("withdrawal", 0) or 0), 2),
            "is_anomaly": bool(transaction.get("is_anomaly")),
        })

    return rows


def _spending_volatility(spending_by_date):
    amounts = [float(amount or 0) for amount in spending_by_date.values() if amount]
    if len(amounts) < 2:
        return 0.0

    mean = sum(amounts) / len(amounts)
    if not mean:
        return 0.0

    variance = sum((amount - mean) ** 2 for amount in amounts) / len(amounts)
    return (variance ** 0.5) / mean


def _statement_period(transactions):
    parsed_dates = [
        transaction.get("parsed_date")
        for transaction in transactions
        if transaction.get("parsed_date") != datetime.min
    ]

    if not parsed_dates:
        return {
            "start": "",
            "end": "",
            "days": 0,
            "label": "Date range unavailable",
        }

    start = min(parsed_dates)
    end = max(parsed_dates)
    days = (end - start).days + 1

    return {
        "start": start.strftime("%d %b %Y"),
        "end": end.strftime("%d %b %Y"),
        "days": days,
        "label": f"{start.strftime('%d %b %Y')} to {end.strftime('%d %b %Y')}",
    }


def _build_credit_profile(
    transactions,
    total_in,
    total_out,
    summary,
    spending_by_date,
    anomalies,
):
    """
    Build an indicative credit-readiness and cash-flow risk profile.

    This is intentionally statement-derived and transparent. It is not a
    replacement for bureau scores such as CIBIL, Experian, Equifax, or CRIF.
    """
    debits = [
        transaction
        for transaction in transactions
        if float(transaction.get("withdrawal", 0) or 0) > 0
    ]
    credits = [
        transaction
        for transaction in transactions
        if float(transaction.get("deposit", 0) or 0) > 0
    ]
    balances = [
        float(transaction.get("balance", 0) or 0)
        for transaction in transactions
    ]

    net_cashflow = total_in - total_out
    savings_rate = _safe_percentage(net_cashflow, total_in)
    expense_income_ratio = _safe_percentage(total_out, total_in)
    balance_floor = min(balances) if balances else 0.0
    average_balance = sum(balances) / len(balances) if balances else 0.0
    ending_balance = float(transactions[0].get("balance", 0) or 0) if transactions else 0.0
    active_days = len(spending_by_date)
    average_daily_spend = total_out / active_days if active_days else 0.0
    largest_payment = max(
        (float(transaction.get("withdrawal", 0) or 0) for transaction in debits),
        default=0.0,
    )
    largest_payment_ratio = _safe_percentage(largest_payment, total_in)
    anomaly_rate = _safe_percentage(len(anomalies), len(debits))
    debt_payment_total = float(summary.get("Debt & Loans", 0) or 0)
    debt_income_ratio = _safe_percentage(debt_payment_total, total_in)
    discretionary_categories = {
        "Food & Dining",
        "Travel & Transport",
        "Entertainment",
        "Shopping",
        "Subscriptions",
    }
    discretionary_spend = sum(
        float(summary.get(category, 0) or 0)
        for category in discretionary_categories
    )
    discretionary_ratio = _safe_percentage(discretionary_spend, total_out)
    volatility = _spending_volatility(spending_by_date)

    score = 650.0
    strengths = []
    risk_factors = []
    recommendations = []

    if total_in > 0:
        if savings_rate >= 30:
            score += 95
            strengths.append("Strong surplus after expenses.")
        elif savings_rate >= 20:
            score += 75
            strengths.append("Healthy monthly cash-flow surplus.")
        elif savings_rate >= 10:
            score += 45
            strengths.append("Positive savings buffer.")
        elif savings_rate >= 0:
            score += 15
            strengths.append("Expenses stay within income.")
        elif savings_rate >= -10:
            score -= 45
            risk_factors.append("Spending is slightly above income.")
            recommendations.append("Reduce flexible spending until monthly outflow is below income.")
        else:
            score -= 95
            risk_factors.append("Spending materially exceeds income.")
            recommendations.append("Prioritise a monthly surplus before taking on new credit.")
    elif total_out > 0:
        score -= 100
        risk_factors.append("No income deposits were detected in this statement.")
        recommendations.append("Upload a full salary or income account statement for a clearer credit view.")

    if expense_income_ratio and expense_income_ratio <= 65:
        score += 25
        strengths.append("Expense-to-income ratio is controlled.")
    elif expense_income_ratio > 100:
        score -= 70
        risk_factors.append("Expense-to-income ratio is above 100%.")
    elif expense_income_ratio > 85:
        score -= 30
        risk_factors.append("Expense-to-income ratio is high.")
        recommendations.append("Aim to keep regular expenses below 80% of income.")

    if balance_floor < 0:
        score -= 90
        risk_factors.append("Balance dropped below zero.")
        recommendations.append("Keep a minimum balance buffer to avoid overdraft stress.")
    elif balance_floor < 1000 and total_out > 0:
        score -= 35
        risk_factors.append("Balance buffer is thin.")
        recommendations.append("Build a small emergency floor before increasing discretionary spends.")
    elif average_balance >= max(average_daily_spend * 7, 1000):
        score += 20
        strengths.append("Average balance covers at least a week of active spending.")

    if debt_income_ratio > 40:
        score -= 70
        risk_factors.append("Debt-like payments consume a large share of income.")
        recommendations.append("Bring EMI and card repayments below 35% of monthly income.")
    elif debt_income_ratio > 25:
        score -= 40
        risk_factors.append("Debt burden is moderately high.")
        recommendations.append("Avoid new borrowing until existing repayments fall.")
    elif 0 < debt_income_ratio <= 25:
        score += 10
        strengths.append("Debt-like payments appear manageable from this statement.")

    if anomaly_rate > 10:
        score -= 35
        risk_factors.append("Several debit transactions look unusual against the statement pattern.")
    elif anomalies:
        score -= 15
        risk_factors.append("One or more unusual transactions need review.")

    if largest_payment_ratio > 50:
        score -= 30
        risk_factors.append("Largest payment is very large relative to income.")
    elif largest_payment_ratio > 30:
        score -= 15
        risk_factors.append("Largest payment is sizeable relative to income.")

    if volatility > 1.4:
        score -= 20
        risk_factors.append("Daily spending is volatile.")
        recommendations.append("Smooth large discretionary payments where possible.")
    elif active_days >= 5:
        score += 10
        strengths.append("Spending pattern is reasonably steady.")

    if credits:
        score += 15
        strengths.append("Income or credit inflows are visible.")
    else:
        risk_factors.append("No credit inflows were visible.")

    if discretionary_ratio > 65:
        risk_factors.append("Discretionary categories dominate spending.")
        recommendations.append("Set category caps for food, travel, shopping, entertainment, and subscriptions.")

    if not recommendations:
        recommendations.append("Keep collecting labelled statements to improve category and risk accuracy.")

    score = int(round(_clamp(score, 300, 900)))
    risk_score = int(round(_clamp(100 - ((score - 300) / 600 * 100), 0, 100)))

    return {
        "score": score,
        "score_min": 300,
        "score_max": 900,
        "band": _score_band(score),
        "risk_score": risk_score,
        "risk_level": _risk_level(risk_score),
        "note": "Indicative statement-based score, not a bureau credit score.",
        "metrics": {
            "net_cashflow": round(net_cashflow, 2),
            "savings_rate": round(savings_rate, 1),
            "expense_income_ratio": round(expense_income_ratio, 1),
            "debt_income_ratio": round(debt_income_ratio, 1),
            "discretionary_ratio": round(discretionary_ratio, 1),
            "average_balance": round(average_balance, 2),
            "ending_balance": round(ending_balance, 2),
            "balance_floor": round(balance_floor, 2),
            "average_daily_spend": round(average_daily_spend, 2),
            "largest_payment_ratio": round(largest_payment_ratio, 1),
            "anomaly_rate": round(anomaly_rate, 1),
            "volatility": round(volatility, 2),
        },
        "strengths": strengths[:4],
        "risk_factors": risk_factors[:5],
        "recommendations": recommendations[:4],
    }


def summarise(transactions):
    model = _load_or_train_model()
    feedback_lookup = _build_feedback_lookup()

    transactions = [dict(t) for t in transactions]

    for transaction in transactions:
        transaction["deposit"] = float(transaction.get("deposit", 0) or 0)
        transaction["withdrawal"] = float(transaction.get("withdrawal", 0) or 0)
        transaction["balance"] = float(transaction.get("balance", 0) or 0)
        transaction["parsed_date"] = _parse_date(transaction.get("date", ""))

    # Build side-effect-free entity profiles from this statement before
    # classification.
    profiles = _build_entity_profiles(transactions)

    transactions.sort(
        key=lambda x: x["parsed_date"],
        reverse=True,
    )

    anomaly_map = _anomaly_scores(transactions)

    total_in = 0.0
    total_out = 0.0
    spending_by_date = defaultdict(float)
    summary = defaultdict(float)
    source_counts = Counter()

    for transaction in transactions:
        entity = transaction.get("_entity", "").upper()
        profile = profiles.get(entity)

        category, confidence, source = _predict_category(
            model,
            transaction.get("merchant", ""),
            transaction.get("is_upi", False),
            profile,
            feedback_lookup,
        )

        transaction["category"] = category
        transaction["category_confidence"] = round(confidence, 3)
        transaction["classification_source"] = source
        transaction["classification_label"] = CLASSIFICATION_SOURCE_LABELS.get(
            source,
            "Classifier",
        )

        transaction["entity"] = entity
        transaction["entity_frequency"] = (
            int(profile.get("frequency", 0))
            if profile else 0
        )

        transaction["is_anomaly"], transaction["anomaly_score"] = anomaly_map[
            id(transaction)
        ]

        # Remove internal processing data before sending it to Jinja.
        transaction.pop("_entity", None)

        total_in += transaction["deposit"]

        if transaction["withdrawal"] > 0:
            total_out += transaction["withdrawal"]
            summary[category] += transaction["withdrawal"]

            if transaction["parsed_date"] != datetime.min:
                spending_by_date[
                    transaction["parsed_date"]
                ] += transaction["withdrawal"]

        source_counts[source] += 1

    date_trend = sorted(spending_by_date.items())

    trend_dates = [
        dt.strftime("%d %b")
        for dt, _ in date_trend
    ]

    trend_amounts = [
        round(amount, 2)
        for _, amount in date_trend
    ]

    behavior_points = []

    if transactions:
        latest = transactions[0]
        behavior_points.append(
            f"Most recent transaction: {latest['merchant']} on {latest['date']}."
        )

    meaningful = {
        k: v
        for k, v in summary.items()
        if k not in ("Peer Transfers", "Savings", "Income")
    }

    if meaningful:
        top_cat = max(meaningful, key=meaningful.get)
        behavior_points.append(
            f"Biggest spending category: {top_cat} at "
            f"₹{meaningful[top_cat]:,.0f}."
        )

    debits = [
        t for t in transactions
        if t["withdrawal"] > 0
    ]

    biggest = max(
        debits,
        key=lambda x: x["withdrawal"],
        default=None,
    )

    if biggest:
        behavior_points.append(
            f"Largest single payment: "
            f"₹{biggest['withdrawal']:,.0f} to "
            f"{biggest['merchant']} on {biggest['date']}."
        )

    active_days = len(spending_by_date)

    if active_days:
        behavior_points.append(
            f"Average daily spending: "
            f"₹{total_out / active_days:,.0f} "
            f"across {active_days} active days."
        )

    anomalies = [
        t for t in transactions
        if t["is_anomaly"]
    ]

    if anomalies:
        behavior_points.append(
            f"ML anomaly detection flagged {len(anomalies)} unusual "
            f"transaction{'s' if len(anomalies) != 1 else ''} for review."
        )
    elif not SKLEARN_AVAILABLE:
        behavior_points.append(
            "scikit-learn is unavailable, so FinSight used the local fallback "
            "classifier and skipped Isolation Forest anomaly detection."
        )

    # Explain the new entity intelligence in the dashboard.
    entity_frequency_categories = Counter(
        t["category"]
        for t in transactions
        if t.get("classification_source") == "entity-frequency"
    )

    if entity_frequency_categories:
        transport_count = entity_frequency_categories.get(
            "Travel & Transport",
            0,
        )
        peer_count = entity_frequency_categories.get(
            "Peer Transfers",
            0,
        )

        if transport_count:
            behavior_points.append(
                f"Recognised {transport_count} unfamiliar low-frequency "
                f"recipient{'s' if transport_count != 1 else ''} as likely "
                f"local transport payments."
            )

        if peer_count:
            behavior_points.append(
                f"Grouped {peer_count} recurring recipient"
                f"{'s' if peer_count != 1 else ''} as peer transfers "
                f"using entity frequency."
            )

    if len(trend_amounts) >= 10:
        recent = trend_amounts[-7:]
        older = trend_amounts[:-7]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if older_avg and recent_avg > older_avg * 1.3:
            behavior_points.append(
                "Spending has increased significantly in the last "
                "7 active days."
            )
        elif older_avg and recent_avg < older_avg * 0.7:
            behavior_points.append(
                "Spending has decreased significantly in the last "
                "7 active days."
            )
        else:
            behavior_points.append(
                "Spending has been relatively consistent over the period."
            )

    if "Savings" in summary:
        behavior_points.append(
            f"₹{summary['Savings']:,.0f} moved into fixed deposits "
            f"during this period."
        )

    low_confidence = sum(
        1
        for t in transactions
        if (
            t["classification_source"] in {"ml", "ml-fallback"}
            and t["category_confidence"] < 0.60
        )
    )

    if low_confidence:
        behavior_points.append(
            f"{low_confidence} transaction classification"
            f"{'s' if low_confidence != 1 else ''} have low ML confidence "
            "and can be corrected below."
        )

    category_breakdown = _build_category_breakdown(summary)
    top_merchants = _top_merchants(transactions, total_out)
    largest_transactions = _largest_transactions(transactions)
    statement_period = _statement_period(transactions)
    credit_profile = _build_credit_profile(
        transactions,
        total_in,
        total_out,
        summary,
        spending_by_date,
        anomalies,
    )
    by_category = {
        k: round(v, 2)
        for k, v in sorted(
            summary.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    # Do not expose internal datetime objects to the template.
    for transaction in transactions:
        transaction.pop("parsed_date", None)

    return {
        "total_income": round(total_in, 2),
        "total_spent": round(total_out, 2),
        "net_cashflow": round(total_in - total_out, 2),
        "by_category": by_category,
        "category_breakdown": category_breakdown,
        "top_merchants": top_merchants,
        "largest_transactions": largest_transactions,
        "statement_period": statement_period,
        "credit_profile": credit_profile,
        "transactions": transactions,
        "trend_dates": trend_dates,
        "trend_amounts": trend_amounts,
        "behavior_points": behavior_points,
        "anomaly_count": len(anomalies),
        "ml_classifications": (
            source_counts.get("ml", 0)
            + source_counts.get("ml-fallback", 0)
        ),
        "rule_classifications": source_counts.get("rule", 0),
        "entity_classifications": source_counts.get(
            "entity-frequency",
            0,
        ),
        "feedback_classifications": (
            source_counts.get("user-feedback", 0)
            + source_counts.get("entity-profile", 0)
        ),
    }
