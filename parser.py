import csv
import io
import os
import re
from datetime import datetime

DATE_RE = re.compile(r"^(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
AMOUNT_RE = re.compile(r"(?:₹\s*)?[-+]?\d[\d,]*(?:\.\d{1,2})?")

# Common transaction/header prefixes found in Indian bank statements.
HEADER_PREFIXES = (
    "UPI/", "UPI ", "INET", "CASH", "FIR/", "ATM", "TD ", "MB-", "MB/",
    "REF-", "ONLINE", "NEFT", "IMPS", "RTGS", "POS/", "POS ", "ACH/",
    "ECS/", "NACH", "BY TRANSFER", "TO TRANSFER", "FROM TRANSFER",
    "RELIANCE", "TRAVEL", "HMS", "ASANGEE", "SMOKE", "KFC", "PAX",
)

CREDIT_PREFIXES = (
    "UPI/CR", "UPI/CR/", "INET-IMPS-CR", "CASH DEPOSIT", "FIR/CR",
    "MB/CR", "REF-TR", "ONLINE PARTIAL", "SBINT", "BY TRANSFER",
    "FROM TRANSFER", "NEFT/CR", "IMPS/CR", "RTGS/CR",
)


def _parse_number(value):
    """Parse Indian-style numbers safely, returning 0.0 for blank/invalid values."""
    if value is None:
        return 0.0
    text = str(value).strip().replace("₹", "").replace("Rs.", "").replace("Rs", "")
    text = text.replace(",", "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalise_date(value):
    text = str(value or "").strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y",
                "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    return text


def _looks_like_header(line):
    upper = line.strip().upper()
    return any(upper.startswith(prefix) for prefix in HEADER_PREFIXES)


def _extract_merchant(header_line):
    text = header_line.strip()

    # UPI examples: UPI/DR/12345/SWIGGY/ORDER...
    match = re.search(r"UPI/(?:DR|CR)/[^/]+/([^/]+)", text, re.I)
    if match:
        return match.group(1).strip()

    # More tolerant UPI format when an ID is omitted.
    match = re.search(r"UPI/(?:DR|CR)/([^/]+)", text, re.I)
    if match:
        candidate = match.group(1).strip()
        if candidate and candidate.upper() not in {"DR", "CR"}:
            return candidate

    upper = text.upper()
    special = {
        "CASH DEPOSIT": "CASH DEPOSIT",
        "INET-IMPS-CR": "SALARY/TRANSFER IN",
        "FIR/CR": "INCOME",
        "ONLINE PARTIAL": "FD REDEMPTION",
        "SBINT": "BANK INTEREST",
    }
    for prefix, merchant in special.items():
        if upper.startswith(prefix):
            return merchant

    # Remove common transaction-type prefixes while retaining merchant text.
    cleaned = re.sub(
        r"^(NEFT|IMPS|RTGS|ATM|POS|ACH|ECS|NACH)[\s/:-]*",
        "",
        text,
        flags=re.I,
    )
    return cleaned[:80].strip() or "Unknown"


def _is_credit(header_line):
    upper = header_line.strip().upper()
    return any(upper.startswith(prefix) for prefix in CREDIT_PREFIXES)


def parse_statement(pdf_path):
    """
    Extract transactions from text-based PDFs.

    The parser deliberately keeps the original backward-scanning strategy:
    a date/amount row is located first, then nearby preceding lines are searched
    for the transaction description/header.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_lines = []
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_lines.extend(line.strip() for line in text.splitlines() if line.strip())

    transactions = []
    for i, line in enumerate(all_lines):
        date_match = DATE_RE.match(line)
        if not date_match:
            continue

        date = _normalise_date(date_match.group("date"))
        amounts = AMOUNT_RE.findall(line)
        numeric_amounts = [_parse_number(a) for a in amounts]

        # Most statements have amount + balance on the dated row.
        if len(numeric_amounts) < 2:
            continue

        amount = numeric_amounts[-2]
        balance = numeric_amounts[-1]
        if amount < 0:
            amount = abs(amount)

        # Search backward, but stop at a previous date to avoid crossing
        # into another transaction.
        header_line = ""
        for j in range(i - 1, max(i - 12, -1), -1):
            previous = all_lines[j]
            if DATE_RE.match(previous):
                break
            if _looks_like_header(previous):
                header_line = previous
                break

        if not header_line:
            # Some statements put the description directly before the date.
            if i > 0 and all_lines[i - 1]:
                header_line = all_lines[i - 1]

        if not header_line:
            continue

        credit = _is_credit(header_line)
        transactions.append({
            "date": date,
            "merchant": _extract_merchant(header_line),
            "deposit": amount if credit else 0.0,
            "withdrawal": 0.0 if credit else amount,
            "balance": balance,
            "is_upi": header_line.upper().startswith("UPI"),
        })

    return _deduplicate_transactions(transactions)


def _deduplicate_transactions(transactions):
    seen = set()
    result = []
    for transaction in transactions:
        key = (
            transaction.get("date"),
            transaction.get("merchant", "").strip().upper(),
            round(float(transaction.get("deposit", 0)), 2),
            round(float(transaction.get("withdrawal", 0)), 2),
            round(float(transaction.get("balance", 0)), 2),
        )
        if key not in seen:
            seen.add(key)
            result.append(transaction)
    return result


def parse_csv(file):
    """Parse CSV statements with tolerant column-name matching."""
    raw = file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = raw

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []

    fields = {str(name).strip().lower(): name for name in reader.fieldnames if name}

    def get(row, *names):
        for name in names:
            if name in fields:
                return row.get(fields[name], "")
        return ""

    transactions = []
    for row in reader:
        date = _normalise_date(get(row, "date", "transaction date", "txn date"))
        merchant = get(row, "merchant", "description", "narration", "transaction description").strip()

        deposit = _parse_number(get(row, "deposit", "credit", "credits", "money in"))
        withdrawal = _parse_number(get(row, "withdrawal", "debit", "debits", "money out"))
        balance = _parse_number(get(row, "balance", "closing balance", "available balance"))

        # Support a single amount column plus a type column.
        amount = _parse_number(get(row, "amount", "transaction amount"))
        transaction_type = get(row, "type", "transaction type", "dr/cr").strip().upper()
        if amount and not deposit and not withdrawal:
            if transaction_type in {"C", "CR", "CREDIT", "CREDITED", "DEPOSIT", "MONEY IN"}:
                deposit = abs(amount)
            elif transaction_type in {
                "D",
                "DR",
                "DEBIT",
                "DEBITED",
                "WITHDRAWAL",
                "MONEY OUT",
            }:
                withdrawal = abs(amount)
            elif amount < 0:
                withdrawal = abs(amount)
            else:
                deposit = amount

        if not date and not merchant and not deposit and not withdrawal:
            continue

        transactions.append({
            "date": date,
            "merchant": merchant or "Unknown",
            "deposit": deposit,
            "withdrawal": withdrawal,
            "balance": balance,
            "is_upi": "UPI" in merchant.upper(),
        })

    return _deduplicate_transactions(transactions)


if __name__ == "__main__":
    pdfs = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    if pdfs:
        txns = parse_statement(pdfs[0])
        print(f"Found {len(txns)} transactions.")
        for t in txns[:15]:
            print(t)
