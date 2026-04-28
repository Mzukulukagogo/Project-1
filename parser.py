import pdfplumber
import re
import os

def parse_statement(pdf_path):
    transactions = []
    
    # Step 1: collect every line from every page into one flat list
    all_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split('\n'):
                all_lines.append(line.strip())
    
    # Step 2: find every transaction line (starts with DD-MM-YYYY)
    for i, line in enumerate(all_lines):
        date_match = re.match(r'^(\d{2}-\d{2}-\d{4})', line)
        if not date_match:
            continue
        
        date = date_match.group(1)
        
        # Extract amounts from this line
        amounts = re.findall(r'[\d,]+\.\d{2}', line)
        if len(amounts) < 2:
            continue
        
        balance = float(amounts[-1].replace(',', ''))
        amount = float(amounts[-2].replace(',', ''))
        
        # Step 3: scan BACKWARDS to find the transaction header line
        # which always starts with UPI/, INET, CASH, FIR/, ATM, TD , MB-, REF, ONLINE
        header_line = ''
        for j in range(i-1, max(i-10, 0), -1):
            prev = all_lines[j]
            if re.match(r'^(UPI/|INET|CASH |FIR/|ATM |TD |MB-|REF-|ONLINE|RELIANCE|TRAVEL|HMS|ASANGEE|SMOKE|KFC|PAX)', prev):
                header_line = prev
                break
        
        if not header_line:
            continue
        
        # Step 4: determine credit or debit from the header line
        is_credit = bool(re.match(r'^(UPI/CR|INET-IMPS-CR|CASH DEPOSIT|FIR/CR|MB/|REF-TR|ONLINE PARTIAL|SBINT)', header_line))
        
        # Step 5: extract clean merchant name
        # UPI format: UPI/DR/123456789/MERCHANTNAME/...
        upi_match = re.match(r'UPI/(?:DR|CR)/\d+/([^/]+)', header_line)
        if upi_match:
            merchant = upi_match.group(1).strip()
        elif header_line.startswith('INET-IMPS-CR'):
            merchant = 'SALARY/TRANSFER IN'
        elif header_line.startswith('CASH DEPOSIT'):
            merchant = 'CASH DEPOSIT'
        elif header_line.startswith('FIR/CR'):
            merchant = 'INCOME'
        elif header_line.startswith('ONLINE PARTIAL'):
            merchant = 'FD REDEMPTION'
        elif header_line.startswith('TD '):
            merchant = 'FIXED DEPOSIT'
        elif header_line.startswith('SBINT'):
            merchant = 'BANK INTEREST'
        else:
            merchant = header_line[:30].strip()
        
        if is_credit:
            deposit = amount
            withdrawal = 0
        else:
            deposit = 0
            withdrawal = amount
        
        transactions.append({
            'date': date,
            'merchant': merchant,
            'deposit': deposit,
            'withdrawal': withdrawal,
            'balance': balance,
            'is_upi': header_line.startswith('UPI/')
        })
    
    return transactions


def parse_csv(file):
    import csv

    transactions = []
    content = file.read().decode('utf-8').splitlines()
    reader = csv.DictReader(content)

    for row in reader:
        transactions.append({
            'date': row.get('date', ''),
            'merchant': row.get('merchant', ''),
            'deposit': float(row.get('deposit', 0) or 0),
            'withdrawal': float(row.get('withdrawal', 0) or 0),
            'balance': float(row.get('balance', 0) or 0)
        })

    return transactions


if __name__ == '__main__':
    pdf_file = [f for f in os.listdir('.') if f.endswith('.pdf')][0]
    print(f"Found PDF: {pdf_file}\n")
    txns = parse_statement(pdf_file)
    print(f"Total transactions found: {len(txns)}\n")
    print(f"{'Date':<12} {'Merchant':<25} {'Deposit':>10} {'Withdrawal':>12} {'Balance':>12}")
    print('-' * 75)
    for t in txns[:15]:
        print(f"{t['date']:<12} {t['merchant']:<25} {t['deposit']:>10.2f} {t['withdrawal']:>12.2f} {t['balance']:>12.2f}")