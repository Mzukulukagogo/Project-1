CATEGORIES = {
    'Food & Dining': [
        'ZOMATO', 'DOMINOS', 'SWIGGY', 'BLINKIT', 'SMARTPED',
        'RELIANCE', 'FRESHCHO', 'LRFOOD', 'TRAVELFO', 'COFFEE',
        'IBACO', 'KFC', 'SWAMI'
    ],
    'Transport': [
        'UBER', 'RAPIDO', 'APSRTC', 'MAKONI', 'KERALA',
        'AIRBNB', 'INDIGO', 'INDIAN', 'MAKEMYTRI', 'STAYFL',
        'WIND', 'OLA', 'RAPIDO', 'RED BUS', 'IRCTC'
    ],
    'Entertainment': [
        'INOX', 'PVR', 'WONDERLA', 'BOOKMYSHOW', 'LIVEYOU',
        'CONNAU', 'HOTSTAR', 'NETFLIX', 'PRIME', 'ZEE5'
    ],
    'Shopping': [
        'AMAZON', 'FLIPKART', 'MEESHO', 'VISHAL', 'ZUDIO',
        'ROTARY', 'MB-IMPS', 'BIGBASKET', 'GROCERY', 'CLOTHING'
    ],
    'Subscriptions': [
        'APPLE', 'OPENAI', 'NETFLIX', 'SPOTIFY', 'JIO', 'AIRTEL', 'VI'
    ],
    'Education': [
        'IIT', 'ADITYA', 'PAX'
    ],
    'Travel': [
        'MAKEMYTRI', 'INDIGO', 'AIRBNB', 'STAYFL', 'HMS',
        'TRAVEL', 'INDIAN'
    ],
    'People/Transfers': [
        'NEFT', 'IMPS', 'RTGS', 'TRANSFER', 'BANK TRANSFER',
        'PAYTM', 'PHONEPE', 'GOOGLEPAY', 'AMAZONPAY', 'BHIM',
        'UPI', 'CASH WITHDRAWAL', 'ATM'
    ],
    'Income': [
        'SALARY', 'INCOME', 'CASH DEPOSIT', 'FD REDEMPTION',
        'BANK INTEREST', 'PRAISEND', 'DLOCAL', 'PRAISE'
    ],
    'Savings': [
        'FIXED DEPOSIT'
    ]
}

def categorise(merchant):
    merchant_upper = merchant.upper()
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in merchant_upper:
                return category
    return 'Other'

def summarise(transactions):
    summary = {}
    total_in = 0
    total_out = 0

    for t in transactions:
        category = categorise(t['merchant'])
        t['category'] = category

        if t['deposit'] > 0:
            total_in += t['deposit']
        if t['withdrawal'] > 0:
            total_out += t['withdrawal']
            if category not in summary:
                summary[category] = 0
            summary[category] += t['withdrawal']

    return {
        'total_income': round(total_in, 2),
        'total_spent': round(total_out, 2),
        'by_category': {k: round(v, 2) for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True)},
        'transactions': transactions
    }


if __name__ == '__main__':
    from parser import parse_statement
    import os

    pdf_file = [f for f in os.listdir('.') if f.endswith('.pdf')][0]
    txns = parse_statement(pdf_file)
    result = summarise(txns)

    print(f"Total Money In:  ₹{result['total_income']:,.2f}")
    print(f"Total Money Out: ₹{result['total_spent']:,.2f}")
    print(f"\nSpending by Category:")
    print('-' * 35)
    for cat, amount in result['by_category'].items():
        print(f"  {cat:<20} ₹{amount:>10,.2f}")