from datetime import datetime
from collections import defaultdict

CATEGORIES = {
    'Food & Dining': [
        'ZOMATO', 'DOMINOS', 'SWIGGY', 'BLINKIT', 'SMARTPED',
        'RELIANCE', 'FRESHCHO', 'LRFOOD', 'TRAVELFO', 'COFFEE',
        'IBACO', 'KFC', 'SWAMI'
    ],
    'Travel & Transport': [
        'UBER', 'RAPIDO', 'APSRTC', 'MAKONI', 'KERALA', 'OLA',
        'RED BUS', 'IRCTC', 'MAKEMYTRI', 'INDIGO', 'AIRBNB',
        'STAYFL', 'HMS', 'TRAVEL', 'INDIAN'
    ],
    'Entertainment': [
        'INOX', 'PVR', 'WONDERLA', 'BOOKMYSHOW', 'LIVEYOU', 'CONNAU'
    ],
    'Shopping': [
        'AMAZON', 'FLIPKART', 'MEESHO', 'VISHAL', 'ZUDIO', 'ROTARY'
    ],
    'Subscriptions': [
        'APPLE', 'OPENAI', 'NETFLIX', 'SPOTIFY'
    ],
    'Education': [
        'IIT', 'ADITYA', 'PAX'
    ],
    'Peer Transfers': [
        'NEFT', 'IMPS', 'RTGS', 'PAYTM', 'PHONEPE', 'GOOGLEPAY',
        'AMAZONPAY', 'BHIM', 'CASH WITHDRAWAL', 'ATM'
    ],
    'Income': [
        'SALARY', 'INCOME', 'CASH DEPOSIT', 'FD REDEMPTION',
        'BANK INTEREST', 'PRAISEND', 'DLOCAL', 'PRAISE'
    ],
    'Savings': [
        'FIXED DEPOSIT'
    ]
}


def categorise(merchant, is_upi=False):
    merchant_upper = merchant.upper()
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in merchant_upper:
                return category
    # If it came through UPI but matched nothing, it's a person-to-person transfer
    if is_upi:
        return 'Peer Transfers'
    return 'Other'


def summarise(transactions):
    behavior_points = []
    summary = {}
    total_in = 0
    total_out = 0
    spending_by_date = defaultdict(float)

    # Parse and sort by date descending
    for t in transactions:
        try:
            t['parsed_date'] = datetime.strptime(t['date'], '%d-%m-%Y')
        except Exception:
            t['parsed_date'] = datetime.min

    transactions.sort(key=lambda x: x['parsed_date'], reverse=True)

    # Categorise and total up
    for t in transactions:
        category = categorise(t['merchant'], is_upi=t.get('is_upi', False))
        t['category'] = category

        if t['deposit'] > 0:
            total_in += t['deposit']

        if t['withdrawal'] > 0:
            total_out += t['withdrawal']
            if category not in summary:
                summary[category] = 0
            summary[category] += t['withdrawal']

            if t['parsed_date'] != datetime.min:
                spending_by_date[t['parsed_date']] += t['withdrawal']

    # Trend data for chart
    date_trend = sorted(spending_by_date.items())
    trend_dates = [dt.strftime('%d %b') for dt, _ in date_trend]
    trend_amounts = [round(amount, 2) for _, amount in date_trend]

    # Insights
    if transactions:
        latest = transactions[0]
        behavior_points.append(
            f"Most recent transaction: {latest['merchant']} on {latest['date']}."
        )

    meaningful = {k: v for k, v in summary.items() if k not in ('Peer Transfers', 'Savings', 'Income')}
    if meaningful:
        top_cat = max(meaningful, key=meaningful.get)
        behavior_points.append(
            f"Biggest spending category: {top_cat} at ₹{meaningful[top_cat]:,.0f}."
        )

    biggest = max(transactions, key=lambda x: x['withdrawal']) if transactions else None
    if biggest and biggest['withdrawal'] > 0:
        behavior_points.append(
            f"Largest single payment: ₹{biggest['withdrawal']:,.0f} to {biggest['merchant']} on {biggest['date']}."
        )

    if spending_by_date:
        avg_daily = total_out / len(spending_by_date)
        behavior_points.append(
            f"Average daily spending: ₹{avg_daily:,.0f} across {len(spending_by_date)} active days."
        )

    if len(trend_amounts) >= 10:
        recent_avg = sum(trend_amounts[-7:]) / 7
        older_avg = sum(trend_amounts[:-7]) / max(1, len(trend_amounts[:-7]))
        if recent_avg > older_avg * 1.3:
            behavior_points.append("Spending has increased significantly in the last 7 days.")
        elif recent_avg < older_avg * 0.7:
            behavior_points.append("Spending has decreased in the last 7 days.")
        else:
            behavior_points.append("Spending has been relatively consistent over the period.")

    if 'Savings' in summary:
        behavior_points.append(
            f"₹{summary['Savings']:,.0f} moved into fixed deposits during this period."
        )

    return {
        'total_income': round(total_in, 2),
        'total_spent': round(total_out, 2),
        'by_category': {k: round(v, 2) for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True)},
        'transactions': transactions,
        'trend_dates': trend_dates,
        'trend_amounts': trend_amounts,
        'behavior_points': behavior_points
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