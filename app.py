import os
from flask import Flask, render_template, request, redirect, url_for
from parser import parse_statement, parse_csv
from categoriser import summarise

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', error=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'statement' not in request.files:
        return render_template('index.html', error='No file uploaded.')
    
    file = request.files['statement']
    if file.filename == '' or not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.csv')):
        return render_template('index.html', error='Please upload a PDF or CSV file.')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'statement.pdf')
    file.save(filepath)
    
    if file.filename.lower().endswith('.pdf'):
        transactions = parse_statement(filepath)
    else:
        file.stream.seek(0)
        transactions = parse_csv(file)
    if not transactions:
        return render_template('index.html', error='No transactions found. Please upload a valid bank statement PDF.')
    
    result = summarise(transactions)

    # PAGINATION
    page = int(request.args.get('page', 1))
    per_page = 20

    start = (page - 1) * per_page
    end = start + per_page

    paginated_transactions = result['transactions'][start:end]

    return render_template(
        'results.html',
        result=result,
        transactions=paginated_transactions,
        page=page
    )
@app.route('/results')
def results():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'statement.pdf')
    if not os.path.exists(filepath):
        return redirect(url_for('index'))

    transactions = parse_statement(filepath)
    result = summarise(transactions)

    page = int(request.args.get('page', 1))
    per_page = 20
    start = (page - 1) * per_page
    end = start + per_page
    paginated_transactions = result['transactions'][start:end]

    return render_template(
        'results.html',
        result=result,
        transactions=paginated_transactions,
        page=page
    )
if __name__ == '__main__':
    app.run(debug=True)