import os
from flask import Flask, render_template, request, redirect, url_for
from parser import parse_statement
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
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return render_template('index.html', error='Please upload a PDF file.')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'statement.pdf')
    file.save(filepath)
    
    transactions = parse_statement(filepath)
    if not transactions:
        return render_template('index.html', error='No transactions found. Please upload a valid bank statement PDF.')
    
    result = summarise(transactions)
    
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)