# FinSight

FinSight is a simple Flask app that analyses PDF bank statements and provides a spending summary, category breakdown, and recent transaction list.

## Features

- Upload a PDF bank statement
- Validate upload format and bank statement content
- Show progress feedback during upload and processing
- Categorise spending into groups such as Food, Transport, People/Transfers, Income, and more
- Display totals for income, spending, and net position
- Fixed footer branding on each page

## Requirements

- Python 3.11+ (tested with Python 3.14)
- Flask
- pdfplumber

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Mzukulukagogo/Project-1.git
   cd Project-1
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install required packages:

   ```powershell
   pip install flask pdfplumber
   ```

## Running the app

```powershell
python app.py
```

Then open a browser to `http://127.0.0.1:5000`.

## Usage

- Select a PDF bank statement from the upload form
- Watch the progress indicator show file selection, upload, and parsing
- View the analysis results page with spending summary and transaction details

## Project structure

- `app.py` — Flask application routes and upload handling
- `parser.py` — PDF parsing logic to extract transactions
- `categoriser.py` — Category mapping and summary generation
- `templates/` — HTML templates for the upload page and results page
- `uploads/` — Temporary uploaded files (ignored by Git)

## Notes

- Only PDF uploads are accepted.
- If the PDF cannot be parsed as a bank statement, the app returns a friendly error message.
- The app currently uses a hard-coded category map and may require adjustments for new merchant text patterns.

## License

This project is open for modification and personal use. Feel free to adapt it to your own bank statement format.
