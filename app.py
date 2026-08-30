import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from parser import parse_statement, parse_csv
from categoriser import CATEGORY_OPTIONS, summarise, record_feedback

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".csv"}


def allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def parse_uploaded_file(filepath, extension):
    if extension == ".pdf":
        return parse_statement(filepath)
    with open(filepath, "rb") as handle:
        return parse_csv(handle)


@app.route("/")
def index():
    return render_template("index.html", error=None)


@app.route("/upload", methods=["POST"])
def upload():
    if "statement" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["statement"]
    original_name = secure_filename(file.filename or "")

    if not original_name:
        return render_template("index.html", error="Please choose a file.")

    extension = os.path.splitext(original_name.lower())[1]
    if extension not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="Please upload a PDF or CSV file.")

    # Use a unique filename so separate uploads cannot overwrite each other.
    stored_name = f"{uuid.uuid4().hex}{extension}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(filepath)

    try:
        transactions = parse_uploaded_file(filepath, extension)
    except Exception as exc:
        app.logger.exception("Statement parsing failed")
        return render_template(
            "index.html",
            error=f"Could not parse the statement: {exc}",
        )

    if not transactions:
        return render_template(
            "index.html",
            error="No transactions found. Please upload a valid bank statement.",
        )

    # Keep the current upload available for the /results route.
    app.config["CURRENT_FILE"] = filepath
    app.config["CURRENT_EXTENSION"] = extension

    return redirect(url_for("results", page=1))


@app.post("/feedback")
def feedback():
    merchant = request.form.get("merchant", "").strip()
    category = request.form.get("category", "").strip()
    if merchant and category:
        record_feedback(merchant, category)

    return redirect(url_for("results", page=request.form.get("page", 1)))


@app.route("/results")
def results():
    filepath = app.config.get("CURRENT_FILE")
    extension = app.config.get("CURRENT_EXTENSION")

    if not filepath or not os.path.exists(filepath):
        return redirect(url_for("index"))

    try:
        transactions = parse_uploaded_file(filepath, extension)
        result = summarise(transactions)
    except Exception as exc:
        app.logger.exception("Result generation failed")
        return render_template("index.html", error=f"Could not analyse the statement: {exc}")

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    per_page = 20
    start = (page - 1) * per_page
    paginated_transactions = result["transactions"][start:start + per_page]

    total_pages = max(1, (len(result["transactions"]) + per_page - 1) // per_page)

    return render_template(
        "results.html",
        result=result,
        transactions=paginated_transactions,
        page=page,
        total_pages=total_pages,
        category_options=CATEGORY_OPTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True)
