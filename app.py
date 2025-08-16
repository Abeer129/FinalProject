import os
from flask import Flask, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=ROOT, static_url_path="")

@app.route("/")
def index():
    for candidate in ("home.html", "final_risk_report.html", "index.html"):
        p = os.path.join(ROOT, candidate)
        if os.path.exists(p):
            return send_from_directory(ROOT, candidate)
    return "App is up but no homepage file was found.", 200

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(ROOT, path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
