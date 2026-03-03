import os
from flask import Flask, render_template, request
from infer import infer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("audio")

        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            score = infer(path)
            result = f"Model output score: {score:.4f}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)