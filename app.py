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

        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            pred = infer(path)

            result = {
                "label": pred["label"],
                "real_prob": f"{pred['real_prob']:.4f}",
                "fake_prob": f"{pred['fake_prob']:.4f}",
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)