from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

model = load_model("./assets/MNIST_model.h5")


def preprocessing(image):
    img = image.convert("L").resize((28, 28))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# simples upload form
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# predict image
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocessing(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return render_template("result.html", prediction=predicted_class)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
