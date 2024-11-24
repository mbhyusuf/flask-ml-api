from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the ML model
model = tf.keras.models.load_model("model/modelfinal.h5")

# Define the image input size (modify this to match your model's input size)
IMG_HEIGHT, IMG_WIDTH = 250, 250

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})
        
        # Save and preprocess the image
        if file:
            # Open the image and preprocess it
            image = Image.open(file)
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to model input size
            image = np.array(image) / 255.0  # Normalize the image
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Predict using the model
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]  # For classification models

            return jsonify({"prediction": int(predicted_class)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
