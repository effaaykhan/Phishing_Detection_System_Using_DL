
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re
from urllib.parse import urlparse

app = Flask(__name__)

model = tf.keras.models.load_model("phishing_detection_model.h5")

with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_data))

def preprocess_url(url):
    url_length = len(url)
    dot_count = url.count('.')
    https_present = 1 if "https" in url else 0
    suspicious_chars = len(re.findall(r'[@\-\_]', url))
    ip_address_present = 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url) else 0
    domain = urlparse(url).netloc
    subdomain_count = domain.count('.')

    additional_features = np.array([url_length, dot_count, https_present, suspicious_chars, ip_address_present, subdomain_count]).reshape(1, -1)

    sequence = tokenizer.texts_to_sequences([url])
    max_len = 100  
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")

    return padded_sequence, additional_features

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            text_features, additional_features = preprocess_url(url)

            prediction = model.predict([text_features, additional_features])[0][0]
            result = "Phishing" if prediction > 0.5 else "Legitimate"
            confidence = round(prediction * 100 if prediction > 0.5 else (1 - prediction) * 100, 2)

            return render_template("index.html", result=result, confidence=confidence, url=url)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)

