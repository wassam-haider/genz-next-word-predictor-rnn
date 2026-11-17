"""
Serves next-word predictions from whichever model the frontend dropdown
selects (rnn / lstm / gru / bidirectional).

pip install flask flask-cors tensorflow
"""

import json
import pickle

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("model_config.json") as f:
    config = json.load(f)
MAX_SEQ_LEN = config["max_seq_len"]

# train.py now saves models in the native Keras format (.keras) instead of
# the legacy HDF5 (.h5) format, and each model's file is the FINAL saved
# model -- i.e. the one with the best-epoch weights already restored via
# EarlyStopping(restore_best_weights=True). Loading "{kind}_model.keras"
# is equivalent to loading "{kind}_model_best.keras" for inference purposes.
MODELS = {
    "rnn": load_model("rnn_model.keras"),
    "lstm": load_model("lstm_model.keras"),
    "gru": load_model("gru_model.keras"),
    "bidirectional": load_model("bidirectional_model.keras"),
}

index_word = {i: w for w, i in tokenizer.word_index.items()}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    model_key = data.get("model", "lstm")
    top_k = int(data.get("top_k", 5))

    if model_key not in MODELS:
        return jsonify({"error": f"unknown model '{model_key}'"}), 400
    if not text.strip():
        return jsonify({"model": model_key, "predictions": []})

    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=MAX_SEQ_LEN - 1, padding="pre")

    probs = MODELS[model_key].predict(seq, verbose=0)[0]
    top_indices = probs.argsort()[-top_k:][::-1]

    # index.html's renderPredictions() expects objects with `word` and
    # `confidence_pct`, not a flat list of words -- the previous version
    # returned plain strings under "suggestions", which would have shown
    # "undefined" for every chip's word and percentage on the frontend.
    predictions = [
        {
            "word": index_word[i],
            "confidence_pct": round(float(probs[i]) * 100, 1),
        }
        for i in top_indices
        if i in index_word
    ]

    return jsonify({"model": model_key, "predictions": predictions})


if __name__ == "__main__":
    app.run(port=5000, debug=True)