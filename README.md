# Genz Next-Word Predictor (RNN)

A next-word prediction tool trained on Gen-Z slang and casual chat text,
comparing four sequence model architectures head-to-head: SimpleRNN, LSTM,
GRU, and Bidirectional LSTM. Includes a Flask API and a browser demo where
you can switch models live and see top-k word suggestions with confidence
scores.

## What it does

Type a partial sentence (e.g. slang like "no cap that's") and the model
predicts the most likely next word(s), the same way a keyboard's predictive
text works — just trained on a Gen-Z/slang corpus instead of generic text.

## Pipeline

1. **`next text prediction/buildcorpus.py`** — merges three slang datasets
   (Kaggle, HuggingFace, and an AI-generated slang set) into one cleaned
   `unified_corpus.csv`, auto-detecting delimiters and deduplicating rows.
2. **`train_models.py`** — builds sliding-window/n-gram training sequences
   from that corpus and trains all four architectures (RNN, LSTM, GRU,
   Bidirectional LSTM) with identical hyperparameters so the comparison is
   fair. Specifically guards against several data-leakage pitfalls:
   - splits train/val **before** fitting the tokenizer, so validation words
     can't get real vocab slots they wouldn't have at real inference time
   - splits **before** building sliding windows, so overlapping prefixes of
     the same sentence never land on both sides of the split
   - picks max sequence length from train data only (99th percentile, not
     a hard max) to avoid one long sentence blowing up padding everywhere
   - uses `EarlyStopping(restore_best_weights=True)` per model
   - see `linkedin-post-data-leakage.md` for the full write-up of this
3. **`serve.py`** — Flask API that loads all four trained `.keras` models
   and returns top-k next-word predictions with confidence percentages for
   whichever model the frontend selects.
4. **`index.html`** — browser demo UI with a model-selector dropdown.

## Results (`comparison_matrix.csv`)

| | RNN | LSTM | GRU | Bidirectional |
|---|---|---|---|---|
| Best epoch | 13 | 23 | 14 | 15 |
| Val accuracy | 15.1% | 16.0% | **16.1%** | 15.9% |
| Val perplexity | 483.4 | 722.8 | **458.7** | 536.6 |
| Trainable params | 1.79M | 1.88M | 1.85M | 2.98M |
| Training time | 48.7s | 126.6s | 78.7s | 124.2s |

GRU came out ahead on both validation accuracy and perplexity while
training faster than LSTM or Bidirectional LSTM — plain SimpleRNN is the
weakest but cheapest to train.

## Running it

```bash
pip install tensorflow pandas scikit-learn flask flask-cors

# 1. build the corpus (optional — unified_corpus.csv is already included)
python "next text prediction/buildcorpus.py"

# 2. train all four models (optional — trained .keras models are included)
python train_models.py

# 3. serve predictions
python serve.py
# open index.html in a browser
```

## Tech stack

Python, TensorFlow/Keras, scikit-learn, pandas, Flask
