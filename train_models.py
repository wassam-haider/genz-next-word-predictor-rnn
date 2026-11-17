"""
Train SimpleRNN, LSTM, GRU, and Bidirectional(LSTM) one by one on the same
sliding-window/n-gram next-word-prediction sequences, then build a
model x metric comparison matrix.

pip install tensorflow pandas scikit-learn
"""

import time
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

CORPUS_CSV    = "unified_corpus.csv"
MAX_VOCAB     = 8000
EMBED_DIM     = 100
HIDDEN_UNITS  = 128
DROPOUT_RATE  = 0.3     # NEW: regularization to fight the overfitting seen in the logs
EPOCHS        = 40      # ceiling -- EarlyStopping below will usually stop sooner
BATCH_SIZE    = 128
PATIENCE      = 3       # lowered from 8: with val_acc peaking ~epoch 13 and val_loss
                         # rising every epoch after ~epoch 4, patience=8 let it run
                         # ~8 pointless extra epochs of pure overfitting
SEED          = 42

# Fix all sources of run-to-run randomness so a 4-model comparison is
# actually comparing architectures, not comparing random seeds.
tf.keras.utils.set_random_seed(SEED)

# ---------- 1. LOAD + SPLIT SENTENCES ----------
df = pd.read_csv(CORPUS_CSV)
texts = df["text"].astype(str).tolist()

n_before = len(texts)
texts = list(dict.fromkeys(texts))  # de-dupe exact-duplicate sentences, order-preserving
n_after = len(texts)
if n_after < n_before:
    print(f"Removed {n_before - n_after} exact-duplicate rows from the corpus "
          f"({n_before} -> {n_after}). Duplicate sentences split across "
          f"train/val are a leakage source, so this must happen before the split.")

# Split BEFORE fitting the tokenizer, not just before windowing. Fitting
# on the full corpus lets validation-only words get a real vocab slot,
# which is a second (smaller) leakage source: at real inference time an
# unseen word should fall back to <UNK>, so validation needs to see that
# same behavior, not know words in advance because they showed up there.
train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=SEED)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
tokenizer.fit_on_texts(train_texts)
vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ---------- 2. SLIDING-WINDOW SEQUENCES, BUILT SEPARATELY PER SPLIT ----------
# Windowing must also happen per-split, not before splitting: overlapping
# prefixes of the same sentence (e.g. [i, like] and [i, like, slow]) must
# not land on opposite sides of the split, or val_accuracy gets inflated
# by near-duplicate examples the model has effectively already seen.
def make_sequences(text_list):
    seqs = []
    for line in text_list:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            seqs.append(token_list[: i + 1])
    return seqs


train_sequences = make_sequences(train_texts)
val_sequences = make_sequences(val_texts)

# max_seq_len is a preprocessing/architecture constant. It must be decided
# from TRAIN data only -- computing it from train+val combined lets the
# model's input shape depend on validation-set content, which is a leakage
# channel (at real inference time you don't get to peek at the deployment
# set to decide your architecture). We use a high percentile of TRAIN
# sequence lengths rather than the raw max, so one freak long training
# sentence can't blow up padding length for every other example. Anything
# longer than this cap (train, val, or at inference) gets truncated from
# the left by pad_sequences(truncating="pre"), which keeps the most recent
# context -- the part most relevant to predicting the next word.
train_lengths = [len(seq) for seq in train_sequences]
max_seq_len = int(np.percentile(train_lengths, 99))
max_seq_len = max(max_seq_len, 2)  # guard against a degenerate corpus

train_sequences = pad_sequences(
    train_sequences, maxlen=max_seq_len, padding="pre", truncating="pre"
)
val_sequences = pad_sequences(
    val_sequences, maxlen=max_seq_len, padding="pre", truncating="pre"
)

X_train, y_train = train_sequences[:, :-1], train_sequences[:, -1]
X_val, y_val = val_sequences[:, :-1], val_sequences[:, -1]

print(
    f"Vocab size: {vocab_size} | Train sequences: {len(train_sequences)} | "
    f"Val sequences: {len(val_sequences)} | Max len (99th pct of train): {max_seq_len}"
)

with open("model_config.json", "w") as f:
    json.dump({"max_seq_len": max_seq_len, "vocab_size": vocab_size}, f)

# ---------- 3. MODEL BUILDER ----------
def build_model(kind: str):
    model = Sequential(name=kind)
    model.add(Embedding(vocab_size, EMBED_DIM, input_length=max_seq_len - 1))
    if kind == "rnn":
        model.add(SimpleRNN(HIDDEN_UNITS))
    elif kind == "lstm":
        model.add(LSTM(HIDDEN_UNITS))
    elif kind == "gru":
        model.add(GRU(HIDDEN_UNITS))
    elif kind == "bidirectional":
        model.add(Bidirectional(LSTM(HIDDEN_UNITS)))
    else:
        raise ValueError(f"unknown model kind: {kind}")
    # NEW: dropout before the softmax head. The training logs showed train
    # accuracy climbing to ~49% while val_accuracy plateaued at ~15.7% and
    # val_loss rose every epoch after epoch 4 -- classic overfitting for a
    # 128-unit recurrent layer on this corpus size. Dropout won't eliminate
    # the gap but should narrow it and let training run longer before
    # val_loss turns upward.
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# ---------- 4. TRAIN ONE BY ONE ----------
results = {}
for kind in ["rnn", "lstm", "gru", "bidirectional"]:
    print(f"\n{'=' * 20} Training {kind.upper()} {'=' * 20}")
    model = build_model(kind)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", mode="max", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(f"{kind}_model_best.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start
    print(f"Stopped after {len(history.history['loss'])} epochs (ceiling was {EPOCHS})")

    # FIX: EarlyStopping(restore_best_weights=True) restores the model's
    # WEIGHTS to the best epoch, but history.history[...][-1] is still the
    # LAST epoch's numbers. If training runs several patience epochs past
    # the peak, "final" metrics silently describe an epoch that is no
    # longer what's actually saved. Report the metrics from the best
    # epoch instead, so the numbers in comparison_matrix.csv match the
    # weights that get saved to disk.
    best_idx = int(np.argmax(history.history["val_accuracy"]))
    best_train_loss = history.history["loss"][best_idx]
    best_val_loss = history.history["val_loss"][best_idx]
    best_train_acc = history.history["accuracy"][best_idx]
    best_val_acc = history.history["val_accuracy"][best_idx]

    results[kind] = {
        "best_epoch": best_idx + 1,
        "epochs_run": len(history.history["loss"]),
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "train_accuracy": best_train_acc,
        "val_accuracy": best_val_acc,
        "train_perplexity": float(np.exp(best_train_loss)),
        "val_perplexity": float(np.exp(best_val_loss)),
        "trainable_params": model.count_params(),
        "training_time_sec": round(elapsed, 1),
    }

    # model already holds the restored best weights (restore_best_weights=True)
    model.save(f"{kind}_model.keras")
    print(f"Saved {kind}_model.keras (best epoch {best_idx + 1}/{len(history.history['loss'])})")

# ---------- 5. COMPARISON MATRIX ----------
# rows = metrics, columns = model (rnn / lstm / gru / bidirectional)
comparison = pd.DataFrame(results)
comparison.to_csv("comparison_matrix.csv")
print("\nComparison matrix (rows = metrics, columns = model):")
print(comparison)