"""
Add plain/"normal" everyday sentences (DailyDialog) to unified_corpus.csv
to balance out the GenZ-slang rows you already collected.

pip install datasets pandas   (before running)
"""

import re
import pandas as pd
from datasets import load_dataset

EXISTING_CORPUS = "unified_corpus.csv"
OUTPUT_CSV      = "unified_corpus.csv"   # overwrite with the merged version
TARGET_NEW_ROWS = 5432                   # the 55% "normal sentence" gap you need

# ---------- LOAD DAILYDIALOG ----------
# NOTE: some daily_dialog mirrors (e.g. roskoN/dailydialog) still ship a
# legacy .py loading script, which recent `datasets` versions refuse to
# run at all (RuntimeError: "Dataset scripts are no longer supported").
# elricwan/dailydialog is stored as plain Parquet, so it loads with no
# script and no code-execution warning.
dd = load_dataset("elricwan/dailydialog", split="train")

# each row's "conversation" field is a list of turns like
# "Person A: Could I have my bill , please ?" -> strip the speaker label
# and flatten to one sentence per row
speaker_prefix = re.compile(r"^\s*(Person [A-Za-z]|[A-Za-z])\s*:\s*")

sentences = []
for row in dd:
    for utt in row["conversation"]:
        sentences.append(speaker_prefix.sub("", utt))

print(f"Flattened DailyDialog utterances: {len(sentences)}")

# ---------- CLEAN (same rules as build_corpus.py) ----------
def clean_text(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[\u201c\u201d]", '"', t)
    t = re.sub(r"[\u2018\u2019]", "'", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(' "\'')
    return t

new_df = pd.DataFrame({"text": [clean_text(s) for s in sentences]})
new_df["source"] = "dailydialog_normal"

# drop empty/too-short/too-long (keep it comparable in length to your slang examples)
new_df = new_df[(new_df["text"].str.len() > 2) & (new_df["text"].str.split().str.len() <= 25)]

# dedupe within itself
new_df = new_df.drop_duplicates(subset="text")
new_df["_lower"] = new_df["text"].str.lower()
new_df = new_df.drop_duplicates(subset="_lower")

# ---------- DEDUPE AGAINST EXISTING CORPUS ----------
existing = pd.read_csv(EXISTING_CORPUS)
existing_lower = set(existing["text"].str.lower())
new_df = new_df[~new_df["_lower"].isin(existing_lower)].drop(columns="_lower")

print(f"New unique normal-sentence rows available: {len(new_df)}")

# ---------- SAMPLE TO TARGET AND MERGE ----------
if len(new_df) < TARGET_NEW_ROWS:
    print(f"WARNING: only {len(new_df)} available, using all of them "
          f"(short by {TARGET_NEW_ROWS - len(new_df)}). Consider adding "
          f"Cornell Movie-Dialogs or Empathetic Dialogues to fill the rest.")
    sampled = new_df
else:
    sampled = new_df.sample(n=TARGET_NEW_ROWS, random_state=42)

merged = pd.concat([existing, sampled], ignore_index=True)
merged.to_csv(OUTPUT_CSV, index=False)

print(f"Final corpus size: {len(merged)} rows")
print(merged["source"].value_counts())