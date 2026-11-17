"""
Merge 3 slang datasets (Kaggle, HuggingFace, AI-generated) into one unified
text corpus for next-word-prediction / LSTM-RNN-GRU chatbot training.

Update the three input paths below, run it, and it drops a clean
unified_corpus.csv with a single 'text' column ready for tokenization.
"""

import re
import pandas as pd

# ---------- CONFIG: update these paths ----------
KAGGLE_CSV = "next text prediction/kaggle_slang.csv"        # Slang, Description, Example, Context
HF_CSV     = "next text prediction/huggingface_chat.csv"    # serial_number, text
AI_CSV     = "next text prediction/ai_generated_slang.csv"  # Word/Phrase, Definition, Example Sentence, Popularity/Trend Level
OUTPUT_CSV = "unified_corpus.csv"


def load_csv(path):
    """Auto-detect the delimiter (comma, tab, semicolon...) instead of
    assuming comma -- this is what was causing 0 rows: the files are
    tab-separated, so read_csv(path) alone was reading each row as ONE
    column instead of four, so row.get('Example') never matched anything."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()  # kill stray leading/trailing spaces in headers
    print(f"{path} -> columns: {list(df.columns)} | rows: {len(df)}")
    return df

# ---------- LOAD ----------
hf_df     = load_csv(HF_CSV)
ai_df     = load_csv(AI_CSV)
kaggle_df = load_csv(KAGGLE_CSV)

records = []

# HuggingFace file: "Example" is a natural sentence -> good training text.
# "Description" is also plain English -> keep it too, it adds more
# unique sentences to a corpus that's otherwise tiny.
for _, row in hf_df.iterrows():
    if pd.notna(row.get("Example")):
        records.append({"text": row["Example"], "source": "hf_example"})
    if pd.notna(row.get("Description")):
        records.append({"text": row["Description"], "source": "hf_definition"})

# AI-generated file: already raw chat-style sentences, use as-is.
for _, row in ai_df.iterrows():
    if pd.notna(row.get("text")):
        records.append({"text": row["text"], "source": "ai_generated"})

# Kaggle file: same shape as HuggingFace file, treat the same way.
for _, row in kaggle_df.iterrows():
    if pd.notna(row.get("Example Sentence")):
        records.append({"text": row["Example Sentence"], "source": "kaggle_example"})
    if pd.notna(row.get("Definition")):
        records.append({"text": row["Definition"], "source": "kaggle_definition"})

corpus = pd.DataFrame(records)
print(f"Raw combined rows: {len(corpus)}")

if corpus.empty:
    raise SystemExit(
        "No rows collected -- check the 'columns:' printout above against "
        "the column names used in the loops (e.g. 'Example', 'text', "
        "'Example Sentence'). A mismatch there (typo, different casing, "
        "extra space) means every row.get(...) returned None."
    )

# ---------- CLEANING ----------
def clean_text(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r"http\S+|www\.\S+", "", t)      # urls
    t = re.sub(r"@\w+", "", t)                  # @mentions
    t = re.sub(r"[\u201c\u201d]", '"', t)       # curly double quotes -> straight
    t = re.sub(r"[\u2018\u2019]", "'", t)       # curly single quotes -> straight
    t = re.sub(r"\s+", " ", t)                  # collapse whitespace
    t = t.strip(' "\'')                         # strip stray leading/trailing quote marks
    return t

corpus["text"] = corpus["text"].apply(clean_text)

# drop empty / too-short rows (not useful sequences to learn from)
corpus = corpus[corpus["text"].str.len() > 2]

# drop exact duplicates
corpus = corpus.drop_duplicates(subset="text")

# drop near-duplicates that only differ by case
corpus["_lower"] = corpus["text"].str.lower()
corpus = corpus.drop_duplicates(subset="_lower").drop(columns="_lower")

corpus = corpus.reset_index(drop=True)
print(f"After cleaning: {len(corpus)} rows")

corpus.to_csv(OUTPUT_CSV, index=False)
print(f"Saved unified corpus -> {OUTPUT_CSV}")