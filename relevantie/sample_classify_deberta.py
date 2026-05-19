"""
Run classify_deberta4 logic on the existing 16-per-skater sample
(sample_16_per_skater.csv) and save results to sample_deberta_labeled.xlsx.
"""

import re
import os
from pathlib import Path

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

_HERE       = Path(__file__).parent
SAMPLE_CSV  = _HERE / "sample_16_per_skater.csv"
OUTPUT_XLSX = _HERE / "sample_deberta_labeled.xlsx"

THRESHOLD = 0.55

_illegal = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def clean(text):
    if isinstance(text, str):
        return _illegal.sub("", text)
    return str(text)

def classify_row(row, classifier):
    comment   = clean(str(row["comment"]))[:400]
    subreddit = str(row["subreddit"])
    thread    = clean(str(row["thread_title"]))[:100]
    persoon   = str(row["persoon"])

    context = f"[Subreddit: r/{subreddit} | Thread: {thread}]\nComment: {comment}"

    result    = classifier(
        context,
        [
            f"this comment is about {persoon} as an athlete or person",
            f"this comment is not about {persoon}",
        ],
        hypothesis_template="{}.",
    )
    top_label = result["labels"][0]
    top_score = round(result["scores"][0], 4)

    if top_score < THRESHOLD:
        label = "NOT_RELEVANT"
    else:
        label = "RELEVANT" if top_label.startswith("this comment is about") else "NOT_RELEVANT"

    return label, top_score


def main():
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"{SAMPLE_CSV} not found — run sample_and_classify.py first.")

    df = pd.read_csv(SAMPLE_CSV)
    print(f"Loaded {len(df)} rows from {SAMPLE_CSV.name}")
    print(df["persoon"].value_counts().sort_index().to_string())

    print("\nLoading DeBERTa model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        device="mps",
    )
    print("Model loaded. Classifying...\n")

    labels, scores = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            label, score = classify_row(row, classifier)
        except Exception as e:
            print(f"  Error on {row['comment_id']}: {e}")
            label, score = "ERROR", None
        labels.append(label)
        scores.append(score)

    df["label"] = labels
    df["score"] = scores

    # Strip illegal Excel characters
    out_df = df.apply(
        lambda col: col.map(lambda v: _illegal.sub("", v) if isinstance(v, str) else v)
    )

    out_df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"\nDone. Saved to {OUTPUT_XLSX.name}")
    print("\nLabel counts:")
    print(out_df["label"].value_counts(dropna=False))
    print("\nBy skater:")
    print(out_df.groupby("persoon")["label"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
