import pandas as pd
import random

random.seed(42)

df = pd.read_csv("comments_all_clean.csv")

personen = ["ireen", "femke", "thomas", "sven", "jutta", "kjeld"]

samples = []
for persoon in personen:
    subset = df[df["persoon"] == persoon].dropna(subset=["comment"])
    drawn = subset.sample(n=min(10, len(subset)), random_state=42)
    samples.append(drawn[["persoon", "gender", "subreddit", "comment"]])

sample_df = pd.concat(samples, ignore_index=True)

# Columns for labelling: one manual + three model slots
sample_df["manual_label"] = ""
sample_df["model_deberta"] = ""
sample_df["model_gpt"] = ""
sample_df["model_claude"] = ""

import re

def clean_for_excel(val):
    if isinstance(val, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', val)
    return val

sample_df = sample_df.map(clean_for_excel)
sample_df.to_excel("sample_relevance_labels.xlsx", index=False)

print(f"Saved {len(sample_df)} comments to sample_relevance_labels.xlsx")
print(sample_df["persoon"].value_counts())
