from transformers import pipeline
import pandas as pd
import re

FULL_NAMES = {
    "sven":   ("Sven Kramer",   "he/him"),
    "jutta":  ("Jutta Leerdam", "she/her"),
    "femke":  ("Femke Kok",     "she/her"),
    "kjeld":  ("Kjeld Nuis",    "he/him"),
    "thomas": ("Thomas Krol",   "he/him"),
    "ireen":  ("Ireen Wüst",    "she/her"),
}

IRRELEVANT_SUBS = {
    "soccer", "fcbayern", "coys", "borussiadortmund", "Bundesliga",
    "feyenoord", "MCFC", "schalke04", "OlympiqueLyonnais", "futebol",
    "Eredivisie", "worldcup", "fcsp",
    "baseball", "minnesotatwins", "angelsbaseball", "Braves",
    "TOP5LEAGUESFM24SAVE", "FifaCareers", "WEPES", "UMF26",
    "indieheads", "byzantium", "Games", "DCcomics", "Wolverine",
    "CrusaderKings", "Magic", "oscarrace", "BareKnuckleFC",
    "generationology", "conspiracy",
}

print("Model laden...")
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
)
print("Model geladen!")

df = pd.read_excel("sample_relevance_labels.xlsx")
df["model_deberta"] = df["model_deberta"].astype(object)

def clean(val):
    if isinstance(val, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', val)
    return val

def label_comment(comment, persoon, subreddit):
    full_name, pronoun = FULL_NAMES[persoon]
    comment_str = clean(str(comment))[:500]

    if subreddit in IRRELEVANT_SUBS:
        candidate_labels = [
            f"this comment evaluates or describes {full_name}, Dutch speed skater ({pronoun})",
            f"this comment is not about {full_name} or contains no evaluation of {full_name}"
        ]
    else:
        candidate_labels = [
            f"this comment evaluates or describes {persoon} ({pronoun})",
            f"this comment is not about {persoon} or contains no evaluation"
        ]

    try:
        result = classifier(comment_str, candidate_labels, hypothesis_template="{}.")
        best_label = result["labels"][0]
        score = result["scores"][0]

        if score < 0.55:
            return "unclear"
        elif "evaluates or describes" in best_label:
            return "relevant"
        else:
            return "not_relevant"
    except Exception as e:
        print(f"  Fout bij rij: {e}")
        return "unknown"

for i, row in df.iterrows():
    label = label_comment(row["comment"], row["persoon"], row["subreddit"])
    df.at[i, "model_deberta"] = label
    print(f"[{i+1}/60] {row['persoon']:8} | {label:12} | {str(row['comment'])[:60]}")

df.to_excel("sample_relevance_labels.xlsx", index=False)
print("\nOpgeslagen: sample_relevance_labels.xlsx")
print("\nDeberta label verdeling:")
print(df["model_deberta"].value_counts())
