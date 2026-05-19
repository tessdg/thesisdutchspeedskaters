from transformers import pipeline
import pandas as pd
import re
import os
from tqdm import tqdm

_HERE      = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(_HERE, "..", "comments_all_clean.csv")
OUTPUT_CSV = os.path.join(_HERE, "comments_classified_deberta.csv")

CHECKPOINT_EVERY = 50
ROW_START = None
ROW_END   = None

THRESHOLD = 0.55

SUBREDDIT_ALLOWLIST = {
    "Speedskating", "speedskatebabes", "JuttaLeerdam", "WinterOlympics2026",
    "MilanOlympics", "OlympicsV2", "olympics", "OlympicBornToday",
    "BeautifulOlympians", "HottestWinterAthletes", "dutchwomanathletes",
    "sports", "nextfuckinglevel", "Damnthatsinteresting", "interestingasfuck",
    "gifs", "coolguides", "Infographics",
    "NUjijDiscussies", "Nederland", "thenetherlands", "Netherlands",
    "Politiek", "PolitiekeMemes", "cirkeltrek", "taalfout", "DeStagiair",
    "tokkiefeesboek", "papgrappen", "HLNFails", "NLCelebs",
    "LAinfluencersnark", "sportsgossips", "ladyladyboners",
    "HottestFemaleAthletes", "ik_ihe", "pics",
}

GENERAL_SUBREDDITS = {
    "coolguides", "Infographics", "Damnthatsinteresting",
    "interestingasfuck", "nextfuckinglevel", "gifs", "sports",
}


def clean(text):
    if isinstance(text, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
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
    print("Loading DeBERTa model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        device="mps",
    )
    print("Model loaded.")

    df = pd.read_csv(INPUT_CSV)
    df = df[df["subreddit"].isin(SUBREDDIT_ALLOWLIST)]

    general_mask  = df["subreddit"].isin(GENERAL_SUBREDDITS)
    name_mentioned = df.apply(
        lambda r: str(r["persoon"]).lower() in str(r["comment"]).lower(), axis=1
    )
    df = df[~general_mask | name_mentioned]

    if ROW_START is not None or ROW_END is not None:
        df = df.iloc[ROW_START:ROW_END]

    print(f"Rows after subreddit filter: {len(df)}")

    if os.path.exists(OUTPUT_CSV):
        done_df  = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["comment_id"].astype(str))
        print(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining.")
    else:
        done_df  = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["comment_id"].astype(str).isin(done_ids)].copy()

    if remaining.empty:
        print("All comments already classified.")
        return

    results = []

    def checkpoint():
        if not results:
            return
        scores_df  = pd.DataFrame(results).drop_duplicates(subset="comment_id", keep="last")
        mapping    = scores_df.set_index("comment_id")[["label", "score"]].to_dict("index")
        scored_ids = set(scores_df["comment_id"].astype(str))
        batch      = remaining[remaining["comment_id"].astype(str).isin(scored_ids)].copy()
        batch[["label", "score"]] = batch["comment_id"].astype(str).map(mapping).apply(pd.Series)
        combined   = pd.concat([done_df, batch], ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False)

    for i, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining), desc="Classifying")):
        try:
            label, score = classify_row(row, classifier)
        except Exception as e:
            print(f"  Error on {row['comment_id']}: {e}")
            label, score = "ERROR", None

        results.append({
            "comment_id": row["comment_id"],
            "label":      label,
            "score":      score,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint()

    # Final save
    scores_df = pd.DataFrame(results).drop_duplicates(subset="comment_id", keep="last")
    mapping   = scores_df.set_index("comment_id")[["label", "score"]].to_dict("index")
    remaining[["label", "score"]] = remaining["comment_id"].astype(str).map(mapping).apply(pd.Series)
    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")

    print("\nLabel counts:")
    print(final["label"].value_counts())
    print("\nBy skater:")
    print(final.groupby("persoon")["label"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
