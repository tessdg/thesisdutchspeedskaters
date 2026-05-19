import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(_HERE, "..", "comments_all_clean.csv")
OUTPUT_CSV = os.path.join(_HERE, "comments_classified_hybrid.csv")
CHECKPOINT_EVERY = 50
ROW_START = None
ROW_END = None

SUBREDDIT_ALLOWLIST = {
    # Speed skating & Olympics
    "Speedskating", "speedskatebabes", "JuttaLeerdam", "WinterOlympics2026",
    "MilanOlympics", "OlympicsV2", "olympics", "OlympicBornToday",
    "BeautifulOlympians", "HottestWinterAthletes", "dutchwomanathletes",
    "sports", "nextfuckinglevel", "Damnthatsinteresting", "interestingasfuck",
    "gifs", "coolguides", "Infographics",
    # Dutch media & culture
    "NUjijDiscussies", "Nederland", "thenetherlands", "Netherlands",
    "Politiek", "PolitiekeMemes", "cirkeltrek", "taalfout", "DeStagiair",
    "tokkiefeesboek", "papgrappen", "HLNFails", "NLCelebs",
    # Grey area: gossip & attractiveness
    "LAinfluencersnark", "sportsgossips", "ladyladyboners",
    "HottestFemaleAthletes", "ik_ihe", "pics",
}

GENERAL_SUBREDDITS = {
    "coolguides", "Infographics", "Damnthatsinteresting",
    "interestingasfuck", "nextfuckinglevel", "gifs", "sports",
}


def classify_sentiment(classifier, comment_id, comment):
    try:
        out = classifier(comment[:512], truncation=True)[0]
        label = out["label"].lower()
        return comment_id, label == "negative", label, round(out["score"], 4)
    except Exception:
        return comment_id, None, None, None


def main():
    print(f"Loading sentiment model: {SENTIMENT_MODEL}")
    classifier = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    df = pd.read_csv(INPUT_CSV)
    df = df[df["subreddit"].isin(SUBREDDIT_ALLOWLIST)]

    general_mask = df["subreddit"].isin(GENERAL_SUBREDDITS)
    name_mentioned = df.apply(
        lambda r: str(r["persoon"]).lower() in str(r["comment"]).lower(), axis=1
    )
    df = df[~general_mask | name_mentioned]

    if ROW_START is not None or ROW_END is not None:
        df = df.iloc[ROW_START:ROW_END]

    print(f"Rows after subreddit filter: {len(df)}")

    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["comment_id"].astype(str))
        print(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["comment_id"].astype(str).isin(done_ids)].copy()

    if remaining.empty:
        print("All comments already classified.")
        return

    results = []

    def checkpoint():
        scores = pd.DataFrame(results).drop_duplicates(subset="comment_id", keep="last")
        scored_ids = set(scores["comment_id"].astype(str))
        batch = remaining[remaining["comment_id"].astype(str).isin(scored_ids)].copy()
        mapping = scores.set_index("comment_id")[["negative", "sentiment_label", "sentiment_score"]].to_dict("index")
        batch[["negative", "sentiment_label", "sentiment_score"]] = (
            batch["comment_id"].astype(str).map(mapping).apply(pd.Series)
        )
        pd.concat([done_df, batch], ignore_index=True).to_csv(OUTPUT_CSV, index=False)

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Classifying"):
        comment_id, negative, label, score = classify_sentiment(
            classifier, row["comment_id"], str(row["comment"])
        )
        results.append({
            "comment_id": comment_id,
            "negative": negative,
            "sentiment_label": label,
            "sentiment_score": score,
        })
        if len(results) % CHECKPOINT_EVERY == 0:
            checkpoint()

    scores = pd.DataFrame(results).drop_duplicates(subset="comment_id", keep="last")
    mapping = scores.set_index("comment_id")[["negative", "sentiment_label", "sentiment_score"]].to_dict("index")
    remaining[["negative", "sentiment_label", "sentiment_score"]] = (
        remaining["comment_id"].astype(str).map(mapping).apply(pd.Series)
    )

    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")

    negative = final[final["negative"] == True]
    print(f"\nNegative comments: {len(negative)} / {len(final)} ({len(negative)/len(final)*100:.1f}%)")
    print("\nBy gender:")
    print(negative["gender"].value_counts())
    print("\nNegative rate by gender:")
    for gender, group in final.groupby("gender"):
        neg_count = int(group["negative"].sum())
        print(f"  {gender}: {neg_count}/{len(group)} ({neg_count/len(group)*100:.1f}%)")


if __name__ == "__main__":
    main()
