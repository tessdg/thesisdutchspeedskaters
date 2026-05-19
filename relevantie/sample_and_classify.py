"""
Sample 16 comments per skater from comments_all_clean.csv and classify
each one as RELEVANT / NOT_RELEVANT using the 2opus classify logic.

Output: sample_labeled.csv  (all columns from clean CSV + label/confidence/reason)
"""

import json
import time
import re
import sys
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---- CONFIG -----------------------------------------------------------------
_HERE        = Path(__file__).parent
INPUT_CSV    = _HERE / ".." / "comments_all_clean.csv"
OUTPUT_XLSX  = _HERE / "sample_labeled.xlsx"
SAMPLE_CSV   = _HERE / "sample_16_per_skater.csv"

SAMPLE_N     = 16            # comments per skater
RANDOM_SEED  = 42

MODEL        = "qwen2.5:7b-instruct"
OLLAMA_URL   = "http://localhost:11434/api/generate"
MAX_RETRIES  = 2
REQUEST_TIMEOUT = 120

SKATER_FULL = {
    "sven":   "Sven Kramer (Dutch long-track speed skater)",
    "ireen":  "Ireen Wüst (Dutch long-track speed skater)",
    "jutta":  "Jutta Leerdam (Dutch long-track speed skater)",
    "kjeld":  "Kjeld Nuis (Dutch long-track speed skater)",
    "thomas": "Thomas Krol (Dutch long-track speed skater) — NOT Thomas Müller the footballer",
    "femke":  "Femke Kok (Dutch long-track speed skater)",
}

# ---- PROMPT (identical to 2opus classify) -----------------------------------
SYSTEM = """You are a research assistant labelling Reddit comments for a thesis on Dutch speed skaters.

For each comment, decide if it is RELEVANT to the TARGET SKATER named in the input.

A comment is RELEVANT if it is about:
  (a) the target skater as a person (personality, looks, behaviour, private life, fans, relationships), OR
  (b) the target skater's sport performance (a race, training, result, competition, career, technique).

IMPORTANT RULE — THREAD CONTEXT:
The thread title tells you what the thread is about. If the thread is clearly about the target skater (e.g. "Femke Kok wins Gold"), then a comment posted in that thread is RELEVANT when it is reacting to that topic — even when:
  - the comment is very short ("Mooi", "Insane run", "Wow", "She is incredible")
  - the comment does NOT repeat the skater's name
  - the comment mentions OTHER people for comparison ("reminds me of X's race")
  - the comment uses pronouns ("she", "her", "he")
  - the comment expresses an emotion or reaction without explanation

A comment is NOT_RELEVANT only when it is clearly off-topic from the thread (random side-conversation, sub-thread bickering, jokes unconnected to the skater, posts about a totally different person).

A comment is NOT_RELEVANT if it is PRIMARILY about:
  - a different person who happens to share a name (very common: footballers, other athletes)
  - unrelated topics, sub-thread arguments between two users
  - the subreddit topic in general but not this skater (e.g. football in r/soccer)

Mentioning another person for comparison does NOT make a comment irrelevant. What matters is whether the comment is RESPONDING TO or ABOUT the target skater in context.

Output ONLY a JSON object, no other text:
{"label": "RELEVANT" or "NOT_RELEVANT", "confidence": "high" or "medium" or "low", "reason": "short phrase, max 12 words"}
"""

FEW_SHOT = """Here are some examples of correct labelling.

Example 1 (short reaction in clearly-about-skater thread → RELEVANT):
Target skater: Femke Kok (Dutch long-track speed skater)
Thread title: Femke Kok wins Gold for Netherlands in Women's 500m Speed Skating
Subreddit: r/olympics
Comment: Mooi
JSON: {"label": "RELEVANT", "confidence": "high", "reason": "positive reaction to her gold-medal race"}

Example 2 (mentions another skater for comparison but praises target's race → RELEVANT):
Target skater: Femke Kok (Dutch long-track speed skater)
Thread title: Jutta Leerdam verovert olympisch goud op 1.000 meter, zilver voor Femke Kok
Subreddit: r/thenetherlands
Comment: Wat een mooie race! Geweldige TV. Doet me aan Gerard van Velde zijn gouden race denken.
JSON: {"label": "RELEVANT", "confidence": "high", "reason": "praises the race; mentions van Velde only as comparison"}

Example 3 (name collision, comment is about a different person → NOT_RELEVANT):
Target skater: Thomas Krol (Dutch long-track speed skater)
Thread title: Match Thread: Bayern Munich vs Borussia Monchengladbach
Subreddit: r/soccer
Comment: Bayern is completely different team without Muller and Lewandowski
JSON: {"label": "NOT_RELEVANT", "confidence": "high", "reason": "about footballer Thomas Müller, not Thomas Krol"}

Example 4 (sub-thread argument unrelated to skater → NOT_RELEVANT):
Target skater: Femke Kok (Dutch long-track speed skater)
Thread title: Jutta Leerdam verovert olympisch goud op 1.000 meter, zilver voor Femke Kok
Subreddit: r/thenetherlands
Comment: Begrijpend lezen blijkt bij deze voor mij te hoog gegrepen zijn haha
JSON: {"label": "NOT_RELEVANT", "confidence": "high", "reason": "sub-thread bickering about reading comprehension"}

Example 5 (short emotional reaction in skater's gif thread → RELEVANT):
Target skater: Jutta Leerdam (Dutch long-track speed skater)
Thread title: Jutta leerdam Dutch speed skater
Subreddit: r/gifs
Comment: I think I just feel in love
JSON: {"label": "RELEVANT", "confidence": "high", "reason": "fan reaction to the skater"}

Now label the next comment.
"""

USER_TEMPLATE = """Target skater: {skater}
Thread title: {thread}
Subreddit: r/{subreddit}
Comment: {comment}

JSON:"""

# ---- OLLAMA -----------------------------------------------------------------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
        "format": "json",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"]

def parse_response(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from: {text[:200]}")

def classify_row(row: dict) -> dict:
    skater  = SKATER_FULL.get(row["persoon"], row["persoon"])
    comment = str(row["comment"])[:1500]
    thread  = str(row["thread_title"])[:300]
    prompt  = FEW_SHOT + USER_TEMPLATE.format(
        skater=skater, thread=thread,
        subreddit=row["subreddit"], comment=comment,
    )
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw    = call_ollama(prompt)
            parsed = parse_response(raw)
            label  = str(parsed.get("label", "")).upper().strip()
            if label not in ("RELEVANT", "NOT_RELEVANT"):
                raise ValueError(f"Bad label: {label!r}")
            return {
                "label":      label,
                "confidence": str(parsed.get("confidence", "")).lower(),
                "reason":     str(parsed.get("reason", ""))[:200],
                "raw":        raw[:300],
            }
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    return {"label": "ERROR", "confidence": "", "reason": f"parse_fail: {last_err}"[:200], "raw": ""}

# ---- MAIN -------------------------------------------------------------------
def main():
    try:
        requests.get("http://localhost:11434/", timeout=3)
    except Exception:
        sys.exit("Ollama is not reachable at localhost:11434. Run `ollama serve` first.")

    # Reuse existing sample if available, so results are reproducible
    if SAMPLE_CSV.exists():
        sample = pd.read_csv(SAMPLE_CSV)
        print(f"Loaded existing sample from {SAMPLE_CSV.name} ({len(sample)} rows)")
    else:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df):,} comments from {INPUT_CSV.name}")
        sample = pd.concat([
            g.sample(min(SAMPLE_N, len(g)), random_state=RANDOM_SEED)
            for _, g in df.groupby("persoon")
        ]).reset_index(drop=True)
        sample.to_csv(SAMPLE_CSV, index=False)
        print(f"Sample saved → {SAMPLE_CSV.name}")

    print(sample["persoon"].value_counts().sort_index().to_string())
    print(f"\nTotal: {len(sample)} comments — starting classification with {MODEL}...\n")

    results = []
    for row_dict in tqdm(sample.to_dict("records"), total=len(sample)):
        out = classify_row(row_dict)
        results.append({**row_dict, **out})

    out_df = pd.DataFrame(results)

    # Strip illegal XML/Excel control characters from all string columns
    import re as _re
    _illegal = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    out_df = out_df.apply(
        lambda col: col.map(lambda v: _illegal.sub("", v) if isinstance(v, str) else v)
    )

    out_df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"\nDone. Wrote {OUTPUT_XLSX.name}")
    print("\nLabel counts:")
    print(out_df["label"].value_counts(dropna=False))
    print("\nLabel by skater:")
    print(out_df.groupby("persoon")["label"].value_counts().unstack(fill_value=0))

if __name__ == "__main__":
    main()
