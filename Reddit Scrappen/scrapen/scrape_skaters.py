"""
Scrape Reddit comments about six Dutch ice skaters.

TWO-TIER STRATEGY
─────────────────
Tier 1 — Dedicated threads (title contains skater's full name):
    Scrape ALL comments. The whole thread is about the skater, so every
    comment captures public reaction (personal, professional, ice skating).

Tier 2 — Mention threads (search returned them, but no name in title):
    Only keep the specific comments that name the skater. This captures
    mentions in Olympics megathreads, sports roundups, etc. without pulling
    in thousands of irrelevant football/general comments.

NAME MATCHING per skater
────────────────────────
Ambiguous last names ("Kok", "Kramer") require the FULL name in Tier 2.
Unique last names ("Wüst", "Leerdam", "Nuis", "Krol") match on last name alone.

No Reddit API credentials needed — uses the public JSON API.
Rate limited to ~1 request/2 seconds to be polite to Reddit.
"""

import requests
import time
import csv
from datetime import datetime, timezone
from collections import Counter

# ── Skater definitions ────────────────────────────────────────────────────────

SKATERS = [
    {
        "naam":           "ireen",
        "full_name":      "Ireen Wüst",
        # Reddit search queries (use multiple to catch ü encoding variations)
        "search_queries": ["Ireen Wust", "Ireen Wüst"],
        # Tier 1: any of these in the thread TITLE → scrape all comments
        "title_variants": ["ireen wüst", "ireen wust"],
        # Tier 2: any of these in the COMMENT TEXT → keep that comment
        # "wüst"/"wust" is unique enough to match alone
        "comment_match":  ["ireen wüst", "ireen wust", "wüst", "wust"],
        "gender":         "vrouw",
    },
    {
        "naam":           "femke",
        "full_name":      "Femke Kok",
        "search_queries": ["Femke Kok"],
        "title_variants": ["femke kok"],
        # "kok" alone is too ambiguous (common Dutch word) — require full name
        "comment_match":  ["femke kok"],
        "gender":         "vrouw",
    },
    {
        "naam":           "thomas",
        "full_name":      "Thomas Krol",
        "search_queries": ["Thomas Krol"],
        "title_variants": ["thomas krol"],
        # "krol" is fairly unique in a skating context
        "comment_match":  ["thomas krol", "krol"],
        "gender":         "man",
    },
    {
        "naam":           "sven",
        "full_name":      "Sven Kramer",
        "search_queries": ["Sven Kramer"],
        "title_variants": ["sven kramer"],
        # "kramer" alone matches Seinfeld characters, Jerry Kramer, etc.
        # require full name in Tier 2 comments
        "comment_match":  ["sven kramer"],
        "gender":         "man",
    },
    {
        "naam":           "jutta",
        "full_name":      "Jutta Leerdam",
        "search_queries": ["Jutta Leerdam"],
        "title_variants": ["jutta leerdam"],
        # "leerdam" is unique
        "comment_match":  ["jutta leerdam", "leerdam"],
        "gender":         "vrouw",
    },
    {
        "naam":           "kjeld",
        "full_name":      "Kjeld Nuis",
        "search_queries": ["Kjeld Nuis"],
        "title_variants": ["kjeld nuis"],
        # "nuis" and "kjeld" are both unusual
        "comment_match":  ["kjeld nuis", "nuis", "kjeld"],
        "gender":         "man",
    },
]

# ── Config ────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "BachelorThesis/1.0 by /u/academic_scraper"
}

# Reddit returns max 100 threads per search page; we paginate up to this total
SEARCH_LIMIT          = 1000
# Cap on how many threads to scrape per skater (sorted by score, highest first)
MAX_THREADS_PER_SKATER = 300
# Seconds between HTTP requests (keeps us well within Reddit's limits)
SLEEP_BETWEEN_REQUESTS = 5.0
OUTPUT_FILE = "comments_all_v2.csv"

FIELDNAMES = [
    "url", "author", "date", "timestamp", "score", "upvotes", "downvotes",
    "golds", "comment", "comment_id", "thread_url", "thread_title",
    "subreddit", "persoon", "gender",
]

# ── Text helpers ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + fold special characters so ü/Ü both match 'u', etc."""
    return (
        text.lower()
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ä", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ë", "e")
    )


def title_is_about_skater(title: str, skater: dict) -> bool:
    """True if the thread title contains the skater's full name."""
    norm = normalize(title)
    return any(normalize(v) in norm for v in skater["title_variants"])


def comment_mentions_skater(body: str, skater: dict) -> bool:
    """True if the comment text contains any recognised name variant."""
    norm = normalize(body)
    return any(normalize(m) in norm for m in skater["comment_match"])


# ── Reddit API calls ──────────────────────────────────────────────────────────

def search_threads(query: str, limit: int = 1000) -> list[dict]:
    """
    Paginate through Reddit search results for an exact-phrase query.
    Returns a list of thread dicts with keys: url, title, subreddit, thread_id, score.
    """
    threads = []
    after = None

    while len(threads) < limit:
        params = {
            "q":     f'"{query}"',
            "sort":  "top",
            "t":     "all",
            "limit": min(100, limit - len(threads)),
            "type":  "link",
        }
        if after:
            params["after"] = after

        data = _get_json("https://www.reddit.com/search.json", params)
        if data is None:
            print("    Search failed, stopping pagination.")
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            p = child["data"]
            threads.append({
                "url":       f"https://www.reddit.com{p['permalink']}",
                "title":     p.get("title", ""),
                "subreddit": p.get("subreddit", ""),
                "thread_id": p.get("id", ""),
                "score":     p.get("score", 0),
            })

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return threads


def _get_json(url: str, params: dict) -> dict | None:
    """
    GET a Reddit JSON endpoint with exponential backoff, and an NSFW fallback.
    Waits 60s → 120s → 300s on repeated 429s before giving up.
    """
    RATE_LIMIT_SLEEPS = [60, 120, 300]

    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                wait = RATE_LIMIT_SLEEPS[attempt - 1]
                print(f"    [rate limited] sleeping {wait}s (attempt {attempt}/3) …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    Poging {attempt} mislukt: {e}")
            if attempt < 3:
                time.sleep(10)

    # NSFW fallback — same trick as the R script's ?over_18=1
    print("    Trying NSFW fallback …")
    try:
        nsfw_params = {**params, "over_18": "1"}
        r = requests.get(url, headers=HEADERS, params=nsfw_params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    Definitief mislukt: {e}")
        return None


def fetch_thread_comments(thread: dict, skater: dict, tier: int) -> list[dict]:
    """
    Fetch comments from one thread.

    tier=1  → return ALL non-deleted comments
    tier=2  → return only comments that mention the skater by name
    """
    url = f"https://www.reddit.com/r/{thread['subreddit']}/comments/{thread['thread_id']}.json"
    comments = []

    data = _get_json(url, {"limit": 500, "depth": 10})
    if data is None:
        return []

    if not isinstance(data, list) or len(data) < 2:
        return []

    def _walk(children):
        for item in children:
            if not isinstance(item, dict) or item.get("kind") != "t1":
                continue
            c    = item["data"]
            body = c.get("body", "")

            if not body or body in ("[deleted]", "[removed]"):
                pass  # still recurse into replies below
            else:
                # Apply tier filter
                if tier == 1 or comment_mentions_skater(body, skater):
                    ts       = int(c.get("created_utc") or 0)
                    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    comments.append({
                        "url":          f"https://www.reddit.com{c.get('permalink', '')}",
                        "author":       c.get("author", ""),
                        "date":         date_str,
                        "timestamp":    ts,
                        "score":        c.get("score", 0),
                        "upvotes":      c.get("ups", 0),
                        "downvotes":    c.get("downs", 0),
                        "golds":        c.get("gilded", 0),
                        "comment":      body,
                        "comment_id":   c.get("id", ""),
                        "thread_url":   thread["url"],
                        "thread_title": thread["title"],
                        "subreddit":    thread["subreddit"],
                        "persoon":      skater["naam"],
                        "gender":       skater["gender"],
                    })

            # Always recurse into replies regardless of filter
            replies = c.get("replies")
            if isinstance(replies, dict):
                _walk(replies.get("data", {}).get("children", []))

    _walk(data[1].get("data", {}).get("children", []))
    return comments


# ── Per-skater orchestration ──────────────────────────────────────────────────

def process_skater(skater: dict) -> list[dict]:
    print(f"\n{'='*58}")
    print(f"  {skater['full_name']}")
    print(f"{'='*58}")

    # 1. Collect unique thread candidates across all search queries
    all_threads: dict[str, dict] = {}
    for query in skater["search_queries"]:
        print(f"  Searching: \"{query}\"")
        found = search_threads(query, limit=SEARCH_LIMIT)
        print(f"    → {len(found)} results")
        for t in found:
            if t["thread_id"]:
                all_threads[t["thread_id"]] = t

    total_found = len(all_threads)
    print(f"  Unique threads from search: {total_found}")

    # 2. Classify threads into tiers
    tier1 = [t for t in all_threads.values() if title_is_about_skater(t["title"], skater)]
    tier2 = [t for t in all_threads.values() if not title_is_about_skater(t["title"], skater)]

    print(f"  Tier 1 (dedicated — scrape all comments): {len(tier1)}")
    print(f"  Tier 2 (mentions  — filter by name):      {len(tier2)}")

    # 3. Sort each tier by score; cap total at MAX_THREADS_PER_SKATER
    tier1.sort(key=lambda x: x["score"], reverse=True)
    tier2.sort(key=lambda x: x["score"], reverse=True)

    # Tier 1 threads get priority; remaining budget goes to Tier 2
    t1_budget = min(len(tier1), MAX_THREADS_PER_SKATER)
    t2_budget = min(len(tier2), MAX_THREADS_PER_SKATER - t1_budget)
    to_scrape = [(t, 1) for t in tier1[:t1_budget]] + [(t, 2) for t in tier2[:t2_budget]]

    print(f"  Scraping: {t1_budget} Tier-1 + {t2_budget} Tier-2 threads\n")

    # 4. Scrape
    all_comments = []
    seen_ids     = set()
    t1_count = t2_count = 0
    failed   = 0

    for i, (thread, tier) in enumerate(to_scrape, 1):
        tier_label = f"T{tier}"
        print(f"  [{i:>3}/{len(to_scrape)}][{tier_label}] {thread['title'][:58]}")

        comments = fetch_thread_comments(thread, skater, tier)

        # Deduplicate by comment_id
        new = [c for c in comments if c["comment_id"] not in seen_ids]
        seen_ids.update(c["comment_id"] for c in new)
        all_comments.extend(new)

        if new:
            if tier == 1:
                t1_count += len(new)
            else:
                t2_count += len(new)
            print(f"           ✓ {len(new)} comments (T1 total: {t1_count} | T2 total: {t2_count})")
        else:
            failed += 1
            print(f"           ✗ 0 kept")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\n  {skater['full_name']}: {len(all_comments)} comments total")
    print(f"    from dedicated threads: {t1_count}")
    print(f"    from mention threads:   {t2_count}")
    print(f"    threads failed/empty:   {failed}")
    return all_comments


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    all_rows = []

    for skater in SKATERS:
        rows = process_skater(skater)
        all_rows.extend(rows)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*58}")
    print(f"DONE — {len(all_rows)} comments saved to {OUTPUT_FILE}")
    print()
    counts = Counter(r["persoon"] for r in all_rows)
    for person, count in counts.most_common():
        print(f"  {person:10s}: {count}")


if __name__ == "__main__":
    main()
