"""Stage 1 + 2 -- ingest PubMedQA and emit chat-formatted JSONL splits.

Run:  python -m src.data_prep

Design notes (these are the decisions worth defending in an interview):

* The 500-row test set comes ONLY from pqa_labeled (expert-annotated) and is
  never touched by training, length-filtering, or balancing. It matches the
  official PubMedQA test split, so the accuracy number is comparable to the
  published leaderboard.
* pqa_artificial is 92.8% "yes" / 7.2% "no" / 0% "maybe". Sampling it blind
  produces a model that answers "yes" to everything and still scores ~55% on
  the test set. We take an equal count per class instead, and mix in the
  leftover expert rows so the model actually sees "maybe" during training.
* Splits are stratified, so val/test class ratios mirror the real distribution.
"""

import json
import random
from collections import Counter, defaultdict

from datasets import load_dataset

from src import config as C


# --------------------------------------------------------------- formatting
def to_chat(row: dict) -> dict:
    """Turn one PubMedQA row into a TRL-compatible chat record."""
    context = "\n".join(row["context"]["contexts"])
    user = C.USER_TEMPLATE.format(question=row["question"].strip(), context=context)
    return {
        "messages": [
            {"role": "system", "content": C.SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["final_decision"].strip().lower()},
        ],
        "label": row["final_decision"].strip().lower(),
        "pubid": row["pubid"],
    }


def prompt_words(rec: dict) -> int:
    return sum(len(m["content"].split()) for m in rec["messages"][:2])


# --------------------------------------------------------------- splitting
def stratified_split(rows: list, sizes: dict, seed: int) -> dict:
    """Split `rows` into named buckets, preserving class ratios.

    `sizes` is ordered; any remainder lands in the LAST bucket.
    """
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    for items in by_label.values():
        rng.shuffle(items)

    total = len(rows)
    out = {name: [] for name in sizes}
    last = list(sizes)[-1]

    for items in by_label.values():
        cursor = 0
        for name, n in sizes.items():
            if name == last:
                continue
            take = round(n * len(items) / total)
            out[name].extend(items[cursor:cursor + take])
            cursor += take
        out[last].extend(items[cursor:])

    for bucket in out.values():
        rng.shuffle(bucket)
    return out


# --------------------------------------------------------------- sampling
def sample_artificial(per_class: int, seed: int) -> list:
    """Equal-count sample per available class from pqa_artificial."""
    print(f"Loading {C.ARTIFICIAL_CONFIG} (~233 MB on first run)...")
    ds = load_dataset(C.HF_DATASET, C.ARTIFICIAL_CONFIG, split="train")
    ds = ds.shuffle(seed=seed)

    picked, counts = [], Counter()
    for row in ds:
        label = row["final_decision"].strip().lower()
        if counts[label] >= per_class:
            if all(counts[l] >= per_class for l in ("yes", "no")):
                break
            continue
        counts[label] += 1
        picked.append(to_chat(row))

    print(f"  sampled from artificial: {dict(counts)}")
    return picked


# --------------------------------------------------------------- write
def write_jsonl(path, records) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):>6} -> {path.name}")


def main() -> None:
    random.seed(C.SEED)

    # ---- expert-annotated rows: the only source of test/val ----------------
    print(f"Loading {C.LABELED_CONFIG}...")
    labeled = [to_chat(r) for r in load_dataset(C.HF_DATASET, C.LABELED_CONFIG, split="train")]
    print(f"  {len(labeled)} expert rows: {Counter(r['label'] for r in labeled)}")

    splits = stratified_split(
        labeled,
        {"test": C.TEST_SIZE, "val": C.VAL_SIZE, "train_expert": 0},
        seed=C.SEED,
    )
    test, val, train_expert = splits["test"], splits["val"], splits["train_expert"]

    # ---- training mix ------------------------------------------------------
    train = sample_artificial(C.ARTIFICIAL_PER_CLASS, C.SEED) + train_expert
    random.shuffle(train)

    # Length filter applies to TRAIN and VAL only -- never to test.
    before = len(train)
    train = [r for r in train if prompt_words(r) <= C.MAX_PROMPT_WORDS]
    val = [r for r in val if prompt_words(r) <= C.MAX_PROMPT_WORDS]
    print(f"Length filter dropped {before - len(train)} train rows "
          f"(> {C.MAX_PROMPT_WORDS} words)")

    # ---- leakage check -----------------------------------------------------
    test_ids = {r["pubid"] for r in test}
    overlap = test_ids & {r["pubid"] for r in train}
    assert not overlap, f"LEAKAGE: {len(overlap)} test pubids appear in train"
    print("Leakage check passed: no test pubid in train")

    # ---- write -------------------------------------------------------------
    write_jsonl(C.TRAIN_FILE, train)
    write_jsonl(C.VAL_FILE, val)
    write_jsonl(C.TEST_FILE, test)

    test_counts = Counter(r["label"] for r in test)
    majority = max(test_counts.values()) / len(test)
    stats = {
        "seed": C.SEED,
        "splits": {
            "train": {"n": len(train), "labels": dict(Counter(r["label"] for r in train))},
            "val": {"n": len(val), "labels": dict(Counter(r["label"] for r in val))},
            "test": {"n": len(test), "labels": dict(test_counts)},
        },
        "majority_class_baseline_on_test": round(majority, 4),
    }
    C.STATS_FILE.write_text(json.dumps(stats, indent=2))

    print(f"\nTest distribution: {dict(test_counts)}")
    print(f"MAJORITY-CLASS BASELINE = {majority:.1%}  <- beat this or you have nothing")


if __name__ == "__main__":
    main()
