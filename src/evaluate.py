"""Stage 4 -- evaluate base vs fine-tuned on the held-out 500.

Run:
    python -m src.evaluate --model base     # score base, cache predictions
    python -m src.evaluate --model tuned    # score tuned, cache predictions
    python -m src.evaluate --report         # compare whatever is cached

or, if you have memory to spare:
    python -m src.evaluate --model both

Method: constrained scoring, not free generation.

One forward pass per item; we read the logits at the final position and compare
the log-probabilities of the tokens for "yes", "no" and "maybe". Highest wins.

Why not model.generate() + string parsing: the base model was never told to
answer in one word, so left free it writes a paragraph and a parser scores that
as a miss. The measured gain would then be mostly "learned the output format",
not "got better at the task". Constrained scoring forces both models to answer
on identical terms, so the delta is attributable to the adapter.

Predictions are cached per model, so a crash in one never costs you the other.
"""

import argparse
import gc
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src import config as C

BATCH_SIZE = 4


def pred_file(name):
    return C.OUTPUT_DIR / f"preds_{name}.json"


def label_token_ids(tokenizer) -> dict:
    ids = {}
    for label in C.LABELS:
        variants = {label, label.capitalize(), f" {label}", f" {label.capitalize()}"}
        found = set()
        for v in variants:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if toks:
                found.add(toks[0])
        ids[label] = sorted(found)
    return ids


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


@torch.no_grad()
def run_model(name: str, adapter: bool, rows: list) -> list:
    """Load, score all rows, then release the GPU completely."""
    tokenizer = get_tokenizer()
    tok_ids = label_token_ids(tokenizer)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"\nLoading {name} model...")
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, str(C.ADAPTER_DIR))
    model.eval()

    preds = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                r["messages"][:2], tokenize=False, add_generation_prompt=True
            )
            for r in batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=C.MAX_SEQ_LEN,
            add_special_tokens=False,
        ).to(model.device)

        logits = model(**enc).logits[:, -1, :].float()
        for row_logits in logits:
            scores = {
                lab: max(row_logits[i].item() for i in ids)
                for lab, ids in tok_ids.items()
            }
            preds.append(max(scores, key=scores.get))

        del enc, logits
        done = min(start + BATCH_SIZE, len(rows))
        print(f"\r  {done}/{len(rows)}", end="", flush=True)
    print()

    pred_file(name).write_text(json.dumps(preds))
    print(f"  cached -> {pred_file(name).name}")
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Release properly: drop EVERY reference, then collect, then empty cache.
    model.cpu()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return preds


def score(name, y_true, y_pred, baseline) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=C.LABELS, zero_division=0)
    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.1%}   (majority-class baseline {baseline:.1%})")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=C.LABELS, zero_division=0, digits=3))
    print("Confusion matrix (rows=true, cols=pred, order: yes/no/maybe)")
    print(confusion_matrix(y_true, y_pred, labels=C.LABELS))
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "report": classification_report(
            y_true, y_pred, labels=C.LABELS, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=C.LABELS).tolist(),
        "prediction_distribution": {l: y_pred.count(l) for l in C.LABELS},
    }


def report(rows) -> None:
    y_true = [r["label"] for r in rows]
    baseline = max(y_true.count(l) for l in C.LABELS) / len(y_true)

    results = {"n_test": len(rows), "majority_class_baseline": round(baseline, 4)}
    for name, title in [("base", "BASE (no adapter)"), ("tuned", "FINE-TUNED (LoRA)")]:
        if not pred_file(name).exists():
            print(f"\n[missing] {pred_file(name).name} -- run --model {name} first")
            continue
        preds = json.loads(pred_file(name).read_text())
        results[name] = score(title, y_true, preds, baseline)

    if "base" in results and "tuned" in results:
        delta = results["tuned"]["accuracy"] - results["base"]["accuracy"]
        results["accuracy_delta"] = round(delta, 4)
        print("\n" + "=" * 46)
        print(f"Baseline (majority class) : {baseline:.1%}")
        print(f"Base model                : {results['base']['accuracy']:.1%}")
        print(f"Fine-tuned                : {results['tuned']['accuracy']:.1%}")
        print(f"Delta                     : {delta:+.1%}")
        print("=" * 46)

    out = C.OUTPUT_DIR / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "tuned", "both"], default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rows = list(load_dataset("json", data_files=str(C.TEST_FILE), split="train"))
    print(f"Test rows: {len(rows)}")

    if args.model in ("base", "both"):
        run_model("base", adapter=False, rows=rows)
    if args.model in ("tuned", "both"):
        run_model("tuned", adapter=True, rows=rows)

    if args.report or args.model:
        report(rows)


if __name__ == "__main__":
    main()
