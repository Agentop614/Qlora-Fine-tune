"""Stage 4b -- threshold calibration for the collapsed "maybe" class.

Run:  python -m src.calibrate

The problem this addresses:

Argmax decoding never predicts "maybe" (0/500 on the test set), because
"maybe" was ~1% of the training rows and its prior sits too low to ever beat
"yes" or "no". But the probability mass is NOT zero -- on genuinely ambiguous
abstracts it reaches 25-30%. The signal is there; the decision rule discards it.

The fix is at the decision layer, not the model:

    if P(maybe) > tau:  predict "maybe"
    else:               predict argmax(yes, no)

tau is chosen on the VALIDATION set and then applied unchanged to the test set.
Choosing it on test would be tuning on the exam -- the resulting number would
mean nothing.

Honest limitation, and it goes in the README: the validation set holds only
~11 "maybe" examples, so tau is estimated from very few points and is
sensitive to the split. A different seed could move it meaningfully.
"""

import gc
import json

import torch
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
TAU_GRID = [round(0.02 * i, 2) for i in range(1, 31)]  # 0.02 .. 0.60


def prob_file(split: str):
    return C.OUTPUT_DIR / f"probs_{split}.json"


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@torch.no_grad()
def compute_probs(splits: dict) -> dict:
    """splits: {name: rows}. Returns {name: [ {yes,no,maybe}, ... ]}."""
    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tok_ids = {}
    for label in C.LABELS:
        variants = {label, label.capitalize(), f" {label}", f" {label.capitalize()}"}
        tok_ids[label] = sorted(
            {tokenizer.encode(v, add_special_tokens=False)[0] for v in variants}
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(C.ADAPTER_DIR))
    model.eval()

    out = {}
    for name, rows in splits.items():
        print(f"Scoring {name} ({len(rows)} rows)...")
        probs = []
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
                raw = torch.tensor(
                    [max(row_logits[i].item() for i in tok_ids[l]) for l in C.LABELS]
                )
                p = torch.softmax(raw, dim=0)
                probs.append({l: round(p[i].item(), 6) for i, l in enumerate(C.LABELS)})

            del enc, logits
            done = min(start + BATCH_SIZE, len(rows))
            print(f"\r  {done}/{len(rows)}", end="", flush=True)
        print()
        prob_file(name).write_text(json.dumps(probs))
        out[name] = probs

    model.cpu()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return out


def argmax_labels(probs):
    return [max(p, key=p.get) for p in probs]


def threshold_labels(probs, tau):
    out = []
    for p in probs:
        if p["maybe"] > tau:
            out.append("maybe")
        else:
            out.append("yes" if p["yes"] >= p["no"] else "no")
    return out


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=C.LABELS, zero_division=0)


def report(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred)
    print(f"\n===== {title} =====")
    print(f"Accuracy : {acc:.1%}")
    print(f"Macro F1 : {mf1:.4f}")
    print(classification_report(y_true, y_pred, labels=C.LABELS, zero_division=0, digits=3))
    print("Confusion matrix (rows=true, cols=pred, order: yes/no/maybe)")
    print(confusion_matrix(y_true, y_pred, labels=C.LABELS))
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "prediction_distribution": {l: y_pred.count(l) for l in C.LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=C.LABELS).tolist(),
    }


def main() -> None:
    val_rows = load_rows(C.VAL_FILE)
    test_rows = load_rows(C.TEST_FILE)

    # Reuse cached probabilities if both files exist.
    if prob_file("val").exists() and prob_file("test").exists():
        print("Using cached probabilities (delete outputs/probs_*.json to recompute).")
        probs = {
            "val": json.loads(prob_file("val").read_text()),
            "test": json.loads(prob_file("test").read_text()),
        }
    else:
        probs = compute_probs({"val": val_rows, "test": test_rows})

    y_val = [r["label"] for r in val_rows]
    y_test = [r["label"] for r in test_rows]

    n_maybe_val = y_val.count("maybe")
    print(f"\nValidation set: {len(y_val)} rows, {n_maybe_val} of them 'maybe'")
    if n_maybe_val < 20:
        print(f"  WARNING: tau is being fitted on only {n_maybe_val} positive examples.")
        print("  Treat the chosen value as indicative, not precise. Document this.")

    # ---- sweep tau on VALIDATION only -------------------------------------
    print("\ntau     val macro-F1   val acc   n_maybe_pred")
    sweep = []
    for tau in TAU_GRID:
        pred = threshold_labels(probs["val"], tau)
        mf1 = macro_f1(y_val, pred)
        acc = accuracy_score(y_val, pred)
        sweep.append({"tau": tau, "macro_f1": round(mf1, 4), "accuracy": round(acc, 4)})
        if int(tau * 100) % 6 == 0:
            print(f"{tau:<7.2f} {mf1:<12.4f} {acc:<9.1%} {pred.count('maybe')}")

    best = max(sweep, key=lambda s: s["macro_f1"])
    tau = best["tau"]
    print(f"\nBest tau on validation: {tau}  (val macro-F1 {best['macro_f1']:.4f})")

    # ---- apply the chosen tau to TEST, once -------------------------------
    base_pred = argmax_labels(probs["test"])
    cal_pred = threshold_labels(probs["test"], tau)

    results = {
        "tau": tau,
        "val_maybe_count": n_maybe_val,
        "val_sweep": sweep,
        "test_argmax": report("TEST -- argmax (original)", y_test, base_pred),
        "test_calibrated": report(f"TEST -- calibrated (tau={tau})", y_test, cal_pred),
    }

    d_acc = results["test_calibrated"]["accuracy"] - results["test_argmax"]["accuracy"]
    d_f1 = results["test_calibrated"]["macro_f1"] - results["test_argmax"]["macro_f1"]
    results["delta_accuracy"] = round(d_acc, 4)
    results["delta_macro_f1"] = round(d_f1, 4)

    print("\n" + "=" * 52)
    print(f"{'':<14}{'argmax':>12}{'calibrated':>14}{'delta':>12}")
    print(f"{'Accuracy':<14}{results['test_argmax']['accuracy']:>11.1%}"
          f"{results['test_calibrated']['accuracy']:>14.1%}{d_acc:>+12.1%}")
    print(f"{'Macro F1':<14}{results['test_argmax']['macro_f1']:>12.4f}"
          f"{results['test_calibrated']['macro_f1']:>14.4f}{d_f1:>+12.4f}")
    print("=" * 52)
    if d_f1 > 0:
        print("Macro F1 improved: the 'maybe' signal was present but mis-ranked.")
    else:
        print("No macro-F1 gain. Report this as a negative result -- it is still a result.")

    out = C.OUTPUT_DIR / "calibration.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
