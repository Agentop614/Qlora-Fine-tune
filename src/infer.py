"""Stage 5 (lite) -- run the fine-tuned model from the command line.

Run:
    python -m src.infer --demo 8        # sample from the held-out test set
    python -m src.infer --interactive   # paste your own question + abstract

Loads the base in 4-bit and attaches the LoRA adapter at runtime. No merge,
no GGUF, no Ollama -- this uses exactly the artifacts src.train produced.

Scoring matches src.evaluate: one forward pass, compare the logits for the
"yes"/"no"/"maybe" tokens, softmax over those three for a confidence figure.
Using the same method here means the demo cannot disagree with the reported
metrics -- a demo that free-generates would drift from the numbers you publish.
"""

import argparse
import json
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src import config as C


def load(adapter: bool = True):
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if adapter:
        if not C.ADAPTER_DIR.exists():
            raise SystemExit(f"No adapter at {C.ADAPTER_DIR} -- run src.train first.")
        model = PeftModel.from_pretrained(model, str(C.ADAPTER_DIR))
    model.eval()

    ids = {}
    for label in C.LABELS:
        variants = {label, label.capitalize(), f" {label}", f" {label.capitalize()}"}
        found = {tok.encode(v, add_special_tokens=False)[0] for v in variants}
        ids[label] = sorted(found)
    return model, tok, ids


@torch.no_grad()
def answer(model, tok, ids, question: str, context: str):
    """Return (predicted_label, {label: probability})."""
    messages = [
        {"role": "system", "content": C.SYSTEM_PROMPT},
        {"role": "user", "content": C.USER_TEMPLATE.format(
            question=question.strip(), context=context.strip())},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=C.MAX_SEQ_LEN, add_special_tokens=False).to(model.device)

    logits = model(**enc).logits[0, -1, :].float()
    raw = torch.tensor([max(logits[i].item() for i in ids[l]) for l in C.LABELS])
    probs = torch.softmax(raw, dim=0)
    scores = {l: probs[i].item() for i, l in enumerate(C.LABELS)}
    return max(scores, key=scores.get), scores


def bar(p: float, width: int = 24) -> str:
    filled = int(round(p * width))
    return "#" * filled + "." * (width - filled)


def show(pred: str, scores: dict, truth: str | None = None) -> None:
    for label in C.LABELS:
        marker = " <-" if label == pred else "   "
        print(f"    {label:<6} {bar(scores[label])} {scores[label]:6.1%}{marker}")
    if truth is not None:
        verdict = "CORRECT" if pred == truth else "WRONG"
        print(f"    prediction: {pred}   truth: {truth}   [{verdict}]")
    else:
        print(f"    prediction: {pred}")


def demo(n: int) -> None:
    rows = [json.loads(l) for l in open(C.TEST_FILE, encoding="utf-8")]
    random.seed(C.SEED)
    sample = random.sample(rows, min(n, len(rows)))

    model, tok, ids = load()
    correct = 0
    for i, row in enumerate(sample, 1):
        user = row["messages"][1]["content"]
        question = user.split("Question:", 1)[1].split("\n\nAbstract:", 1)[0].strip()
        context = user.split("Abstract:", 1)[1].rsplit("\n\nAnswer:", 1)[0].strip()

        pred, scores = answer(model, tok, ids, question, context)
        correct += pred == row["label"]

        print(f"\n[{i}/{len(sample)}] {question[:110]}")
        print(f"    abstract: {len(context.split())} words")
        show(pred, scores, row["label"])

    print(f"\n{'=' * 46}")
    print(f"Sample accuracy: {correct}/{len(sample)} = {correct / len(sample):.1%}")
    print(f"(Full test-set accuracy is in outputs/eval_results.json)")
    print("=" * 46)


def read_block(prompt: str) -> str:
    print(prompt)
    lines = []
    for line in sys.stdin:
        if line.strip() == "END":
            break
        lines.append(line)
    return "".join(lines).strip()


def interactive() -> None:
    model, tok, ids = load()
    print("\nPaste a question, then the abstract. Type END on its own line to submit.")
    print("Ctrl+C to quit.\n")
    while True:
        question = read_block("QUESTION (END to finish):")
        if not question:
            break
        context = read_block("ABSTRACT (END to finish):")
        if not context:
            break
        pred, scores = answer(model, tok, ids, question, context)
        print()
        show(pred, scores)
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, metavar="N", help="sample N held-out test items")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        interactive()
    else:
        demo(args.demo or 5)


if __name__ == "__main__":
    main()
