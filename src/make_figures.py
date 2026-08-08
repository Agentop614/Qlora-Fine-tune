"""Generate README figures from artifacts already on disk.

Run:  python -m src.make_figures

Reads:
    outputs/eval_results.json        (src.evaluate)
    outputs/calibration.json         (src.calibrate)
    outputs/adapter/checkpoint-*/trainer_state.json   (src.train)

Writes PNGs to docs/. Nothing here re-runs the model.
"""

import json
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C

DOCS = C.ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def draw_cm(ax, cm, title, subtitle=""):
    cm = np.array(cm)
    # Row-normalise so colour reflects recall, not class size.
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), C.LABELS)
    ax.set_yticks(range(3), C.LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=10)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=11)


def figure_confusion():
    ev = json.loads((C.OUTPUT_DIR / "eval_results.json").read_text())
    cal_path = C.OUTPUT_DIR / "calibration.json"
    panels = [
        (ev["base"]["confusion_matrix"], "Base model",
         f"acc {ev['base']['accuracy']:.1%} / macro-F1 {ev['base']['macro_f1']:.3f}"),
        (ev["tuned"]["confusion_matrix"], "Fine-tuned (argmax)",
         f"acc {ev['tuned']['accuracy']:.1%} / macro-F1 {ev['tuned']['macro_f1']:.3f}"),
    ]
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        panels.append((
            cal["test_calibrated"]["confusion_matrix"],
            f"Fine-tuned (tau={cal['tau']})",
            f"acc {cal['test_calibrated']['accuracy']:.1%} / "
            f"macro-F1 {cal['test_calibrated']['macro_f1']:.3f}",
        ))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (cm, title, sub) in zip(np.atleast_1d(axes), panels):
        draw_cm(ax, cm, title, sub)
    fig.suptitle("Confusion matrices on the held-out 500-item test set", fontsize=12)
    fig.tight_layout()
    out = DOCS / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_calibration():
    path = C.OUTPUT_DIR / "calibration.json"
    if not path.exists():
        print("skip calibration figure (no calibration.json)")
        return
    cal = json.loads(path.read_text())
    taus = [s["tau"] for s in cal["val_sweep"]]
    f1s = [s["macro_f1"] for s in cal["val_sweep"]]
    accs = [s["accuracy"] for s in cal["val_sweep"]]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(taus, f1s, marker="o", ms=3, label="macro F1 (val)")
    ax.plot(taus, accs, marker="s", ms=3, ls="--", label="accuracy (val)")
    ax.axvline(cal["tau"], color="crimson", lw=1.2,
               label=f"chosen tau = {cal['tau']}")
    ax.set_xlabel("tau  —  threshold on P(maybe)")
    ax.set_ylabel("score on validation set")
    ax.set_title("Threshold sweep, fitted on validation only\n"
                 f"(only {cal['val_maybe_count']} 'maybe' examples — curve is noisy)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DOCS / "calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_training():
    states = sorted(glob(str(C.ADAPTER_DIR / "checkpoint-*" / "trainer_state.json")))
    if not states:
        print("skip training figure (no checkpoint-*/trainer_state.json)")
        return
    # highest step count = most complete log history
    state = max(states, key=lambda p: len(json.loads(open(p).read())["log_history"]))
    logs = json.loads(open(state).read())["log_history"]

    tr = [(l["step"], l["loss"]) for l in logs if "loss" in l]
    ev = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if tr:
        ax.plot(*zip(*tr), lw=1.2, label="train loss")
    if ev:
        ax.plot(*zip(*ev), marker="o", ms=4, label="eval loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("QLoRA training — Qwen2.5-3B-Instruct on PubMedQA", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DOCS / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_summary():
    ev = json.loads((C.OUTPUT_DIR / "eval_results.json").read_text())
    baseline = ev["majority_class_baseline"]
    names, accs, f1s = ["Baseline\n(majority)"], [baseline], [0]
    names += ["Base model"]; accs += [ev["base"]["accuracy"]]; f1s += [ev["base"]["macro_f1"]]
    names += ["Fine-tuned\n(argmax)"]; accs += [ev["tuned"]["accuracy"]]; f1s += [ev["tuned"]["macro_f1"]]

    cal_path = C.OUTPUT_DIR / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        names += [f"Fine-tuned\n(tau={cal['tau']})"]
        accs += [cal["test_calibrated"]["accuracy"]]
        f1s += [cal["test_calibrated"]["macro_f1"]]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, accs, 0.4, label="accuracy")
    ax.bar(x + 0.2, f1s, 0.4, label="macro F1")
    ax.axhline(baseline, color="crimson", ls="--", lw=1.2,
               label=f"majority baseline ({baseline:.1%})")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("score on 500-item test set")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs macro F1 — why accuracy alone is misleading", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(xi - 0.2, a + 0.015, f"{a:.1%}", ha="center", fontsize=8)
        if f:
            ax.text(xi + 0.2, f + 0.015, f"{f:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = DOCS / "results_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    figure_summary()
    figure_confusion()
    figure_calibration()
    figure_training()
    print(f"\nAll figures in {DOCS}")
