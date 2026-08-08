"""Single source of truth for the whole pipeline.

Every stage (ingest, train, eval, export) imports from here so that a run is
reproducible from one file.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"
STATS_FILE = DATA_DIR / "dataset_stats.json"

# ---------------------------------------------------------------- dataset
HF_DATASET = "qiaojin/PubMedQA"
LABELED_CONFIG = "pqa_labeled"        # 1k expert-annotated
ARTIFICIAL_CONFIG = "pqa_artificial"  # 211k machine-labeled

SEED = 42
LABELS = ["yes", "no", "maybe"]

# pqa_labeled (1000 rows) is carved into:
TEST_SIZE = 500   # held out, NEVER trained on, NEVER filtered
VAL_SIZE = 100    # early-stopping / sanity signal
# remaining 400 go into the training mix so the model sees real "maybe" examples

# pqa_artificial is 92.8% yes / 7.2% no / 0% maybe.
# We take an equal number of each available class instead of sampling blind.
ARTIFICIAL_PER_CLASS = 2000

# Training rows whose prompt is longer than this get dropped (test set never is).
MAX_PROMPT_WORDS = 380

# ---------------------------------------------------------------- prompt
SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Using only the provided abstract, "
    "answer the research question. Reply with exactly one word: yes, no, or maybe."
)

USER_TEMPLATE = "Question: {question}\n\nAbstract:\n{context}\n\nAnswer:"

# ---------------------------------------------------------------- model
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # ungated, no HF login needed
# Swap to Llama once you have accepted the license on HF:
# BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

MAX_SEQ_LEN = 512

# ---------------------------------------------------------------- LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# All attention + MLP projections. Attention-only (q,v) trains fewer params but
# consistently underperforms on classification-style tasks.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------- training
ADAPTER_DIR = OUTPUT_DIR / "adapter"
MERGED_DIR = OUTPUT_DIR / "merged"

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 2            # per-device micro-batch; drop to 1 if you OOM
GRAD_ACCUM = 8            # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50