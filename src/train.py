"""Stage 3 -- QLoRA fine-tune on the PubMedQA splits.

Run:  python -m src.train

Decisions worth defending:

* 4-bit NF4 quantization with double-quant. A 3B model in fp16 is ~6GB of
  weights alone, which leaves nothing for activations on a 6GB card. In NF4
  it is ~2GB.
* LoRA on all attention AND MLP projections. Attention-only (q_proj/v_proj)
  is the common tutorial default and it underfits classification tasks.
* Prompt/completion format, NOT plain chat. TRL computes loss only on the
  completion for this format, so the model is graded on the one token we
  actually care about ("yes"/"no"/"maybe") instead of on reconstructing
  400 words of abstract it will never need to generate.
* paged_adamw_8bit. Optimizer state is the usual OOM culprit, not weights.
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src import config as C


def to_prompt_completion(row: dict) -> dict:
    """messages -> {prompt, completion} so TRL masks loss on the prompt."""
    return {
        "prompt": row["messages"][:2],       # system + user
        "completion": [row["messages"][2]],  # assistant answer
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible -- fix the environment first.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {C.BASE_MODEL}")

    # ---- data --------------------------------------------------------------
    ds = load_dataset(
        "json",
        data_files={"train": str(C.TRAIN_FILE), "validation": str(C.VAL_FILE)},
    )
    ds = ds.map(to_prompt_completion, remove_columns=["messages", "label", "pubid"])
    print(f"train={len(ds['train'])}  val={len(ds['validation'])}")

    # ---- tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4-bit quantization ------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ---- LoRA --------------------------------------------------------------
    peft_config = LoraConfig(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- training args (TRL v1: max_length lives HERE) ---------------------
    args = SFTConfig(
        output_dir=str(C.ADAPTER_DIR),
        max_length=C.MAX_SEQ_LEN,
        num_train_epochs=C.NUM_EPOCHS,
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.BATCH_SIZE,
        gradient_accumulation_steps=C.GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=C.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=C.WARMUP_RATIO,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=C.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=C.EVAL_STEPS,
        save_strategy="steps",
        save_steps=C.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=C.SEED,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        },
    )

    # ---- train -------------------------------------------------------------
    trainer = SFTTrainer(
        model=C.BASE_MODEL,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,   # TRL v1: NOT tokenizer=
    )

    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(str(C.ADAPTER_DIR))
    tokenizer.save_pretrained(str(C.ADAPTER_DIR))
    print(f"\nAdapter saved to {C.ADAPTER_DIR}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
