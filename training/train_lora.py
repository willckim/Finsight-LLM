# training/train_lora.py
import os
import json
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

# --- Config via .env (with safe defaults) ---
MODEL_NAME  = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "models/qwen3b-lora")
HF_TOKEN    = os.getenv("HF_TOKEN", None)

# Data config
USE_LOCAL_FINANCE   = True
LOCAL_JSONL_PATH    = Path("data/raw/finance_tiny.jsonl")

# Reproducibility
SEED = int(os.getenv("SEED", "42"))

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# Optional: login with HF token (helps with gated repos)
try:
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
except Exception:
    # Non-fatal: proceed without explicit login if cached creds exist
    pass


def ensure_local_finance_jsonl() -> None:
    """Create a tiny finance JSONL if it doesn't exist so you can train immediately."""
    LOCAL_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_JSONL_PATH.exists():
        samples: List[Dict[str, str]] = [
            {
                "prompt": "Explain price-to-earnings ratio (P/E) in simple terms.",
                "response": "P/E compares a company’s share price to its earnings per share; higher P/E suggests investors expect more growth.",
            },
            {
                "prompt": "What is free cash flow (FCF)?",
                "response": "FCF is cash generated after operating expenses and capital expenditures; it’s the money available to pay debt, dividends, or reinvest.",
            },
            {
                "prompt": "Give a one-sentence summary of EBITDA.",
                "response": "EBITDA is earnings before interest, taxes, depreciation, and amortization—used to compare core operating profitability across firms.",
            },
            {
                "prompt": "Define beta in finance.",
                "response": "Beta measures a stock’s volatility relative to the market; beta > 1 is more volatile than the market, while beta < 1 is less volatile.",
            },
        ]
        with LOCAL_JSONL_PATH.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


def format_example(prompt: str, response: str, tokenizer: AutoTokenizer) -> str:
    """
    Build a chat-style training string.
    Prefer the model's chat template (best for Qwen/Mistral).
    Fallback to a simple user/assistant format if template missing.
    """
    messages = [
        {"role": "system", "content": "You are a helpful finance tutor."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Fallback generic format
    return f"<|system|>\nYou are a helpful finance tutor.\n<|user|>\n{prompt}\n<|assistant|>\n{response}\n"


def get_dataset(tokenizer: AutoTokenizer) -> Dataset:
    """
    Prefer a local finance JSONL. Optionally fall back to a tiny public dataset.
    Returns a HuggingFace Dataset with a 'text' column.
    """
    if USE_LOCAL_FINANCE:
        ensure_local_finance_jsonl()
        ds = load_dataset("json", data_files=str(LOCAL_JSONL_PATH), split="train")
        def to_text(example: Dict[str, Any]) -> Dict[str, str]:
            return {"text": format_example(example["prompt"], example["response"], tokenizer)}
        return ds.map(to_text, remove_columns=ds.column_names)
    else:
        # Small, public, and reliable; good for quick scaffolding
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
        ds = ds.rename_column("text", "text")
        return ds


def main() -> None:
    # Reproducibility
    set_seed(SEED)

    # Detect device
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    print(f"Device: {device}")

    # Tokenizer (trust_remote_code helps for Qwen/Mistral custom code)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    # Ensure pad token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    dtype = torch.bfloat16 if has_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        token=HF_TOKEN,
        device_map="auto" if has_cuda else "cpu",   # <- force CPU mapping on non-GPU to avoid offload_dir issues
        low_cpu_mem_usage=False,                     # simpler on Windows CPU
    )

    # Reduce memory pressure during training
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    # Optional: gradient checkpointing (better for GPU RAM; slows training on CPU)
    if has_cuda and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA config (covers common attention/MLP projections across Qwen/Mistral)
    lora_cfg = LoraConfig(
        r=8,                    # light for CPU; bump to 16 on GPU
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Data -> tokenize
    ds = get_dataset(tokenizer)
    print(f"Loaded {len(ds)} training samples")

    def tok_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding=False,   # dynamic padding via collator
        )

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training args
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if not has_cuda else 8,
        learning_rate=1e-4 if not has_cuda else 2e-4,
        num_train_epochs=1,                # bump when you have more data/GPU
        logging_steps=5,
        save_steps=100,
        save_total_limit=1,
        bf16=has_cuda,                     # True only if CUDA available
        fp16=False,                        # use bf16 if CUDA; fp16 off by default
        optim="adamw_torch",
        report_to=[],                      # no wandb by default
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Saved LoRA adapter + tokenizer to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
