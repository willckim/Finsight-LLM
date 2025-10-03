# eval/run_eval.py
import os, json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import evaluate as hf_eval  # Hugging Face evaluate

# --------- Env / Config ----------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")  # optional
MODEL_DIR = os.getenv("OUTPUT_DIR", "models/mistral7b-lora")      # local LoRA dir
FALLBACK_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
DEV_PATH = os.getenv("DEV_JSONL", "data/dev.jsonl")
USE_BASELINE = os.getenv("USE_BASELINE", "0") == "1"
MAX_NEW_TOKENS = int(os.getenv("EVAL_MAX_NEW_TOKENS", "64"))      # shorter = fairer vs short refs
SYSTEM_MSG = os.getenv("EVAL_SYSTEM_MSG", "You are a concise finance tutor. Answer in 1–2 sentences.")

# Silence HF’s info/warning spam (you can override by setting TRANSFORMERS_VERBOSITY)
if os.getenv("TRANSFORMERS_VERBOSITY") is None:
    hf_logging.set_verbosity_error()


# --------- Helpers ----------
def is_valid_lora_dir(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    # Treat as valid if an adapter or a full config exists
    return (p / "adapter_config.json").exists() or (p / "config.json").exists()


def format_chat(tok, user_msg: str, system_msg: str = SYSTEM_MSG):
    """Use the model's chat template (critical for instruct models)."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_and_tok():
    if USE_BASELINE:
        model_id = FALLBACK_MODEL
    else:
        model_id = MODEL_DIR if is_valid_lora_dir(MODEL_DIR) else FALLBACK_MODEL

    print(f"Loading model from: {model_id}")

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True, token=HF_TOKEN
    )

    # CPU-friendly defaults; will auto-use CUDA if available
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=not torch.cuda.is_available(),
        token=HF_TOKEN,
    )
    return model, tok


def postprocess(text: str) -> str:
    return text.replace("Assistant:", "").strip()


def generate(model, tok, prompt, max_new_tokens=MAX_NEW_TOKENS):
    chat_str = format_chat(tok, prompt)
    inputs = tok(chat_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # greedy for stable metrics
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return postprocess(tok.decode(out[0], skip_special_tokens=True))


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# --------- Main ----------
def main():
    # Metrics
    bleu = hf_eval.load("sacrebleu")
    rouge = hf_eval.load("rouge")

    # Data
    if not Path(DEV_PATH).exists():
        raise FileNotFoundError(f"Dev set not found at {DEV_PATH}. Create a JSONL first.")
    dataset = load_jsonl(DEV_PATH)
    print(f"Loaded {len(dataset)} dev samples from {DEV_PATH}")

    # Model
    model, tok = load_model_and_tok()

    # Inference
    preds, refs = [], []
    for row in dataset:
        gen = generate(model, tok, row["prompt"])
        preds.append(gen)
        refs.append([row["ref"]])

    # Scores
    results = {
        "bleu": bleu.compute(predictions=preds, references=refs),
        "rouge": rouge.compute(predictions=preds, references=[r[0] for r in refs]),
    }
    print(json.dumps(results, indent=2))

    # Persist results + predictions (timestamped)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("eval").mkdir(parents=True, exist_ok=True)
    Path(f"eval/results_{ts}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    dump = [{"prompt": d["prompt"], "ref": d["ref"], "pred": p} for d, p in zip(dataset, preds)]
    Path(f"eval/preds_{ts}.jsonl").write_text("\n".join(json.dumps(x) for x in dump), encoding="utf-8")
    print(f"Saved eval to eval/results_{ts}.json and eval/preds_{ts}.jsonl")


if __name__ == "__main__":
    main()
