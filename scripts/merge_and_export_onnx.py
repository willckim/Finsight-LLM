import os
from pathlib import Path
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from optimum.onnxruntime import ORTModelForCausalLM

load_dotenv()

BASE_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
LORA_DIR   = os.getenv("OUTPUT_DIR", "models/qwen3b-lora")
MERGED_DIR = os.getenv("MERGED_DIR", "models/qwen3b-merged")
ONNX_DIR   = os.getenv("ONNX_DIR", "models/qwen3b-onnx")
HF_TOKEN   = os.getenv("HF_TOKEN", None)

def safe_load_lora_config():
    """
    Avoid reading adapter_config.json (which can be empty/corrupt on Windows) by
    supplying a minimal, known-good LoraConfig that matches how we trained.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        task_type="CAUSAL_LM",
        inference_mode=True,
    )

def merge_lora():
    print(f"Loading base: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,     # Qwen is safe with this
        token=HF_TOKEN
    )

    # Try the normal route; if it fails due to JSON error, fall back to manual config.
    try:
        print(f"Loading adapter (auto-config): {LORA_DIR}")
        lora = PeftModel.from_pretrained(base, LORA_DIR)
    except Exception as e:
        print("Auto-config load failed, falling back to manual LoraConfig. Reason:", repr(e))
        lcfg = safe_load_lora_config()
        # Create a PEFT-wrapped model and load the adapter weights from dir
        peft_model = get_peft_model(base, lcfg)
        print(f"Loading adapter weights from: {LORA_DIR}")
        peft_model.load_adapter(LORA_DIR, adapter_name="default")
        lora = peft_model

    print("Merging LoRA into base...")
    merged = lora.merge_and_unload()

    print(f"Saving merged model to: {MERGED_DIR}")
    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)

    # Save tokenizer from base (or lora dir if you prefer)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN, trust_remote_code=True)
    tok.save_pretrained(MERGED_DIR)

def export_onnx():
    print(f"Exporting ONNX to: {ONNX_DIR}")
    Path(ONNX_DIR).mkdir(parents=True, exist_ok=True)
    model = ORTModelForCausalLM.from_pretrained(
        MERGED_DIR,
        export=True,
        provider="CPUExecutionProvider"
    )
    model.save_pretrained(ONNX_DIR)

    tok = AutoTokenizer.from_pretrained(MERGED_DIR, use_fast=True)
    tok.save_pretrained(ONNX_DIR)
    print("ONNX export complete.")

if __name__ == "__main__":
    merge_lora()
    export_onnx()
    print("Done. Merged:", MERGED_DIR, "| ONNX:", ONNX_DIR)
