import os, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_DIR   = os.path.abspath("models/qwen3b-lora")
OUT_DIR    = os.path.abspath("./merged")

CFG_PATH   = os.path.join(LORA_DIR, "adapter_config.json")
WEIGHTS    = os.path.join(LORA_DIR, "adapter_model.safetensors")

print("Base :", BASE_MODEL)
print("LoRA :", LORA_DIR)
print("Conf :", CFG_PATH)
print("Weights:", WEIGHTS)

# Read JSON with BOM-friendly codec
with open(CFG_PATH, "r", encoding="utf-8-sig") as f:
    cfg = json.load(f)

lcfg = LoraConfig(
    r=cfg["r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg.get("lora_dropout", 0.0),
    bias=cfg.get("bias", "none"),
    target_modules=cfg["target_modules"],
    task_type=cfg["task_type"],
    base_model_name_or_path=cfg.get("base_model_name_or_path", BASE_MODEL),
)

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tok  = AutoTokenizer.from_pretrained(BASE_MODEL)

peft_model = get_peft_model(base, lcfg)
state_dict = safe_load(WEIGHTS)
set_peft_model_state_dict(peft_model, state_dict)

merged = peft_model.merge_and_unload()
os.makedirs(OUT_DIR, exist_ok=True)
merged.save_pretrained(OUT_DIR, safe_serialization=True)
tok.save_pretrained(OUT_DIR)

print(f"✅ Merged model saved to {OUT_DIR}")
