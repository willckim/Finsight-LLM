# serving/app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

load_dotenv()
ONNX_DIR = os.getenv("ONNX_DIR", "models/qwen3b-onnx")

app = FastAPI(title="Domain LLM Inference (ONNX)")

# --- CORS: allow Next.js dev to call this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inp(BaseModel):
    prompt: str
    max_new_tokens: int = 128  # you can bump default to 512 if you like

tok = AutoTokenizer.from_pretrained(ONNX_DIR, use_fast=True)
model = ORTModelForCausalLM.from_pretrained(
    ONNX_DIR,
    provider="CPUExecutionProvider",
    file_name="model.onnx",   # works with model.onnx + model.onnx_data
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(inp: Inp):
    """
    Returns:
      {
        completion: str,           # generated text (without echoing the prompt)
        finish_reason: "stop"|"length",
        usage: {prompt_tokens, completion_tokens, total_tokens},
        max_new_tokens: int
      }
    """
    # 1) Tokenize
    inputs = tok(inp.prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # 2) Generate
    gen = model.generate(
        **inputs,
        max_new_tokens=inp.max_new_tokens,
        use_cache=False,
        # make sure generation can pad/stop cleanly
        pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # 3) Slice off the prompt so we don't echo it back
    gen_len = gen.shape[1]
    new_tokens = max(gen_len - input_len, 0)
    completion_ids = gen[0][input_len:]
    completion = tok.decode(completion_ids, skip_special_tokens=True).strip()

    # 4) Heuristic finish_reason: if we hit the cap, call it "length"
    finish_reason = "length" if new_tokens >= inp.max_new_tokens else "stop"

    return {
        "completion": completion,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": int(input_len),
            "completion_tokens": int(new_tokens),
            "total_tokens": int(input_len + new_tokens),
        },
        "max_new_tokens": inp.max_new_tokens,
    }
