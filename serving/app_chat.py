# serving/app_chat.py
import os, json
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5-mini")
HF_URL         = os.getenv("HF_INFERENCE_URL", "")   # e.g. https://xyz.endpoints.huggingface.cloud
HF_TOKEN       = os.getenv("HF_TOKEN", "")
DEFAULT        = os.getenv("PROVIDER_DEFAULT", "finetuned").lower()

# CORS: add your Vercel domain
CORS_ORIGINS   = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,https://*.vercel.app"
).split(",")]

app = FastAPI(title="Domain-FT Chat Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"ok": True, "provider_default": DEFAULT}

def join_messages(messages: List[Dict[str, Any]]) -> str:
    # Convert OpenAI chat format → a single prompt string (for HF endpoints)
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    provider = (req.headers.get("x-llm") or DEFAULT).lower()

    if provider == "finetuned":
        if not HF_URL:
            return {"error": "HF_INFERENCE_URL not set"}, 500
        prompt = join_messages(messages)
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.2}}
        headers = {"Content-Type": "application/json"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(HF_URL, headers=headers, json=payload)
        r.raise_for_status()
        out = r.json()
        # Handle common HF return shapes
        if isinstance(out, list) and out and "generated_text" in out[0]:
            text = out[0]["generated_text"].split("assistant:", 1)[-1].strip()
        elif isinstance(out, dict) and "generated_text" in out:
            text = out["generated_text"]
        else:
            text = json.dumps(out)
        return {"provider": "finetuned", "text": text}

    # ---- OpenAI fallback
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY missing"}, 500

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    r.raise_for_status()
    out = r.json()
    text = out["choices"][0]["message"]["content"]
    return {"provider": "openai", "text": text}
