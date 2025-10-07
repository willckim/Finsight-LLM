import type { NextRequest } from "next/server";

const HF_URL = process.env.HF_INFERENCE_URL!;
const HF_TOKEN = process.env.HF_TOKEN || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const DEFAULT_PROVIDER = (process.env.PROVIDER_DEFAULT || "finetuned").toLowerCase();

// Match server config
const SERVER_MAX_INPUT = 3072;
const SERVER_MAX_TOTAL = 4096;
const SERVER_MAX_NEW = SERVER_MAX_TOTAL - SERVER_MAX_INPUT; // 1024

function joinMessages(messages: { role: string; content: string }[]) {
  return messages.map((m) => `${m.role}: ${m.content}`).join("\n") + "\nassistant:";
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const messages = body?.messages ?? [];
  const provider = (req.headers.get("x-llm") || DEFAULT_PROVIDER).toLowerCase();

  // Clamp requested max_new_tokens to stay within server budget
  const reqMaxNew = Number(body?.max_new_tokens ?? 256);
  const maxNew = Math.max(
    16,
    Math.min(SERVER_MAX_NEW, Number.isFinite(reqMaxNew) ? Math.floor(reqMaxNew) : 256)
  );

  if (provider === "finetuned") {
    if (!HF_URL) {
      return Response.json({ error: "HF_INFERENCE_URL not set" }, { status: 500 });
    }

    const r = await fetch(HF_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {}),
      },
      body: JSON.stringify({
        inputs: joinMessages(messages),
        parameters: {
          max_new_tokens: maxNew,
          temperature: 0.2,
          // stop: ["\nuser:", "\nUser:", "\nassistant:"], // optional guards
        },
      }),
    });

    if (!r.ok) {
      return Response.json({ error: `HF error: ${r.status} ${await r.text()}` }, { status: 502 });
    }

    const out = await r.json();
    // TGI non-stream may return array or object; normalize:
    const text =
      (Array.isArray(out) && out[0]?.generated_text
        ? String(out[0].generated_text)
        : (out?.generated_text ?? "")) as string;

    // If using a chat-style prompt format, trim any leading "assistant:"
    const cleaned = text.split("assistant:").pop()?.trim() ?? text.trim();

    return Response.json({ provider: "finetuned", text: cleaned });
  }

  // OpenAI fallback
  if (!OPENAI_API_KEY) {
    return Response.json({ error: "OPENAI_API_KEY missing" }, { status: 500 });
  }

  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      messages,
      temperature: 0.2,
      max_tokens: maxNew, // mirror same user control
    }),
  });

  if (!r.ok) {
    return Response.json({ error: `OpenAI error: ${r.status} ${await r.text()}` }, { status: 502 });
  }

  const j = await r.json();
  return Response.json({
    provider: "openai",
    text: j.choices?.[0]?.message?.content ?? "",
  });
}
