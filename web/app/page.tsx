"use client";

import { useState } from "react";

const API_BASE = ""; // so fetch goes to "/api/chat"

// Match your HF Endpoint settings
const SERVER_MAX_INPUT = 3072;
const SERVER_MAX_TOTAL = 4096;
const SERVER_MAX_NEW = SERVER_MAX_TOTAL - SERVER_MAX_INPUT; // 1024

type Role = "system" | "user" | "assistant";
type Message = { role: Role; content: string };

type ChatOK = { provider: string; text: string };
type ChatErr = { error: string };
type ChatResp = ChatOK | ChatErr;

function isChatErr(r: ChatResp): r is ChatErr {
  return (r as ChatErr).error !== undefined;
}

const msg = (role: Role, content: string): Message => ({ role, content });

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([
    msg("system", "You are a finance expert. Keep responses concise and helpful."),
    msg("user", "Explain the difference between P/E and PEG with a simple example."),
  ]);

  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(512);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");
  const [provider, setProvider] = useState<"finetuned" | "openai">("finetuned");

  function endsCleanly(s: string) {
    return /[.?!]["')\]]?\s*$/.test(s.trim());
  }

  async function callChat(history: Message[], requestedMaxNew: number) {
    const safeMaxNew = Math.max(16, Math.min(SERVER_MAX_NEW, Math.floor(requestedMaxNew)));

    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-llm": provider,
      },
      body: JSON.stringify({
        messages: history,
        max_new_tokens: safeMaxNew,
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return (await res.json()) as ChatResp;
  }

  async function run() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setErr("");

    try {
      // append new user message
      const newMessages: Message[] = [...messages, msg("user", prompt)];
      setMessages(newMessages);

      const out = await callChat(newMessages, maxNewTokens);
      if (isChatErr(out)) throw new Error(out.error);

      const text = out.text ?? "";
      const updated: Message[] = [...newMessages, msg("assistant", text)];
      setMessages(updated);
      setResult((prev) => prev + (prev ? "\n\n" : "") + text);
      setPrompt("");
    } catch (e: unknown) {
      const msgText = e instanceof Error ? e.message : String(e);
      setErr(msgText || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 md:grid-cols-[1.15fr_1fr]">
      {/* Left column */}
      <section className="space-y-6">
        <div className="card">
          <div className="card-title">Ask a Finance Question</div>
          <p className="card-subtitle">
            Ratios • Filings (10-K/10-Q) • Valuation (DCF, comps) • Statements
          </p>

          <div className="input-shell">
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              placeholder="e.g., What does a low PEG ratio mean?"
              className="input-textarea"
            />

            <div className="controls-row">
              <div className="pill">
                Max new tokens
                <input
                  aria-label="Max new tokens"
                  type="number"
                  min={16}
                  max={SERVER_MAX_NEW}
                  value={maxNewTokens}
                  onChange={(e) => {
                    const n = parseInt(e.target.value || "512", 10);
                    const clamped = Math.max(
                      16,
                      Math.min(SERVER_MAX_NEW, Number.isNaN(n) ? 512 : n)
                    );
                    setMaxNewTokens(clamped);
                  }}
                  className="pill-input"
                />
              </div>

              <div className="pill">
                Model
                <select
                  aria-label="Model provider"
                  value={provider}
                  onChange={(e) =>
                    setProvider(e.target.value as "finetuned" | "openai")
                  }
                  className="pill-input"
                >
                  <option value="finetuned">Fine-tuned</option>
                  <option value="openai">OpenAI</option>
                </select>
              </div>

              <button
                onClick={run}
                disabled={loading || !prompt.trim()}
                className="btn-primary"
              >
                <SendIcon />
                {loading ? "Running…" : "Ask FinSight"}
              </button>
            </div>

            {err && <div className="alert error">{err}</div>}
            <p className="card-subtitle">
              Server limits: input ≤ {SERVER_MAX_INPUT}, total ≤ {SERVER_MAX_TOTAL}, so new ≤ {SERVER_MAX_NEW}.
            </p>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Quick Finance Prompts</div>
          <div className="chips">
            {[
              "Explain EBITDA in plain English.",
              "P/E vs PEG — when is PEG better?",
              "What’s beta and how is it used?",
              "Walk me through a 3-statement model at a high level.",
              "DCF in 5 steps.",
            ].map((q) => (
              <button
                key={q}
                className="chip"
                onClick={() => setPrompt(q)}
                type="button"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Right column */}
      <section className="space-y-6">
        <div className="card">
          <div className="card-title">Conversation</div>
          {!messages.length && !loading && (
            <p className="card-subtitle">Your chat will appear here.</p>
          )}
          {loading && <p className="card-subtitle">Generating…</p>}
          <div className="result-pre">
            {messages
              .filter((m) => m.role !== "system")
              .map((m, i) => (
                <p key={i}>
                  <strong>{m.role === "user" ? "You" : "FinSight"}:</strong>{" "}
                  {m.content}
                </p>
              ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Tips</div>
          <ul className="tips">
            <li>Ask for comparisons: “P/E vs EV/EBITDA for capital-intensive firms.”</li>
            <li>Request examples: “Show PEG math with 20% growth.”</li>
            <li>Constrain output: “Explain free cash flow in 4 bullets.”</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

function SendIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      aria-hidden="true"
      className="-translate-y-[1px]"
    >
      <path d="M2 21l20-9L2 3v7l14 2-14 2v7z" fill="currentColor" />
    </svg>
  );
}
