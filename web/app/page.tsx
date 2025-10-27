"use client";

import { useState } from "react";

const API_BASE = "";

// ⚠️ UPDATED LIMITS - matches route.ts
const SERVER_MAX_INPUT = 2048;
const SERVER_MAX_TOTAL = 4096;
const SERVER_MAX_NEW = 2048;  // DOUBLED from 1024

const MAX_HISTORY_TURNS = 12;

type Role = "system" | "user" | "assistant";
type Message = { role: Role; content: string };

type ChatOK = { provider: string; text: string };
type ChatErr = { error: string };
type ChatResp = ChatOK | ChatErr;

const isChatErr = (r: ChatResp): r is ChatErr => (r as ChatErr).error !== undefined;
const msg = (role: Role, content: string): Message => ({ role, content });

const SYSTEM_PROMPT =
  "You are a finance expert. Keep responses concise and helpful.";

function trimHistory(messages: Message[]): Message[] {
  const system = messages.find((m) => m.role === "system") ?? msg("system", SYSTEM_PROMPT);
  const rest = messages.filter((m) => m.role !== "system");
  const trimmed = rest.slice(-MAX_HISTORY_TURNS);
  return [system, ...trimmed];
}

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([msg("system", SYSTEM_PROMPT)]);
  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(1024);  // Higher default
  const [provider, setProvider] = useState<"finetuned" | "openai">("finetuned");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");
  const [compactView, setCompactView] = useState(false);

  async function callChat(history: Message[], requestedMaxNew: number) {
    const safeMaxNew = Math.max(16, Math.min(SERVER_MAX_NEW, Math.floor(requestedMaxNew)));
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-llm": provider,
      },
      body: JSON.stringify({ messages: history, max_new_tokens: safeMaxNew }),
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
      const withUser = [...messages, msg("user", prompt)];
      const history = trimHistory(withUser);
      setMessages(withUser);

      const out = await callChat(history, maxNewTokens);
      if (isChatErr(out)) throw new Error(out.error);

      const text = out.text ?? "";
      setMessages((prev) => [...prev, msg("assistant", text)]);
      setPrompt("");
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e) || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([msg("system", SYSTEM_PROMPT)]);
    setPrompt("");
    setErr("");
  }

  function deleteMessageAt(index: number) {
    setMessages((prev) => prev.filter((_, i) => i !== index || prev[i].role === "system"));
  }

  async function copyLatest() {
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (last?.content) await navigator.clipboard.writeText(last.content);
  }

  const rendered = (() => {
    const list = messages
      .map((m, idx) => ({ m, idx }))
      .filter((x) => x.m.role !== "system");

    if (compactView) {
      const latest = [...list].reverse().find((x) => x.m.role === "assistant");
      return latest ? [latest] : [];
    }
    return list;
  })();

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
              placeholder="e.g., P/E vs PEG — when is PEG better?"
              className="input-textarea"
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
              }}
            />

            <div className="controls-row flex-wrap gap-2">
              <div className="pill">
                Max new tokens
                <input
                  aria-label="Max new tokens"
                  type="number"
                  min={16}
                  max={SERVER_MAX_NEW}
                  value={maxNewTokens}
                  onChange={(e) => {
                    const n = parseInt(e.target.value || "1024", 10);
                    const clamped = Math.max(
                      16,
                      Math.min(SERVER_MAX_NEW, Number.isNaN(n) ? 1024 : n)
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
                {loading ? "Running…" : "Ask FinSight"}
              </button>

              <button onClick={newChat} type="button" className="btn-secondary">
                New chat
              </button>

              <label className="pill cursor-pointer">
                <input
                  type="checkbox"
                  checked={compactView}
                  onChange={(e) => setCompactView(e.target.checked)}
                />
                <span className="ml-2">Compact view</span>
              </label>

              <button onClick={copyLatest} type="button" className="btn-secondary">
                Copy latest
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
              "What does a low PEG mean?",
              "Walk me through a 3-statement model.",
              "DCF in 5 steps.",
            ].map((q) => (
              <button key={q} className="chip" onClick={() => setPrompt(q)} type="button">
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

          {messages.filter((m) => m.role !== "system").length === 0 && !loading && (
            <p className="card-subtitle">Start a new conversation above.</p>
          )}
          {loading && <p className="card-subtitle">Generating…</p>}

          <div className="result-pre">
            {rendered.map(({ m, idx }, i) => (
              <div key={`${idx}-${i}`} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <strong>{m.role === "user" ? "You" : "FinSight"}:</strong> {m.content}
                </div>
                <button
                  className="btn-secondary"
                  style={{ padding: "2px 8px", lineHeight: 1 }}
                  onClick={() => deleteMessageAt(idx)}
                  aria-label="Delete message"
                  title="Delete this message"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Tips</div>
          <ul className="tips">
            <li>Submit with ⌘/Ctrl + Enter.</li>
            <li>Ask for comparisons: "P/E vs EV/EBITDA for capital-intensive firms."</li>
            <li>Request examples: "Show PEG math with 20% growth."</li>
            <li>Constrain output: "Explain free cash flow in 4 bullets."</li>
          </ul>
        </div>
      </section>
    </div>
  );
}