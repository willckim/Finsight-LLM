"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { PlusCircle, Copy, Trash2 } from "lucide-react";

const API_BASE = ""; // use your /api/chat route

// Match your HF Endpoint settings
const SERVER_MAX_INPUT = 3072;
const SERVER_MAX_TOTAL = 4096;
const SERVER_MAX_NEW = SERVER_MAX_TOTAL - SERVER_MAX_INPUT; // 1024

// Keep history light: system + last N messages (user/assistant turns)
const MAX_HISTORY_TURNS = 12;

type Role = "system" | "user" | "assistant";
type Message = { role: Role; content: string };

type ChatOK = { provider: string; text: string };
type ChatErr = { error: string };
type ChatResp = ChatOK | ChatErr;

const isChatErr = (r: ChatResp): r is ChatErr => (r as ChatErr).error !== undefined;
const msg = (role: Role, content: string): Message => ({ role, content });

const SYSTEM_PROMPT = "You are a finance expert. Keep responses concise and helpful.";

function trimHistory(messages: Message[]): Message[] {
  const system = messages.find((m) => m.role === "system") ?? msg("system", SYSTEM_PROMPT);
  const rest = messages.filter((m) => m.role !== "system");
  const trimmed = rest.slice(-MAX_HISTORY_TURNS);
  return [system, ...trimmed];
}

export default function Page() {
  // Start BLANK (only system)
  const [messages, setMessages] = useState<Message[]>([msg("system", SYSTEM_PROMPT)]);
  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(512);
  const [provider, setProvider] = useState<"finetuned" | "openai">("finetuned");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");
  const [compactView, setCompactView] = useState(false);
  const [copied, setCopied] = useState(false);

  const listRef = useRef<HTMLDivElement>(null);

  // Smooth scroll to latest message
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const safeMaxNew = useMemo(
    () => Math.max(16, Math.min(SERVER_MAX_NEW, Math.floor(maxNewTokens))),
    [maxNewTokens]
  );

  async function callChat(history: Message[], requestedMaxNew: number) {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-llm": provider,
      },
      body: JSON.stringify({
        messages: history,
        max_new_tokens: Math.max(16, Math.min(SERVER_MAX_NEW, Math.floor(requestedMaxNew))),
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
      const withUser = [...messages, msg("user", prompt.trim())];
      const history = trimHistory(withUser);
      setMessages(withUser);

      const out = await callChat(history, safeMaxNew);
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
    // Reset to only the system message
    setMessages([msg("system", SYSTEM_PROMPT)]);
    setPrompt("");
    setErr("");
  }

  function clearAll() {
    // Same as newChat for now; keep as separate action if you later persist threads
    newChat();
  }

  function deleteMessageAt(index: number) {
    // Prevent removing the system message
    setMessages((prev) => prev.filter((_, i) => i !== index || prev[i].role === "system"));
  }

  async function copyLatest() {
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (!last?.content) return;
    await navigator.clipboard.writeText(last.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  // Render list (hide system). If compact, show only latest assistant reply.
  const rendered = (() => {
    const list = messages
      .map((m, idx) => ({ m, idx }))
      .filter((x) => x.m.role !== "system");

    if (compactView) {
      const latestAssistant = [...list].reverse().find((x) => x.m.role === "assistant");
      return latestAssistant ? [latestAssistant] : [];
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

            {/* Controls row */}
            <div className="flex flex-wrap items-center gap-3 justify-between">
              <div className="flex flex-wrap items-center gap-3">
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
                      const clamped = Math.max(16, Math.min(SERVER_MAX_NEW, Number.isNaN(n) ? 512 : n));
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
                    onChange={(e) => setProvider(e.target.value as "finetuned" | "openai")}
                    className="pill-input"
                  >
                    <option value="finetuned">Fine-tuned</option>
                    <option value="openai">OpenAI</option>
                  </select>
                </div>
              </div>

              <button onClick={run} disabled={loading || !prompt.trim()} className="btn-primary">
                {loading ? "Running…" : "Ask FinSight"}
              </button>
            </div>

            {/* Action bar */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                onClick={newChat}
                type="button"
                className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-sm hover:bg-white/10"
                title="Start a fresh thread"
              >
                <PlusCircle className="h-4 w-4" />
                New chat
              </button>

              <button
                onClick={copyLatest}
                type="button"
                className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-sm hover:bg-white/10"
                title="Copy the latest assistant reply"
              >
                <Copy className="h-4 w-4" />
                {copied ? "Copied!" : "Copy latest"}
              </button>

              <button
                onClick={clearAll}
                type="button"
                className="inline-flex items-center gap-2 rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-sm hover:bg-red-500/20"
                title="Clear everything"
              >
                <Trash2 className="h-4 w-4" />
                Clear all
              </button>

              <label className="ml-auto inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={compactView}
                  onChange={(e) => setCompactView(e.target.checked)}
                  className="h-4 w-4 accent-white/80"
                />
                Compact view
              </label>
            </div>

            {err && <div className="alert error mt-2">{err}</div>}
            <p className="card-subtitle mt-1">
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

          {/* Scrollable, soft-wrapped list to prevent cut-offs */}
          <div
            ref={listRef}
            className="max-h-[60vh] overflow-y-auto pr-1 space-y-2"
          >
            {rendered.map(({ m, idx }) => (
              <div
                key={idx}
                className={[
                  "flex items-start justify-between gap-3 rounded-xl border px-3 py-2",
                  m.role === "user" ? "bg-white/5 border-white/10" : "bg-white/3 border-white/10",
                ].join(" ")}
                style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
              >
                <div className="min-w-0">
                  <strong>{m.role === "user" ? "You" : "FinSight"}: </strong>
                  {m.content}
                </div>
                <button
                  className="inline-flex items-center justify-center rounded-md border border-white/10 px-2 text-sm hover:bg-white/10"
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
            <li>Ask for comparisons: “P/E vs EV/EBITDA for capital-intensive firms.”</li>
            <li>Request examples: “Show PEG math with 20% growth.”</li>
            <li>Constrain output: “Explain free cash flow in 4 bullets.”</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
