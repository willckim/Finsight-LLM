"use client";

import { useState } from "react";

const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");

type ChatOK = { provider: string; text: string };
type ChatErr = { error: string };
type ChatResp = ChatOK | ChatErr;

function isChatErr(r: ChatResp): r is ChatErr {
  return (r as ChatErr).error !== undefined;
}

export default function Page() {
  const [prompt, setPrompt] = useState(
    "Explain the difference between P/E and PEG with a simple example."
  );
  const [maxNewTokens, setMaxNewTokens] = useState(768);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");
  const [provider, setProvider] = useState<"finetuned" | "openai">("finetuned");

  function endsCleanly(s: string) {
    return /[.?!]["')\]]?\s*$/.test(s.trim());
  }

  async function callChat(userContent: string) {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-llm": provider, // "finetuned" | "openai"
      },
      body: JSON.stringify({
        // OpenAI-style messages; backend will join them for HF endpoint
        messages: [
          {
            role: "system",
            content: `You are a finance expert. Keep responses under ~${maxNewTokens} tokens.`,
          },
          { role: "user", content: userContent },
        ],
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
    setResult("");

    try {
      const CONTINUE =
        "\n\nContinue from where you left off. Do not repeat earlier text.";
      let acc = "";
      let rounds = 0;
      const MAX_ROUNDS = 3;

      while (rounds < MAX_ROUNDS) {
        const userMsg = rounds === 0 ? prompt : prompt + CONTINUE;
        const out = await callChat(userMsg);

        if (isChatErr(out)) throw new Error(out.error);
        const text = out.text ?? "";
        acc += (acc ? "\n" : "") + text;
        setResult(acc);

        if (endsCleanly(text)) break;
        rounds += 1;
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg || "Request failed");
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

          <label htmlFor="prompt" className="sr-only">
            Prompt
          </label>
          <div className="input-shell">
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              placeholder="e.g., Walk me through DCF like I’m in high school."
              className="input-textarea"
            />

            <div className="controls-row">
              <div className="pill">
                Max tokens
                <input
                  aria-label="Max new tokens"
                  type="number"
                  min={16}
                  max={4096}
                  value={maxNewTokens}
                  onChange={(e) => {
                    const n = parseInt(e.target.value || "768", 10);
                    setMaxNewTokens(Number.isNaN(n) ? 768 : n);
                  }}
                  className="pill-input"
                />
              </div>

              {/* Model picker */}
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
          <div className="card-title">Response</div>
          {!result && !loading && (
            <p className="card-subtitle">Your answer will appear here.</p>
          )}
          {loading && <p className="card-subtitle">Generating…</p>}
          {result && <pre className="result-pre">{result}</pre>}
        </div>

        <div className="card">
          <div className="card-title">Tips</div>
          <ul className="tips">
            <li>
              Ask for comparisons: “P/E vs EV/EBITDA for capital-intensive
              firms.”
            </li>
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
