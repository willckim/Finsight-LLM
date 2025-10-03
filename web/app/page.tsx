"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

type InferResponse = {
  completion?: string;
  finish_reason?: "stop" | "length" | string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  [k: string]: any; // tolerate extra fields
};

export default function Page() {
  const [prompt, setPrompt] = useState(
    "Explain the difference between P/E and PEG with a simple example."
  );
  const [maxNewTokens, setMaxNewTokens] = useState(768); // ↑ default for longer finance answers
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");

  function endsCleanly(s: string) {
    // ends with ., !, ?, or closing quote/bracket after punctuation
    return /[.?!]["')\]]?\s*$/.test(s.trim());
  }

  async function callInfer(body: { prompt: string; max_new_tokens: number }) {
    const res = await fetch(`${API_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    const json = (await res.json()) as InferResponse;
    const text =
      typeof json.completion === "string"
        ? json.completion
        : JSON.stringify(json);
    const reason = json.finish_reason;
    return { text, reason };
  }

  async function run() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setErr("");
    setResult("");

    try {
      const CONTINUE = "\n\nContinue from where you left off. Do not repeat earlier text.";
      const CAP = Math.max(128, maxNewTokens); // never below 128
      let acc = "";
      let rounds = 0;
      const MAX_ROUNDS = 3; // original + up to 2 continuations

      while (rounds < MAX_ROUNDS) {
        const isFirst = rounds === 0;
        const payload = {
          prompt: isFirst ? prompt : prompt + CONTINUE,
          max_new_tokens: CAP,
        };

        const { text, reason } = await callInfer(payload);
        acc += (acc ? "\n" : "") + text;
        setResult(acc);

        // stop if not length-limited OR looks like a natural sentence end
        if (reason !== "length" || endsCleanly(text)) break;

        rounds += 1;
      }
    } catch (e: any) {
      setErr(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 md:grid-cols-[1.15fr_1fr]">
      {/* Left column */}
      <section className="space-y-6">
        {/* Hero card */}
        <div className="card">
          <div className="card-title">Ask a Finance Question</div>
          <p className="card-subtitle">
            Ratios • Filings (10-K/10-Q) • Valuation (DCF, comps) • Statements
          </p>

          <label htmlFor="prompt" className="sr-only">Prompt</label>
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
                  onChange={(e) =>
                    setMaxNewTokens(parseInt(e.target.value || "768", 10))
                  }
                  className="pill-input"
                />
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

        {/* Quick prompts */}
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
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" className="-translate-y-[1px]">
      <path d="M2 21l20-9L2 3v7l14 2-14 2v7z" fill="currentColor" />
    </svg>
  );
}
