export async function chat(messages: {role: "user"|"assistant"|"system"; content: string;}[], provider: "finetuned"|"openai"="finetuned") {
  const base = process.env.NEXT_PUBLIC_API_BASE!;
  const r = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-llm": provider },
    body: JSON.stringify({ messages }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ provider: string; text: string }>;
}
