"use client";
import { useState } from "react";

export default function ModelPicker({ onChange }: { onChange: (v: "finetuned"|"openai") => void }) {
  const [v, setV] = useState<"finetuned"|"openai">("finetuned");
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm">Model:</label>
      <select
        className="border rounded p-2"
        value={v}
        onChange={(e) => { const nv = e.target.value as "finetuned"|"openai"; setV(nv); onChange(nv); }}
      >
        <option value="finetuned">Fine-tuned</option>
        <option value="openai">OpenAI</option>
      </select>
    </div>
  );
}
