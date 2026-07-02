"use client";

import { useState, useRef, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

type Citation = {
  source_doc: string;
  page_number: number;
  score: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  confidence?: number;
  processingTime?: number;
  isError?: boolean;
};

type HealthData = {
  status: string;
  ollama_connected: boolean;
  postgres_connected: boolean;
  vector_count: number;
};

const SUGGESTED = [
  "What are my rights if my employer doesn't pay salary on time?",
  "How do I file an RTI application?",
  "What legal protection do I have against domestic violence?",
  "What is the punishment for theft under IPC?",
];

const ACT_LABELS: Record<string, string> = {
  "IPC_1860.pdf": "IPC 1860",
  "CrPC_1973.pdf": "CrPC 1973",
  "RTI_Act_2005.pdf": "RTI Act",
  "POCSO_2012.pdf": "POCSO",
  "DV_Act_2005.pdf": "DV Act",
  "Consumer_Protection_2019.pdf": "Consumer Protection",
  "Payment_of_Wages_1936.pdf": "Payment of Wages",
  "LARR_2013.pdf": "LARR 2013",
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [portal, setPortal] = useState<"nagarik" | "vakeel">("nagarik");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((d) => { setHealth(d); setHealthLoading(false); })
      .catch(() => { setHealth(null); setHealthLoading(false); });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendQuery(text?: string) {
    const query = (text ?? input).trim();
    if (!query || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, language: "english", portal }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMessages((m) => [...m, {
        role: "assistant",
        content: data.answer ?? "No answer returned.",
        citations: data.citations,
        confidence: data.confidence_score,
        processingTime: data.processing_time_ms,
      }]);
    } catch {
      setMessages((m) => [...m, {
        role: "assistant",
        content: "Backend unreachable. Ensure Docker containers are running:\n\ndocker-compose up -d",
        isError: true,
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-screen bg-[#0A0A0F] text-white overflow-hidden">

      {/* ── Sidebar ── */}
      <aside className="w-64 shrink-0 border-r border-white/5 flex flex-col bg-[#0D0D15]">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[#F59E0B] to-[#DC2626] flex items-center justify-center text-base">
              ⚖️
            </div>
            <div>
              <p className="font-bold text-sm tracking-wide text-white">Nyaya Setu</p>
              <p className="text-[10px] text-white/30 tracking-wider uppercase">Indian Legal AI</p>
            </div>
          </div>
        </div>

        {/* Portal Toggle */}
        <div className="px-4 pt-5">
          <p className="text-[10px] uppercase tracking-widest text-white/25 mb-2 px-1">Portal</p>
          <div className="flex flex-col gap-1">
            {(["nagarik", "vakeel"] as const).map((p) => (
              <button
                key={p}
                onClick={() => setPortal(p)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all text-left ${
                  portal === p
                    ? "bg-[#F59E0B]/10 text-[#F59E0B] border border-[#F59E0B]/20"
                    : "text-white/40 hover:text-white/70 hover:bg-white/5"
                }`}
              >
                <span className="text-base">{p === "nagarik" ? "🏛️" : "⚖️"}</span>
                <div>
                  <p className="capitalize leading-tight">{p}</p>
                  <p className="text-[10px] opacity-60 font-normal">
                    {p === "nagarik" ? "Citizen guidance" : "Legal professionals"}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Indexed Acts */}
        <div className="px-4 pt-6">
          <p className="text-[10px] uppercase tracking-widest text-white/25 mb-2 px-1">Indexed Acts</p>
          <div className="flex flex-col gap-0.5">
            {Object.values(ACT_LABELS).map((label) => (
              <div key={label} className="flex items-center gap-2 px-3 py-1.5">
                <div className="w-1 h-1 rounded-full bg-[#F59E0B]/50" />
                <p className="text-xs text-white/35">{label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Status */}
        <div className="mt-auto px-4 pb-5">
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
            {healthLoading ? (
              <p className="text-xs text-white/25 animate-pulse">Checking status...</p>
            ) : health ? (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <p className="text-[10px] text-white/30 uppercase tracking-wider">System</p>
                  <span className="text-[10px] text-emerald-400 font-medium">Online</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <p className="text-xs text-white/40">{health.vector_count.toLocaleString()} chunks indexed</p>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className={`w-1.5 h-1.5 rounded-full ${health.ollama_connected ? "bg-emerald-500" : "bg-red-500"}`} />
                  <p className="text-xs text-white/40">Local LLM (Qwen 1.5B)</p>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className={`w-1.5 h-1.5 rounded-full ${health.postgres_connected ? "bg-emerald-500" : "bg-red-500"}`} />
                  <p className="text-xs text-white/40">PostgreSQL + pgvector</p>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500" />
                <p className="text-xs text-red-400">Backend offline</p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* ── Main chat area ── */}
      <main className="flex-1 flex flex-col min-w-0">

        {/* Top bar */}
        <header className="shrink-0 border-b border-white/5 px-8 py-4 flex items-center justify-between bg-[#0A0A0F]/80 backdrop-blur">
          <div>
            <h1 className="text-sm font-semibold text-white/80">
              {portal === "nagarik" ? "Nagarik Portal" : "Vakeel Portal"}
            </h1>
            <p className="text-xs text-white/25 mt-0.5">
              {portal === "nagarik"
                ? "Know your rights under Indian law"
                : "Legal research & clause drafting"}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-white/25">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
            Offline · Private · No data leaves your machine
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6 min-h-0">
          {/* Empty state */}
          {isEmpty && (
            <div className="max-w-2xl mx-auto pt-16">
              <div className="text-center mb-10">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#F59E0B] to-[#DC2626] flex items-center justify-center text-3xl mx-auto mb-4">
                  ⚖️
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">
                  Nyaya Setu
                </h2>
                <p className="text-white/40 text-sm max-w-sm mx-auto leading-relaxed">
                  Ask about your rights under Indian law. I'll cite the exact law section and page — no guessing.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {SUGGESTED.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => sendQuery(q)}
                    className="text-left px-4 py-3.5 rounded-xl border border-white/8 bg-white/[0.02] hover:bg-white/5 hover:border-[#F59E0B]/30 transition-all group"
                  >
                    <p className="text-xs text-white/50 group-hover:text-white/80 leading-relaxed transition-colors">
                      {q}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-4 max-w-3xl ${m.role === "user" ? "ml-auto flex-row-reverse" : ""}`}>
              {/* Avatar */}
              <div className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold ${
                m.role === "user"
                  ? "bg-[#F59E0B] text-black"
                  : "bg-gradient-to-br from-[#F59E0B] to-[#DC2626] text-white"
              }`}>
                {m.role === "user" ? "U" : "⚖"}
              </div>

              {/* Bubble */}
              <div className={`flex-1 min-w-0 ${m.role === "user" ? "items-end flex flex-col" : ""}`}>
                <div className={`rounded-2xl px-5 py-4 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-[#F59E0B] text-black font-medium rounded-tr-sm max-w-lg"
                    : m.isError
                    ? "bg-red-500/10 border border-red-500/20 text-red-300 rounded-tl-sm font-mono text-xs"
                    : "bg-white/[0.04] border border-white/8 text-white/85 rounded-tl-sm"
                }`}>
                  <p className="whitespace-pre-wrap">{m.content}</p>

                  {/* Citations */}
                  {m.role === "assistant" && m.citations && m.citations.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-white/8">
                      <p className="text-[10px] uppercase tracking-widest text-white/25 mb-2">Sources</p>
                      <div className="flex flex-wrap gap-2">
                        {m.citations.slice(0, 5).map((c, ci) => (
                          <span key={ci} className="inline-flex items-center gap-1.5 text-[11px] bg-[#F59E0B]/10 border border-[#F59E0B]/20 text-[#F59E0B] rounded-full px-3 py-1">
                            📖 {ACT_LABELS[c.source_doc] ?? c.source_doc} · p.{c.page_number}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Meta row */}
                {m.role === "assistant" && m.confidence !== undefined && (
                  <div className="mt-2 flex items-center gap-3 px-1">
                    <div className="flex items-center gap-1.5">
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        m.confidence >= 0.5 ? "bg-emerald-500" :
                        m.confidence >= 0.25 ? "bg-amber-500" : "bg-red-500"
                      }`} />
                      <span className="text-[11px] text-white/25">
                        {Math.round(m.confidence * 100)}% confidence
                      </span>
                    </div>
                    {m.processingTime && (
                      <span className="text-[11px] text-white/20">
                        {(m.processingTime / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="flex gap-4 max-w-3xl">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#F59E0B] to-[#DC2626] flex items-center justify-center text-sm shrink-0">
                ⚖
              </div>
              <div className="bg-white/[0.04] border border-white/8 rounded-2xl rounded-tl-sm px-5 py-4 flex items-center gap-1.5">
                {[0, 150, 300].map((delay, i) => (
                  <div
                    key={i}
                    className="w-2 h-2 rounded-full bg-[#F59E0B]/60 animate-bounce"
                    style={{ animationDelay: `${delay}ms` }}
                  />
                ))}
                <span className="text-xs text-white/25 ml-2">Searching {health?.vector_count?.toLocaleString() ?? "25,412"} legal chunks...</span>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 border-t border-white/5 px-8 py-5 bg-[#0A0A0F]">
          <div className="max-w-3xl mx-auto">
            <div className="flex gap-3 items-end">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={`Ask a legal question as a ${portal}... (Enter to send)`}
                  rows={1}
                  className="w-full bg-white/[0.04] border border-white/10 rounded-xl px-4 py-3.5 text-sm text-white placeholder:text-white/20 focus:outline-none focus:border-[#F59E0B]/40 focus:bg-white/[0.06] resize-none transition-all"
                  style={{ minHeight: "52px", maxHeight: "160px" }}
                  onInput={(e) => {
                    const t = e.target as HTMLTextAreaElement;
                    t.style.height = "auto";
                    t.style.height = Math.min(t.scrollHeight, 160) + "px";
                  }}
                />
              </div>
              <button
                onClick={() => sendQuery()}
                disabled={loading || !input.trim()}
                className="shrink-0 w-12 h-12 rounded-xl bg-[#F59E0B] hover:bg-[#FBBF24] disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all active:scale-95"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-black -rotate-45 translate-x-0.5">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <p className="text-center text-[11px] text-white/15 mt-3">
              For educational purposes only · Not a substitute for a registered advocate
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}