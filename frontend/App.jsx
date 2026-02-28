import { useState, useEffect, useRef, useCallback } from "react";

// Monaco Editor loaded via CDN in index.html
// Language detection utility
const detectLanguage = (code) => {
  if (code.includes("def ") || code.includes("import ") && code.includes(":")) return "python";
  if (code.includes("function ") || code.includes("const ") || code.includes("=>")) return "javascript";
  if (code.includes("public class") || code.includes("void main")) return "java";
  if (code.includes("func ") && code.includes("package ")) return "go";
  if (code.includes("fn ") && code.includes("let mut")) return "rust";
  if (code.includes("#include")) return "cpp";
  return "python";
};

const SEVERITY_CONFIG = {
  "🚨": { label: "Critical", color: "#ff4757", bg: "rgba(255,71,87,0.1)", border: "rgba(255,71,87,0.3)" },
  "⚠️": { label: "Warning", color: "#ffa502", bg: "rgba(255,165,2,0.1)", border: "rgba(255,165,2,0.3)" },
  "💡": { label: "Suggestion", color: "#2ed573", bg: "rgba(46,213,115,0.1)", border: "rgba(46,213,115,0.3)" },
};

const WS_URL = "ws://localhost:8000/ws/review";

// ── Review Block Parser ──────────────────────────────────────────────────────
function parseReview(text) {
  const blocks = [];
  const lines = text.split("\n");
  let current = null;

  for (const line of lines) {
    const emoji = ["🚨", "⚠️", "💡"].find((e) => line.startsWith(e));
    if (emoji) {
      if (current) blocks.push(current);
      current = { emoji, text: line.slice(emoji.length + 1).trim(), code: null };
    } else if (current) {
      if (line.startsWith("```")) {
        if (!current.code && !line.endsWith("```")) {
          current.code = "";
        } else if (current.code !== null && line === "```") {
          // end of code block — keep
        }
      } else if (current.code !== null) {
        current.code += (current.code ? "\n" : "") + line;
      } else {
        current.text += "\n" + line;
      }
    } else {
      if (blocks.length === 0 && line.trim()) {
        blocks.push({ emoji: null, text: line, code: null });
      } else if (blocks.length > 0) {
        blocks[blocks.length - 1].text += "\n" + line;
      }
    }
  }
  if (current) blocks.push(current);
  return blocks;
}

// ── Components ───────────────────────────────────────────────────────────────
function ReviewBlock({ block, index, animate }) {
  const config = block.emoji ? SEVERITY_CONFIG[block.emoji] : null;
  return (
    <div
      style={{
        opacity: animate ? 1 : 0,
        transform: animate ? "translateY(0)" : "translateY(12px)",
        transition: `all 0.4s cubic-bezier(0.16,1,0.3,1) ${index * 80}ms`,
        borderLeft: config ? `3px solid ${config.border}` : "3px solid rgba(255,255,255,0.1)",
        background: config ? config.bg : "rgba(255,255,255,0.03)",
        borderRadius: "0 8px 8px 0",
        padding: "14px 18px",
        marginBottom: "12px",
      }}
    >
      {config && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{ fontSize: 16 }}>{block.emoji}</span>
          <span style={{ color: config.color, fontFamily: "'Space Mono', monospace", fontSize: 11, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase" }}>
            {config.label}
          </span>
        </div>
      )}
      <p style={{ color: "#c9d1d9", lineHeight: 1.7, margin: 0, fontSize: 14, fontFamily: "'Inter', sans-serif", whiteSpace: "pre-wrap" }}>
        {block.text.trim()}
      </p>
      {block.code && (
        <pre style={{
          background: "rgba(0,0,0,0.4)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 6,
          padding: "12px 16px",
          marginTop: 12,
          overflowX: "auto",
          fontSize: 12,
          fontFamily: "'Space Mono', monospace",
          color: "#a8ff78",
          lineHeight: 1.6,
        }}>
          {block.code}
        </pre>
      )}
    </div>
  );
}

function StreamingCursor() {
  return (
    <span style={{
      display: "inline-block",
      width: 8,
      height: 16,
      background: "#58a6ff",
      marginLeft: 2,
      verticalAlign: "middle",
      animation: "blink 1s step-end infinite",
    }} />
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function CodeReviewApp() {
  const [code, setCode] = useState(`def authenticate(username, password):
    # TODO: review this auth function
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    if result:
        session['user'] = username
        return True
    return False`);
  const [language, setLanguage] = useState("python");
  const [review, setReview] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [animateBlocks, setAnimateBlocks] = useState(false);

  const wsRef = useRef(null);
  const reviewEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll review panel
  useEffect(() => {
    reviewEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [review]);

  // Animate blocks when streaming stops
  useEffect(() => {
    if (!streaming && review) {
      setTimeout(() => setAnimateBlocks(true), 50);
    } else {
      setAnimateBlocks(false);
    }
  }, [streaming]);

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    ws.onopen = () => { setConnected(true); setError(null); };
    ws.onclose = () => { setConnected(false); setStreaming(false); };
    ws.onerror = () => { setError("Connection failed. Is the backend running?"); setConnected(false); };
    ws.onmessage = (e) => {
      if (e.data === "__DONE__") {
        setStreaming(false);
      } else {
        setReview((prev) => prev + e.data);
      }
    };
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();
    return () => wsRef.current?.close();
  }, [connectWS]);

  const handleReview = () => {
    if (!code.trim()) return;
    if (!connected) { connectWS(); return; }

    setReview("");
    setStreaming(true);
    setAnimateBlocks(false);
    setError(null);

    wsRef.current.send(JSON.stringify({ code, language }));
  };

  const handleCodeChange = (e) => {
    const val = e.target.value;
    setCode(val);
    setLanguage(detectLanguage(val));
  };

  const blocks = parseReview(review);

  const LANGUAGES = ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "csharp"];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0e17; }
        @keyframes blink { 50% { opacity: 0; } }
        @keyframes pulse-dot { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.4); opacity: 0.6; } }
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes scanline {
          0% { background-position: 0 0; }
          100% { background-position: 0 100px; }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        textarea:focus { outline: none; }
        .review-btn:hover { background: rgba(88,166,255,0.15) !important; border-color: rgba(88,166,255,0.6) !important; transform: translateY(-1px); }
        .review-btn:active { transform: translateY(0); }
        .review-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none !important; }
        .lang-btn { cursor: pointer; transition: all 0.15s; }
        .lang-btn:hover { background: rgba(255,255,255,0.08) !important; }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "#0a0e17",
        fontFamily: "'Inter', sans-serif",
        color: "#e6edf3",
        display: "flex",
        flexDirection: "column",
      }}>

        {/* Header */}
        <header style={{
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          padding: "16px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "rgba(255,255,255,0.02)",
          backdropFilter: "blur(10px)",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 32,
              height: 32,
              background: "linear-gradient(135deg, #58a6ff, #1f6feb)",
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
            }}>⚡</div>
            <div>
              <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 14, fontWeight: 700, letterSpacing: 1 }}>
                CodeReview<span style={{ color: "#58a6ff" }}>AI</span>
              </div>
              <div style={{ fontSize: 10, color: "#8b949e", letterSpacing: 1.5, textTransform: "uppercase", fontFamily: "'Space Mono', monospace" }}>
                DeepSeek-Coder · Fine-tuned
              </div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: connected ? "#3fb950" : "#f85149",
              animation: connected ? "pulse-dot 2s infinite" : "none",
            }} />
            <span style={{ fontSize: 12, color: "#8b949e", fontFamily: "'Space Mono', monospace" }}>
              {connected ? "connected" : "disconnected"}
            </span>
          </div>
        </header>

        {/* Main Content */}
        <div style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 0,
          height: "calc(100vh - 65px)",
        }}>

          {/* LEFT: Code Editor */}
          <div style={{
            borderRight: "1px solid rgba(255,255,255,0.06)",
            display: "flex",
            flexDirection: "column",
          }}>
            {/* Editor toolbar */}
            <div style={{
              padding: "12px 20px",
              borderBottom: "1px solid rgba(255,255,255,0.06)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              background: "rgba(255,255,255,0.01)",
            }}>
              <div style={{ display: "flex", gap: 6 }}>
                {LANGUAGES.map((lang) => (
                  <button
                    key={lang}
                    className="lang-btn"
                    onClick={() => setLanguage(lang)}
                    style={{
                      padding: "3px 10px",
                      borderRadius: 4,
                      border: "1px solid",
                      borderColor: language === lang ? "rgba(88,166,255,0.5)" : "rgba(255,255,255,0.08)",
                      background: language === lang ? "rgba(88,166,255,0.1)" : "transparent",
                      color: language === lang ? "#58a6ff" : "#8b949e",
                      fontSize: 11,
                      fontFamily: "'Space Mono', monospace",
                      cursor: "pointer",
                    }}
                  >
                    {lang}
                  </button>
                ))}
              </div>

              <button
                className="review-btn"
                onClick={handleReview}
                disabled={streaming || !code.trim()}
                style={{
                  padding: "8px 20px",
                  background: "rgba(88,166,255,0.08)",
                  border: "1px solid rgba(88,166,255,0.3)",
                  borderRadius: 6,
                  color: "#58a6ff",
                  fontFamily: "'Space Mono', monospace",
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: "pointer",
                  letterSpacing: 1,
                  transition: "all 0.2s",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                {streaming ? (
                  <>
                    <span style={{ display: "inline-block", animation: "pulse-dot 0.8s infinite" }}>●</span>
                    ANALYZING...
                  </>
                ) : (
                  <>⚡ REVIEW CODE</>
                )}
              </button>
            </div>

            {/* Code textarea */}
            <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
              {/* Line numbers */}
              <div style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: 48,
                background: "rgba(0,0,0,0.2)",
                borderRight: "1px solid rgba(255,255,255,0.04)",
                overflowY: "hidden",
                paddingTop: 20,
              }}>
                {code.split("\n").map((_, i) => (
                  <div key={i} style={{
                    height: "1.65em",
                    textAlign: "right",
                    paddingRight: 12,
                    fontSize: 12,
                    fontFamily: "'Space Mono', monospace",
                    color: "rgba(255,255,255,0.2)",
                    lineHeight: "1.65em",
                  }}>
                    {i + 1}
                  </div>
                ))}
              </div>

              <textarea
                ref={textareaRef}
                value={code}
                onChange={handleCodeChange}
                spellCheck={false}
                style={{
                  position: "absolute",
                  inset: 0,
                  paddingLeft: 60,
                  paddingTop: 20,
                  paddingRight: 20,
                  paddingBottom: 20,
                  background: "transparent",
                  border: "none",
                  color: "#e6edf3",
                  fontFamily: "'Space Mono', monospace",
                  fontSize: 13,
                  lineHeight: "1.65em",
                  resize: "none",
                  width: "100%",
                  height: "100%",
                  tabSize: 4,
                }}
                placeholder="// Paste your code here..."
              />
            </div>
          </div>

          {/* RIGHT: Review Panel */}
          <div style={{
            display: "flex",
            flexDirection: "column",
            background: "#0d1117",
          }}>
            {/* Panel header */}
            <div style={{
              padding: "12px 20px",
              borderBottom: "1px solid rgba(255,255,255,0.06)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              background: "rgba(255,255,255,0.01)",
            }}>
              <span style={{
                fontFamily: "'Space Mono', monospace",
                fontSize: 11,
                color: "#8b949e",
                letterSpacing: 1.5,
                textTransform: "uppercase",
              }}>
                Review Output
              </span>
              {review && !streaming && (
                <div style={{ display: "flex", gap: 8 }}>
                  {Object.entries(SEVERITY_CONFIG).map(([emoji, cfg]) => {
                    const count = blocks.filter((b) => b.emoji === emoji).length;
                    if (!count) return null;
                    return (
                      <div key={emoji} style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 4,
                        padding: "2px 8px",
                        borderRadius: 12,
                        background: cfg.bg,
                        border: `1px solid ${cfg.border}`,
                      }}>
                        <span style={{ fontSize: 11 }}>{emoji}</span>
                        <span style={{ fontSize: 11, color: cfg.color, fontFamily: "'Space Mono', monospace", fontWeight: 700 }}>{count}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Review content */}
            <div style={{
              flex: 1,
              overflowY: "auto",
              padding: "20px",
            }}>
              {error && (
                <div style={{
                  background: "rgba(248,81,73,0.1)",
                  border: "1px solid rgba(248,81,73,0.3)",
                  borderRadius: 8,
                  padding: "16px 20px",
                  color: "#f85149",
                  fontFamily: "'Space Mono', monospace",
                  fontSize: 13,
                  animation: "fadeSlideIn 0.3s ease",
                }}>
                  ❌ {error}
                  <div style={{ marginTop: 8, fontSize: 11, color: "#8b949e" }}>
                    Make sure your backend is running: <code>python server.py</code>
                  </div>
                </div>
              )}

              {!review && !streaming && !error && (
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                  gap: 16,
                  opacity: 0.4,
                }}>
                  <div style={{ fontSize: 48 }}>🔍</div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 13, marginBottom: 6 }}>Ready to review</div>
                    <div style={{ fontSize: 12, color: "#8b949e" }}>Paste your code and click Review</div>
                  </div>
                </div>
              )}

              {streaming && !review && (
                <div style={{ color: "#8b949e", fontFamily: "'Space Mono', monospace", fontSize: 13 }}>
                  Analyzing code <StreamingCursor />
                </div>
              )}

              {/* Streaming raw view */}
              {streaming && review && (
                <div style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: 13,
                  lineHeight: 1.8,
                  color: "#c9d1d9",
                  whiteSpace: "pre-wrap",
                }}>
                  {review}
                  <StreamingCursor />
                </div>
              )}

              {/* Parsed block view (after streaming) */}
              {!streaming && review && (
                <div>
                  {blocks.map((block, i) => (
                    <ReviewBlock key={i} block={block} index={i} animate={animateBlocks} />
                  ))}
                </div>
              )}

              <div ref={reviewEndRef} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
