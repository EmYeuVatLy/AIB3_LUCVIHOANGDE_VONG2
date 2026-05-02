import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { gsap } from "gsap";
import {
  Activity,
  BarChart3,
  Building2,
  CheckCircle2,
  Database,
  Dot,
  AlertTriangle,
  Cpu,
  FileText,
  Leaf,
  Play,
  Search,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";

const defaultText = `VINAMILK công bố thông tin quản trị, phát triển bền vững và hiệu quả vận hành trong năm 2024.

Hội đồng quản trị giám sát chiến lược ESG, quản lý rủi ro và công bố thông tin cho cổ đông.

Doanh nghiệp báo cáo các chỉ tiêu liên quan đến năng lượng, nước, chất thải, lao động, đào tạo và đóng góp cộng đồng.`;

const sectors = [
  "Financials",
  "Consumer Staples",
  "Information Technology",
  "Energy",
  "Materials",
  "Industrials",
];

function parseLogLine(line) {
  const text = String(line || "").trim();
  if (!text) {
    return { tone: "neutral", label: "LOG", title: "", detail: "" };
  }

  if (text.startsWith("❌") || text.includes("[ERROR]")) {
    return { tone: "error", label: "ERROR", title: text.replace("❌", "").trim(), detail: "" };
  }
  if (text.startsWith("⚠")) {
    return { tone: "warning", label: "WARNING", title: text.replace("⚠", "").trim(), detail: "" };
  }
  if (text.startsWith("✅")) {
    return { tone: "success", label: "OK", title: text.replace("✅", "").trim(), detail: "" };
  }
  if (/^\[\d\/\d\]/.test(text)) {
    const [, phase, detail = ""] = text.match(/^\[(\d\/\d)\]\s*(.*)$/) || [];
    return { tone: "phase", label: phase || "PHASE", title: detail, detail: "" };
  }
  if (text.startsWith("[SCREENING]")) {
    return { tone: "screening", label: "SCREEN", title: text.replace("[SCREENING]", "").trim(), detail: "" };
  }
  if (text.startsWith("[SCORING]")) {
    return { tone: "scoring", label: "SCORING", title: text.replace("[SCORING]", "").trim(), detail: "" };
  }
  if (text.startsWith("[RETRIEVAL AUDIT]")) {
    return { tone: "audit", label: "AUDIT", title: text.replace("[RETRIEVAL AUDIT]", "").trim(), detail: "" };
  }
  if (/^\[(SL\d+)\]/.test(text)) {
    const [, rule] = text.match(/^\[(SL\d+)\]\s*(.*)$/) || [];
    const content = text.replace(/^\[(SL\d+)\]\s*/, "");
    const tone = content.includes("Không vi phạm") ? "success" : "warning";
    return { tone, label: rule || "SL", title: content, detail: "" };
  }
  if (/^\[\d+\/\d+\]/.test(text)) {
    const [, progress, rest = ""] = text.match(/^\[(\d+\/\d+)\]\s*(.*)$/) || [];
    return { tone: "progress", label: progress || "STEP", title: rest, detail: "" };
  }
  if (text.startsWith("Retrieval preflight:")) {
    return { tone: "summary", label: "PREFLIGHT", title: "Retrieval preflight", detail: text.replace("Retrieval preflight:", "").trim() };
  }
  if (text.startsWith("Retrieval audit:")) {
    return { tone: "summary", label: "AUDIT", title: "Retrieval audit", detail: text.replace("Retrieval audit:", "").trim() };
  }
  if (text.startsWith("  📄") || text.startsWith("📄")) {
    return { tone: "artifact", label: "FILE", title: text.replace("📄", "").trim(), detail: "" };
  }
  return { tone: "neutral", label: "LOG", title: text, detail: "" };
}

function LogIcon({ tone }) {
  if (tone === "error") return <XCircle size={16} />;
  if (tone === "warning") return <AlertTriangle size={16} />;
  if (tone === "success") return <CheckCircle2 size={16} />;
  if (tone === "phase") return <Cpu size={16} />;
  if (tone === "audit") return <Search size={16} />;
  if (tone === "summary") return <BarChart3 size={16} />;
  if (tone === "artifact") return <FileText size={16} />;
  return <Dot size={16} />;
}

export default function App() {
  const orbRef = useRef(null);
  const logConsoleRef = useRef(null);
  const [form, setForm] = useState({
    company_name: "VNM",
    industry_sector: "Consumer Staples",
    year: "2024",
    report_text: defaultText,
    skip_preflight_gate: true,
  });
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("idle");
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.to(orbRef.current, {
        x: 30,
        y: 22,
        duration: 5,
        scale: 1.05,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      });
    });
    return () => ctx.revert();
  }, []);

  useEffect(() => {
    const node = logConsoleRef.current;
    if (!node) {
      return;
    }
    node.scrollTop = node.scrollHeight;
  }, [logs]);

  useEffect(() => {
    if (!jobId) {
      return undefined;
    }

    const source = new EventSource(`http://localhost:8000/api/jobs/${jobId}/stream`);
    source.onmessage = async (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "log") {
        setLogs((current) => [...current, payload.message]);
      }
      if (payload.type === "done") {
        source.close();
        const response = await fetch(`http://localhost:8000/api/jobs/${jobId}`);
        const data = await response.json();
        setStatus(data.status);
        setResult(data.result);
        setError(data.error?.message || "");
      }
    };
    source.onerror = () => {
      source.close();
      setStatus("failed");
      setError("Không kết nối được stream log từ API server.");
    };

    return () => source.close();
  }, [jobId]);

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus("submitting");
    setLogs([]);
    setResult(null);
    setError("");
    setJobId("");

    try {
      const response = await fetch("http://localhost:8000/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company_name: form.company_name,
          industry_sector: form.industry_sector,
          year: Number(form.year),
          report_text: form.report_text,
          skip_preflight_gate: form.skip_preflight_gate,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Không thể bắt đầu scoring.");
      }
      setJobId(data.job_id);
      setStatus("running");
    } catch (submitError) {
      setStatus("failed");
      setError(submitError.message);
    }
  }

  const score100 = result?.scores?.score_100 ?? result?.scores?.percentage ?? 0;

  return (
    <div className="page-shell">
      <div className="page-noise" />
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Leaf size={18} />
          </div>
          <div>
            <p>ESG Interactive Console</p>
            <span>AI Scoring Engine v2.0</span>
          </div>
        </div>
        <div className="status-indicator">
          <div className={`status-dot ${status === "running" ? "running" : status === "done" ? "success" : status === "failed" ? "failed" : "idle"}`} />
          <span>{status.toUpperCase()}</span>
        </div>
      </header>

      <main>
        <section className="hero interactive-hero">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65 }}
          >
            <span className="eyebrow">
              <Sparkles size={14} />
              Interactive ESG scoring
            </span>
            <h1>
              Nhập toàn bộ `txt`, chạy scoring thật, và xem <span>log realtime</span>.
            </h1>
            <p>
              Frontend này gọi trực tiếp vào API Python, ghi nhận tiến trình
              pipeline và hiển thị kết quả ESG sau khi job hoàn tất.
            </p>
          </motion.div>

          <div className="hero-visual interactive-visual">
            <div className="orb" ref={orbRef} />
            <div className="score-panel compact-panel">
              <div className="panel-top">
                <span className="chip">Pipeline status</span>
                <Activity size={18} />
              </div>
              <div className="status-stack">
                <div>
                  <p>Trạng thái</p>
                  <strong>{status}</strong>
                </div>
                <div>
                  <p>Số dòng log</p>
                  <strong>{logs.length}</strong>
                </div>
                <div>
                  <p>Job ID</p>
                  <strong>{jobId ? jobId.slice(0, 8) : "--"}</strong>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <motion.form
            className="input-card"
            onSubmit={handleSubmit}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.55 }}
          >
            <div className="card-head">
              <div>
                <span>Input workspace</span>
                <h3>Bộ dữ liệu text cho scoring</h3>
              </div>
              <FileText size={20} />
            </div>

            <div className="field-grid">
              <label>
                <span>Doanh nghiệp</span>
                <input
                  value={form.company_name}
                  onChange={(event) => setForm({ ...form, company_name: event.target.value })}
                />
              </label>
              <label>
                <span>Năm</span>
                <input
                  type="number"
                  value={form.year}
                  onChange={(event) => setForm({ ...form, year: event.target.value })}
                />
              </label>
              <label className="full">
                <span>Ngành</span>
                <select
                  value={form.industry_sector}
                  onChange={(event) => setForm({ ...form, industry_sector: event.target.value })}
                >
                  {sectors.map((sector) => (
                    <option key={sector} value={sector}>
                      {sector}
                    </option>
                  ))}
                </select>
              </label>
              <label className="full">
                <span>Nội dung báo cáo</span>
                <textarea
                  value={form.report_text}
                  onChange={(event) => setForm({ ...form, report_text: event.target.value })}
                  rows={16}
                />
              </label>
              <label className="full toggle-field">
                <input
                  type="checkbox"
                  checked={form.skip_preflight_gate}
                  onChange={(event) => setForm({ ...form, skip_preflight_gate: event.target.checked })}
                />
                <div>
                  <span>Bỏ qua preflight gate</span>
                  <small>Phù hợp cho chế độ interactive khi text ngắn hoặc chỉ là bản tóm tắt.</small>
                </div>
              </label>
            </div>

            <button className="primary-btn submit-btn" type="submit" disabled={status === "running"}>
              <Play size={16} />
              {status === "running" ? "Đang chạy scoring" : "Chạy ESG scoring"}
            </button>

            {error ? <p className="error-text">{error}</p> : null}
          </motion.form>

          <div className="side-stack">
            <motion.section
              className="result-card"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.18, duration: 0.55 }}
            >
              <div className="card-head">
                <div>
                  <span>Result snapshot</span>
                  <h3>Kết quả tổng hợp</h3>
                </div>
                <BarChart3 size={20} />
              </div>

              <div className="result-grid">
                <div className="metric-box">
                  <p>OVERALL SCORE</p>
                  <strong>{Number(score100 || 0).toFixed(2)}</strong>
                  <div className="metric-bar-container">
                    <div className="metric-bar g" style={{ width: `${score100}%` }} />
                  </div>
                </div>
                <div className="metric-box">
                  <p>ENVIRONMENTAL (E)</p>
                  <strong>{Number(result?.scores?.E || 0).toFixed(1)}</strong>
                  <div className="metric-bar-container">
                    <div className="metric-bar e" style={{ width: `${(result?.scores?.E / 100) * 100}%` }} />
                  </div>
                </div>
                <div className="metric-box">
                  <p>SOCIAL (S)</p>
                  <strong>{Number(result?.scores?.S || 0).toFixed(1)}</strong>
                  <div className="metric-bar-container">
                    <div className="metric-bar s" style={{ width: `${(result?.scores?.S / 100) * 100}%` }} />
                  </div>
                </div>
                <div className="metric-box">
                  <p>GOVERNANCE (G)</p>
                  <strong>{Number(result?.scores?.G || 0).toFixed(1)}</strong>
                  <div className="metric-bar-container">
                    <div className="metric-bar g" style={{ width: `${(result?.scores?.G / 100) * 100}%` }} />
                  </div>
                </div>
              </div>
            </motion.section>

            <motion.section
              className="log-card"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.26, duration: 0.55 }}
            >
              <div className="card-head">
                <div>
                  <span>Live execution log</span>
                  <h3>Timeline thực thi</h3>
                </div>
                <Database size={20} />
              </div>
              <div className="log-console" ref={logConsoleRef}>
                {logs.length ? (
                  logs.map((line, index) => {
                    const parsed = parseLogLine(line);
                    return (
                      <div className={`log-entry log-${parsed.tone}`} key={`${index}-${line}`}>
                        <div className="log-entry-icon">
                          <LogIcon tone={parsed.tone} />
                        </div>
                        <div className="log-entry-body">
                          <div className="log-entry-top">
                            <span className="log-badge">{parsed.label}</span>
                            <strong>{parsed.title}</strong>
                          </div>
                          {parsed.detail ? <p>{parsed.detail}</p> : null}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="log-empty">Chưa có log. Bắt đầu job để xem timeline pipeline chạy.</div>
                )}
              </div>
            </motion.section>

            <motion.section
              className="result-card"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.34, duration: 0.55 }}
            >
              <div className="card-head">
                <div>
                  <span>What is wired</span>
                  <h3>Luồng interactive</h3>
                </div>
                <ShieldCheck size={20} />
              </div>
              <div className="mini-list">
                <div><Building2 size={16} /> Frontend gửi text, doanh nghiệp, ngành, năm</div>
                <div><Database size={16} /> `api_server.py` tạo job và stream stdout</div>
                <div><BarChart3 size={16} /> `main.py` scoring trên input `.txt`</div>
              </div>
            </motion.section>
          </div>
        </section>
      </main>
    </div>
  );
}
