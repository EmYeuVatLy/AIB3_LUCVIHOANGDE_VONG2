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

function CountingNumber({ value, duration = 1 }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = parseFloat(value) || 0;
    if (start === end) return;
    const increment = (end - start) / (duration * 60);
    const handle = setInterval(() => {
      start += increment;
      if ((increment > 0 && start >= end) || (increment < 0 && start <= end)) {
        setDisplay(end);
        clearInterval(handle);
      } else {
        setDisplay(start);
      }
    }, 1000 / 60);
    return () => clearInterval(handle);
  }, [value, duration]);
  return <span>{display.toFixed(display > 0 && display < 100 ? 1 : 1)}</span>;
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
        x: "random(-20, 20)",
        y: "random(-15, 15)",
        duration: 8,
        scale: "random(0.95, 1.1)",
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      });
    });
    return () => ctx.revert();
  }, []);

  useEffect(() => {
    const node = logConsoleRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    if (!jobId) return undefined;
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
      if (!response.ok) throw new Error(data.error || "Không thể bắt đầu scoring.");
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
          <motion.div 
            className="brand-mark"
            whileHover={{ rotate: 180 }}
            transition={{ duration: 0.6 }}
          >
            <Leaf size={18} />
          </motion.div>
          <div>
            <p>Interactive Console</p>
            <span>ESG AI Scoring Engine v2.0</span>
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
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <span className="eyebrow">
              <Sparkles size={14} />
              AI-Powered ESG Analysis
            </span>
            <h1>
              Phân tích dữ liệu ESG <span>thông minh</span>.
            </h1>
            <p>
              Hệ thống tự động quét báo cáo, trích xuất bằng chứng và chấm điểm 
              theo bộ tiêu chuẩn VNSI dựa trên mô hình ngôn ngữ lớn Qwen3-30B.
            </p>
          </motion.div>

          <motion.div 
            className="hero-visual interactive-visual"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
          >
            <div className={`orb ${status === "running" ? "active" : ""}`} ref={orbRef} />
            <div className="score-panel compact-panel">
              <div className="panel-top">
                <span className="chip">Real-time Pipeline</span>
                <Activity size={18} className={status === "running" ? "animate-pulse" : ""} />
              </div>
              <div className="status-stack">
                <div className="status-item">
                  <p>Trạng thái</p>
                  <strong>{status === "idle" ? "Sẵn sàng" : status.toUpperCase()}</strong>
                </div>
                <div className="status-item">
                  <p>Công ty</p>
                  <strong>{form.company_name}</strong>
                </div>
                <div className="status-item">
                  <p>Ngành</p>
                  <strong>{form.industry_sector.split(" ")[0]}</strong>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        <section className="workspace-grid">
          <motion.form
            className="input-card"
            onSubmit={handleSubmit}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            <div className="card-head">
              <div>
                <span>Analysis Input</span>
                <h3>Dữ liệu đầu vào</h3>
              </div>
              <FileText size={20} />
            </div>

            <div className="field-grid">
              <label>
                <span>Tên công ty</span>
                <input
                  value={form.company_name}
                  onChange={(event) => setForm({ ...form, company_name: event.target.value })}
                />
              </label>
              <label>
              <span>Năm báo cáo</span>
                <input
                  type="number"
                  value={form.year}
                  onChange={(event) => setForm({ ...form, year: event.target.value })}
                />
              </label>
              <label className="full">
                <span>Lĩnh vực</span>
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
                <span>Văn bản báo cáo</span>
                <textarea
                  placeholder="Dán nội dung báo cáo tại đây..."
                  value={form.report_text}
                  onChange={(event) => setForm({ ...form, report_text: event.target.value })}
                  rows={12}
                />
              </label>
              <label className="full toggle-field">
                <input
                  type="checkbox"
                  checked={form.skip_preflight_gate}
                  onChange={(event) => setForm({ ...form, skip_preflight_gate: event.target.checked })}
                />
                <div>
                  <span>Bỏ qua kiểm tra sơ bộ (Skip Preflight)</span>
                </div>
              </label>
            </div>

            <motion.button 
              className="primary-btn submit-btn" 
              type="submit" 
              disabled={status === "running"}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Play size={16} />
              {status === "running" ? "Đang xử lý..." : "Bắt đầu chấm điểm"}
            </motion.button>
          </motion.form>

          <div className="side-stack">
            <motion.section
              className="result-card"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
            >
              <div className="card-head">
                <div>
                  <span>Scoring Results</span>
                  <h3>Kết quả ESG</h3>
                </div>
                <BarChart3 size={20} />
              </div>

              <div className="result-grid">
                <motion.div 
                  className="metric-box large"
                  animate={{ scale: result ? [1, 1.02, 1] : 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <p>TỔNG ĐIỂM</p>
                  <strong><CountingNumber value={score100} /><span>%</span></strong>
                </motion.div>
                
                {[
                  { label: "MÔI TRƯỜNG (E)", val: result?.scores?.E, cls: "e" },
                  { label: "XÃ HỘI (S)", val: result?.scores?.S, cls: "s" },
                  { label: "QUẢN TRỊ (G)", val: result?.scores?.G, cls: "g" },
                ].map((item, idx) => (
                  <motion.div 
                    key={item.label}
                    className="metric-box"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 + idx * 0.1 }}
                  >
                    <p>{item.label}</p>
                    <div className="score-row">
                      <strong><CountingNumber value={item.val} /></strong>
                      <div className="metric-bar-container">
                        <motion.div 
                          className={`metric-bar ${item.cls}`} 
                          initial={{ width: 0 }}
                          animate={{ width: `${(item.val / 100) * 100}%` }}
                          transition={{ duration: 1.5, ease: "circOut" }}
                        />
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.section>

            <motion.section
              className="log-card"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.6 }}
            >
              <div className="card-head">
                <div>
                  <span>Execution Stream</span>
                  <h3>Nhật ký hệ thống</h3>
                </div>
                <Database size={20} className={status === "running" ? "animate-spin-slow" : ""} />
              </div>
              <div className="log-console" ref={logConsoleRef}>
                {logs.length ? (
                  logs.map((line, index) => {
                    const parsed = parseLogLine(line);
                    return (
                      <motion.div 
                        className={`log-entry log-${parsed.tone}`} 
                        key={`${index}-${line}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3 }}
                      >
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
                      </motion.div>
                    );
                  })
                ) : (
                  <div className="log-empty">Hệ thống đang chờ lệnh...</div>
                )}
              </div>
            </motion.section>
          </div>
        </section>
      </main>
    </div>
  );
}
