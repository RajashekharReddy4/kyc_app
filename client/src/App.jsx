import React, { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:4000";

function FileRow({ label, multiple, onChange, accept }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: 12, alignItems: "center", marginBottom: 10 }}>
      <div style={{ fontWeight: 600 }}>{label}</div>
      <input type="file" accept={accept} multiple={multiple} onChange={(e) => onChange(multiple ? e.target.files : e.target.files[0])} />
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 14, padding: 16, background: "white", boxShadow: "0 1px 10px rgba(0,0,0,0.05)" }}>
      <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
}

export default function App() {
  const [caseId, setCaseId] = useState(`CASE_${Date.now()}`);
  const [lang, setLang] = useState("eng");

  const [docs, setDocs] = useState(null);
  const [idFront, setIdFront] = useState(null);
  const [idBack, setIdBack] = useState(null);
  const [selfie, setSelfie] = useState(null);
  const [audios, setAudios] = useState(null);

  const [status, setStatus] = useState("Idle");
  const [analysis, setAnalysis] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const canUpload = useMemo(() => {
    return (docs && docs.length) || idFront || idBack || selfie || (audios && audios.length);
  }, [docs, idFront, idBack, selfie, audios]);

  async function uploadCase() {
    setError(null);
    setStatus("Uploading...");
    const fd = new FormData();
    fd.append("caseId", caseId);
    if (docs) Array.from(docs).forEach((f) => fd.append("docs", f));
    if (idFront) fd.append("idFront", idFront);
    if (idBack) fd.append("idBack", idBack);
    if (selfie) fd.append("selfie", selfie);
    if (audios) Array.from(audios).forEach((f) => fd.append("audios", f));

    const res = await fetch(`${API_BASE}/api/cases`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setStatus("Uploaded");
    return data.caseId;
  }

  async function analyzeCase(cid) {
    setStatus("Analyzing (worker)... this can take a bit for OCR/audio");
    const res = await fetch(`${API_BASE}/api/cases/${cid}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setAnalysis(data);
    setStatus("Analysis complete");
  }

  async function fetchReport(cid) {
    const res = await fetch(`${API_BASE}/api/cases/${cid}/report`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setReport(data);
  }

  async function run() {
    try {
      if (!canUpload) return;
      const cid = await uploadCase();
      await analyzeCase(cid);
      await fetchReport(cid);
    } catch (e) {
      setError(String(e?.message || e));
      setStatus("Error");
    }
  }

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", background: "#f8fafc", minHeight: "100vh" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{ margin: 0 }}>KYC Authenticity Analyzer</h1>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
          <Card title="1) Upload inputs">
            <div style={{ display: "grid", gap: 10 }}>
              <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: 12, alignItems: "center" }}>
                <div style={{ fontWeight: 600 }}>Case ID</div>
                <input value={caseId} onChange={(e) => setCaseId(e.target.value)} style={{ padding: 8, borderRadius: 10, border: "1px solid #e5e7eb" }} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: 12, alignItems: "center" }}>
                <div style={{ fontWeight: 600 }}>OCR lang</div>
                <input value={lang} onChange={(e) => setLang(e.target.value)} style={{ padding: 8, borderRadius: 10, border: "1px solid #e5e7eb" }} />
              </div>

              <div style={{ marginTop: 6 }}>
                <FileRow label="Documents (PDF)" multiple accept=".pdf" onChange={setDocs} />
                <FileRow label="ID front" accept="image/*" onChange={setIdFront} />
                <FileRow label="ID back" accept="image/*" onChange={setIdBack} />
                <FileRow label="Selfie" accept="image/*" onChange={setSelfie} />
                <FileRow label="Audio clips" multiple accept="audio/*" onChange={setAudios} />
              </div>

              <button
                onClick={run}
                disabled={!canUpload}
                style={{
                  padding: "10px 12px",
                  borderRadius: 12,
                  border: "1px solid #0f172a",
                  background: canUpload ? "#0f172a" : "#94a3b8",
                  color: "white",
                  cursor: canUpload ? "pointer" : "not-allowed",
                  fontWeight: 700
                }}
              >
                Run analysis
              </button>

              <div style={{ color: "#334155" }}><b>Status:</b> {status}</div>
              {error && <div style={{ color: "#b91c1c", whiteSpace: "pre-wrap" }}>{error}</div>}
            </div>
          </Card>

          <Card title="2) Result summary">
            {!analysis ? (
              <div style={{ color: "#64748b" }}>Run an analysis to see results.</div>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 10 }}>
                  <div style={{ fontWeight: 700 }}>Decision</div>
                  <div>{analysis.decision}</div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 10 }}>
                  <div style={{ fontWeight: 700 }}>Score</div>
                  <div>{analysis.score_0_100} / 100</div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 10 }}>
  <div style={{ fontWeight: 700 }}>Fraud Confidence</div>
  <div
    style={{
      fontWeight: 700,
      color:
        analysis.fraud_confidence > 0.75
          ? "#dc2626"
          : analysis.fraud_confidence > 0.45
          ? "#f59e0b"
          : "#16a34a"
    }}
  >
    {(analysis.report.fraud_confidence * 100).toFixed(1)}%
  </div>
</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>Risks</div>
                <pre style={{ background: "#0b1220", color: "#e2e8f0", padding: 12, borderRadius: 12, overflowX: "auto" }}>
{analysis?.report.risks
  ? JSON.stringify(analysis.report.risks, null, 2)
  : "No risks available"}
</pre>
              </div>
            )}
          </Card>
        </div>

        <div style={{ marginTop: 14 }}>
          <Card title="3) Full report JSON (from MongoDB)">
            {!report ? (
              <div style={{ color: "#64748b" }}>Report will appear here after analysis.</div>
            ) : (
              <pre style={{ background: "#0b1220", color: "#e2e8f0", padding: 12, borderRadius: 12, overflowX: "auto" }}>
{JSON.stringify(report, null, 2)}
              </pre>
            )}
          </Card>
        </div>

        <div style={{ marginTop: 14, color: "#64748b" }}>
          Tip: uploaded files are not stored in the application's database.
        </div>
      </div>
    </div>
  );
}
