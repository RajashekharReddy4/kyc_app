import React, { useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:4000";

function FileRow({ label, multiple, onChange, accept }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "180px 1fr",
        gap: 12,
        alignItems: "center",
        marginBottom: 10,
      }}
    >
      <div style={{ fontWeight: 600 }}>{label}</div>
      <input
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={(e) =>
          onChange(multiple ? e.target.files : e.target.files[0])
        }
      />
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 14,
        padding: 16,
        background: "white",
        boxShadow: "0 1px 10px rgba(0,0,0,0.05)",
      }}
    >
      <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

const decisionCopy = (decision) => {
  console.log("Decision:", decision);
  const d = (decision || "").toLowerCase();
  if (d === "pass")
    return "Verification Successful — no material inconsistencies detected.";
  if (d === "review")
    return "Manual Review Recommended — additional validation is advised.";
  if (d === "fail")
    return "Integrity Concerns Identified — verification criteria were not satisfied.";
  return "Status unavailable.";
};

export default function App() {
  const [caseId, setCaseId] = useState(`CASE_${Date.now()}`);
  const [lang, setLang] = useState("eng");

  const [docs, setDocs] = useState(null);
  const [idFront, setIdFront] = useState(null);
  const [selfie, setSelfie] = useState(null);
  const [audios, setAudios] = useState(null);

  const [isRecording, setIsRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [liveness, setLiveness] = useState(null);

  const [mode, setMode] = useState("all"); // all | documents | images | audios | video

  const videoRef = useRef(null);

  const [status, setStatus] = useState("Idle");
  const [analysis, setAnalysis] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const canUpload = useMemo(() => {
    return (
      (docs && docs.length) ||
      idFront ||
      selfie ||
      (audios && audios.length) ||
      videoBlob
    );
  }, [docs, idFront, selfie, audios, videoBlob]);

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    videoRef.current.srcObject = stream;
  }
  function stopCamera() {
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  }

  function recordVideo() {
    const stream = videoRef.current?.srcObject;
    if (!stream) {
      alert("Start camera first");
      return;
    }

    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      setVideoBlob(blob);
      setIsRecording(false);
      setStatus("Video recorded");
    };

    setIsRecording(true);

    let secondsLeft = 6;
    setStatus(`Recording ${secondsLeft} seconds...`);

    const timer = setInterval(() => {
      secondsLeft -= 1;
      setStatus(`Recording ${secondsLeft} seconds...`);

      if (secondsLeft <= 0) {
        clearInterval(timer);
        recorder.stop();
        setStatus("Recording complete");
        stopCamera();
      }
    }, 1000);

    recorder.start();
    setTimeout(() => recorder.stop(), 6000);
  }

  async function uploadCase() {
    setError(null);
    setStatus("Uploading...");
    const fd = new FormData();
    fd.append("caseId", caseId);

    if (docs) Array.from(docs).forEach((f) => fd.append("docs", f));
    if (idFront) fd.append("idFront", idFront);
    if (selfie) fd.append("selfie", selfie);
    if (audios) Array.from(audios).forEach((f) => fd.append("audios", f));
    if (videoBlob) fd.append("video", videoBlob, "liveness.webm");

    const res = await fetch(`${API_BASE}/api/cases`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setStatus("Uploaded");
    return data.caseId;
  }

  async function analyzeByMode(cid) {
    setAnalysis(null);
    setReport(null);
    setLiveness(null);

    if (mode === "video") {
      setStatus("Running video liveness verification...");
      const res = await fetch(`${API_BASE}/api/cases/${cid}/video`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setLiveness(data);
      setStatus("Video verification complete");
      return;
    }

    const url =
      mode === "all"
        ? `${API_BASE}/api/cases/${cid}/analyze`
        : `${API_BASE}/api/cases/${cid}/analyze/${mode}`;

    setStatus(`Analyzing (${mode})...`);
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setAnalysis(data);
    setStatus("Analysis complete");
  }

  async function fetchReport(cid) {
    const res = await fetch(`${API_BASE}/api/cases/${cid}/report`);
    if (!res.ok) return; // report only exists for full analyze route
    const data = await res.json();
    setReport(data);
  }

  async function run() {
    try {
      if (!canUpload) return;

      // Ensure video is recorded before video-only analysis
      if (mode === "video" && !videoBlob) {
        alert("Please record the liveness video first.");
        return;
      }

      const cid = await uploadCase();
      await analyzeByMode(cid);

      // Only full pipeline writes report.json
      if (mode === "all") await fetchReport(cid);
    } catch (e) {
      setError(String(e?.message || e));
      setStatus("Error");
    }
  }

  return (
    <div
      style={{
        fontFamily: "system-ui",
        background: "#f8fafc",
        minHeight: "100vh",
      }}
    >
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: 22 }}>
        <h1 style={{ margin: 0 }}>KYC Authenticity Analyzer</h1>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 14,
            marginTop: 14,
          }}
        >
          <Card title="1) Inputs & Run">
            <div style={{ display: "grid", gap: 10 }}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "120px 1fr",
                  gap: 12,
                  alignItems: "center",
                }}
              >
                <div style={{ fontWeight: 600 }}>Case ID</div>
                <input
                  value={caseId}
                  onChange={(e) => setCaseId(e.target.value)}
                  style={{
                    padding: 8,
                    borderRadius: 10,
                    border: "1px solid #e5e7eb",
                  }}
                />
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "120px 1fr",
                  gap: 12,
                  alignItems: "center",
                }}
              >
                <div style={{ fontWeight: 600 }}>Mode</div>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  style={{
                    padding: 8,
                    borderRadius: 10,
                    border: "1px solid #e5e7eb",
                  }}
                >
                  <option value="all">All (PDF + Image + Audio + Video)</option>
                  <option value="documents">PDF only</option>
                  <option value="images">Image only</option>
                  <option value="audios">Audio only</option>
                  <option value="video">Video (Liveness)</option>
                </select>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "120px 1fr",
                  gap: 12,
                  alignItems: "center",
                }}
              >
                <div style={{ fontWeight: 600 }}>OCR lang</div>
                <input
                  value={lang}
                  onChange={(e) => setLang(e.target.value)}
                  style={{
                    padding: 8,
                    borderRadius: 10,
                    border: "1px solid #e5e7eb",
                  }}
                />
              </div>

              <div style={{ marginTop: 6 }}>
                <FileRow
                  label="Documents (PDF)"
                  multiple
                  accept=".pdf"
                  onChange={setDocs}
                />
                <FileRow
                  label="ID Front / Doc Image"
                  accept="image/*"
                  onChange={setIdFront}
                />
                <FileRow label="Selfie" accept="image/*" onChange={setSelfie} />
                <FileRow
                  label="Audio clips"
                  multiple
                  accept="audio/*"
                  onChange={setAudios}
                />
              </div>

              <div
                style={{
                  marginTop: 8,
                  padding: 10,
                  border: "1px dashed #cbd5e1",
                  borderRadius: 12,
                }}
              >
                <div style={{ fontWeight: 700, marginBottom: 6 }}>
                  Video Liveness
                </div>
                <div
                  style={{ fontSize: 13, color: "#64748b", marginBottom: 8 }}
                >
                  Blink once and slowly turn your face left and right.
                </div>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={{
                    width: "100%",
                    maxHeight: 250,
                    borderRadius: 10,
                    background: "#000",
                  }}
                />
                <div
                  style={{
                    marginTop: 10,
                    display: "flex",
                    gap: 10,
                    flexWrap: "wrap",
                  }}
                >
                  <button
                    onClick={startCamera}
                    style={{
                      padding: "8px 12px",
                      borderRadius: 10,
                      border: "1px solid #0f172a",
                      background: "#0f172a",
                      color: "white",
                      cursor: "pointer",
                      fontWeight: 700,
                    }}
                  >
                    Start Camera
                  </button>
                  <button
                    onClick={recordVideo}
                    disabled={isRecording}
                    style={{
                      padding: "8px 12px",
                      borderRadius: 10,
                      border: "1px solid #16a34a",
                      background: isRecording ? "#94a3b8" : "#16a34a",
                      color: "white",
                      cursor: isRecording ? "not-allowed" : "pointer",
                      fontWeight: 700,
                    }}
                  >
                    {isRecording ? "Recording..." : "Record 6s"}
                  </button>
                  <div
                    style={{
                      fontSize: 13,
                      color: "#475569",
                      alignSelf: "center",
                    }}
                  >
                    {videoBlob ? "Video ready ✅" : "No video recorded"}
                  </div>
                </div>
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
                  fontWeight: 800,
                }}
              >
                Run verification
              </button>

              <div style={{ color: "#334155" }}>
                <b>Status:</b> {status}
              </div>
              {error && (
                <div style={{ color: "#b91c1c", whiteSpace: "pre-wrap" }}>
                  {error}
                </div>
              )}
            </div>
          </Card>

          <Card title="2) Results">
            {!analysis && !liveness ? (
              <div style={{ color: "#64748b" }}>
                Run verification to see results.
              </div>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {analysis && (
                  <>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "160px 1fr",
                        gap: 10,
                      }}
                    >
                      <div style={{ fontWeight: 800 }}>Decision</div>
                      <div>{decisionCopy(analysis.decision)}</div>
                    </div>

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "160px 1fr",
                        gap: 10,
                      }}
                    >
                      <div style={{ fontWeight: 800 }}>Score</div>
                      <div>{analysis.score_0_100} / 100</div>
                    </div>

                    <div style={{ fontWeight: 800, marginTop: 4 }}>Risks</div>
                    <pre
                      style={{
                        background: "#0b1220",
                        color: "#e2e8f0",
                        padding: 12,
                        borderRadius: 12,
                        overflowX: "auto",
                      }}
                    >
                      {analysis?.report?.risks
                        ? JSON.stringify(analysis.report.risks, null, 2)
                        : "No risks available"}
                    </pre>
                  </>
                )}

                {liveness && (
                  <>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "160px 1fr",
                        gap: 10,
                      }}
                    >
                      <div style={{ fontWeight: 800 }}>Video Decision</div>
                      <div>{decisionCopy(liveness.decision)}</div>
                    </div>
                    <div
                      style={{
                        marginTop: 4,
                        padding: 12,
                        borderRadius: 12,
                        background: liveness.report?.results?.video
                          ?.liveness_pass
                          ? "#dcfce7"
                          : "#fee2e2",
                        color: liveness.report?.results?.video?.liveness_pass
                          ? "#166534"
                          : "#991b1b",
                        fontWeight: 700,
                      }}
                    >
                      {liveness.report?.results?.video?.liveness_pass
                        ? "Active presence confirmed (blink + head movement detected)."
                        : "Manual review recommended (insufficient liveness indicators)."}
                      <div
                        style={{
                          fontSize: 12,
                          fontWeight: 600,
                          marginTop: 6,
                          opacity: 0.9,
                        }}
                      >
                        Blinks:{" "}
                        {liveness.report?.results?.video?.blink_count ?? "-"} ·
                        Head turn:{" "}
                        {String(
                          liveness.report?.results?.video?.head_turn_detected ??
                            "-",
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </Card>
        </div>

        <div style={{ marginTop: 14 }}>
          <Card title="3) Full report JSON (only for All mode)">
            {!report ? (
              <div style={{ color: "#64748b" }}>
                Run All mode to generate the full report.
              </div>
            ) : (
              <pre
                style={{
                  background: "#0b1220",
                  color: "#e2e8f0",
                  padding: 12,
                  borderRadius: 12,
                  overflowX: "auto",
                }}
              >
                {JSON.stringify(report, null, 2)}
              </pre>
            )}
          </Card>
        </div>

        <div style={{ marginTop: 14, color: "#64748b" }}>
          Note: Uploaded files are stored on the server filesystem for the demo.
        </div>
      </div>
    </div>
  );
}
