import express from "express";
import cors from "cors";
import morgan from "morgan";
import multer from "multer";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

import { callWorkerAnalyze } from "./workerClient.js";

dotenv.config();

const PORT = process.env.PORT || 4000;
const WORKER_URL = process.env.WORKER_URL || "http://localhost:8000";
const UPLOAD_DIR = process.env.UPLOAD_DIR || "./uploads";

const app = express();
app.use(
  cors({
    origin: [
      "http://localhost:5173",
      "https://kyc-app-jade.vercel.app"
    ],
    methods: ["GET", "POST", "PUT", "DELETE"],
    credentials: true
  })
);
app.options("*", cors());
app.use(express.json({ limit: "10mb" }));
app.use(morgan("dev"));

// Ensure upload dir exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

function caseDir(caseId) {
  return path.join(UPLOAD_DIR, caseId);
}
function caseMetaPath(caseId) {
  return path.join(caseDir(caseId), "case.json");
}
function reportPath(caseId) {
  return path.join(caseDir(caseId), "report.json");
}

function readJson(p) {
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}
function writeJson(p, obj) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(obj, null, 2), "utf-8");
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const caseId = req.body.caseId || "case";
    const dir = caseDir(caseId);
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_");
    cb(null, `${Date.now()}_${safe}`);
  }
});
const upload = multer({ storage });

app.get("/health", (req, res) => res.json({ ok: true, storage: "filesystem" }));

// Create / upload case (stores metadata in uploads/<caseId>/case.json)
app.post(
  "/api/cases",
  upload.fields([
    { name: "docs", maxCount: 10 },
    { name: "idFront", maxCount: 1 },
    { name: "idBack", maxCount: 1 },
    { name: "selfie", maxCount: 1 },
    { name: "audios", maxCount: 10 }
  ]),
  async (req, res) => {
    const caseId = req.body.caseId || `CASE_${Date.now()}`;
    const dir = caseDir(caseId);

    const files = req.files || {};
    const payload = {
      docs: (files.docs || []).map((f) => f.path),
      idFront: (files.idFront || [])[0]?.path || null,
      idBack: (files.idBack || [])[0]?.path || null,
      selfie: (files.selfie || [])[0]?.path || null,
      audios: (files.audios || []).map((f) => f.path)
    };

    const meta = {
      caseId,
      createdAt: new Date().toISOString(),
      status: "UPLOADED",
      files: payload
    };

    writeJson(caseMetaPath(caseId), meta);

    res.json({ caseId, uploadDir: dir, storage: "filesystem" });
  }
);

// Trigger analysis (calls worker, writes uploads/<caseId>/report.json)
app.post("/api/cases/:caseId/analyze", async (req, res) => {
  const { caseId } = req.params;
  const lang = req.body?.lang || "eng";

  const meta = readJson(caseMetaPath(caseId));
  if (!meta) return res.status(404).json({ error: "Case not found" });

  try {
    const result = await callWorkerAnalyze({
      workerUrl: WORKER_URL,
      caseId,
      lang,
      files: meta.files
    });

    const out = {
      caseId,
      createdAt: meta.createdAt,
      analyzedAt: new Date().toISOString(),
      decision: result.decision,
      score_0_100: result.score_0_100,
      risks: result.risks,
      report: result.report
    };

    writeJson(reportPath(caseId), out);

    meta.status = "ANALYZED";
    writeJson(caseMetaPath(caseId), meta);

    res.json(result);
  } catch (e) {
    meta.status = "ERROR";
    meta.error = String(e?.message || e);
    writeJson(caseMetaPath(caseId), meta);
    res.status(500).json({ error: meta.error });
  }
});

// Get report (reads uploads/<caseId>/report.json)
app.get("/api/cases/:caseId/report", async (req, res) => {
  const { caseId } = req.params;
  const r = readJson(reportPath(caseId));
  if (!r) return res.status(404).json({ error: "Report not found" });
  res.json(r);
});

// Serve uploaded artifacts/files (demo only)
app.use("/uploads", express.static(path.resolve(UPLOAD_DIR)));

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Worker URL: ${WORKER_URL}`);
  console.log(`Storage: filesystem at ${UPLOAD_DIR}`);
});
