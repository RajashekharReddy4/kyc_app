from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional, List


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kyc_analyzer.config import AnalyzerConfig
from kyc_analyzer.models import OptionalModelHooks
from kyc_analyzer.analyzers.document import analyze_document
from kyc_analyzer.analyzers.image import analyze_image
from kyc_analyzer.analyzers.audio import analyze_audio
from kyc_analyzer.analyzers.correlation import correlate_images, correlate_documents, correlate_audio_embeddings
from kyc_analyzer.scoring import score_case
from kyc_analyzer.utils import ensure_dir, decision_from_score
from kyc_analyzer.quality_gate import image_quality_gate
from kyc_analyzer.quality_gate_audio import audio_quality_gate
from kyc_analyzer.analyzers.video import analyze_video_kyc



app = FastAPI(title="KYC Analyzer Worker", version="0.2.0")
cfg = AnalyzerConfig()
hooks = OptionalModelHooks()

DATA_DIR = Path("/app/data")
ensure_dir(DATA_DIR)

@app.get("/")
def root():
    return {"service": "KYC Analyzer Worker", "status": "running"}

@app.get("/health")
def health():
    return {"ok": True}

def _save_upload(upload: UploadFile, folder: Path) -> Path:
    ensure_dir(folder)
    ext = Path(upload.filename or "file").suffix
    out = folder / f"{uuid.uuid4().hex}{ext}"
    with out.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return out

class AnalyzeResponse(BaseModel):
    decision: str
    score_0_100: float
    risks: dict
    report: dict

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), case_id: str = Form("case")):
    case_dir = ensure_dir(DATA_DIR / case_id)
    out_dir = ensure_dir(case_dir / "out")
    ensure_dir(out_dir / "artifacts")

    saved = _save_upload(file, case_dir / "in" / "video")

    # Run video analyzer
    vr = analyze_video_kyc(str(saved))

    # 🚨 DO NOT call score_case here
    # Return video decision directly

    return {
        "decision": vr["decision"],
        "score_0_100": vr["score_0_100"],
        "risks": {
            "liveness": 1.0 if vr["decision"] != "pass" else 0.0
        },
        "report": {
            "decision": vr["decision"],
            "score_0_100": vr["score_0_100"],
            "risks": vr["risks"],
            "fraud_confidence": 1.0 if vr["decision"] == "fail" else 0.5,
            "results": {
                "case_id": case_id,
                "video": vr["report"]["video"]
            },
            "worker_artifacts_dir": str(out_dir / "artifacts"),
            "notes": [
                "Video-only KYC decision.",
                "Scoring engine bypassed for video endpoint."
            ]
        }
    }

@app.post("/analyze/documents")
async def analyze_documents(
    case_id: str = Form("case"),
    lang: str = Form("eng"),
    docs: Optional[List[UploadFile]] = File(default=None),
):
    # Reuse /analyze by calling core with only docs
    return await analyze(case_id=case_id, lang=lang, docs=docs, id_front=None, selfie=None, audios=None, video=None)

@app.post("/analyze/images")
async def analyze_images(
    case_id: str = Form("case"),
    id_front: Optional[UploadFile] = File(default=None),
    selfie: Optional[UploadFile] = File(default=None),
):
    return await analyze(case_id=case_id, lang="eng", docs=None, id_front=id_front, selfie=selfie, audios=None, video=None)

@app.post("/analyze/audios")
async def analyze_audios(
    case_id: str = Form("case"),
    audios: Optional[List[UploadFile]] = File(default=None),
):
    return await analyze(case_id=case_id, lang="eng", docs=None, id_front=None, selfie=None, audios=audios, video=None)
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    case_id: str = Form("case"),
    lang: str = Form("eng"),
    docs: Optional[List[UploadFile]] = File(default=None),
    id_front: Optional[UploadFile] = File(default=None),
    selfie: Optional[UploadFile] = File(default=None),
    audios: Optional[List[UploadFile]] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
):
    case_dir = ensure_dir(DATA_DIR / case_id)
    out_dir = ensure_dir(case_dir / "out")
    ensure_dir(out_dir / "artifacts")

    # ---------------------------
    # SAVE INPUTS
    # ---------------------------
    saved_docs = []
    if docs:
        for d in docs:
            saved_docs.append(_save_upload(d, case_dir / "in" / "docs"))

    saved_images = {}
    if id_front:
        saved_images["id_front"] = _save_upload(id_front, case_dir / "in" / "images")
    if selfie:
        saved_images["selfie"] = _save_upload(selfie, case_dir / "in" / "images")

    saved_audios = []
    if audios:
        for a in audios:
            saved_audios.append(_save_upload(a, case_dir / "in" / "audios"))

    saved_video = None
    if video:
        saved_video = _save_upload(video, case_dir / "in" / "video")

    results = {"case_id": case_id}

    # ---------------------------
    # VIDEO ANALYSIS (GATING)
    # ---------------------------
# ---------------------------
# VIDEO ANALYSIS (STRICT GATE)
# ---------------------------
    video_decision = None

    if saved_video:
        vr = analyze_video_kyc(str(saved_video))

        results["video"] = {
            "decision": vr["decision"],
            "score": vr["score_0_100"],
            "risks": vr["risks"],
            "analysis": vr["report"]["video"]
        }

        video_decision = vr["decision"]

        # 🚨 HARD STOP — if video fails
        if video_decision == "fail":
            return {
                "decision": "fail",
                "score_0_100": vr["score_0_100"],
                "risks": {"liveness": 1.0},
                "report": {
                    "decision": "fail",
                    "score_0_100": vr["score_0_100"],
                    "risks": {"liveness": 1.0},
                    "fraud_confidence": 1.0,
                    "results": results,
                    "notes": ["Video liveness failed — case automatically rejected."]
                }
            }

    # ---------------------------
    # VIDEO GATING FIX
    # ---------------------------
    if "video" in results:
        video_decision = results["video"]["decision"]

        if video_decision == "fail":
            return {
                "decision": "fail",
                "score_0_100": 0,
                "risks": {"liveness": 1.0},
                "report": {
                    "decision": "fail",
                    "score_0_100": 0,
                    "risks": {"liveness": 1.0},
                    "fraud_confidence": 1.0,
                    "results": results,
                    "notes": ["Video liveness failed."]
                }
            }

        if video_decision == "review":
            # force review later
            force_review = True
        else:
            force_review = False
    else:
        force_review = False

    # ---------------------------
    # DOCUMENT ANALYSIS
    # ---------------------------
    doc_results = []
    doc_texts = []

    for d in saved_docs:
        dr = analyze_document(d, out_dir, lang=lang, hooks=hooks)
        doc_results.append(dr)

        txt_path = out_dir / "artifacts" / "document" / "extracted_text.txt"
        if txt_path.exists():
            doc_texts.append(txt_path.read_text(encoding="utf-8", errors="ignore"))

    if doc_results:
        results["documents"] = doc_results

    # ---------------------------
    # IMAGE ANALYSIS
    # ---------------------------
    image_results = []
    image_paths = []

    if "id_front" in saved_images:
        p = saved_images["id_front"]
        image_paths.append(p)
        image_results.append(
            analyze_image(p, out_dir, ela_quality=cfg.ela_quality, expected="id", hooks=hooks)
        )

    if "selfie" in saved_images:
        p = saved_images["selfie"]
        image_paths.append(p)
        image_results.append(
            analyze_image(p, out_dir, ela_quality=cfg.ela_quality, expected="selfie", hooks=hooks)
        )

    if image_results:
        results["images"] = image_results

    # ---------------------------
    # AUDIO ANALYSIS
    # ---------------------------
    audio_results = []
    embeddings = []

    for a in saved_audios:
        ar = analyze_audio(a, out_dir, target_sr=cfg.audio_sr, hooks=hooks)
        audio_results.append(ar)

        emb = ar.get("analysis", {}).get("embedding_mfcc_mean")
        if isinstance(emb, list):
            embeddings.append(emb)

    if audio_results:
        results["audios"] = audio_results

    # ---------------------------
    # CORRELATION
    # ---------------------------
    corr = {}

    if image_paths:
        corr["cross_image"] = correlate_images(
            image_paths,
            ensure_dir(out_dir / "artifacts" / "correlation"),
            cfg
        )

    if doc_texts:
        corr["cross_document"] = correlate_documents(doc_texts)

    if embeddings:
        corr["cross_audio"] = correlate_audio_embeddings(embeddings)

    if corr:
        results["correlation"] = corr

    # --- VIDEO GATING / OVERRIDE ---
    has_docs = bool(saved_docs)
    has_images = bool(saved_images)
    has_audios = bool(saved_audios)
    has_video = saved_video is not None

    only_video = has_video and (not has_docs) and (not has_images) and (not has_audios)

    # If only video uploaded, return the video decision as the final decision
    if only_video:
        vr = results["video"]  # the object you stored earlier
        return {
            "decision": vr["decision"],
            "score_0_100": vr["score"],
            "risks": {"liveness": 1.0 if vr["decision"] != "pass" else 0.0},
            "report": {
                "decision": vr["decision"],
                "score_0_100": vr["score"],
                "risks": {"liveness": 1.0 if vr["decision"] != "pass" else 0.0},
                "fraud_confidence": 0.9 if vr["decision"] == "fail" else 0.5,
                "results": results,
                "worker_artifacts_dir": str(out_dir / "artifacts"),
                "notes": ["Video-only submission. Final decision derived from liveness verification."]
            }
        }

    # ---------------------------
    # GLOBAL SCORING
    # ---------------------------
    scoring = score_case(results, weights=cfg.weights.model_dump(), profile="strict")

    final_decision = scoring["decision"]

    # If video was REVIEW, cap overall decision to REVIEW
    if video_decision == "review" and final_decision == "pass":
        final_decision = "review"

    report = {
        "decision": final_decision,
        "score_0_100": scoring["score_0_100"],
        "risks": scoring["risks"],
        "fraud_confidence": scoring["fraud_confidence"],
        "results": results,
        "worker_artifacts_dir": str(out_dir / "artifacts"),
        "notes": [
            "Video liveness acts as a gating control.",
            "Treat output as risk signals; use human review."
        ],
    }

    return {
        "decision": final_decision,
        "score_0_100": scoring["score_0_100"],
        "risks": scoring["risks"],
        "report": report
    }