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

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    case_id: str = Form("case"),
    lang: str = Form("eng"),
    docs: Optional[List[UploadFile]] = File(default=None),
    id_front: Optional[UploadFile] = File(default=None),
    id_back: Optional[UploadFile] = File(default=None),
    selfie: Optional[UploadFile] = File(default=None),
    audios: Optional[List[UploadFile]] = File(default=None),
):
    case_dir = ensure_dir(DATA_DIR / case_id)
    out_dir = ensure_dir(case_dir / "out")
    ensure_dir(out_dir / "artifacts")

    # Save uploads
    saved_docs = []
    if docs:
        for d in docs:
            saved_docs.append(_save_upload(d, case_dir / "in" / "docs"))

    saved_images = {}
    if id_front:
        saved_images["id_front"] = _save_upload(id_front, case_dir / "in" / "images")
    if id_back:
        saved_images["id_back"] = _save_upload(id_back, case_dir / "in" / "images")
    if selfie:
        saved_images["selfie"] = _save_upload(selfie, case_dir / "in" / "images")

    saved_audios = []
    if audios:
        for a in audios:
            saved_audios.append(_save_upload(a, case_dir / "in" / "audios"))

    results = {"case_id": case_id}

    # Documents
    doc_results = []
    doc_texts = []
    for d in saved_docs:
        dr = analyze_document(d, out_dir, lang=lang, hooks=hooks)
        doc_results.append(dr)
        # extract text from artifacts
        txt_path = out_dir / "artifacts" / "document" / "extracted_text.txt"
        if txt_path.exists():
            doc_texts.append(txt_path.read_text(encoding="utf-8", errors="ignore"))
    if doc_results:
        results["documents"] = doc_results

    # Images
    image_results = []
    image_paths = []
    if "id_front" in saved_images:
        p = saved_images["id_front"]; image_paths.append(p)
        image_results.append(analyze_image(p, out_dir, ela_quality=cfg.ela_quality, expected="id", hooks=hooks))
    if "id_back" in saved_images:
        p = saved_images["id_back"]; image_paths.append(p)
        image_results.append(analyze_image(p, out_dir, ela_quality=cfg.ela_quality, expected="id", hooks=hooks))
    if "selfie" in saved_images:
        p = saved_images["selfie"]; image_paths.append(p)
        image_results.append(analyze_image(p, out_dir, ela_quality=cfg.ela_quality, expected="selfie", hooks=hooks))
    if image_results:
        results["images"] = image_results

    # Audio
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

    # Correlation
    corr = {}
    if image_paths:
        corr["cross_image"] = correlate_images(image_paths, ensure_dir(out_dir / "artifacts" / "correlation"), cfg)
    if doc_texts:
        corr["cross_document"] = correlate_documents(doc_texts)
    if embeddings:
        corr["cross_audio"] = correlate_audio_embeddings(embeddings)
    if corr:
        results["correlation"] = corr

    override_decision, override_reason = image_quality_gate(results)

    if override_decision:
        report = {
            "decision": override_decision,
            "score_0_100": 100.0,
            "risks": {},
            "fraud_confidence": 1.0,
            "results": results,
            "worker_artifacts_dir": str(out_dir / "artifacts"),
            "notes": [override_reason],
        }

        return {
            "decision": override_decision,
            "score_0_100": 100.0,
            "risks": {},
            "report": report
        }
    
    override_decision, override_reason = audio_quality_gate(results)

    if override_decision:
        report = {
        "decision": override_decision,
        "score_0_100": 100.0,
        "risks": {},
        "fraud_confidence": 1.0,
        "results": results,
        "worker_artifacts_dir": str(out_dir / "artifacts"),
        "notes": [override_reason],
                }
        return {"decision": override_decision, "score_0_100": 100.0, "risks": {}, "report": report}

    scoring = score_case(results, weights=cfg.weights.model_dump(), profile="strict")
    decision = scoring["decision"]
    fraud_confidence = scoring["fraud_confidence"]
    report = {
        "decision": scoring["decision"],
        "score_0_100": scoring["score_0_100"],
        "risks": scoring["risks"],
        "fraud_confidence": scoring["fraud_confidence"],
        "results": results,
        "worker_artifacts_dir": str(out_dir / "artifacts"),
        "notes": [
            "Some checks are heuristic or depend on external tools/models.",
            "Treat output as risk signals; use human review.",
        ],
    }
    return {"decision": decision, "score_0_100": scoring["score_0_100"], "risks": scoring["risks"], "report": report}
