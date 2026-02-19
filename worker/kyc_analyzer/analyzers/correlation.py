from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image

from ..utils import ensure_dir, cosine
from ..config import AnalyzerConfig

def _flag(code, severity, message, category):
    return {"code": code, "severity": severity, "message": message, "category": category}

def phash(image_path: Path, hash_size: int = 32, highfreq_factor: int = 8) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img = img.resize((hash_size*highfreq_factor, hash_size*highfreq_factor))
    arr = np.asarray(img, dtype=float)
    dct = cv2.dct(arr)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:,1:])
    bits = (dct_low > med).astype(np.uint8)
    return bits.flatten()

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def correlate_images(image_paths: List[Path], artifacts_dir: Path, cfg: AnalyzerConfig) -> Dict[str, Any]:
    flags=[]; metrics={"pairs": []}
    if len(image_paths) < 2:
        flags.append(_flag("cross_image_insufficient","info","Need >=2 images for cross-image correlation.","correlation"))
        return {"metrics": metrics, "flags": flags}

    hashes = {p: phash(p, cfg.phash_size, cfg.phash_highfreq) for p in image_paths}
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            a=image_paths[i]; b=image_paths[j]
            d=hamming(hashes[a], hashes[b])
            metrics["pairs"].append({"a": str(a), "b": str(b), "phash_hamming": d})
            if d < 20:
                flags.append(_flag("images_very_similar","low",f"Images {a.name} and {b.name} look extremely similar (pHash).","correlation"))
    return {"metrics": metrics, "flags": flags}

def correlate_documents(doc_texts: List[str]) -> Dict[str, Any]:
    flags=[]; metrics={}
    if len(doc_texts) < 2:
        flags.append(_flag("cross_doc_insufficient","info","Need >=2 documents for cross-document correlation.","correlation"))
        return {"metrics": metrics, "flags": flags}
    token_sets=[]
    for t in doc_texts:
        toks=set([w.lower() for w in t.split() if len(w) > 4])
        token_sets.append(toks)
    inter=set.intersection(*token_sets) if token_sets else set()
    metrics["common_token_count"]=len(inter)
    if len(inter) < 10:
        flags.append(_flag("documents_low_overlap","low","Low overlap across documents; verify consistency of identity fields.","correlation"))
    return {"metrics": metrics, "flags": flags}

def correlate_audio_embeddings(embeddings: List[List[float]]) -> Dict[str, Any]:
    flags=[]; metrics={"pairs": []}
    if len(embeddings) < 2:
        flags.append(_flag("cross_audio_insufficient","info","Need >=2 audio samples for cross-audio correlation.","correlation"))
        return {"metrics": metrics, "flags": flags}
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim=cosine(embeddings[i], embeddings[j])
            metrics["pairs"].append({"i": i, "j": j, "cosine": sim})
            if sim < 0.6:
                flags.append(_flag("voice_mismatch_possible","medium","Low similarity between voice embeddings; possible different speaker.","correlation"))
    return {"metrics": metrics, "flags": flags}
