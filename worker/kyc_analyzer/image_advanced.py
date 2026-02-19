from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import os
import math

import cv2
import numpy as np


@dataclass
class AdvancedCfg:
    # Document boundary detection
    doc_min_area_ratio: float = 0.35   # doc contour should occupy >=35% of frame
    doc_max_area_ratio: float = 0.97   # if nearly full frame, likely cropped too tight
    doc_max_skew_deg: float = 18.0     # too much skew -> bad capture / boundary not reliable
    doc_rectangularity_min: float = 0.72  # how rectangle-like the contour is

    # Screenshot detection
    screenshot_border_px: int = 12
    screenshot_border_edge_density_min: float = 0.10
    screenshot_flat_bg_std_max: float = 18.0

    # Screen recapture detection (photo of screen)
    banding_row_std_min: float = 28.0
    banding_col_std_min: float = 28.0
    moire_fft_peak_ratio_min: float = 0.020

    # Synthetic / GAN heuristics
    gan_patch_var_std_min: float = 900.0
    gan_patch_var_mean_min: float = 500.0

    # Optional ONNX model path (you provide)
    gan_onnx_path: Optional[str] = None
    gan_onnx_input_size: int = 224


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback for normal paths
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_document_boundary(image_path: str, artifacts_dir: str, cfg: AdvancedCfg) -> Dict[str, Any]:
    """
    Finds largest document-like contour, checks rectangularity, area ratio, skew.
    Writes a debug overlay image when found.
    """
    img = _read_bgr(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    best_area = 0.0
    best_quad = None

    for c in contours[:25]:
        area = float(cv2.contourArea(c))
        if area < 0.05 * (w * h):
            break

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            best = c
            best_area = area
            best_quad = approx.reshape(4, 2)
            break

    flags = []
    metrics: Dict[str, Any] = {}

    if best_quad is None:
        flags.append({
            "code": "doc_boundary_not_found",
            "severity": "medium",
            "message": "No strong 4-corner document boundary found (possible crop/low contrast).",
            "category": "integrity"
        })
        return {"metrics": metrics, "flags": flags}

    area_ratio = best_area / float(w * h)
    rect = _order_points(best_quad.astype("float32"))

    # Compute skew via edge angles
    # angle of top edge relative to horizontal
    (tl, tr, br, bl) = rect
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1e-6 else 90.0
    skew_deg = abs(angle)

    # Rectangularity: contour area / bounding box area
    x, y, bw, bh = cv2.boundingRect(best_quad.astype(np.int32))
    bb_area = float(bw * bh) if bw > 0 and bh > 0 else 1.0
    rectangularity = best_area / bb_area

    metrics.update({
        "doc_area_ratio": area_ratio,
        "doc_skew_deg": skew_deg,
        "doc_rectangularity": rectangularity,
        "doc_bbox": [int(x), int(y), int(bw), int(bh)]
    })

    # Save overlay artifact
    overlay = img.copy()
    cv2.polylines(overlay, [best_quad.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    out_path = os.path.join(artifacts_dir, "doc_boundary_overlay.png")
    cv2.imencode(".png", overlay)[1].tofile(out_path)

    # Rules
    if area_ratio < cfg.doc_min_area_ratio:
        flags.append({
            "code": "doc_too_small_in_frame",
            "severity": "medium",
            "message": "Document occupies too little of the frame (likely zoomed out / incorrect crop).",
            "category": "integrity"
        })

    if area_ratio > cfg.doc_max_area_ratio:
        flags.append({
            "code": "doc_overcropped",
            "severity": "medium",
            "message": "Document occupies almost entire frame (likely over-cropped / edges missing).",
            "category": "integrity"
        })

    if skew_deg > cfg.doc_max_skew_deg:
        flags.append({
            "code": "doc_high_skew",
            "severity": "low",
            "message": "Document is highly skewed; capture quality may be insufficient.",
            "category": "integrity"
        })

    if rectangularity < cfg.doc_rectangularity_min:
        flags.append({
            "code": "doc_boundary_irregular",
            "severity": "medium",
            "message": "Detected boundary is not rectangular enough (possible crop/occlusion).",
            "category": "integrity"
        })

    metrics["overlay_path"] = out_path
    return {"metrics": metrics, "flags": flags}


def detect_screenshot(image_path: str, cfg: AdvancedCfg) -> Dict[str, Any]:
    """
    Screenshot heuristics:
    - borders often include UI bars / solid bands
    - flat background (low std) + sharp edges
    """
    img = _read_bgr(image_path)
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    b = min(cfg.screenshot_border_px, h // 20, w // 20)
    border = np.concatenate([
        g[:b, :].ravel(),
        g[h - b:h, :].ravel(),
        g[:, :b].ravel(),
        g[:, w - b:w].ravel()
    ])

    border_std = float(np.std(border))
    border_mean = float(np.mean(border))

    # Edge density on border
    edges = cv2.Canny(g, 60, 160)
    border_mask = np.zeros_like(g, dtype=np.uint8)
    border_mask[:b, :] = 1
    border_mask[h - b:h, :] = 1
    border_mask[:, :b] = 1
    border_mask[:, w - b:w] = 1
    border_edge_density = float(np.sum(edges[border_mask == 1] > 0) / max(1, np.sum(border_mask)))

    flags = []
    metrics = {
        "border_std": border_std,
        "border_mean": border_mean,
        "border_edge_density": border_edge_density
    }

    # “Flat border” + noticeable border edges often indicates UI chrome / screenshot frame
    if border_std < cfg.screenshot_flat_bg_std_max and border_edge_density > cfg.screenshot_border_edge_density_min:
        flags.append({
            "code": "possible_screenshot",
            "severity": "medium",
            "message": "Border looks like UI chrome (possible screenshot capture).",
            "category": "print_recapture"
        })

    return {"metrics": metrics, "flags": flags}


def detect_screen_recapture(image_path: str, cfg: AdvancedCfg) -> Dict[str, Any]:
    """
    Detect photo-of-screen (recapture) heuristics:
    - row/col banding (rolling shutter)
    - moire periodic peaks in FFT
    """
    img = _read_bgr(image_path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Row/col banding
    row_means = g.mean(axis=1)
    col_means = g.mean(axis=0)

    row_std = float(np.std(row_means))
    col_std = float(np.std(col_means))

    # FFT peak ratio
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    # remove center (low freq)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 10
    mag2 = mag.copy()
    mag2[cy - r:cy + r, cx - r:cx + r] = 0

    top = np.partition(mag2.ravel(), int(mag2.size * 0.99))[-int(mag2.size * 0.01):]
    top1pct_mean = float(np.mean(top))
    overall_mean = float(np.mean(mag2))
    fft_peak_ratio = float(top1pct_mean / (overall_mean + 1e-6))

    flags = []
    metrics = {
        "row_mean_std": row_std,
        "col_mean_std": col_std,
        "fft_peak_ratio": fft_peak_ratio
    }

    if row_std > cfg.banding_row_std_min or col_std > cfg.banding_col_std_min:
        flags.append({
            "code": "possible_screen_banding",
            "severity": "medium",
            "message": "Row/column banding detected; possible photo of a screen.",
            "category": "print_recapture"
        })

    if fft_peak_ratio > cfg.moire_fft_peak_ratio_min:
        flags.append({
            "code": "possible_moire_pattern",
            "severity": "low",
            "message": "Strong periodic frequency peaks; possible moire from screen recapture.",
            "category": "print_recapture"
        })

    return {"metrics": metrics, "flags": flags}


def detect_gan_synthetic(image_path: str, cfg: AdvancedCfg) -> Dict[str, Any]:
    """
    GAN detection:
    - If cfg.gan_onnx_path is provided, run model (optional).
    - Otherwise run strong heuristics: patch variance distribution.
    """
    img = _read_bgr(image_path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Heuristic patch variance stats
    ph = max(32, g.shape[0] // 12)
    pw = max(32, g.shape[1] // 12)
    vars_ = []
    for y in range(0, g.shape[0] - ph, ph):
        for x in range(0, g.shape[1] - pw, pw):
            patch = g[y:y + ph, x:x + pw]
            vars_.append(float(np.var(patch)))

    if not vars_:
        vars_ = [float(np.var(g))]

    vmean = float(np.mean(vars_))
    vstd = float(np.std(vars_))

    flags = []
    metrics = {"patch_var_mean": vmean, "patch_var_std": vstd, "model_used": None}

    # Heuristic flag (NOT definitive)
    if vmean > cfg.gan_patch_var_mean_min and vstd > cfg.gan_patch_var_std_min:
        flags.append({
            "code": "possible_synthetic_heuristic",
            "severity": "low",
            "message": "Patch variance distribution unusual; could be synthetic or heavy processing (heuristic).",
            "category": "synthetic"
        })

    # Optional ONNX model hook (you provide weights)
    if cfg.gan_onnx_path and os.path.exists(cfg.gan_onnx_path):
        try:
            import onnxruntime as ort

            inp = cv2.resize(img, (cfg.gan_onnx_input_size, cfg.gan_onnx_input_size))
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, ...]  # NCHW

            sess = ort.InferenceSession(cfg.gan_onnx_path, providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out = sess.run(None, {in_name: inp})[0]

            # Assume output is probability fake in [0,1]
            p_fake = float(np.squeeze(out))
            metrics["model_used"] = "onnx"
            metrics["p_fake"] = p_fake

            if p_fake >= 0.75:
                flags.append({
                    "code": "synthetic_model_high",
                    "severity": "high",
                    "message": "Synthetic model indicates high probability of generated/manipulated image.",
                    "category": "synthetic"
                })
            elif p_fake >= 0.45:
                flags.append({
                    "code": "synthetic_model_medium",
                    "severity": "medium",
                    "message": "Synthetic model indicates moderate probability of generated/manipulated image.",
                    "category": "synthetic"
                })
        except Exception:
            flags.append({
                "code": "synthetic_model_error",
                "severity": "skipped",
                "message": "Synthetic ONNX model configured but failed to run; using heuristics only.",
                "category": "synthetic"
            })

    return {"metrics": metrics, "flags": flags}
