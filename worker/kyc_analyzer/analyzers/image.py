from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import cv2
from PIL import Image, ImageChops

from ..utils import which, run_cmd, sha256_file, ensure_dir
from ..models import OptionalModelHooks

from kyc_analyzer.image_advanced import (
    AdvancedCfg,
    detect_document_boundary,
    detect_screenshot,
    detect_screen_recapture,
    detect_gan_synthetic,
)

def _flag(code, severity, message, category):
    return {"code": code, "severity": severity, "message": message, "category": category}

def file_and_encoding_sanity(image_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; info={"sha256": sha256_file(image_path), "size_bytes": image_path.stat().st_size}
    try:
        im = Image.open(image_path); im.verify()
        info["pil_verified"]=True
    except Exception as e:
        info["pil_verified"]=False
        flags.append(_flag("image_decode_fail_pil","high",f"PIL verify failed: {e}","integrity"))
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        flags.append(_flag("image_decode_fail_opencv","high","OpenCV failed to decode image.","integrity"))
    else:
        h,w=img.shape[:2]; info["width"]=int(w); info["height"]=int(h)
    return {"info": info, "flags": flags}

def metadata_and_capture_history(image_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; meta={}
    if which("exiftool"):
        rc,out,err = run_cmd(["exiftool","-j",str(image_path)], timeout=30)
        (artifacts_dir/"exiftool_image.json").write_text(out if out else "[]", encoding="utf-8")
        if rc==0 and out.strip().startswith("["):
            try:
                arr=json.loads(out); meta=arr[0] if arr else {}
            except Exception:
                meta={}
        else:
            flags.append(_flag("image_metadata_read_fail","low","ExifTool failed to read metadata.","metadata"))
    else:
        flags.append(_flag("exiftool_missing","skipped","ExifTool not installed; skipped metadata extraction.","metadata"))

    software=str(meta.get("Software","")).lower()
    if any(k in software for k in ["photoshop","gimp","snapseed","lightroom"]):
        flags.append(_flag("image_edited_software","medium","EXIF Software tag suggests editing.","metadata"))
    return {"metadata": meta, "flags": flags}

def source_classification(image_path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    flags=[]; cls={"type":"unknown","confidence":0.3,"signals":{}}
    make=str(meta.get("Make","")); model=str(meta.get("Model",""))
    if make or model:
        cls["type"]="camera_photo"; cls["confidence"]=0.65; cls["signals"]["device"]=f"{make} {model}".strip()
    software=str(meta.get("Software","")).lower()
    if "screenshot" in software:
        cls["type"]="screenshot"; cls["confidence"]=0.7; cls["signals"]["software"]=software
    try:
        img=cv2.imread(str(image_path)); h,w=img.shape[:2]
        if (w,h) in [(1920,1080),(2560,1440),(3840,2160)] and not (make or model):
            cls["type"]="screenshot_or_recapture"; cls["confidence"]=0.6; cls["signals"]["resolution"]=[w,h]
    except Exception:
        pass
    return {"classification": cls, "flags": flags}

def resolution_and_geometry(image_path: Path, expected: str | None = None) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path))
    if img is None:
        return {"metrics": metrics, "flags":[_flag("image_load_fail","high","Could not load image for geometry checks.","integrity")]}
    h,w=img.shape[:2]
    metrics={"width":int(w),"height":int(h),"aspect": float(w)/max(1.0,float(h))}
    if w < 400 or h < 400:
        flags.append(_flag("image_low_resolution","medium","Image resolution is quite low; may hinder verification.","integrity"))
    if expected=="id" and metrics["aspect"] < 1.2:
        flags.append(_flag("id_aspect_unexpected","low","ID image aspect ratio unexpected; verify cropping/rotation.","integrity"))
    if expected=="selfie" and metrics["aspect"] > 1.8:
        flags.append(_flag("selfie_aspect_wide","low","Selfie is unusually wide; could be screenshot or crop.","integrity"))
    return {"metrics": metrics, "flags": flags}

def global_quality(image_path: Path) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("image_quality_load_fail","high","Could not load image for quality checks.","integrity")]}
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["sharpness_laplacian_var"]=sharp
    if sharp < 50:
        flags.append(_flag("low_sharpness","medium","Low sharpness; may indicate blur/scan/recapture.","print_recapture"))
    dark=float(np.mean(gray < 15)); bright=float(np.mean(gray > 240))
    metrics["dark_ratio"]=dark; metrics["bright_ratio"]=bright
    if dark > 0.25:
        flags.append(_flag("underexposed","low","Large dark regions; may reduce readability.","integrity"))
    if bright > 0.25:
        flags.append(_flag("overexposed","low","Large bright regions; may wash out text.","integrity"))
    noise=float(np.std(cv2.GaussianBlur(gray,(0,0),1.2) - gray))
    metrics["noise_proxy_std"]=noise
    return {"metrics": metrics, "flags": flags}

def ela(image_path: Path, artifacts_dir: Path, quality: int = 90) -> Dict[str, Any]:
    flags=[]
    try:
        im=Image.open(image_path).convert("RGB")
        tmp=artifacts_dir/"ela_recompressed.jpg"
        im.save(tmp,"JPEG",quality=quality)
        im2=Image.open(tmp).convert("RGB")
        diff=ImageChops.difference(im,im2)
        arr=np.asarray(diff).astype(np.float32)
        maxv=float(arr.max()) if arr.size else 0.0
        scale=255.0/max(1.0,maxv)
        hm=np.clip(arr*scale,0,255).astype(np.uint8)
        out=Image.fromarray(hm)
        out_path=artifacts_dir/"ela_heatmap.png"
        out.save(out_path)
        if maxv>40:
            flags.append(_flag("ela_high_signal","medium","High ELA signal; inspect highlighted areas.","tamper"))
        return {"heatmap_path": str(out_path), "max_diff": maxv, "flags": flags}
    except Exception as e:
        return {"flags":[_flag("ela_failed","low",f"ELA failed: {e}","tamper")]}

def texture_frequency(image_path: Path) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("fft_load_fail","high","Could not load image for FFT analysis.","tamper")]}
    f=np.fft.fft2(img); mag=np.log(np.abs(np.fft.fftshift(f))+1.0)
    flat=mag.flatten(); k=max(1,int(0.01*flat.size))
    top=np.partition(flat,-k)[-k:]
    ratio=float(top.sum()/(flat.sum()+1e-9))
    metrics["fft_top1pct_ratio"]=ratio
    if ratio>0.22:
        flags.append(_flag("fft_periodic_energy","low","Concentrated FFT energy; possible moiré/screen pattern (heuristic).","print_recapture"))
    return {"metrics": metrics, "flags": flags}

def lighting_shadow_consistency(image_path: Path) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("lighting_load_fail","high","Could not load image for lighting checks.","integrity")]}
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    gx=cv2.Sobel(gray, cv2.CV_32F, 1,0,ksize=3)
    gy=cv2.Sobel(gray, cv2.CV_32F, 0,1,ksize=3)
    ang=np.arctan2(gy, gx)
    hist, _ = np.histogram(ang, bins=36, range=(-np.pi, np.pi))
    peak=float(hist.max()/max(1,hist.sum()))
    metrics["gradient_direction_peak_ratio"]=peak
    if peak < 0.08:
        flags.append(_flag("lighting_direction_incoherent","low","No dominant lighting direction; could be complex scene or compositing.","tamper"))
    return {"metrics": metrics, "flags": flags}

def color_chromatic(image_path: Path) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("color_load_fail","high","Could not load image for color checks.","integrity")]}
    b,g,r=cv2.split(img.astype(np.float32))
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)
    def edge_mean(ch):
        return float(np.mean(ch[edges>0])) if np.any(edges>0) else 0.0
    metrics["edge_mean_r"]=edge_mean(r); metrics["edge_mean_g"]=edge_mean(g); metrics["edge_mean_b"]=edge_mean(b)
    dif = max(abs(metrics["edge_mean_r"]-metrics["edge_mean_g"]), abs(metrics["edge_mean_b"]-metrics["edge_mean_g"]))
    metrics["edge_channel_diff_max"]=float(dif)
    if dif > 25:
        flags.append(_flag("chromatic_inconsistency","low","Large channel differences on edges; could indicate compositing.","tamper"))
    return {"metrics": metrics, "flags": flags}

def ai_generated_detection(image_path: Path, hooks: OptionalModelHooks | None) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("synthetic_load_fail","high","Could not load image for synthetic detection.","synthetic")]}
    h,w=img.shape[:2]
    ps=64
    vars=[]
    for yy in range(0, h-ps, ps):
        for xx in range(0, w-ps, ps):
            patch=img[yy:yy+ps,xx:xx+ps].astype(np.float32)
            vars.append(float(np.var(patch)))
    if vars:
        vstd=float(np.std(vars)); vmean=float(np.mean(vars))
        metrics["patch_var_mean"]=vmean; metrics["patch_var_std"]=vstd
        if vmean>0 and (vstd/(vmean+1e-9)) < 0.25:
            flags.append(_flag("synthetic_uniform_texture","low","Texture variance unusually uniform; possible synthetic/overprocessed image.","synthetic"))

    if hooks:
        res=hooks.detect_image_synthetic(image_path)
        if res is None:
            flags.append(_flag("image_synth_model_missing","skipped","No image synthetic model configured; used heuristics only.","synthetic"))
        else:
            metrics["model_score"]=float(res.score_0_1)
            if res.score_0_1>0.7:
                flags.append(_flag("image_synth_model_high","high","Model indicates high likelihood of synthetic image.","synthetic"))
    else:
        flags.append(_flag("image_synth_hooks_none","skipped","No model hooks provided; used heuristics only.","synthetic"))
    return {"metrics": metrics, "flags": flags}

def object_region_consistency(image_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("region_load_fail","high","Could not load image for region consistency.","tamper")]}
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    metrics["faces_detected"]=int(len(faces))
    if len(faces)==0:
        flags.append(_flag("face_not_detected","info","No face detected (heuristic); region consistency limited.","tamper"))
        return {"metrics": metrics, "flags": flags}
    x,y,w,h = max(faces, key=lambda t: t[2]*t[3])
    face = gray[y:y+h, x:x+w]
    face_sharp=float(cv2.Laplacian(face, cv2.CV_64F).var())
    global_sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["face_sharpness"]=face_sharp; metrics["global_sharpness"]=global_sharp
    if abs(face_sharp - global_sharp) / (max(1.0, global_sharp)) > 1.2:
        flags.append(_flag("region_sharpness_mismatch","medium","Face sharpness differs strongly from background; possible compositing.","tamper"))
    preview = img.copy()
    cv2.rectangle(preview, (x,y), (x+w,y+h), (0,255,0), 2)
    out = artifacts_dir/"face_bbox.png"
    cv2.imwrite(str(out), preview)
    metrics["face_bbox_preview"]=str(out)
    return {"metrics": metrics, "flags": flags}

def print_scan_recapture(image_path: Path) -> Dict[str, Any]:
    flags=[]; metrics={}
    img=cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"metrics": metrics, "flags":[_flag("recapture_load_fail","high","Could not load image for recapture signals.","print_recapture")]}
    fm=float(cv2.Laplacian(img, cv2.CV_64F).var())
    metrics["focus_var"]=fm
    if fm<50:
        flags.append(_flag("recapture_low_sharpness","medium","Low sharpness suggests scan/recapture.","print_recapture"))
    row_var=float(np.std(np.mean(img, axis=1)))
    col_var=float(np.std(np.mean(img, axis=0)))
    metrics["row_mean_std"]=row_var; metrics["col_mean_std"]=col_var
    if row_var>12 and row_var>col_var*1.3:
        flags.append(_flag("possible_screen_banding","low","Row-wise banding detected; possible photo of a screen.","print_recapture"))
    return {"metrics": metrics, "flags": flags}

def analyze_image(
    image_path: Path,
    out_dir: Path,
    ela_quality: int = 90,
    expected: str | None = None,
    hooks: OptionalModelHooks | None = None
) -> Dict[str, Any]:

    artifacts = ensure_dir(out_dir / "artifacts" / f"image_{image_path.stem}")

    # Core checks
    sanity = file_and_encoding_sanity(image_path, artifacts)
    meta = metadata_and_capture_history(image_path, artifacts)
    src = source_classification(image_path, meta.get("metadata", {}))
    geom = resolution_and_geometry(image_path, expected=expected)
    qual = global_quality(image_path)
    ela_out = ela(image_path, artifacts, quality=ela_quality)
    fft = texture_frequency(image_path)
    light = lighting_shadow_consistency(image_path)
    color = color_chromatic(image_path)
    synth = ai_generated_detection(image_path, hooks)
    region = object_region_consistency(image_path, artifacts)
    rec = print_scan_recapture(image_path)

    # ------------------------------------
    # ADVANCED BANK-GRADE CHECKS
    # ------------------------------------

    adv_cfg = AdvancedCfg()

    img_art_dir = ensure_dir(
        out_dir / "artifacts" / f"image_{image_path.stem}"
    )
    img_art_dir_str = str(img_art_dir)

    # 1) Document contour boundary detection
    boundary = detect_document_boundary(
        image_path,
        img_art_dir_str,
        adv_cfg
    )

    # 2) Screenshot detection
    screenshot = detect_screenshot(
        image_path,
        adv_cfg
    )

    # 3) Screen recapture detection
    screen_recap = detect_screen_recapture(
        image_path,
        adv_cfg
    )

    # 4) GAN synthetic detection
    gan_synth = detect_gan_synthetic(
        image_path,
        adv_cfg
    )

    # ------------------------------------
    # FINAL RESULT OBJECT
    # ------------------------------------

    result = {
        "path": str(image_path),
        "integrity": sanity,
        "metadata": meta,
        "source": src,
        "resolution_geometry": geom,
        "global_quality": qual,
        "tamper_localized": ela_out,
        "texture_frequency": fft,
        "lighting_shadow": light,
        "color_chromatic": color,
        "synthetic": synth,
        "object_region": region,
        "print_recapture": rec,

        # Advanced modules
        "document_boundary": boundary,
        "screenshot": screenshot,
        "screen_recapture": screen_recap,
        "gan_synthetic": gan_synth,
    }

    return result
