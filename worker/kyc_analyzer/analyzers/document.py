from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re, json
import numpy as np
import cv2
from PyPDF2 import PdfReader

from ..utils import which, run_cmd, sha256_file, ensure_dir
from ..models import OptionalModelHooks

def _flag(code, severity, message, category):
    return {"code": code, "severity": severity, "message": message, "category": category}

def pdf_integrity_and_structure(pdf_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; info={"sha256": sha256_file(pdf_path), "size_bytes": pdf_path.stat().st_size}

    if which("qpdf"):
        rc,out,err = run_cmd(["qpdf","--check",str(pdf_path)], timeout=60)
        (artifacts_dir/"qpdf_check.txt").write_text(out+"\n"+err, encoding="utf-8")
        info["qpdf_check_rc"]=rc
        if rc != 0:
            flags.append(_flag("pdf_integrity_qpdf_fail","high","qpdf --check reported issues.","integrity"))
    else:
        flags.append(_flag("qpdf_missing","skipped","qpdf not installed; skipped structural integrity check.","integrity"))

    try:
        reader = PdfReader(str(pdf_path))
        info["pages"]=len(reader.pages)
        info["is_encrypted"]=bool(getattr(reader,"is_encrypted",False))
        if info["is_encrypted"]:
            flags.append(_flag("pdf_encrypted","medium","PDF is encrypted; analysis may be incomplete.","integrity"))
        meta={}
        try:
            md=reader.metadata or {}
            meta={k.strip("/") if isinstance(k,str) else str(k): str(v) for k,v in md.items()}
        except Exception:
            meta={}
        (artifacts_dir/"pypdf2_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        info["pypdf2_metadata"]=meta
    except Exception as e:
        flags.append(_flag("pdf_parse_fail","high",f"PyPDF2 failed to parse PDF: {e}","integrity"))
        return {"info": info, "flags": flags}

    return {"info": info, "flags": flags}

def pdf_metadata_analysis(pdf_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; meta={}
    if which("exiftool"):
        rc,out,err = run_cmd(["exiftool","-j",str(pdf_path)], timeout=60)
        (artifacts_dir/"exiftool_pdf.json").write_text(out if out else "[]", encoding="utf-8")
        if rc==0 and out.strip().startswith("["):
            try:
                arr=json.loads(out); meta=arr[0] if arr else {}
            except Exception:
                meta={}
        else:
            flags.append(_flag("pdf_metadata_read_fail","low","ExifTool failed to read PDF metadata.","metadata"))
    else:
        flags.append(_flag("exiftool_missing","skipped","ExifTool not installed; skipped metadata extraction.","metadata"))

    producer=(str(meta.get("Producer",""))+" "+str(meta.get("Creator",""))).lower()
    if any(k in producer for k in ["photoshop","illustrator","gimp","pdf editor","acrobat","canva"]):
        flags.append(_flag("pdf_metadata_edit_software","medium","Metadata suggests editing software used.","metadata"))
    return {"metadata": meta, "flags": flags}

def pdf_source_classification(struct_info: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    flags=[]; classification={"type":"unknown","confidence":0.3,"signals":{}}
    pmeta = (struct_info.get("info", {}).get("pypdf2_metadata") or {})
    producer=(str(meta.get("Producer",""))+" "+str(meta.get("Creator",""))+" "+str(pmeta.get("Producer",""))).lower()
    if any(k in producer for k in ["scanner","scan","epson","canon","hp"]):
        classification["type"]="scan_pdf"; classification["confidence"]=0.7; classification["signals"]["producer"]=producer
    elif any(k in producer for k in ["word","libreoffice","latex","canva","indesign","acrobat"]):
        classification["type"]="digital_authored_pdf"; classification["confidence"]=0.7; classification["signals"]["producer"]=producer
    else:
        classification["signals"]["producer"]=producer
    return {"classification": classification, "flags": flags}

def pdf_page_consistency_checks(pdf_path: Path) -> Dict[str, Any]:
    flags=[]; info={}
    try:
        reader=PdfReader(str(pdf_path))
        info["pages"]=len(reader.pages)
        obj_ids=[]
        for i,p in enumerate(reader.pages):
            try:
                obj_ids.append(str(p.indirect_reference.idnum))
            except Exception:
                obj_ids.append(str(i))
        dup=len(obj_ids)-len(set(obj_ids))
        info["duplicate_page_objects"]=dup
        if dup>0:
            flags.append(_flag("pdf_duplicate_page_objects","low","Detected duplicate page object ids; verify page order/coherence.","integrity"))
    except Exception as e:
        flags.append(_flag("pdf_page_check_fail","low",f"Page consistency checks failed: {e}","integrity"))
    return {"info": info, "flags": flags}

def render_pdf_to_images(pdf_path: Path, artifacts_dir: Path, max_pages: int = 8) -> Tuple[List[Path], List[Dict[str, Any]]]:
    flags=[]; out_paths=[]
    if which("pdftoppm"):
        prefix=artifacts_dir/"page"
        rc,out,err=run_cmd(["pdftoppm","-jpeg","-r","200",str(pdf_path),str(prefix)], timeout=180)
        (artifacts_dir/"pdftoppm.log").write_text(out+"\n"+err, encoding="utf-8")
        if rc==0:
            out_paths=sorted(artifacts_dir.glob("page-*.jpg"))[:max_pages]
        else:
            flags.append(_flag("pdftoppm_fail","medium","Failed to render PDF pages using pdftoppm.","integrity"))
    else:
        flags.append(_flag("pdftoppm_missing","skipped","pdftoppm not installed; skipped PDF rendering.","integrity"))
    return out_paths, flags

def ocr_pdf(pdf_path: Path, artifacts_dir: Path, lang: str = "eng") -> Dict[str, Any]:
    flags=[]; text=""; method=None
    if which("ocrmypdf"):
        out_pdf=artifacts_dir/"ocr_output.pdf"
        rc,out,err=run_cmd(["ocrmypdf","--skip-text","--force-ocr","-l",lang,str(pdf_path),str(out_pdf)], timeout=420)
        (artifacts_dir/"ocrmypdf.log").write_text(out+"\n"+err, encoding="utf-8")
        if rc==0 and out_pdf.exists():
            method="ocrmypdf"
            if which("pdftotext"):
                txt_path=artifacts_dir/"ocr.txt"
                rc2,out2,err2=run_cmd(["pdftotext",str(out_pdf),str(txt_path)], timeout=60)
                if rc2==0 and txt_path.exists():
                    text=txt_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    flags.append(_flag("pdftotext_fail","low","OCR succeeded but pdftotext failed.","ocr_semantic"))
            else:
                flags.append(_flag("pdftotext_missing","skipped","pdftotext not installed; skipped text extraction.","ocr_semantic"))
        else:
            flags.append(_flag("ocrmypdf_fail","medium","OCRmyPDF failed; trying fallback OCR.","ocr_semantic"))

    if not text and which("tesseract"):
        pages, rflags = render_pdf_to_images(pdf_path, artifacts_dir, max_pages=5)
        flags.extend(rflags)
        if pages:
            method=method or "tesseract_raster"
            texts=[]
            for jpg in pages:
                outbase=artifacts_dir/jpg.stem
                rc,out,err=run_cmd(["tesseract",str(jpg),str(outbase),"-l",lang,"hocr"], timeout=120)
                (artifacts_dir/f"tesseract_{jpg.stem}.log").write_text(out+"\n"+err, encoding="utf-8")
                txt=outbase.with_suffix(".txt")
                if txt.exists():
                    texts.append(txt.read_text(encoding="utf-8", errors="ignore"))
            text="\n\n".join(texts)
        else:
            flags.append(_flag("ocr_no_renders","medium","No rendered pages available for OCR fallback.","ocr_semantic"))
    elif not text:
        flags.append(_flag("ocr_unavailable","skipped","No OCR tool available (install ocrmypdf or tesseract).","ocr_semantic"))

    (artifacts_dir/"extracted_text.txt").write_text(text or "", encoding="utf-8")
    return {"method": method, "text": text, "flags": flags}

def semantic_and_logical_validation(text: str) -> Dict[str, Any]:
    flags=[]; checks={}
    if not text or len(text.strip()) < 30:
        flags.append(_flag("ocr_text_too_short","medium","Extracted text is very short; OCR may have failed.","ocr_semantic"))
        return {"checks": checks, "flags": flags}

    years=[int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)]
    if years and max(years) > 2100:
        flags.append(_flag("future_year_detected","high","Contains far-future year; likely OCR error or manipulation.","ocr_semantic"))

    page_nums=re.findall(r"\bPage\s*(\d+)\s*(?:of|/)\s*(\d+)\b", text, flags=re.I)
    if page_nums:
        nums=[(int(a),int(b)) for a,b in page_nums]
        checks["page_number_samples"]=nums[:10]
        totals={b for a,b in nums}
        if len(totals) > 1:
            flags.append(_flag("page_total_inconsistent","medium","Detected inconsistent total page counts in text.","ocr_semantic"))
    else:
        flags.append(_flag("page_numbers_not_found","info","Could not find page numbering patterns in extracted text.","ocr_semantic"))

    if re.search(r"[A-Za-z].*[А-Яа-я]|[А-Яа-я].*[A-Za-z]", text):
        flags.append(_flag("mixed_alphabets","medium","Mixed Latin/Cyrillic characters detected; verify for homoglyph spoofing.","ocr_semantic"))
    return {"checks": checks, "flags": flags}

def layout_template_validation(rendered_pages: List[Path]) -> Dict[str, Any]:
    flags=[]; metrics={}
    if not rendered_pages:
        flags.append(_flag("layout_no_pages","skipped","No rendered pages available for layout checks.","integrity"))
        return {"metrics": metrics, "flags": flags}

    densities=[]; edge_inks=[]
    for p in rendered_pages:
        img=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        h,w=img.shape[:2]
        bw=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
        densities.append(float(np.mean(bw>0)))
        edge=20
        top=float(np.mean(bw[:edge,:]>0)); bottom=float(np.mean(bw[h-edge:,:]>0))
        left=float(np.mean(bw[:, :edge]>0)); right=float(np.mean(bw[:, w-edge:]>0))
        edge_inks.append((top+bottom+left+right)/4.0)

    if densities:
        metrics["ink_density_mean"]=float(np.mean(densities))
        metrics["ink_density_std"]=float(np.std(densities))
        if metrics["ink_density_std"] > 0.08:
            flags.append(_flag("layout_density_inconsistent","medium","Ink density varies across pages; verify coherence.","integrity"))
    if edge_inks:
        metrics["edge_ink_mean"]=float(np.mean(edge_inks))
        if metrics["edge_ink_mean"] > 0.12:
            flags.append(_flag("layout_edge_ink_high","low","High ink near edges; possible cropping/overlay issues.","integrity"))
    return {"metrics": metrics, "flags": flags}

def localized_tamper_detection(rendered_pages: List[Path], artifacts_dir: Path) -> Dict[str, Any]:
    flags=[]; outputs=[]
    if not rendered_pages:
        flags.append(_flag("pdf_tamper_no_pages","skipped","No rendered pages for localized tamper detection.","tamper"))
        return {"outputs": outputs, "flags": flags}
    from PIL import Image, ImageChops
    for p in rendered_pages:
        try:
            im=Image.open(p).convert("RGB")
            tmp=artifacts_dir/f"{p.stem}_re.jpg"
            im.save(tmp,"JPEG",quality=90)
            im2=Image.open(tmp).convert("RGB")
            diff=ImageChops.difference(im,im2)
            arr=np.asarray(diff).astype(np.float32)
            maxv=float(arr.max())
            scale=255.0/max(1.0,maxv)
            hm=np.clip(arr*scale,0,255).astype(np.uint8)
            out=Image.fromarray(hm)
            out_path=artifacts_dir/f"{p.stem}_ela.png"
            out.save(out_path)
            outputs.append({"page": str(p), "ela_heatmap": str(out_path), "max_diff": maxv})
            if maxv > 40:
                flags.append(_flag("pdf_ela_high_signal","medium",f"High ELA signal on {p.name}; inspect highlighted regions.","tamper"))
        except Exception as e:
            flags.append(_flag("pdf_ela_fail","low",f"ELA failed on {p.name}: {e}","tamper"))
    return {"outputs": outputs, "flags": flags}

def signature_and_stamp_checks(rendered_pages: List[Path]) -> Dict[str, Any]:
    flags=[]; metrics={"signature_like_regions":0,"stamp_like_circles":0}
    if not rendered_pages:
        flags.append(_flag("sigstamp_no_pages","skipped","No rendered pages for signature/stamp analysis.","tamper"))
        return {"metrics": metrics, "flags": flags}
    for p in rendered_pages:
        img=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        h,w=img.shape[:2]
        roi=img[int(h*0.65):,:]
        edges=cv2.Canny(roi,50,150)
        lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,minLineLength=60,maxLineGap=10)
        if lines is not None and len(lines) > 25:
            metrics["signature_like_regions"] += 1
        blur=cv2.medianBlur(roi,5)
        circles=cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=100, param2=30, minRadius=20, maxRadius=160)
        if circles is not None:
            metrics["stamp_like_circles"] += int(circles.shape[1])
    if metrics["signature_like_regions"] == 0:
        flags.append(_flag("signature_not_detected","info","No obvious signature-like region detected (heuristic).","tamper"))
    if metrics["stamp_like_circles"] == 0:
        flags.append(_flag("stamp_not_detected","info","No obvious stamp/seal-like circles detected (heuristic).","tamper"))
    return {"metrics": metrics, "flags": flags}

def print_scan_detection(rendered_pages: List[Path]) -> Dict[str, Any]:
    flags=[]; metrics={}
    if not rendered_pages:
        flags.append(_flag("pdf_printscan_no_pages","skipped","No rendered pages for print-scan detection.","print_recapture"))
        return {"metrics": metrics, "flags": flags}
    fms=[]; moires=[]
    for p in rendered_pages:
        img=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        fm=float(cv2.Laplacian(img, cv2.CV_64F).var())
        fms.append(fm)
        f=np.fft.fft2(img)
        mag=np.log(np.abs(np.fft.fftshift(f))+1.0)
        flat=mag.flatten(); k=max(1,int(0.01*flat.size))
        top=np.partition(flat,-k)[-k:]
        moires.append(float(top.sum()/(flat.sum()+1e-9)))
    if fms:
        metrics["focus_var_mean"]=float(np.mean(fms))
        if metrics["focus_var_mean"] < 60:
            flags.append(_flag("pdf_low_sharpness","medium","Low sharpness across pages; may indicate scan/recapture.","print_recapture"))
    if moires:
        metrics["fft_top1pct_ratio_mean"]=float(np.mean(moires))
        if metrics["fft_top1pct_ratio_mean"] > 0.22:
            flags.append(_flag("pdf_possible_moire","low","FFT suggests periodic patterns consistent with screen recapture.","print_recapture"))
    return {"metrics": metrics, "flags": flags}

def ai_generated_document_detection(pdf_path: Path, text: str, meta: Dict[str, Any], hooks: OptionalModelHooks | None) -> Dict[str, Any]:
    flags=[]; metrics={}
    if text:
        lines=[ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            rep=1.0 - (len(set(lines))/max(1,len(lines)))
            metrics["line_repeat_ratio"]=float(rep)
            if rep > 0.35:
                flags.append(_flag("doc_repetitive_lines","medium","High repeated-line ratio; OCR artifact or synthetic doc.","synthetic"))
    prod=(str(meta.get("Producer",""))+" "+str(meta.get("Creator",""))).lower()
    if any(k in prod for k in ["ai","stable","midjourney","dalle","sdxl"]):
        flags.append(_flag("doc_metadata_ai_hint","high","Metadata contains AI tool hints.","synthetic"))
    if hooks:
        res=hooks.detect_document_synthetic(pdf_path)
        if res is None:
            flags.append(_flag("doc_synth_model_missing","skipped","No document synthetic model configured; heuristics only.","synthetic"))
        else:
            metrics["model_score"]=float(res.score_0_1)
            if res.score_0_1 > 0.7:
                flags.append(_flag("doc_synth_model_high","high","Model indicates high likelihood of synthetic document.","synthetic"))
    else:
        flags.append(_flag("doc_synth_hooks_none","skipped","No model hooks; heuristics only.","synthetic"))
    return {"metrics": metrics, "flags": flags}

def analyze_document(pdf_path: Path, out_dir: Path, lang: str = "eng", hooks: OptionalModelHooks | None = None) -> Dict[str, Any]:
    artifacts=ensure_dir(out_dir/"artifacts"/"document")
    integrity=pdf_integrity_and_structure(pdf_path, artifacts)
    metadata=pdf_metadata_analysis(pdf_path, artifacts)
    source=pdf_source_classification(integrity, metadata.get("metadata", {}))
    page_cons=pdf_page_consistency_checks(pdf_path)

    rendered, rflags = render_pdf_to_images(pdf_path, artifacts, max_pages=8)
    integrity["flags"].extend(rflags)

    layout=layout_template_validation(rendered)
    ocr=ocr_pdf(pdf_path, artifacts, lang=lang)
    semantic=semantic_and_logical_validation(ocr.get("text",""))
    tamper=localized_tamper_detection(rendered, artifacts)
    sigstamp=signature_and_stamp_checks(rendered)
    printscan=print_scan_detection(rendered)
    synthetic=ai_generated_document_detection(pdf_path, ocr.get("text",""), metadata.get("metadata", {}), hooks)

    return {
        "path": str(pdf_path),
        "integrity": integrity,
        "metadata": metadata,
        "source": source,
        "page_consistency": page_cons,
        "layout_template": layout,
        "ocr": {"method": ocr.get("method"), "flags": ocr.get("flags", [])},
        "semantic": semantic,
        "tamper_localized": tamper,
        "signature_stamp": sigstamp,
        "print_scan": printscan,
        "synthetic": synthetic,
    }
