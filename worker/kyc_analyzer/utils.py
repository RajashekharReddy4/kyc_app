from __future__ import annotations
import json, shutil, subprocess, hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout running: {' '.join(cmd)}"

def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def decision_from_score(score_0_100: float, pass_max: int, review_max: int) -> str:
    if score_0_100 <= pass_max:
        return "PASS"
    if score_0_100 <= review_max:
        return "REVIEW"
    return "FAIL"

def risk_from_flags(flags: List[Dict[str, Any]]) -> float:
    if not flags:
        return 0.0
    sev_map = {"info": 0.05, "low": 0.2, "medium": 0.5, "high": 0.9, "critical": 1.0, "skipped": 0.15}
    vals = [sev_map.get(str(f.get("severity","low")), 0.2) for f in flags]
    prod = 1.0
    for v in vals:
        prod *= (1.0 - clamp01(v))
    return 1.0 - prod

def cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a) + 1e-9); nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))
