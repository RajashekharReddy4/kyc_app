from __future__ import annotations
from typing import Dict, Any, List
from .utils import risk_from_flags, clamp01
import math
def collect_flags(obj: Any) -> List[dict]:
    flags = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "flags" and isinstance(v, list):
                flags.extend(v)
            else:
                flags.extend(collect_flags(v))
    elif isinstance(obj, list):
        for it in obj:
            flags.extend(collect_flags(it))
    return flags

def apply_tamper_tiers(max_diff, risks, profile):
    """
    Tiered tamper severity logic based on ELA max_diff.
    """

    decision_override = None

    if max_diff < 20:
        # Low noise
        risks["tamper"] = min(1.0, risks.get("tamper", 0) + 0.05)

    elif 20 <= max_diff < 40:
        # Medium suspicion
        risks["tamper"] = min(1.0, risks.get("tamper", 0) + 0.2)

    elif 40 <= max_diff < 60:
        # High tamper suspicion
        risks["tamper"] = min(1.0, risks.get("tamper", 0) + 0.4)

        if profile in ["strict", "aggressive"]:
            risks["tamper"] += 0.2

    else:  # >= 60
        # Critical tamper
        risks["tamper"] = 1.0

        if profile == "conservative":
            decision_override = "REVIEW"
        elif profile == "strict":
            decision_override = "FAIL"
        else:  # aggressive
            decision_override = "FAIL"

    return risks, decision_override

def score_case(results: Dict[str, Any], weights: Dict[str, float], profile: str = "strict") -> Dict[str, Any]:

    flags = collect_flags(results)

    buckets = {k: [] for k in weights.keys()}

    for f in flags:
        cat = f.get("category")
        if cat in buckets:
            buckets[cat].append(f)
        else:
            buckets.setdefault("metadata", []).append(f)

    # Base risk from flags
    risks = {k: risk_from_flags(v) for k, v in buckets.items()}

    # ------------------------------------
    # Extract max ELA diff
    # ------------------------------------

    max_diff = 0

    try:
        for doc in results.get("documents", []):
            for o in doc.get("tamper_localized", {}).get("outputs", []):
                max_diff = max(max_diff, o.get("max_diff", 0))
    except:
        pass

    decision_override = None

    # ------------------------------------
    # Tiered Tamper Severity (Fixed)
    # ------------------------------------

    if max_diff < 20:
        risks["tamper"] = min(1.0, risks.get("tamper", 0) + 0.05)

    elif 20 <= max_diff < 40:
        risks["tamper"] = min(1.0, risks.get("tamper", 0) + 0.2)

    elif 40 <= max_diff < 60:
        risks["tamper"] = min(1.0, risks.get("tamper", 0) + 0.4)

        if profile in ["strict", "aggressive"]:
            risks["tamper"] = min(1.0, risks["tamper"] + 0.2)

        # High tamper escalation
        if profile == "aggressive":
            decision_override = "FAIL"
        elif profile == "strict":
            decision_override = "REVIEW"

    else:  # >= 60 (CRITICAL)
        risks["tamper"] = 1.0

        if profile == "conservative":
            decision_override = "REVIEW"
        else:
            decision_override = "FAIL"

    # ------------------------------------
    # CRITICAL SAFETY CHECK
    # Tamper should never PASS if = 1
    # ------------------------------------

    if risks.get("tamper", 0) >= 1.0:
        if profile == "conservative":
            decision_override = decision_override or "REVIEW"
        else:
            decision_override = "FAIL"

    # ------------------------------------
    # Weighted scoring
    # ------------------------------------

    final = 0.0
    for k, w in weights.items():
        final += w * risks.get(k, 0.0)

    score = round(clamp01(final) * 100.0, 2)

    # ------------------------------------
    # Decision Logic
    # ------------------------------------
    

    if decision_override:
        decision = decision_override
    else:
        if profile == "conservative":
            decision = "PASS" if score <= 30 else "REVIEW" if score <= 60 else "FAIL"

        elif profile == "strict":
            decision = "PASS" if score <= 20 else "REVIEW" if score <= 45 else "FAIL"

        else:  # aggressive
            decision = "PASS" if score <= 15 else "REVIEW" if score <= 30 else "FAIL"
        
    fraud_confidence = compute_fraud_confidence(risks, decision)
    return {
        "risks": risks,
        "score_0_100": score,
        "decision": decision,
        "fraud_confidence": fraud_confidence
    }



def compute_fraud_confidence(risks: Dict[str, float], decision: str) -> float:
    """
    Converts risk signals into a probabilistic fraud confidence score.
    """

    # Base risk aggregation (non-linear)
    base = (
        1.8 * risks.get("tamper", 0) +
        1.2 * risks.get("ocr_semantic", 0) +
        1.0 * risks.get("metadata", 0) +
        0.8 * risks.get("print_recapture", 0) +
        0.6 * risks.get("synthetic", 0) +
        0.5 * risks.get("correlation", 0)
    )

    # Sigmoid scaling (keeps result between 0 and 1)
    confidence = 1 / (1 + math.exp(-base + 2))

    # Decision influence
    if decision == "FAIL":
        confidence = max(confidence, 0.75)
    elif decision == "REVIEW":
        confidence = max(confidence, 0.45)

    return round(min(confidence, 1.0), 4)

