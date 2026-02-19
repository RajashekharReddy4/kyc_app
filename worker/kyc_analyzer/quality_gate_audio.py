from typing import Dict, Any, Tuple

def audio_quality_gate(results: Dict[str, Any]) -> Tuple[str | None, str | None]:
    for a in results.get("audios", []):
        meta = a.get("metadata", {}).get("metadata", {})
        analysis = a.get("analysis", {})

        size_bytes = meta.get("size_bytes", 0)
        ffprobe = meta.get("ffprobe", {})
        duration = None
        try:
            duration = float(ffprobe.get("format", {}).get("duration"))
        except Exception:
            duration = None

        silence = analysis.get("silence_boundary", {}).get("metrics", {})
        spectral = analysis.get("spectral", {}).get("metrics", {})
        voice = analysis.get("voice_clone", {}).get("metrics", {})

        silent_ratio = float(silence.get("silent_ratio", 0.0))
        centroid = float(spectral.get("centroid_mean", 0.0))
        f0_std = float(voice.get("f0_std", 0.0)) if voice else 0.0

        # 1) silent / mostly silent -> FAIL
        if silent_ratio >= 0.90:
            return "FAIL", "Audio is silent or mostly silent"

        # 2) spectrum looks empty -> FAIL
        if centroid == 0.0:
            return "FAIL", "Audio has no spectral content"

        # 3) duration/size mismatch (often “empty container”) -> FAIL
        if duration is not None and duration > 30 and size_bytes < 200_000:
            return "FAIL", "Audio duration/size mismatch (likely empty or invalid audio)"

        # Optional: if you require actual speech (not just any sound)
        if f0_std == 0.0:
            return "REVIEW", "No clear voice pitch detected (may be non-speech audio)"

    return None, None
