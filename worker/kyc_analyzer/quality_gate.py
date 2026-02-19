from typing import Dict, Any, Tuple


def image_quality_gate(results: Dict[str, Any]) -> Tuple[str | None, str | None]:
    """
    Bank-grade adaptive image validation.
    Handles ID and Selfie separately.
    Returns:
        (decision_override, reason)
    """

    for img in results.get("images", []):

        expected = img.get("expected_type")  # must be set in analyze_image()

        metrics_global = img.get("global_quality", {}).get("metrics", {})
        metrics_geom = img.get("resolution_geometry", {}).get("metrics", {})
        metrics_face = img.get("object_region", {}).get("metrics", {})
        metrics_rec = img.get("print_recapture", {}).get("metrics", {})
        boundary_flags = img.get("document_boundary", {}).get("flags", [])
        recapture_flags = img.get("screen_recapture", {}).get("flags", [])

        sharpness = metrics_global.get("sharpness_laplacian_var", 0)
        width = metrics_geom.get("width", 0)
        height = metrics_geom.get("height", 0)
        faces = metrics_face.get("faces_detected", 0)

        if height == 0:
            continue

        megapixels = (width * height) / 1_000_000
        aspect_ratio = max(width, height) / min(width, height)

        boundary_codes = {f.get("code") for f in boundary_flags}
        recapture_codes = {f.get("code") for f in recapture_flags}

        # ============================================================
        # UNIVERSAL HARD FAILS
        # ============================================================

        # Severe blur
        if sharpness < 15:
            return "FAIL", "Severely blurred image"

        # Extremely low resolution
        if megapixels < 0.3:
            return "FAIL", "Image resolution too low"

        # Strong screen recapture
        if "possible_screen_banding" in recapture_codes:
            return "FAIL", "Screen recapture detected"

        # ============================================================
        # ID CARD VALIDATION
        # ============================================================

        if expected == "id":

            # Strict megapixel rule for ID
            if megapixels < 0.8:
                return "FAIL", "ID image resolution too low"

            # ID card crop ratio
            if not (1.35 <= aspect_ratio <= 1.75):
                return "FAIL", "Incorrect ID crop or full-frame capture"

            # Boundary must exist
            if "doc_boundary_not_found" in boundary_codes:
                return "FAIL", "Document boundary not detected"

            # Face required
            if faces == 0:
                return "FAIL", "ID must contain visible face"

            # Moderate blur → review
            if sharpness < 50:
                return "REVIEW", "ID slightly blurred"

        # ============================================================
        # SELFIE VALIDATION
        # ============================================================

        if expected == "selfie":

            # Selfie megapixel softer threshold
            if megapixels < 0.5:
                return "REVIEW", "Low resolution selfie"

            # Exactly one face
            if faces != 1:
                return "FAIL", "Exactly one face required in selfie"

            face_sharpness = metrics_face.get("face_sharpness", 0)

            if sharpness > 0:
                ratio = face_sharpness / sharpness

                if ratio < 0.5:
                    return "REVIEW", "Face region too small or low clarity"

            # Slight blur → review
            if sharpness < 40:
                return "REVIEW", "Selfie slightly blurred"

    return None, None
