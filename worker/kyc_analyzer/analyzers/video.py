import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _ear(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return float((A + B) / (2.0 * C + 1e-6))

def analyze_video_kyc(video_path: str, max_seconds: float = 10.0, sample_fps: int = 12):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "decision": "fail",
            "score_0_100": 0,
            "risks": ["VIDEO_UNREADABLE"],
            "report": {"video": {"error": "Cannot open video"}}
        }

    # WebM often reports wrong fps; protect it
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 10 or fps > 120:
        fps = 30.0

    step = max(int(round(fps / float(sample_fps))), 1)
    max_ms = float(max_seconds) * 1000.0

    face_frames = 0
    total_sampled = 0
    max_no_face_streak = 0
    no_face_streak = 0

    ear_series = []
    yaw_series = []

    blur_scores = []
    brightness_vals = []

    # thresholds
    MIN_FACE_RATIO = 0.50
    BLUR_TH = 55.0
    DARK_TH = 40.0
    BRIGHT_TH = 220.0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as fm:

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ✅ time-based stop (fixes "sampled_frames=2" webm problem)
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if t_ms > max_ms:
                break

            idx += 1
            if idx % step != 0:
                continue

            total_sampled += 1

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness_vals.append(float(np.mean(gray)))
            blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if not res.multi_face_landmarks:
                no_face_streak += 1
                max_no_face_streak = max(max_no_face_streak, no_face_streak)
                continue

            face_frames += 1
            no_face_streak = 0

            lms = res.multi_face_landmarks[0].landmark

            # EAR
            left_eye = np.array([[lms[i].x * w, lms[i].y * h] for i in LEFT_EYE], dtype=np.float32)
            right_eye = np.array([[lms[i].x * w, lms[i].y * h] for i in RIGHT_EYE], dtype=np.float32)
            ear = (_ear(left_eye) + _ear(right_eye)) / 2.0
            ear_series.append(float(ear))

            # Head turn (stable): nose x relative to face width
            left_face = lms[234].x
            right_face = lms[454].x
            nose = lms[1].x
            face_w = max(right_face - left_face, 1e-6)
            yaw_norm = float((nose - (left_face + right_face) / 2.0) / face_w)
            yaw_series.append(yaw_norm)

    cap.release()

    face_ratio = (face_frames / total_sampled) if total_sampled else 0.0

    # Hard fail: no face
    if total_sampled == 0 or face_frames == 0 or face_ratio < 0.10:
        return {
            "decision": "fail",
            "score_0_100": 0,
            "risks": ["NO_FACE_DETECTED"],
            "report": {
                "video": {
                    "face_ratio": round(face_ratio, 3),
                    "blink_count": 0,
                    "head_turn_detected": False,
                    "liveness_pass": False,
                    "sampled_frames": int(total_sampled),
                    "max_no_face_streak": int(max_no_face_streak),
                }
            }
        }

    # ---- Adaptive blink detection ----
    blink_count = 0
    if len(ear_series) >= 6:
        sorted_ear = np.sort(np.array(ear_series))
        top = sorted_ear[int(len(sorted_ear) * 0.4):]
        baseline = float(np.median(top)) if len(top) else float(np.median(sorted_ear))

        th = baseline * 0.75
        low_run = 0
        for e in ear_series:
            if e < th:
                low_run += 1
            else:
                if low_run >= 2:
                    blink_count += 1
                low_run = 0
        if low_run >= 2:
            blink_count += 1

    # ---- Head turn detection ----
    head_turn_detected = False
    if len(yaw_series) >= 6:
        head_turn_detected = (max(yaw_series) - min(yaw_series)) > 0.25

    avg_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
    avg_brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0

    too_blurry = avg_blur < BLUR_TH
    too_dark = avg_brightness < DARK_TH
    too_bright = avg_brightness > BRIGHT_TH

    liveness_pass = (blink_count >= 1) or head_turn_detected

    risks = []
    score = 100

    if face_ratio < MIN_FACE_RATIO:
        risks.append("FACE_NOT_STABLE")
        score -= 35
    if too_blurry:
        risks.append("VIDEO_TOO_BLURRY")
        score -= 10
    if too_dark:
        risks.append("VIDEO_TOO_DARK")
        score -= 10
    if too_bright:
        risks.append("VIDEO_TOO_BRIGHT")
        score -= 10
    if not liveness_pass:
        risks.append("LIVENESS_NOT_CONFIRMED")
        score -= 30

    score = int(max(0, min(100, score)))

    if face_ratio >= MIN_FACE_RATIO and liveness_pass and score >= 70:
        decision = "pass"
    elif score >= 40:
        decision = "review"
    else:
        decision = "fail"

    return {
        "decision": decision,
        "score_0_100": score,
        "risks": risks,
        "report": {
            "video": {
                "face_ratio": round(face_ratio, 3),
                "blink_count": int(blink_count),
                "head_turn_detected": bool(head_turn_detected),
                "liveness_pass": bool(liveness_pass),
                "avg_blur": round(avg_blur, 2),
                "avg_brightness": round(avg_brightness, 2),
                "sampled_frames": int(total_sampled),
                "max_no_face_streak": int(max_no_face_streak),
            }
        }
    }