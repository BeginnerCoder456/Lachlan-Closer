"""
Prank calibration tool.
Run this BEFORE the calculator to check:
  1. Your photo loads and a face is found in it
  2. What confidence scores the webcam sees for your face
  3. Whether the shutdown command actually works on this PC

Lower confidence = better match. The calculator triggers shutdown
when confidence < CONFIDENCE_THRESHOLD (set in calculator.py).
"""

import cv2
import numpy as np
import os
import sys
import platform
import subprocess

PHOTO = "lachlan.jpg"
FACE_SIZE = (100, 100)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def test_shutdown():
    """Schedule a shutdown 30 seconds away, then immediately cancel it.
    This confirms the shutdown command works WITHOUT actually shutting down."""
    system = platform.system()
    print("\n[TEST] Testing shutdown command (will schedule then immediately cancel)...")
    if system == "Windows":
        r = subprocess.run(["shutdown", "/s", "/f", "/t", "30"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["shutdown", "/a"], capture_output=True)
            print("[OK] Shutdown command works! (/a cancelled the test)")
        else:
            print(f"[FAIL] Shutdown command failed: {r.stderr.strip()}")
            print("  → Try running this script as Administrator.")
    elif system == "Darwin":
        print("[INFO] Skipping shutdown test on Mac — run manually if needed.")
    else:
        print("[INFO] Skipping shutdown test on Linux — run manually if needed.")


def load_photo():
    if not os.path.exists(PHOTO):
        print(f"[ERROR] {PHOTO} not found in this folder!")
        sys.exit(1)

    img = cv2.imread(PHOTO)
    if img is None:
        print(f"[ERROR] Could not read {PHOTO} — is it a valid image file?")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("[ERROR] No face detected in lachlan.jpg!")
        print("  → Try a clearer, front-on, well-lit photo.")
        sys.exit(1)

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
    print(f"[OK] Face found in {PHOTO}  (region: x={x} y={y} w={w} h={h})")

    preview = img.copy()
    cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("lachlan_preview.jpg", preview)
    print("[OK] Saved lachlan_preview.jpg — open it to confirm the green box is on the face.")

    return face_crop


def train(face_crop):
    samples, labels = [], []
    for flip in (False, True):
        img_ = cv2.flip(face_crop, 1) if flip else face_crop
        for brightness in (0, 20, -20):
            adjusted = np.clip(img_.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
            samples.append(adjusted)
            labels.append(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(labels))
    return recognizer


def run_live(recognizer):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("\n[LIVE] Webcam open. Sit in front of the camera like Lachlan would.")
    print("  Confidence scores print below. LOWER = better match.")
    print("  Press Q in the video window to quit.\n")

    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for (fx, fy, fw, fh) in faces:
            face_crop = cv2.resize(gray[fy:fy+fh, fx:fx+fw], FACE_SIZE)
            label, confidence = recognizer.predict(face_crop)
            conf_int = int(confidence)
            scores.append(conf_int)

            if conf_int < 60:
                verdict = "STRONG MATCH"
                color = (0, 200, 0)
            elif conf_int < 100:
                verdict = "GOOD MATCH"
                color = (0, 180, 80)
            elif conf_int < 130:
                verdict = "WEAK MATCH"
                color = (0, 140, 255)
            else:
                verdict = "no match"
                color = (0, 0, 255)

            print(f"  confidence = {conf_int:4d}   {verdict}")

            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
            cv2.putText(frame, f"{conf_int} — {verdict}", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Prank Calibration — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if scores:
        avg = int(sum(scores) / len(scores))
        low = min(scores)
        print(f"\n─────────────────────────────────────────")
        print(f"Results: lowest={low}  average={avg}")
        print(f"Recommended CONFIDENCE_THRESHOLD: {low + 15}")
        print(f"Set this in calculator.py line ~41")
        print(f"─────────────────────────────────────────")
    else:
        print("\n[!] No faces were detected at all during the live test.")
        print("  → Check lachlan_preview.jpg to see if the photo is valid.")
        print("  → Make sure your webcam is not covered and lighting is decent.")


if __name__ == "__main__":
    test_shutdown()
    face = load_photo()
    rec  = train(face)
    run_live(rec)
