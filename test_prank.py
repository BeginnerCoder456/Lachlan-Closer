"""
Prank calibration tool.
Run this BEFORE the calculator to check:
  1. Your photo loads and a face is found in it
  2. What confidence scores the webcam sees for your face

Lower confidence = better match. The calculator triggers shutdown
when confidence < CONFIDENCE_THRESHOLD (currently 90).
This tool will tell you what number to use.
"""

import cv2
import numpy as np
import os
import sys

PHOTO = "lachlan.jpg"
FACE_SIZE = (100, 100)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


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

    # Draw box and save preview so user can verify it found the right face
    preview = img.copy()
    cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("lachlan_preview.jpg", preview)
    print("[OK] Saved lachlan_preview.jpg — open it to confirm the face box looks right.")

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

    print("\n[LIVE] Webcam open. Look at the camera.")
    print("  Confidence scores will print below.")
    print("  LOWER score = better match.")
    print("  Press Q in the video window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (fx, fy, fw, fh) in faces:
            face_crop = cv2.resize(gray[fy:fy+fh, fx:fx+fw], FACE_SIZE)
            label, confidence = recognizer.predict(face_crop)
            conf_int = int(confidence)

            if conf_int < 60:
                verdict = "STRONG MATCH"
            elif conf_int < 100:
                verdict = "GOOD MATCH"
            elif conf_int < 130:
                verdict = "WEAK MATCH"
            else:
                verdict = "no match"

            print(f"  confidence = {conf_int:4d}   {verdict}")

            color = (0, 255, 0) if conf_int < 100 else (0, 0, 255)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
            cv2.putText(frame, f"{conf_int} {verdict}", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Prank Calibration — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n─────────────────────────────────────────")
    print("Look at the confidence values printed above.")
    print("Your face when detected = those numbers.")
    print("Set CONFIDENCE_THRESHOLD in calculator.py to")
    print("~20 points ABOVE your typical score.")
    print("e.g. if you see 65–80, set threshold = 100")
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    face = load_photo()
    rec  = train(face)
    run_live(rec)
