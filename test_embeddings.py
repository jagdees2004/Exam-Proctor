"""
test_embeddings.py — Diagnostic: test if MobileFaceNet produces discriminative embeddings.

Usage:
  python test_embeddings.py

Steps:
  1. Press SPACE to capture person 1's face
  2. Switch to person 2, press SPACE again
  3. Script prints the cosine similarity score
  
If the score is > 0.4 for two DIFFERENT people, the model needs tuning.
If the score is > 0.6 for the SAME person, the model is working.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
YUNET_MODEL = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = str(MODELS_DIR / "mobilefacenet.onnx")

ALIGNMENT_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame, face):
    landmarks = np.array([
        [face[4], face[5]],
        [face[6], face[7]],
        [face[8], face[9]],
        [face[10], face[11]],
        [face[12], face[13]],
    ], dtype=np.float32)
    transform = cv2.estimateAffinePartial2D(landmarks, ALIGNMENT_TEMPLATE, method=cv2.LMEDS)[0]
    if transform is None:
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        crop = frame[max(0, y):y+h, max(0, x):x+w]
        return cv2.resize(crop if crop.size > 0 else frame, (112, 112))
    return cv2.warpAffine(frame, transform, (112, 112))


def extract_embedding(session, input_name, output_name, aligned_face):
    blob = aligned_face.astype(np.float32)
    blob = blob[:, :, ::-1]  # BGR → RGB
    blob = (blob - 127.5) / 127.5
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    embedding = session.run([output_name], {input_name: blob})[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.flatten()


def main():
    # Load models
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.6, 0.3, 10)
    session = ort.InferenceSession(SFACE_MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("\n" + "=" * 60)
    print("  MobileFaceNet Embedding Quality Test")
    print("=" * 60)
    print("\nInstructions:")
    print("  1. Person 1: Look at the camera, press SPACE to capture")
    print("  2. Person 2: Switch to a different person, press SPACE")
    print("  3. Press Q to quit")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    embeddings = []
    labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None and len(faces) > 0:
            for face in faces:
                x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                cv2.rectangle(display, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(display, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        capture_num = len(embeddings) + 1
        cv2.putText(display, f"Press SPACE to capture person {capture_num}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Embedding Test", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if faces is not None and len(faces) > 0:
                aligned = align_face(frame, faces[0])
                emb = extract_embedding(session, input_name, output_name, aligned)
                embeddings.append(emb)
                label = f"Person {len(embeddings)}"
                labels.append(label)
                print(f"\n  ✓ Captured {label}")
                print(f"    Embedding norm: {np.linalg.norm(emb):.4f}")
                print(f"    Embedding sample: [{', '.join(f'{v:.4f}' for v in emb[:5])}...]")

                # Compare with all previous
                if len(embeddings) > 1:
                    print(f"\n  --- Similarity Matrix ---")
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            score = float(np.dot(embeddings[i], embeddings[j]))
                            verdict = "MATCH ✓" if score >= 0.30 else "DIFFERENT ✗"
                            print(f"    {labels[i]} vs {labels[j]}: {score:.4f}  → {verdict}")
            else:
                print("  ✗ No face detected, try again")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) >= 2:
        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                score = float(np.dot(embeddings[i], embeddings[j]))
                if score >= 0.30:
                    print(f"  {labels[i]} vs {labels[j]}: {score:.4f} → SAME PERSON (above 0.30)")
                else:
                    print(f"  {labels[i]} vs {labels[j]}: {score:.4f} → DIFFERENT PERSON (below 0.30)")
        print()
        print("  If two DIFFERENT people score > 0.3, the model may")
        print("  need threshold adjustments.")
        print("=" * 60)


if __name__ == "__main__":
    main()
