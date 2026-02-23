"""
engine.py — Face Detection (YuNet) + Face Recognition (MobileFaceNet) engine.

Uses:
  - cv2.FaceDetectorYN (YuNet) for face detection
  - onnxruntime for MobileFaceNet inference (512-D embeddings)
  - Brightness check to detect covered/off camera
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass

MODELS_DIR = Path(__file__).parent / "models"

YUNET_MODEL = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
FACE_REC_MODEL = str(MODELS_DIR / "mobilefacenet.onnx")

# ── Thresholds ───────────────────────────────────────────────────────────────
FACE_SCORE_THRESHOLD = 0.6       # YuNet detection confidence
FACE_NMS_THRESHOLD = 0.3
COSINE_SIMILARITY_THRESHOLD = 0.30  # Cosine similarity: MobileFaceNet uses higher-dim embeddings, will be tuned after testing
BRIGHTNESS_THRESHOLD = 40         # Mean pixel value below this = too dark / camera off
VARIANCE_THRESHOLD = 15           # Pixel variance below this = uniform image (camera covered)


# Standard face alignment template (for 112×112 crop)
ALIGNMENT_TEMPLATE = np.array([
    [38.2946, 51.6963],   # right eye
    [73.5318, 51.5014],   # left eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # right mouth corner
    [70.7299, 92.2041],   # left mouth corner
], dtype=np.float32)


@dataclass
class FaceResult:
    """Result of a face verification check."""
    face_count: int = 0
    identity_match: bool = False
    similarity_score: float = 0.0
    status: str = "no_face"  # ok | no_face | multiple_faces | identity_mismatch | camera_blocked


class ProctorEngine:
    """
    Lightweight face detection + recognition engine.

    - Checks frame brightness (detects covered/off camera)
    - Detects faces using YuNet (cv2.FaceDetectorYN)
    - Aligns face crops manually (affine transform)
    - Extracts 512-D embeddings using MobileFaceNet via onnxruntime
    - Stores reference embeddings in-memory per user_id
    """

    def __init__(self) -> None:
        # Face detector — YuNet
        self._detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=(320, 320),
            score_threshold=FACE_SCORE_THRESHOLD,
            nms_threshold=FACE_NMS_THRESHOLD,
            top_k=10,
        )

        # Face recognizer — MobileFaceNet via onnxruntime
        self._session = ort.InferenceSession(
            FACE_REC_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # In-memory reference embeddings
        self._embeddings: dict[str, np.ndarray] = {}

        print("[ProctorEngine] Initialized with YuNet + MobileFaceNet (onnxruntime)")

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_frame_too_dark(frame: np.ndarray) -> bool:
        """Check if the frame is too dark or too uniform (camera covered/off)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        variance = float(np.std(gray))
        print(f"  [Brightness] mean={mean_brightness:.1f}, variance={variance:.1f} (thresholds: bright<{BRIGHTNESS_THRESHOLD}, var<{VARIANCE_THRESHOLD})")
        # Camera is blocked if too dark OR image is too uniform (solid color = covered)
        return mean_brightness < BRIGHTNESS_THRESHOLD or variance < VARIANCE_THRESHOLD

    def _detect_faces(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces in frame. Returns Nx15 array (or empty)."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is not None:
            count = len(faces)
            scores = [f"{faces[i][14]:.3f}" for i in range(min(count, 5))]
            print(f"  [FaceDetect] Found {count} face(s), scores={scores}")
            return faces
        print("  [FaceDetect] No faces found")
        return np.array([])

    def _align_face(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        Align and crop a face to 112×112 using the 5 landmarks from YuNet.

        YuNet face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
                            x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        landmarks = np.array([
            [face[4], face[5]],   # right eye
            [face[6], face[7]],   # left eye
            [face[8], face[9]],   # nose tip
            [face[10], face[11]], # right mouth corner
            [face[12], face[13]], # left mouth corner
        ], dtype=np.float32)

        transform = cv2.estimateAffinePartial2D(
            landmarks, ALIGNMENT_TEMPLATE, method=cv2.LMEDS
        )[0]

        if transform is None:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            x, y = max(0, x), max(0, y)
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                crop = frame
            return cv2.resize(crop, (112, 112))

        aligned = cv2.warpAffine(frame, transform, (112, 112))
        return aligned

    def _extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Run MobileFaceNet ONNX model to extract 512-D face embedding."""
        blob = aligned_face.astype(np.float32)
        blob = blob[:, :, ::-1]  # BGR → RGB (MobileFaceNet expects RGB)
        blob = (blob - 127.5) / 127.5  # normalize to [-1, 1]
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)  # (1, 3, 112, 112)

        embedding = self._session.run(
            [self._output_name], {self._input_name: blob}
        )[0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.flatten()

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb1, emb2))

    # ── Public API ───────────────────────────────────────────────────────

    def register_face(self, frame: np.ndarray, user_id: str) -> dict:
        """
        Register a user's face from a webcam frame.
        """
        print(f"\n[REGISTER] user_id={user_id}, frame={frame.shape}")

        # Check brightness
        if self._is_frame_too_dark(frame):
            return {
                "success": False,
                "message": "Camera appears blocked or too dark. Please ensure good lighting.",
                "face_count": 0,
            }

        faces = self._detect_faces(frame)
        face_count = len(faces)

        if face_count == 0:
            return {
                "success": False,
                "message": "No face detected. Please look at the camera.",
                "face_count": 0,
            }

        if face_count > 1:
            return {
                "success": False,
                "message": "Multiple faces detected. Only one person should be visible.",
                "face_count": face_count,
            }

        # Align face and extract embedding
        aligned = self._align_face(frame, faces[0])
        embedding = self._extract_embedding(aligned)
        self._embeddings[user_id] = embedding

        print(f"  [REGISTER] ✓ Embedding saved for '{user_id}'")

        return {
            "success": True,
            "message": f"Face registered for user '{user_id}'.",
            "face_count": 1,
        }

    def verify_face(self, frame: np.ndarray, user_id: str) -> FaceResult:
        """
        Verify the current frame against the registered reference for user_id.
        """
        print(f"\n[VERIFY] user_id={user_id}, frame={frame.shape}")
        result = FaceResult()

        if user_id not in self._embeddings:
            result.status = "not_registered"
            print("  [VERIFY] Not registered!")
            return result

        # Check brightness — camera blocked / off
        if self._is_frame_too_dark(frame):
            result.status = "camera_blocked"
            print("  [VERIFY] ✗ Camera blocked (too dark)")
            return result

        faces = self._detect_faces(frame)
        result.face_count = len(faces)

        if result.face_count == 0:
            result.status = "no_face"
            print("  [VERIFY] ✗ No face")
            return result

        if result.face_count > 1:
            result.status = "multiple_faces"
            print(f"  [VERIFY] ✗ Multiple faces ({result.face_count})")
            return result

        # Single face — verify identity
        aligned = self._align_face(frame, faces[0])
        current_embedding = self._extract_embedding(aligned)
        ref_embedding = self._embeddings[user_id]
        score = self._cosine_similarity(ref_embedding, current_embedding)

        result.similarity_score = round(score, 4)
        print(f"  [VERIFY] Similarity score: {score:.4f} (threshold: {COSINE_SIMILARITY_THRESHOLD})")

        if score >= COSINE_SIMILARITY_THRESHOLD:
            result.identity_match = True
            result.status = "ok"
            print("  [VERIFY] ✓ Identity match!")
        else:
            result.identity_match = False
            result.status = "identity_mismatch"
            print("  [VERIFY] ✗ Identity mismatch!")

        return result

    def is_registered(self, user_id: str) -> bool:
        return user_id in self._embeddings
