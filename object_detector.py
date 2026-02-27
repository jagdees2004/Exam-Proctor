"""
object_detector.py — YOLOv8n ONNX object detector with letterbox preprocessing.

Uses onnxruntime for inference. Scans frames for forbidden objects during exams:
  - COCO class 62: tv / monitor / tablet
  - COCO class 63: laptop
  - COCO class 67: cell phone
  - COCO class 74: clock / watch
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass

MODELS_DIR = Path(__file__).parent / "models"

# Try FP32 model first (better quality), fall back to INT8
_FP32_MODEL = MODELS_DIR / "yolov8n.onnx"
_INT8_MODEL = MODELS_DIR / "yolov8n_int8.onnx"
YOLO_MODEL = str(_FP32_MODEL if _FP32_MODEL.exists() else _INT8_MODEL)

# ── COCO class mapping (only the ones we care about) ─────────────────────────
FORBIDDEN_CLASSES: dict[int, str] = {
    62: "tv/monitor",
    63: "laptop",
    67: "cell phone",
    74: "clock/watch",
}

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.15   # Low threshold for INT8 (outputs lower scores)
NMS_THRESHOLD = 0.45
INPUT_SIZE = 640


@dataclass
class Detection:
    """A single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list  # [x, y, w, h]


class ObjectDetector:
    """
    YOLOv8n ONNX object detector using onnxruntime.

    Only reports detections for forbidden exam objects
    (phones, laptops, monitors, watches/clocks).
    """

    def __init__(self) -> None:
        self._session = ort.InferenceSession(
            YOLO_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Read actual input size from model
        inp_shape = self._session.get_inputs()[0].shape
        if isinstance(inp_shape[2], int) and isinstance(inp_shape[3], int):
            self._input_h = inp_shape[2]
            self._input_w = inp_shape[3]
        else:
            self._input_h = INPUT_SIZE
            self._input_w = INPUT_SIZE

        print(f"[ObjectDetector] Loaded {Path(YOLO_MODEL).name} "
              f"(input: {self._input_w}x{self._input_h})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8n inference on a frame.

        Returns:
            List of Detection objects for forbidden items only.
        """
        h_orig, w_orig = frame.shape[:2]

        # ── Preprocess with letterbox ────────────────────────────────────
        blob, ratio, (pad_w, pad_h) = self._letterbox(frame)

        # ── Forward pass ─────────────────────────────────────────────────
        outputs = self._session.run(None, {self._input_name: blob})

        # YOLOv8 output shape: (1, 84, N) → transpose to (N, 84)
        predictions = outputs[0][0].T

        # ── Postprocess ──────────────────────────────────────────────────
        boxes_raw = predictions[:, :4]      # cx, cy, w, h
        class_scores = predictions[:, 4:]   # 80 class scores

        # Extract scores ONLY for forbidden classes
        forbidden_ids = list(FORBIDDEN_CLASSES.keys())
        forbidden_scores = class_scores[:, forbidden_ids]  # (N, 4)

        # Best forbidden class per detection
        best_idx = np.argmax(forbidden_scores, axis=1)
        best_conf = forbidden_scores[np.arange(len(best_idx)), best_idx]

        # Keep only detections above threshold
        valid = best_conf > CONFIDENCE_THRESHOLD

        # Debug: log top forbidden scores every call
        if len(best_conf) > 0:
            top_score = best_conf.max()
            top_idx = best_idx[best_conf.argmax()]
            top_name = FORBIDDEN_CLASSES[forbidden_ids[top_idx]]
            n_above = valid.sum()
            print(f"  [ObjectDetect] Top forbidden: {top_name}={top_score:.3f}, "
                  f"{n_above} detections above {CONFIDENCE_THRESHOLD}")

        if not np.any(valid):
            return []

        boxes_valid = boxes_raw[valid]
        class_ids = np.array([forbidden_ids[i] for i in best_idx[valid]])
        confidences = best_conf[valid]

        # Convert cx,cy,w,h → x,y,w,h (top-left corner) in letterboxed coords
        boxes = np.zeros_like(boxes_valid)
        boxes[:, 0] = boxes_valid[:, 0] - boxes_valid[:, 2] / 2
        boxes[:, 1] = boxes_valid[:, 1] - boxes_valid[:, 3] / 2
        boxes[:, 2] = boxes_valid[:, 2]
        boxes[:, 3] = boxes_valid[:, 3]

        # Undo letterbox: remove padding then scale back to original size
        boxes[:, 0] = (boxes[:, 0] - pad_w) / ratio
        boxes[:, 1] = (boxes[:, 1] - pad_h) / ratio
        boxes[:, 2] = boxes[:, 2] / ratio
        boxes[:, 3] = boxes[:, 3] / ratio

        # Clip to frame bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        # ── Build final detection list ───────────────────────────────────
        detections: list[Detection] = []
        for i in indices:
            cid = int(class_ids[i])
            conf = float(confidences[i])
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=FORBIDDEN_CLASSES[cid],
                    confidence=round(conf, 3),
                    bbox=boxes[i].astype(int).tolist(),
                )
            )

        if detections:
            det_str = ", ".join(f"{d.class_name}({d.confidence:.2f})" for d in detections)
            print(f"  [ObjectDetect] 🚨 Found: {det_str}")

        return detections

    def _letterbox(self, frame: np.ndarray) -> tuple:
        """
        Letterbox resize: scale image to fit input size while maintaining
        aspect ratio, padding the shorter side with gray (114).

        Returns:
            (blob, ratio, (pad_w, pad_h))
        """
        h, w = frame.shape[:2]
        target_h, target_w = self._input_h, self._input_w

        # Scale ratio (new / old)
        ratio = min(target_w / w, target_h / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        # Padding
        pad_w = (target_w - new_w) / 2
        pad_h = (target_h - new_h) / 2

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Ensure exact size (rounding can be off by 1)
        if padded.shape[:2] != (target_h, target_w):
            padded = cv2.resize(padded, (target_w, target_h))

        # Normalize and convert BGR→RGB, HWC→CHW
        blob = padded.astype(np.float32) / 255.0
        blob = blob[:, :, ::-1]     # BGR → RGB
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, 0)  # (1, 3, H, W)
        return np.ascontiguousarray(blob), ratio, (pad_w, pad_h)
