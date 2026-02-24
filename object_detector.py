"""
object_detector.py — Lightweight YOLOv8n INT8 ONNX object detector.

Uses onnxruntime for inference (cv2.dnn can't load dynamically quantized
ONNX models with DynamicQuantizeLinear nodes).

Scans frames for forbidden objects during exams:
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
YOLO_MODEL = str(MODELS_DIR / "yolov8n_int8.onnx")

# ── COCO class mapping (only the ones we care about) ─────────────────────────
FORBIDDEN_CLASSES: dict[int, str] = {
    62: "tv/monitor",
    63: "laptop",
    67: "cell phone",
    74: "clock/watch",
}

# Detection thresholds — lowered for INT8 quantized model
CONFIDENCE_THRESHOLD = 0.2   # Lower threshold to catch more objects (INT8 outputs lower scores)
NMS_THRESHOLD = 0.45
INPUT_SIZE = 320


@dataclass
class Detection:
    """A single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list  # [x, y, w, h]


class ObjectDetector:
    """
    YOLOv8n INT8 ONNX object detector using onnxruntime.

    Only reports detections for forbidden exam objects
    (phones, laptops, monitors, watches/clocks).
    """

    def __init__(self) -> None:
        self._session = ort.InferenceSession(
            YOLO_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        print("[ObjectDetector] Loaded YOLOv8n INT8 ONNX model (onnxruntime)")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8n inference on a frame.

        Returns:
            List of Detection objects for forbidden items only.
        """
        h_orig, w_orig = frame.shape[:2]

        # ── Preprocess ───────────────────────────────────────────────────
        # Letterbox resize to maintain aspect ratio
        blob = self._preprocess(frame)

        # ── Forward pass ─────────────────────────────────────────────────
        outputs = self._session.run(None, {self._input_name: blob})

        # YOLOv8 output shape: (1, 84, 8400) → transpose to (8400, 84)
        predictions = outputs[0][0].T  # (8400, 84)

        # ── Postprocess ──────────────────────────────────────────────────
        boxes_raw = predictions[:, :4]      # cx, cy, w, h
        class_scores = predictions[:, 4:]   # 80 class scores

        # FIRST filter: only keep detections for our forbidden classes
        # This avoids false positives from other classes leaking through
        forbidden_mask = np.zeros(len(predictions), dtype=bool)
        for cls_id in FORBIDDEN_CLASSES:
            cls_scores = class_scores[:, cls_id]
            forbidden_mask |= (cls_scores > CONFIDENCE_THRESHOLD)

        if not np.any(forbidden_mask):
            return []

        # Apply forbidden-class pre-filter
        boxes_filtered = boxes_raw[forbidden_mask]
        scores_filtered = class_scores[forbidden_mask]

        # Get best class per remaining detection
        class_ids = np.argmax(scores_filtered, axis=1)
        confidences = scores_filtered[np.arange(len(class_ids)), class_ids]

        # Only keep if best class IS a forbidden class AND above threshold
        final_mask = np.array([
            int(cid) in FORBIDDEN_CLASSES and conf > CONFIDENCE_THRESHOLD
            for cid, conf in zip(class_ids, confidences)
        ], dtype=bool)

        if not np.any(final_mask):
            return []

        boxes_final = boxes_filtered[final_mask]
        class_ids = class_ids[final_mask]
        confidences = confidences[final_mask]

        # Convert cx,cy,w,h → x,y,w,h (top-left corner)
        boxes = np.zeros_like(boxes_final)
        boxes[:, 0] = boxes_final[:, 0] - boxes_final[:, 2] / 2
        boxes[:, 1] = boxes_final[:, 1] - boxes_final[:, 3] / 2
        boxes[:, 2] = boxes_final[:, 2]
        boxes[:, 3] = boxes_final[:, 3]

        # Scale back to original frame size
        x_scale = w_orig / INPUT_SIZE
        y_scale = h_orig / INPUT_SIZE
        boxes[:, 0] *= x_scale
        boxes[:, 1] *= y_scale
        boxes[:, 2] *= x_scale
        boxes[:, 3] *= y_scale

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

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLOv8: resize, normalize, BGR→RGB, NCHW."""
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        blob = resized.astype(np.float32) / 255.0
        blob = blob[:, :, ::-1]  # BGR → RGB
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)  # (1, 3, 640, 640)
        return np.ascontiguousarray(blob)
