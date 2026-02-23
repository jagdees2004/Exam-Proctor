"""
setup_models.py — One-time model download & quantization script.

Downloads:
  1. YuNet  (face detection)   ~0.22 MB  ONNX from OpenCV Zoo
  2. SFace  (face recognition)  ~37 MB  ONNX → quantized to INT8 via onnxruntime
  3. YOLOv8n (object detection) ~12 MB  ONNX → quantized to INT8 via onnxruntime

All models are saved into  ./models/
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# ── URLs ─────────────────────────────────────────────────────────────────────
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return
    print(f"  ↓ Downloading {dest.name} …")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved {dest.name}  ({size_mb:.2f} MB)")


def _quantize_onnx(src: Path, dst: Path, label: str) -> None:
    """Dynamically quantize an ONNX model to UINT8 using onnxruntime."""
    if dst.exists():
        print(f"  ✓ Already exists: {dst.name}")
        return
    print(f"  ⚙ Quantizing {label} to INT8 …")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QUInt8,
    )
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  ✓ Quantized → {dst.name}  ({size_mb:.2f} MB)")


def _export_yolov8n(dst_fp32: Path) -> None:
    """Download yolov8n.pt via ultralytics and export to ONNX."""
    if dst_fp32.exists():
        print(f"  ✓ Already exists: {dst_fp32.name}")
        return
    print("  ↓ Downloading & exporting YOLOv8n to ONNX …")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # auto-downloads from Ultralytics hub
    export_path = model.export(format="onnx", imgsz=640, simplify=True)

    # Move the exported ONNX to our models directory
    export_file = Path(export_path)
    shutil.move(str(export_file), str(dst_fp32))

    size_mb = dst_fp32.stat().st_size / (1024 * 1024)
    print(f"  ✓ Exported → {dst_fp32.name}  ({size_mb:.2f} MB)")

    # Clean up the downloaded .pt file
    pt_file = Path("yolov8n.pt")
    if pt_file.exists():
        pt_file.unlink()
        print("  🗑 Cleaned up yolov8n.pt")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  AI Exam Proctor — Model Setup")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)

    # 1. YuNet — Face Detection (tiny, no quantization needed)
    print("\n[1/3] YuNet (Face Detection)")
    yunet_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    _download(YUNET_URL, yunet_path)

    # 2. SFace — Face Recognition (download full → quantize → delete full)
    print("\n[2/3] SFace (Face Recognition)")
    sface_full = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    sface_int8 = MODELS_DIR / "face_recognition_sface_int8.onnx"
    _download(SFACE_URL, sface_full)
    _quantize_onnx(sface_full, sface_int8, "SFace")
    if sface_full.exists() and sface_int8.exists():
        sface_full.unlink()
        print(f"  🗑 Removed original SFace to save space")

    # 3. YOLOv8n — Object Detection (export to ONNX → quantize → delete FP32)
    print("\n[3/3] YOLOv8n (Object Detection)")
    yolo_fp32 = MODELS_DIR / "yolov8n_fp32.onnx"
    yolo_int8 = MODELS_DIR / "yolov8n_int8.onnx"
    _export_yolov8n(yolo_fp32)
    _quantize_onnx(yolo_fp32, yolo_int8, "YOLOv8n")
    if yolo_fp32.exists() and yolo_int8.exists():
        yolo_fp32.unlink()
        print(f"  🗑 Removed FP32 YOLO to save space")

    # Summary
    print("\n" + "=" * 60)
    print("  Setup Complete! Model sizes:")
    print("=" * 60)
    total = 0
    for f in sorted(MODELS_DIR.glob("*.onnx")):
        size = f.stat().st_size / (1024 * 1024)
        total += size
        print(f"  {f.name:50s} {size:6.2f} MB")
    print(f"  {'TOTAL':50s} {total:6.2f} MB")
    if total < 10:
        print("  ✅ Under 10 MB budget!")
    else:
        print(f"  ⚠️  {total:.1f} MB — models ready for use")
    print()


if __name__ == "__main__":
    main()
