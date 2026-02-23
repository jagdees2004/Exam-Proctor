"""
setup_mobilefacenet.py — Download MobileFaceNet ONNX model from InsightFace.

Downloads the 'buffalo_sc' model pack from InsightFace GitHub releases,
extracts the w600k_mbf.onnx face recognition model (~12 MB), and saves
it to the models/ directory.

Usage:
    python setup_mobilefacenet.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# InsightFace buffalo_sc model pack (contains MobileFaceNet)
BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
TARGET_MODEL = "w600k_mbf.onnx"
FINAL_NAME = "mobilefacenet.onnx"


def download_with_progress(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"  ↓ Downloading from: {url}")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  ↓ {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()  # newline after progress


def main() -> None:
    print("=" * 60)
    print("  MobileFaceNet ONNX — Model Setup")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)
    final_path = MODELS_DIR / FINAL_NAME

    # Check if already downloaded
    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"\n  ✓ Already exists: {FINAL_NAME} ({size_mb:.2f} MB)")
        print("  Delete it manually if you want to re-download.")
        return

    # Download the buffalo_sc.zip
    zip_path = MODELS_DIR / "buffalo_sc.zip"
    print(f"\n[1/3] Downloading InsightFace buffalo_sc model pack...")

    try:
        download_with_progress(BUFFALO_SC_URL, str(zip_path))
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        print(f"\n  Manual download instructions:")
        print(f"  1. Go to: {BUFFALO_SC_URL}")
        print(f"  2. Download buffalo_sc.zip")
        print(f"  3. Extract '{TARGET_MODEL}' from the zip")
        print(f"  4. Rename it to '{FINAL_NAME}' and place in models/ folder")
        sys.exit(1)

    # Extract the recognition model
    print(f"\n[2/3] Extracting {TARGET_MODEL}...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            # List contents to find the model
            names = zf.namelist()
            target_file = None
            for name in names:
                if name.endswith(TARGET_MODEL):
                    target_file = name
                    break

            if target_file is None:
                print(f"  ✗ Could not find {TARGET_MODEL} in the zip!")
                print(f"  Contents: {names}")
                sys.exit(1)

            # Extract just the recognition model
            with zf.open(target_file) as src, open(str(final_path), 'wb') as dst:
                shutil.copyfileobj(src, dst)

        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Extracted → {FINAL_NAME} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        sys.exit(1)

    # Cleanup
    print(f"\n[3/3] Cleaning up...")
    if zip_path.exists():
        zip_path.unlink()
        print(f"  🗑 Removed buffalo_sc.zip")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  ✅ MobileFaceNet ready!")
    print(f"  Model: {final_path}")
    print(f"  Size:  {size_mb:.2f} MB")
    print(f"  Type:  MobileFaceNet (WebFace600K, 512-D embeddings)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
