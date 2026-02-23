# 🛡️ AI Exam Proctoring System

A **production-ready, ultra-lightweight** AI-powered exam proctoring system that monitors students in real-time using face recognition and object detection — all running on CPU with **< 14 MB total model size**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [AI Models](#-ai-models)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Model Setup](#model-setup)
  - [Running the Application](#running-the-application)
- [API Documentation](#-api-documentation)
- [Frontend Guide](#-frontend-guide)
- [Backend Deep Dive](#-backend-deep-dive)
- [How It Works](#-how-it-works)
- [Configuration & Thresholds](#-configuration--thresholds)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔐 Face Registration** | Captures user face via webcam, extracts a 128-D embedding, stores it in memory as the identity reference |
| **👤 Identity Verification** | Continuously matches the live face against the registered reference using cosine similarity |
| **👥 Multiple Face Detection** | Detects if more than one person is in the frame (someone helping the student) |
| **🚫 No Face Detection** | Flags when the student looks away or leaves the camera view |
| **📵 Forbidden Object Detection** | Scans every frame for cell phones, laptops, monitors/tablets, and watches/clocks |
| **📷 Camera Block Detection** | Detects when the webcam is covered or turned off (brightness analysis) |
| **🟢🟡🔴 Real-time Status UI** | Dynamic border colors — Green (OK), Yellow (Warning), Red (Flagged/Cheating) |
| **📊 Live Dashboard** | Shows identity match status, face count, similarity score, and detected objects |
| **🚩 Flag History** | Logs every violation with timestamps for audit trail |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │ Webcam   │→ │ Canvas    │→ │ JPEG Blob → FormData │  │
│  │ Stream   │  │ Capture   │  │ POST every 3 seconds │  │
│  └──────────┘  └───────────┘  └──────────┬───────────┘  │
│                                          │               │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Status UI: Green / Yellow / Red borders          │   │
│  │ Live Stats: Identity, Faces, Similarity, Objects │   │
│  │ Flag History: Timestamped violation log           │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP POST (multipart/form-data)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                        │
│                                                          │
│  POST /exam/start  ─→  Register face embedding           │
│  POST /exam/verify ─→  Verify identity + detect objects   │
│                                                          │
│  ┌────────────────────┐    ┌─────────────────────────┐  │
│  │   ProctorEngine    │    │   ObjectDetector         │  │
│  │                    │    │                           │  │
│  │ 1. Brightness check│    │ 1. Resize to 640×640     │  │
│  │ 2. YuNet face det. │    │ 2. YOLOv8n INT8 ONNX    │  │
│  │ 3. Face alignment  │    │ 3. NMS postprocessing    │  │
│  │ 4. SFace embedding │    │ 4. Filter forbidden cls  │  │
│  │ 5. Cosine similarity│   │                           │  │
│  └────────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **FastAPI** | Async REST API framework |
| **Uvicorn** | ASGI server with auto-reload |
| **OpenCV (headless)** | Face detection via `cv2.FaceDetectorYN`, image processing |
| **ONNXRuntime** | Inference engine for quantized models (SFace INT8, YOLOv8n INT8) |
| **NumPy** | Numerical operations, embedding math |
| **python-multipart** | Parsing `multipart/form-data` uploads in FastAPI |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **Vite** | Lightning-fast dev server & bundler |
| **Native Browser APIs** | `navigator.mediaDevices.getUserMedia` for webcam — **no external camera packages** |
| **Canvas API** | Frame capture from video stream → JPEG blob conversion |
| **Google Fonts (Inter)** | Premium typography |

### Setup-Only (not needed at runtime)
| Technology | Purpose |
|------------|---------|
| **Ultralytics** | Download YOLOv8n.pt and export to ONNX |
| **onnx** | ONNX model manipulation |
| **onnxruntime.quantization** | Dynamic INT8 quantization for SFace and YOLOv8n |

---

## 🤖 AI Models

All models are stored in the `models/` directory (auto-created by `setup_models.py`).

| Model | File | Task | Size | Source |
|-------|------|------|------|--------|
| **YuNet** | `face_detection_yunet_2023mar.onnx` | Face detection (bounding box + 5 landmarks) | **0.22 MB** | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) |
| **SFace INT8** | `face_recognition_sface_int8.onnx` | Face recognition (128-D embedding) | **9.44 MB** | OpenCV Zoo → quantized with `onnxruntime` |
| **YOLOv8n INT8** | `yolov8n_int8.onnx` | Object detection (80 COCO classes, filtered to 4) | **3.34 MB** | Ultralytics → quantized with `onnxruntime` |
| | | **Total** | **~13 MB** | |

### How Models Are Used

**YuNet (Face Detection)**
- Loaded via `cv2.FaceDetectorYN` (OpenCV native API)
- Input: any size BGR frame → internally resized
- Output: `Nx15` array — `[x, y, w, h, 5×landmark_pairs, confidence_score]`
- Detects multiple faces simultaneously

**SFace INT8 (Face Recognition)**
- Loaded via `onnxruntime.InferenceSession` (can't use `cv2.FaceRecognizerSF` with dynamically quantized ONNX)
- Input: aligned `112×112` face crop, normalized to `[-1, 1]`
- Output: `128-D` embedding vector (L2-normalized)
- Identity matching via **cosine similarity**

**YOLOv8n INT8 (Object Detection)**
- Loaded via `onnxruntime.InferenceSession`
- Input: `640×640` RGB frame, normalized to `[0, 1]`
- Output: `8400` anchor predictions → filtered by confidence + NMS
- Only reports **forbidden classes**: `tv/monitor (62)`, `laptop (63)`, `cell phone (67)`, `clock/watch (74)`

---

## 📁 Project Structure

```
d:\face\
│
├── requirements.txt          # Python dependencies (runtime + setup)
├── setup_models.py           # One-time script: download + quantize models
├── engine.py                 # ProctorEngine class (face detection + recognition)
├── object_detector.py        # ObjectDetector class (YOLO forbidden object detection)
├── main.py                   # FastAPI server (REST endpoints)
│
├── models/                   # Auto-created by setup_models.py
│   ├── face_detection_yunet_2023mar.onnx       (0.22 MB)
│   ├── face_recognition_sface_int8.onnx        (9.44 MB)
│   └── yolov8n_int8.onnx                       (3.34 MB)
│
└── frontend/                 # React + Vite
    ├── package.json          # NPM config & dependencies
    ├── vite.config.js        # Vite config with API proxy
    ├── index.html            # HTML entry point
    └── src/
        ├── main.jsx          # React entry point
        ├── App.jsx           # Main component (webcam, API calls, UI)
        └── App.css           # Dark glassmorphism theme
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Node.js 18+** — [Download](https://nodejs.org/)
- **Git** — [Download](https://git-scm.com/)
- **Webcam** — Built-in or USB camera

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-exam-proctor.git
cd ai-exam-proctor
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi` — REST API framework
- `uvicorn[standard]` — ASGI server
- `opencv-python-headless` — Computer vision (no GUI)
- `numpy` — Numerical computing
- `python-multipart` — Form data parsing
- `onnxruntime` — Model inference engine
- `ultralytics` — YOLOv8 model export (setup only)
- `onnx` — ONNX model format (setup only)

#### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Model Setup

Run the one-time setup script to download and quantize all AI models:

```bash
python setup_models.py
```

**What this does:**
1. **Downloads YuNet** face detection model (~0.22 MB) from OpenCV Zoo
2. **Downloads SFace** face recognition model (~37 MB) from OpenCV Zoo → **quantizes to INT8** (~9.44 MB)
3. **Downloads YOLOv8n** object detection model (~6 MB .pt) via Ultralytics → **exports to ONNX** → **quantizes to INT8** (~3.34 MB)

After setup, you'll see:
```
============================================================
  Setup Complete! Model sizes:
============================================================
  face_detection_yunet_2023mar.onnx                    0.22 MB
  face_recognition_sface_int8.onnx                     9.44 MB
  yolov8n_int8.onnx                                    3.34 MB
  TOTAL                                               13.00 MB
============================================================
```

> ⚠️ **Note:** The `ultralytics` and `onnx` packages are only needed for this setup step. You can uninstall them after if you want to minimize your environment.

### Running the Application

You need **two terminals** — one for the backend, one for the frontend.

#### Terminal 1: Start Backend

```bash
python main.py
```

This starts the FastAPI server on `http://localhost:8000` with auto-reload enabled.

You should see:
```
[ProctorEngine] Initialized with YuNet + SFace (onnxruntime)
[ObjectDetector] Loaded YOLOv8n INT8 ONNX model (onnxruntime)
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

#### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

This starts the Vite dev server on `http://localhost:3000`.

#### Open in Browser

Navigate to **http://localhost:3000** in Chrome (recommended).

1. **Allow camera access** when prompted
2. Enter any **Student ID** (e.g., `student1`)
3. Click **🚀 Start Exam** — your face gets registered
4. Monitoring starts automatically (every 3 seconds)
5. Click **⏹ End Exam** to stop

---

## 📡 API Documentation

The backend exposes a REST API. You can view the interactive Swagger docs at: **http://localhost:8000/docs**

### `POST /exam/start` — Register Face

Registers the student's face for the exam session.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | ✅ | Unique student identifier |
| `file` | file (JPEG/PNG) | ✅ | Webcam frame image |

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Face registered for user 'student1'.",
  "face_count": 1
}
```

**Error Responses:**
| Status | Reason |
|--------|--------|
| `400` | No face detected in the frame |
| `400` | Multiple faces detected |
| `400` | Camera too dark / blocked |
| `422` | Missing required fields |

---

### `POST /exam/verify` — Verify Identity + Detect Objects

Verifies the student's identity and scans for forbidden objects. Called by the frontend every 3 seconds.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | ✅ | Same user ID used in `/exam/start` |
| `file` | file (JPEG/PNG) | ✅ | Current webcam frame |

**Response (200 OK):**
```json
{
  "identity_match": true,
  "face_count": 1,
  "similarity_score": 0.7823,
  "status": "ok",
  "forbidden_objects": [],
  "flagged": false
}
```

**Possible `status` values:**

| Status | Meaning | Flagged? |
|--------|---------|----------|
| `ok` | Identity verified, single face, no forbidden objects | ❌ |
| `no_face` | No face detected in frame | ✅ |
| `multiple_faces` | More than one face detected | ✅ |
| `identity_mismatch` | Face doesn't match registered user | ✅ |
| `camera_blocked` | Camera is covered or turned off (frame too dark) | ✅ |
| `not_registered` | User ID not found in memory | ✅ |

**When forbidden objects are detected:**
```json
{
  "identity_match": true,
  "face_count": 1,
  "similarity_score": 0.8012,
  "status": "ok",
  "forbidden_objects": [
    {"class_name": "cell phone", "confidence": 0.456}
  ],
  "flagged": true
}
```

---

### `GET /health` — Health Check

```json
{"status": "healthy", "service": "AI Exam Proctor"}
```

---

### Connecting Frontend to Backend

The frontend uses **Vite's proxy** to forward API calls to the backend. This is configured in `frontend/vite.config.js`:

```javascript
server: {
  port: 3000,
  proxy: {
    '/exam': 'http://localhost:8000',    // /exam/start, /exam/verify
    '/health': 'http://localhost:8000',  // /health
  },
}
```

**In development:** Frontend runs on `http://localhost:3000`, API calls like `fetch('/exam/start')` are proxied to `http://localhost:8000/exam/start`. No CORS issues.

**In production:** You would either:
- Serve the frontend build (`npm run build`) from FastAPI using `StaticFiles`
- Deploy frontend and backend separately and update `API_BASE` in `App.jsx`

---

## 🎨 Frontend Guide

### Component: `App.jsx`

The entire frontend is a single React component with the following structure:

**State Management:**
| State | Type | Purpose |
|-------|------|---------|
| `userId` | string | Student ID input value |
| `phase` | string | Current phase: `setup` → `registering` → `monitoring` |
| `status` | string | Current proctoring status: `idle`, `ok`, `no_face`, `flagged`, etc. |
| `message` | string | Status message displayed to the user |
| `stats` | object | Latest verification response from API |
| `flagLog` | array | History of all flagged violations |
| `cameraReady` | boolean | Whether webcam is active |

**Key Functions:**

| Function | What It Does |
|----------|-------------|
| `startCamera()` | Requests webcam access via `getUserMedia`, sets video stream |
| `captureFrame()` | Draws current video frame to hidden canvas, converts to JPEG blob |
| `handleStart()` | Captures frame → POSTs to `/exam/start` → registers face → starts monitoring |
| `startMonitoring()` | Sets up 3-second `setInterval` loop → captures frame → POSTs to `/exam/verify` |
| `handleStop()` | Clears interval → resets state → returns to setup phase |
| `addFlag(reason)` | Adds timestamped entry to the flag history log (max 20 entries) |

**Webcam Flow:**
```
getUserMedia → video element → canvas.drawImage → canvas.toBlob → JPEG Blob
→ FormData.append('file', blob) → fetch('/exam/verify') → update UI
```

### Styling: `App.css`

- **Dark glassmorphism theme** with backdrop blur and translucent cards
- **Inter font** from Google Fonts for premium typography
- **JetBrains Mono** for numerical/code values
- **Status border animations:**
  - 🟢 Green glow + steady border = identity verified
  - 🟡 Yellow pulse = warning (no face / looking away)
  - 🔴 Red pulse + shake animation = flagged (cheating detected)
- **Responsive layout** — adapts to tablet and mobile screens
- **Micro-animations** — slide-in flag entries, spinner on registration, smooth transitions

---

## 🔧 Backend Deep Dive

### `engine.py` — ProctorEngine Class

Handles all face-related processing.

**Initialization:**
- Creates `cv2.FaceDetectorYN` instance with YuNet model
- Creates `onnxruntime.InferenceSession` with SFace INT8 model
- Initializes empty `{user_id: embedding}` dictionary

**Methods:**

| Method | Description |
|--------|-------------|
| `register_face(frame, user_id)` | Detects single face → aligns → extracts 128-D embedding → stores in memory |
| `verify_face(frame, user_id)` | Brightness check → face detection → alignment → embedding → cosine similarity vs reference |
| `_is_frame_too_dark(frame)` | Converts to grayscale → checks mean brightness < 25 (camera covered/off) |
| `_detect_faces(frame)` | Runs YuNet → returns `Nx15` face array with landmarks |
| `_align_face(frame, face)` | Extracts 5 landmarks → computes affine transform → warps to 112×112 aligned crop |
| `_extract_embedding(aligned_face)` | Preprocesses face → runs SFace ONNX → L2-normalizes → returns 128-D vector |
| `_cosine_similarity(emb1, emb2)` | Dot product of two L2-normalized embeddings |

**Face Alignment Pipeline:**
```
YuNet landmarks (5 points) → estimateAffinePartial2D → warpAffine → 112×112 aligned face
```

The alignment template maps the 5 detected landmarks (eyes, nose, mouth corners) to a standard face template, ensuring consistent face positioning regardless of head pose.

---

### `object_detector.py` — ObjectDetector Class

Handles forbidden object detection using YOLOv8n.

**Initialization:**
- Creates `onnxruntime.InferenceSession` with YOLOv8n INT8 model

**Methods:**

| Method | Description |
|--------|-------------|
| `detect(frame)` | Full detection pipeline → returns list of `Detection` objects for forbidden items only |
| `_preprocess(frame)` | Resize to 640×640, normalize [0,1], BGR→RGB, HWC→NCHW |

**Detection Pipeline:**
```
Frame → Resize 640×640 → Normalize → RGB → NCHW → ONNX inference
→ Transpose (8400, 84) → Filter forbidden classes → Confidence threshold
→ cx,cy,w,h → x,y,w,h → Scale to original → NMS → Final detections
```

**Forbidden COCO Classes:**

| Class ID | Label | Examples |
|----------|-------|----------|
| 62 | `tv/monitor` | Monitors, tablets, TVs |
| 63 | `laptop` | Open laptops |
| 67 | `cell phone` | Smartphones |
| 74 | `clock/watch` | Wall clocks, wristwatches |

---

### `main.py` — FastAPI Server

**Endpoints:** See [API Documentation](#-api-documentation) above.

**Key Details:**
- CORS enabled for all origins (development)
- Models loaded once at startup (global instances)
- Auto-reload via `uvicorn --reload` for development
- Debug logging prints detection results to terminal for every frame

---

### `setup_models.py` — Model Setup Script

Run once to prepare all AI models.

**Functions:**

| Function | Description |
|----------|-------------|
| `_download(url, dest)` | Downloads a file from URL, skips if already exists |
| `_quantize_onnx(src, dst, label)` | Applies `onnxruntime.quantization.quantize_dynamic` (UINT8 weights) |
| `_export_yolov8n(dst)` | Downloads YOLOv8n.pt via Ultralytics, exports to ONNX |
| `main()` | Orchestrates the full download → quantize → cleanup pipeline |

---

## ⚙️ How It Works

### Registration Flow (`/exam/start`)

```
1. Student enters ID + clicks "Start Exam"
2. Frontend captures webcam frame as JPEG blob
3. POST to /exam/start with user_id + file
4. Backend:
   a. Check brightness (reject if camera blocked)
   b. Run YuNet face detection
   c. Reject if 0 or >1 faces
   d. Align the single face to 112×112
   e. Extract 128-D embedding via SFace
   f. Store embedding in memory: {user_id → embedding}
5. Return success → Frontend starts monitoring loop
```

### Monitoring Flow (`/exam/verify`)

```
Every 3 seconds:
1. Frontend captures webcam frame → POST to /exam/verify
2. Backend runs TWO parallel checks:

   CHECK 1: Face Verification (ProctorEngine)
   ├─ Is frame too dark? → "camera_blocked" 🔴
   ├─ No faces found? → "no_face" 🟡
   ├─ Multiple faces? → "multiple_faces" 🔴
   └─ Single face:
      ├─ Align face → Extract embedding
      ├─ Compare with reference (cosine similarity)
      ├─ Score ≥ 0.3 → "ok" 🟢
      └─ Score < 0.3 → "identity_mismatch" 🔴

   CHECK 2: Object Detection (ObjectDetector)
   ├─ Run YOLOv8n on frame
   ├─ Filter to forbidden classes only
   └─ Any phone/laptop/monitor/watch found → flagged 🔴

3. Backend returns combined result
4. Frontend updates UI (border color, status, flag log)
```

---

## 🎛️ Configuration & Thresholds

All thresholds are defined as constants at the top of each file. You can tune them:

### `engine.py`

| Constant | Default | Purpose |
|----------|---------|---------|
| `FACE_SCORE_THRESHOLD` | `0.6` | Minimum YuNet confidence to count as a valid face |
| `FACE_NMS_THRESHOLD` | `0.3` | Non-Maximum Suppression threshold for face detection |
| `COSINE_SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to confirm identity match |
| `BRIGHTNESS_THRESHOLD` | `25` | Mean pixel value below this → camera blocked |

### `object_detector.py`

| Constant | Default | Purpose |
|----------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.2` | Minimum YOLO confidence (low for INT8 model accuracy) |
| `NMS_THRESHOLD` | `0.45` | Non-Maximum Suppression threshold for object detection |
| `INPUT_SIZE` | `640` | YOLO input resolution |

### `App.jsx`

| Constant | Default | Purpose |
|----------|---------|---------|
| Verification interval | `3000` ms | How often frames are sent to `/exam/verify` |
| JPEG quality | `0.85` | Canvas-to-blob compression quality |

---

## 🐛 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `Camera access denied` | Allow camera permissions in browser settings → reload page |
| `422 Unprocessable Entity` | Ensure the frontend sends the file field as `file` (not `frame`) |
| `DynamicQuantizeLinear error` | You're trying to load a quantized model with `cv2.dnn` — use `onnxruntime` instead (already handled) |
| Server port already in use | Kill process: `netstat -ano \| findstr 8000` then `taskkill /PID <pid> /F` |
| Phone not detected | Lower `CONFIDENCE_THRESHOLD` in `object_detector.py` (try `0.15`) |
| False face matches | Increase `COSINE_SIMILARITY_THRESHOLD` in `engine.py` (try `0.4`) |
| Camera blocked not triggering | Adjust `BRIGHTNESS_THRESHOLD` in `engine.py` (try `30` for well-lit rooms) |
| `npm start` doesn't work | Use `npm run dev` instead (Vite uses `dev` script, not `start`) |

### Debug Logging

The backend prints detailed logs to the terminal for every verification call:

```
[VERIFY] user_id=student1, frame=(480, 640, 3)
  [Brightness] mean=142.3 (threshold=25)
  [FaceDetect] Found 1 face(s), scores=['0.987']
  [VERIFY] Similarity score: 0.7823 (threshold: 0.3)
  [VERIFY] ✓ Identity match!
  [RESPONSE] status=ok flagged=False objects=[]
```

When a phone is detected:
```
  [ObjectDetect] 🚨 Found: cell phone(0.34)
  [RESPONSE] status=ok flagged=True objects=['cell phone']
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — YuNet and SFace models
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 model
- [ONNXRuntime](https://onnxruntime.ai/) — Fast model inference and quantization
