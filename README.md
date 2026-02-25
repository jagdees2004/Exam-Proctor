# 🛡️ AI Exam Proctoring System

A **production-ready, ultra-lightweight** AI-powered exam proctoring system that monitors students in real-time using face recognition, object detection, and voice activity detection — all running on CPU with **< 10 MB total model size** via WebSocket.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔐 Face Registration** | Captures user face via webcam, extracts a 512-D embedding, stores in memory |
| **👤 Identity Verification** | Continuously matches live face against the registered reference using cosine similarity |
| **👥 Multiple Face Detection** | Detects if more than one person is in the frame |
| **🚫 No Face Detection** | Flags when the student leaves the camera view |
| **📵 Forbidden Object Detection** | Scans frames for cell phones, laptops, monitors/tablets, and watches/clocks |
| **📷 Camera Block Detection** | Detects when the webcam is covered or turned off (brightness + variance analysis) |
| **🎤 Voice Activity Detection** | Detects speech during the exam using Silero VAD |
| **⚡ Real-time WebSocket** | Self-pacing WebSocket loop for low-latency, real-time AI analysis |
| **🟢🟡🔴 Status UI** | Dynamic border colors — Green (OK), Yellow (Warning), Red (Flagged) |
| **📊 Live Dashboard** | Shows identity, face count, similarity score, objects, and audio status |
| **🚩 Flag History** | Logs every violation with timestamps |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      REACT FRONTEND                          │
│  ┌──────────┐  ┌───────────┐  ┌───────────────────────────┐ │
│  │ Webcam   │→ │ Canvas    │→ │ Base64 JPEG → WebSocket   │ │
│  │ Stream   │  │ Capture   │  │ Self-pacing loop    │ │
│  └──────────┘  └───────────┘  └────────────┬──────────────┘ │
│  ┌──────────┐                              │                 │
│  │ Audio    │→ WAV encode → base64 ────────┤                 │
│  │ Capture  │  (every 3 seconds)           │                 │
│  └──────────┘                              │                 │
│  ┌─────────────────────────────────────────┘                 │
│  │ Live Analysis: Identity, Faces, Similarity, Objects, Audio│
│  │ Flag History: Timestamped violation log                   │
│  └───────────────────────────────────────────────────────────┘
└───────────────────────────┬──────────────────────────────────┘
                            │ WebSocket (ws:// or wss://)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                          │
│                                                              │
│  POST /exam/start     → Register face embedding (HTTP)       │
│  WS   /exam/ws/{id}   → Real-time frame + audio analysis     │
│                                                              │
│  ┌──────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  ProctorEngine   │  │ ObjectDetector │  │ AudioDetector│ │
│  │                  │  │                │  │              │ │
│  │ 1. Brightness    │  │ YOLOv8n INT8   │  │ Silero VAD   │ │
│  │ 2. YuNet face    │  │ 640×640 input  │  │ 16kHz mono   │ │
│  │ 3. Align 112×112 │  │ NMS + filter   │  │ Speech prob  │ │
│  │ 4. MobileFaceNet │  │ Forbidden only │  │              │ │
│  │ 5. Cosine sim    │  │                │  │              │ │
│  └──────────────────┘  └────────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Models

All models are stored in the `models/` directory (~9.1 MB total).

| Model | File | Task | Size |
|-------|------|------|------|
| **YuNet** | `face_detection_yunet_2023mar.onnx` | Face detection (bbox + 5 landmarks) | **0.22 MB** |
| **MobileFaceNet INT8** | `mobilefacenet_int8.onnx` | Face recognition (512-D embedding) | **3.35 MB** |
| **YOLOv8n INT8** | `yolov8n_int8.onnx` | Object detection (filtered to 4 classes) | **3.34 MB** |
| **Silero VAD** | `silero_vad.onnx` | Voice activity detection | **2.2 MB** |
| | | **Total** | **~9.1 MB** |

---

## 📁 Project Structure

```
d:\face\
├── main.py                 # FastAPI server (HTTP + WebSocket endpoints)
├── engine.py               # ProctorEngine (face detection + recognition)
├── object_detector.py      # ObjectDetector (YOLOv8n forbidden objects)
├── vad_engine.py           # AudioDetector (Silero VAD speech detection)
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── README.md
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx   (0.22 MB)
│   ├── mobilefacenet_int8.onnx             (3.35 MB)
│   ├── yolov8n_int8.onnx                   (3.34 MB)
│   └── silero_vad.onnx                     (2.2 MB)
│
└── frontend/               # React + Vite
    ├── package.json
    ├── vite.config.js      # Dev proxy + WebSocket config
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx         # Main component (webcam, WebSocket, UI)
        └── App.css         # Dark glassmorphism theme
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Node.js 18+** — [Download](https://nodejs.org/)
- **Webcam** — Built-in or USB camera

### Installation

```bash
# Clone the repository
git clone https://github.com/jagdees2004/Exam-Proctor.git
cd Exam-Proctor

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running the Application

You need **two terminals**:

#### Terminal 1: Backend
```bash
python main.py
```
Output:
```
[ProctorEngine] Initialized with YuNet + MobileFaceNet (onnxruntime)
[ObjectDetector] Loaded YOLOv8n INT8 ONNX model (onnxruntime)
[AudioDetector] Loaded Silero VAD ONNX model
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```

#### Open in Browser

Navigate to **http://localhost:3000**:

1. **Allow camera + microphone** when prompted
2. Enter a **Student ID** (e.g., `student1`)
3. Click **🚀 Start Exam** — face is registered
4. Real-time monitoring starts automatically via WebSocket
5. Click **⏹ End Exam** to stop

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/exam/start` | POST | Register face (multipart: `user_id` + `file`) |
| `/exam/ws/{user_id}` | WebSocket | Real-time frame + audio analysis |
| `/exam/verify` | POST | One-shot face verification (legacy) |
| `/exam/objects` | POST | One-shot object detection (legacy) |
| `/exam/audio` | POST | One-shot audio analysis (legacy) |
| `/health` | GET | Health check |

### WebSocket Protocol (`/exam/ws/{user_id}`)

**Send frame:**
```json
{"type": "frame", "data": "data:image/jpeg;base64,..."}
```

**Receive video result:**
```json
{
  "type": "video_result",
  "identity_match": true,
  "face_count": 1,
  "similarity_score": 0.7823,
  "status": "ok",
  "forbidden_objects": [],
  "flagged": false
}
```

**Send audio:**
```json
{"type": "audio", "data": "<base64 WAV>"}
```

**Receive audio result:**
```json
{
  "type": "audio_result",
  "is_talking": false,
  "speech_prob": 0.03,
  "flagged": false
}
```

**Status values:** `ok`, `no_face`, `multiple_faces`, `identity_mismatch`, `camera_blocked`, `not_registered`, `error`

---

## 🚀 Deployment

### Backend (Render)

The project includes `render.yaml` for one-click deployment to [Render](https://render.com):

```yaml
services:
  - type: web
    name: proctor-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Frontend (Vercel / Netlify)

Deploy the `frontend/` directory. Set the environment variable:

```
VITE_API_URL=https://your-render-backend.onrender.com
```

---

## ⚙️ Configuration & Thresholds

### `engine.py`
| Constant | Default | Purpose |
|----------|---------|---------|
| `FACE_SCORE_THRESHOLD` | `0.6` | YuNet face detection confidence |
| `COSINE_SIMILARITY_THRESHOLD` | `0.30` | Minimum similarity for identity match |
| `BRIGHTNESS_THRESHOLD` | `40` | Mean pixel brightness below this → camera blocked |
| `VARIANCE_THRESHOLD` | `15` | Pixel variance below this → camera covered |

### `object_detector.py`
| Constant | Default | Purpose |
|----------|---------|---------|
| `CONFIDENCE_THRESHOLD` | `0.2` | YOLO detection confidence |
| `NMS_THRESHOLD` | `0.45` | Non-Maximum Suppression |
| `INPUT_SIZE` | `640` | Fixed ONNX input size (do not change) |

### `vad_engine.py`
| Constant | Default | Purpose |
|----------|---------|---------|
| `SPEECH_THRESHOLD` | `0.5` | Probability above this → speech detected |
| `TARGET_SAMPLE_RATE` | `16000` | Required by Silero VAD |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera access denied | Allow permissions in browser settings → reload |
| WebSocket not connecting | Ensure `vite.config.js` has `ws: true` in the proxy |
| `Got: 320 Expected: 640` | Never change `INPUT_SIZE` — the INT8 model has fixed shape |
| Server port in use | Kill process: `netstat -ano \| findstr 8000` then `taskkill /PID <pid> /F` |
| Phone not detected | Lower `CONFIDENCE_THRESHOLD` in `object_detector.py` (try `0.15`) |
| False face matches | Increase `COSINE_SIMILARITY_THRESHOLD` in `engine.py` (try `0.4`) |
| Slow on Render free tier | Expected — YOLOv8 at 640×640 is CPU-heavy; upgrade to paid plan for speed |

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — YuNet face detection model
- [MobileFaceNet](https://github.com/nicholaspat/MobileFaceNet_PyTorch) — Face recognition model
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 model
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection
- [ONNXRuntime](https://onnxruntime.ai/) — Fast model inference and quantization
