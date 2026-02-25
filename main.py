"""
main.py — FastAPI server for the AI Exam Proctoring System.

Endpoints:
  POST /exam/start   → Register a user's face (multipart/form-data)
  POST /exam/verify  → Verify identity + detect forbidden objects
  POST /exam/audio   → Detect speech via Voice Activity Detection
"""

import cv2
import numpy as np
import uvicorn
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict

from engine import ProctorEngine
from object_detector import ObjectDetector
from vad_engine import AudioDetector

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Exam Proctor",
    description="Lightweight AI-powered exam proctoring system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize engines (loaded once at startup) ─────────────────────────────

engine = ProctorEngine()
detector = ObjectDetector()
audio_detector = AudioDetector()


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _read_frame(file: UploadFile) -> np.ndarray:
    """Read an uploaded image file into an OpenCV BGR frame."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file.")
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")
    return frame


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/exam/start")
async def exam_start(
    user_id: str = Form(..., description="Unique user identifier"),
    file: UploadFile = File(..., description="Webcam frame (JPEG/PNG)"),
):
    """
    Register the user's face for the exam session.

    Expects multipart/form-data with:
      - user_id: string
      - file: image file (JPEG or PNG)
    """
    img = await _read_frame(file)
    result = engine.register_face(img, user_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/exam/verify")
async def exam_verify(
    user_id: str = Form(..., description="Unique user identifier"),
    file: UploadFile = File(..., description="Webcam frame (JPEG/PNG)"),
):
    """
    Verify the user's identity (face only — fast).

    Called every ~1.5 seconds by the React frontend.
    Object detection runs on a separate endpoint (/exam/objects) in parallel.
    """
    img = await _read_frame(file)

    # Face verification only
    face_result = engine.verify_face(img, user_id)

    flagged = face_result.status in ("no_face", "multiple_faces", "identity_mismatch", "camera_blocked")

    response = {
        "identity_match": face_result.identity_match,
        "face_count": face_result.face_count,
        "similarity_score": face_result.similarity_score,
        "status": face_result.status,
        "flagged": flagged,
    }

    print(f"  [RESPONSE] status={face_result.status} flagged={flagged}")

    return response


@app.post("/exam/objects")
async def exam_objects(
    user_id: str = Form(..., description="Unique user identifier"),
    file: UploadFile = File(..., description="Webcam frame (JPEG/PNG)"),
):
    """
    Scan a frame for forbidden objects (phones, laptops, etc).

    Runs in parallel with /exam/verify for faster detection.
    """
    img = await _read_frame(file)

    print(f"\n[OBJECTS] user_id={user_id}, frame={img.shape}")
    detections = detector.detect(img)
    forbidden = [
        {"class_name": d.class_name, "confidence": d.confidence}
        for d in detections
    ]

    flagged = len(forbidden) > 0
    if flagged:
        print(f"  [OBJECTS] 🚨 Found: {[d['class_name'] for d in forbidden]}")

    return {
        "forbidden_objects": forbidden,
        "flagged": flagged,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Exam Proctor"}


@app.post("/exam/audio")
async def exam_audio(
    user_id: str = Form(..., description="Unique user identifier"),
    audio: UploadFile = File(..., description="Audio chunk (WAV)"),
):
    """
    Analyze audio for human speech using Voice Activity Detection.

    Called by the frontend alongside /exam/verify to detect talking.
    Accepts WAV audio blobs from MediaRecorder.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        return {"is_talking": False, "speech_prob": 0.0, "flagged": False}

    print(f"\n[AUDIO] user_id={user_id}, audio_size={len(audio_bytes)} bytes")
    result = audio_detector.detect_speech(audio_bytes)
    return result

@app.websocket("/exam/ws/{user_id}")
async def exam_ws(websocket: WebSocket, user_id: str):
    """
    Continuous real-time monitoring via WebSocket.
    Accepts JSON messages with base64 encoded video frames ('frame') or audio chunks ('audio').
    """
    await websocket.accept()
    print(f"\n[WS] Client connected: {user_id}")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "frame":
                try:
                    b64_data = message.get("data", "")
                    if "," in b64_data:
                        b64_data = b64_data.split(",")[1]
                    
                    img_bytes = base64.b64decode(b64_data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        print(f"[WS] Failed to decode image, sending fallback response")
                        await websocket.send_json({
                            "type": "video_result",
                            "identity_match": False,
                            "face_count": 0,
                            "similarity_score": 0.0,
                            "status": "no_face",
                            "forbidden_objects": [],
                            "flagged": True,
                        })
                        continue

                    print(f"[WS] Processing frame: {img.shape} for user {user_id}")
                    
                    # Debug: log to file so we don't lose output during reload
                    with open("ws_debug.log", "a") as f:
                        f.write(f"\n[WS] Processing frame: {img.shape} for user {user_id}\n")
                        f.write(f"[WS] User registered: {engine.is_registered(user_id)}\n")

                    # Face verification
                    face_result = engine.verify_face(img, user_id)
                    face_flagged = face_result.status in ("no_face", "multiple_faces", "identity_mismatch", "camera_blocked")

                    # Object detection
                    detections = detector.detect(img)
                    forbidden = [
                        {"class_name": d.class_name, "confidence": d.confidence}
                        for d in detections
                    ]
                    obj_flagged = len(forbidden) > 0

                    response = {
                        "type": "video_result",
                        "identity_match": face_result.identity_match,
                        "face_count": face_result.face_count,
                        "similarity_score": face_result.similarity_score,
                        "status": "camera_blocked" if face_result.status == "camera_blocked" else face_result.status,
                        "forbidden_objects": forbidden,
                        "flagged": face_flagged or obj_flagged,
                    }
                    print(f"[WS] Sending response: status={response['status']}, flagged={response['flagged']}")
                    await websocket.send_json(response)

                except Exception as frame_err:
                    import traceback
                    traceback.print_exc()
                    print(f"[WS] Frame processing error: {frame_err}")
                    # Log to file for debugging
                    with open("ws_debug.log", "a") as f:
                        f.write(f"\n[WS] FRAME ERROR: {frame_err}\n")
                        traceback.print_exc(file=f)
                    # Still send a response so the frontend doesn't deadlock
                    await websocket.send_json({
                        "type": "video_result",
                        "identity_match": False,
                        "face_count": 0,
                        "similarity_score": 0.0,
                        "status": "error",
                        "forbidden_objects": [],
                        "flagged": False,
                    })

            elif msg_type == "audio":
                try:
                    b64_data = message.get("data", "")
                    if "," in b64_data:
                        b64_data = b64_data.split(",")[1]
                        
                    audio_bytes = base64.b64decode(b64_data)
                    if len(audio_bytes) > 0:
                        result = audio_detector.detect_speech(audio_bytes)
                        response = {
                            "type": "audio_result",
                            "is_talking": result["is_talking"],
                            "speech_prob": result["speech_prob"],
                            "flagged": result["flagged"]
                        }
                        await websocket.send_json(response)
                except Exception as audio_err:
                    import traceback
                    traceback.print_exc()
                    print(f"[WS] Audio processing error: {audio_err}")

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {user_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WS] Fatal error: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
