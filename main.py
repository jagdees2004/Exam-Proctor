"""
main.py — FastAPI server for the AI Exam Proctoring System.

Endpoints:
  POST /exam/start           → Register a user's face
  GET  /health               → Health check
  WS   /exam/ws/{user_id}    → Real-time monitoring via WebSocket
"""

import asyncio
import cv2
import numpy as np
import uvicorn
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/exam/start")
async def exam_start(
    user_id: str = Form(..., description="Unique user identifier"),
    file: UploadFile = File(..., description="Webcam frame (JPEG/PNG)"),
):
    """Register the user's face for the exam session."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file.")
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    result = engine.register_face(img, user_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Exam Proctor"}


# ── WebSocket: Real-time Monitoring ──────────────────────────────────────────

@app.websocket("/exam/ws/{user_id}")
async def exam_ws(websocket: WebSocket, user_id: str):
    """
    Continuous real-time monitoring via WebSocket.
    Accepts JSON messages with base64 encoded video frames ('frame') or audio chunks ('audio').

    Processing intervals (with ~2s frame pacing from frontend):
      - Face verification: every 3rd frame (~6s)
      - Object detection:  every 5th frame (~10s)
      - Audio VAD:         every message received (~5s)
    """
    await websocket.accept()
    print(f"\n[WS] Client connected: {user_id}")

    frame_count = 0
    last_face = {"identity_match": False, "face_count": 0, "similarity_score": 0.0, "status": "ok"}
    last_forbidden = []

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
                        await websocket.send_json({
                            "type": "video_result",
                            "identity_match": False, "face_count": 0,
                            "similarity_score": 0.0, "status": "no_face",
                            "forbidden_objects": [], "flagged": True,
                        })
                        continue

                    frame_count += 1

                    # Face verification every 3rd frame (~6s at 2s pacing)
                    if frame_count % 3 == 1:
                        face_result = await asyncio.to_thread(engine.verify_face, img, user_id)
                        last_face = {
                            "identity_match": face_result.identity_match,
                            "face_count": face_result.face_count,
                            "similarity_score": face_result.similarity_score,
                            "status": face_result.status,
                        }
                        print(f"[WS] Frame #{frame_count} face: {last_face['status']}")

                    # Object detection every 5th frame (~10s at 2s pacing)
                    if frame_count % 5 == 1:
                        detections = await asyncio.to_thread(detector.detect, img)
                        last_forbidden = [
                            {"class_name": d.class_name, "confidence": d.confidence}
                            for d in detections
                        ]

                    face_flagged = last_face["status"] in (
                        "no_face", "multiple_faces", "identity_mismatch", "camera_blocked"
                    )
                    obj_flagged = len(last_forbidden) > 0

                    await websocket.send_json({
                        "type": "video_result",
                        **last_face,
                        "forbidden_objects": last_forbidden,
                        "flagged": face_flagged or obj_flagged,
                    })

                except Exception as frame_err:
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "video_result",
                        "identity_match": False, "face_count": 0,
                        "similarity_score": 0.0, "status": "error",
                        "forbidden_objects": [], "flagged": False,
                    })

            elif msg_type == "audio":
                try:
                    b64_data = message.get("data", "")
                    if "," in b64_data:
                        b64_data = b64_data.split(",")[1]

                    audio_bytes = base64.b64decode(b64_data)
                    if len(audio_bytes) > 0:
                        result = await asyncio.to_thread(audio_detector.detect_speech, audio_bytes)
                        await websocket.send_json({
                            "type": "audio_result",
                            "is_talking": result["is_talking"],
                            "speech_prob": result["speech_prob"],
                            "flagged": result["flagged"],
                        })
                except Exception:
                    import traceback
                    traceback.print_exc()

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {user_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WS] Fatal error: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
