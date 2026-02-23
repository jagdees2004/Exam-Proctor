"""
vad_engine.py — Voice Activity Detection using Silero VAD ONNX.

Detects human speech in audio chunks to flag talking during exams.
Uses onnxruntime for inference — no PyTorch or heavy ML deps needed.

Model: Silero VAD v5 (~2 MB ONNX)
Input: 16kHz mono audio (WAV or raw PCM)
Output: Speech probability [0.0 – 1.0]
"""

import io
import wave
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
VAD_MODEL = str(MODELS_DIR / "silero_vad.onnx")

# VAD thresholds
SPEECH_THRESHOLD = 0.5  # Probability above this = speech detected
TARGET_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz
CHUNK_SIZE = 512  # Number of audio samples per inference chunk (32ms at 16kHz)
CONTEXT_SIZE = 64  # Context samples to prepend from previous chunk (required by model)


class AudioDetector:
    """
    Silero VAD v5 ONNX voice activity detector.

    Each chunk fed to the model must be prepended with a 64-sample context
    from the previous chunk's tail. Without this, the model can't detect
    speech patterns and always returns near-zero probability.

    Model interface (v5):
      Inputs:  input (1, 576), state (2, 1, 128), sr (int64)
      Outputs: output (1, 1), stateN (2, 1, 128)
    """

    def __init__(self) -> None:
        self._session = ort.InferenceSession(
            VAD_MODEL,
            providers=["CPUExecutionProvider"],
        )

        # Silero VAD v5: single combined state tensor
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        # Context: last CONTEXT_SIZE samples from previous chunk
        self._context = np.zeros(CONTEXT_SIZE, dtype=np.float32)

        print("[AudioDetector] Loaded Silero VAD ONNX model")

    def reset_states(self) -> None:
        """Reset the RNN hidden states and context."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    def detect_speech(self, audio_bytes: bytes) -> dict:
        """
        Analyze audio data for human speech.

        Args:
            audio_bytes: Raw WAV file bytes from the browser.

        Returns:
            dict with is_talking, speech_prob, flagged
        """
        try:
            audio = self._decode_wav(audio_bytes)
        except Exception as e:
            print(f"  [VAD] ✗ Failed to decode audio: {e}")
            return {"is_talking": False, "speech_prob": 0.0, "flagged": False}

        if len(audio) == 0:
            return {"is_talking": False, "speech_prob": 0.0, "flagged": False}

        # Process audio in CHUNK_SIZE-sample chunks with context prepended
        max_prob = 0.0
        num_chunks = len(audio) // CHUNK_SIZE

        if num_chunks == 0:
            padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
            padded[:len(audio)] = audio
            max_prob = self._run_chunk(padded)
        else:
            for i in range(num_chunks):
                chunk = audio[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                prob = self._run_chunk(chunk)
                max_prob = max(max_prob, prob)

        is_talking = max_prob > SPEECH_THRESHOLD
        result = {
            "is_talking": is_talking,
            "speech_prob": round(float(max_prob), 3),
            "flagged": is_talking,
        }

        if is_talking:
            print(f"  [VAD] 🎤 Speech detected! prob={max_prob:.3f}")
        else:
            print(f"  [VAD] Silent. prob={max_prob:.3f}")

        return result

    def _run_chunk(self, chunk: np.ndarray) -> float:
        """
        Run Silero VAD on a single chunk.

        Prepends CONTEXT_SIZE samples from the previous chunk (critical!)
        then updates context with this chunk's tail for next call.
        """
        # Prepend context to chunk: [context(64) + chunk(512)] = 576 samples
        with_context = np.concatenate([self._context, chunk])
        input_data = with_context.reshape(1, -1).astype(np.float32)

        sr = np.array(TARGET_SAMPLE_RATE, dtype=np.int64)

        ort_inputs = {
            "input": input_data,
            "state": self._state,
            "sr": sr,
        }

        output, new_state = self._session.run(None, ort_inputs)
        self._state = new_state

        # Save last CONTEXT_SIZE samples as context for next chunk
        self._context = chunk[-CONTEXT_SIZE:].copy()

        return float(output[0][0])

    @staticmethod
    def _decode_wav(audio_bytes: bytes) -> np.ndarray:
        """
        Decode WAV bytes to float32 numpy array at 16kHz mono.

        Handles WAV files from browser (typically 44.1kHz or 48kHz).
        """
        buf = io.BytesIO(audio_bytes)

        with wave.open(buf, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        # Convert raw bytes to float32
        if sampwidth == 2:  # 16-bit PCM (most common)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Stereo to mono
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Resample to 16kHz if needed
        if framerate != TARGET_SAMPLE_RATE:
            duration = len(samples) / framerate
            target_len = int(duration * TARGET_SAMPLE_RATE)
            if target_len > 0:
                indices = np.linspace(0, len(samples) - 1, target_len)
                samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

        return samples
