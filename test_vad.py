"""Test VAD on actual browser audio"""
import numpy as np
import onnxruntime as ort
import wave, io

# Load model
s = ort.InferenceSession('models/silero_vad.onnx', providers=['CPUExecutionProvider'])

# Read actual browser audio  
with wave.open('debug_audio.wav', 'rb') as wf:
    framerate = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

print(f"Original: {len(samples)} samples at {framerate}Hz, max_amp={np.max(np.abs(samples)):.4f}")

# Resample to 16kHz
target_sr = 16000
duration = len(samples) / framerate
target_len = int(duration * target_sr)
indices = np.linspace(0, len(samples) - 1, target_len)
resampled = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
print(f"Resampled: {len(resampled)} samples at {target_sr}Hz, max_amp={np.max(np.abs(resampled)):.4f}")

# Process with VAD
state = np.zeros((2, 1, 128), dtype=np.float32)
chunk_size = 512
probs = []
for i in range(len(resampled) // chunk_size):
    chunk = resampled[i * chunk_size : (i + 1) * chunk_size]
    x = chunk.reshape(1, -1)
    out, state = s.run(None, {'input': x, 'state': state, 'sr': np.array(target_sr, dtype=np.int64)})
    probs.append(float(out[0][0]))

print(f"\nResults: {len(probs)} chunks processed")
print(f"Probs: min={min(probs):.4f}, max={max(probs):.4f}, mean={np.mean(probs):.4f}")
print(f"Chunks > 0.5: {sum(1 for p in probs if p > 0.5)}/{len(probs)}")
print(f"Chunks > 0.3: {sum(1 for p in probs if p > 0.3)}/{len(probs)}")
print(f"Chunks > 0.1: {sum(1 for p in probs if p > 0.1)}/{len(probs)}")
print(f"\nTop 10 probs: {sorted(probs, reverse=True)[:10]}")

# Also try with 1536-sample chunks (another size supported by v5)
print("\n--- Testing with 1536-sample chunks ---")
state = np.zeros((2, 1, 128), dtype=np.float32)
chunk_size = 1536
probs2 = []
for i in range(len(resampled) // chunk_size):
    chunk = resampled[i * chunk_size : (i + 1) * chunk_size]
    x = chunk.reshape(1, -1)
    out, state = s.run(None, {'input': x, 'state': state, 'sr': np.array(target_sr, dtype=np.int64)})
    probs2.append(float(out[0][0]))

print(f"Probs: min={min(probs2):.4f}, max={max(probs2):.4f}, mean={np.mean(probs2):.4f}")
print(f"Chunks > 0.5: {sum(1 for p in probs2 if p > 0.5)}/{len(probs2)}")
print(f"Top 10 probs: {sorted(probs2, reverse=True)[:10]}")
