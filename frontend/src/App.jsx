import { useState, useRef, useEffect, useCallback } from 'react';

// Get the API URL from Vite environment variables (set in Netlify/Vercel)
// Fallback to relative path if not set (useful if hosted together)
const API_BASE = import.meta.env.VITE_API_URL || '';

// ── WAV Encoder ─────────────────────────────────────────────────────────────
// Converts Float32 PCM samples to a proper WAV binary buffer for the backend
function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);           // chunk size
    view.setUint16(20, 1, true);            // PCM format
    view.setUint16(22, 1, true);            // mono
    view.setUint32(24, sampleRate, true);   // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);            // block align
    view.setUint16(34, 16, true);           // bits per sample
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM data (float32 → int16)
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buffer;
}

function App() {
    // ── State ──────────────────────────────────────────────────────────────
    const [userId, setUserId] = useState('');
    const [phase, setPhase] = useState('setup');        // setup | registering | monitoring
    const [status, setStatus] = useState('idle');        // idle | ok | no_face | multiple_faces | identity_mismatch | flagged
    const [message, setMessage] = useState('Enter your ID to begin the exam.');
    const [stats, setStats] = useState(null);
    const [flagLog, setFlagLog] = useState([]);
    const [cameraReady, setCameraReady] = useState(false);
    const [audioStatus, setAudioStatus] = useState(null); // { is_talking, speech_prob }

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const intervalRef = useRef(null);
    const audioContextRef = useRef(null);
    const audioBufferRef = useRef([]); // accumulates raw Float32 PCM samples

    // ── Camera Setup ───────────────────────────────────────────────────────
    useEffect(() => {
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' },
                    audio: true,
                });
                streamRef.current = stream;
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    setCameraReady(true);
                }
                // Audio will be set up when user clicks Start Exam (requires user gesture)
            } catch (err) {
                setMessage('⚠ Camera access denied. Please allow camera permissions.');
                console.error('Camera error:', err);
            }
        }
        startCamera();
        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((t) => t.stop());
            }
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (audioContextRef.current) audioContextRef.current.close();
        };
    }, []);

    // ── Capture Frame ──────────────────────────────────────────────────────
    const captureFrame = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return null;

        // Check if camera track is still active
        const videoTrack = streamRef.current?.getVideoTracks()?.[0];
        if (!videoTrack || videoTrack.readyState === 'ended' || !videoTrack.enabled) {
            return 'dark'; // Signal camera is off
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Quick darkness check: sample pixels to detect covered/off camera
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        let totalBrightness = 0;
        const sampleCount = 200;
        const step = Math.max(1, Math.floor(data.length / 4 / sampleCount));
        for (let i = 0; i < sampleCount; i++) {
            const idx = (i * step) * 4;
            totalBrightness += (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        }
        const avgBrightness = totalBrightness / sampleCount;
        if (avgBrightness < 15) {
            return 'dark'; // Frame is too dark — camera blocked
        }

        return new Promise((resolve) => {
            canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85);
        });
    }, []);

    // ── Register Face ──────────────────────────────────────────────────────
    const handleStart = async () => {
        if (!userId.trim()) {
            setMessage('Please enter a valid user ID.');
            return;
        }
        if (!cameraReady) {
            setMessage('Camera is not ready yet.');
            return;
        }

        setPhase('registering');
        setMessage('📸 Capturing your face…');

        const blob = await captureFrame();
        console.log('[Proctor] Captured blob:', blob, 'size:', blob?.size);

        if (!blob || blob.size === 0) {
            setMessage('Failed to capture frame. Make sure camera is active and visible.');
            setPhase('setup');
            return;
        }

        const formData = new FormData();
        formData.append('user_id', userId.trim());
        formData.append('file', blob, 'frame.jpg');

        console.log('[Proctor] Sending registration for:', userId.trim());

        try {
            const res = await fetch(`${API_BASE}/exam/start`, {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();
            console.log('[Proctor] Registration response:', res.status, data);

            if (res.ok) {
                // Start audio capture NOW (inside user gesture so AudioContext won't be suspended)
                try {
                    if (!audioContextRef.current && streamRef.current) {
                        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        await audioCtx.resume(); // ensure it's running
                        const source = audioCtx.createMediaStreamSource(streamRef.current);
                        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
                        processor.onaudioprocess = (e) => {
                            const inputData = e.inputBuffer.getChannelData(0);
                            audioBufferRef.current.push(new Float32Array(inputData));
                        };
                        source.connect(processor);
                        processor.connect(audioCtx.destination);
                        audioContextRef.current = audioCtx;
                        console.log('[Proctor] Audio capture started at', audioCtx.sampleRate, 'Hz');
                    }
                } catch (audioErr) {
                    console.warn('[Proctor] Audio setup failed:', audioErr);
                }

                setMessage(`✅ ${data.message} — Monitoring started.`);
                setStatus('ok');
                setPhase('monitoring');
                startMonitoring();
            } else {
                // FastAPI detail can be a string or array of validation errors
                const errMsg = typeof data.detail === 'string'
                    ? data.detail
                    : Array.isArray(data.detail)
                        ? data.detail.map((e) => e.msg || e).join(', ')
                        : data.message || 'Registration failed';
                setMessage(`❌ ${errMsg}`);
                setPhase('setup');
            }
        } catch (err) {
            setMessage(`❌ Server error: ${err.message}`);
            setPhase('setup');
        }
    };

    // ── Monitoring Loop ────────────────────────────────────────────────────
    const inFlightRef = useRef(false);
    const tickRef = useRef(0); // Add a tick counter to stagger heavy requests

    const startMonitoring = useCallback(() => {
        if (intervalRef.current) clearInterval(intervalRef.current);

        intervalRef.current = setInterval(async () => {
            // Skip if previous request is still in flight (prevents stacking)
            if (inFlightRef.current) return;
            inFlightRef.current = true;
            tickRef.current += 1;

            try {
                const blob = await captureFrame();
                if (!blob) return;

                // Camera blocked detected client-side
                if (blob === 'dark') {
                    setStatus('flagged');
                    setMessage('🚨 CAMERA BLOCKED — Your camera appears to be covered or off.');
                    setStats(prev => prev ? { ...prev, status: 'camera_blocked', flagged: true } : prev);
                    addFlag('Camera blocked / off');
                    return;
                }

                // Fire face + audio requests every tick (1.5s)
                const formData = new FormData();
                formData.append('user_id', userId.trim());
                formData.append('file', blob, 'frame.jpg');

                const facePromise = fetch(`${API_BASE}/exam/verify`, {
                    method: 'POST',
                    body: formData,
                }).then(r => r.json()).catch(() => null);

                // Run heavy object detection every 3rd tick (~4.5s) to prevent backend OOM
                let objectsPromise = Promise.resolve(null);
                if (tickRef.current % 3 === 0) {
                    const objForm = new FormData();
                    objForm.append('user_id', userId.trim());
                    objForm.append('file', blob, 'frame.jpg');

                    objectsPromise = fetch(`${API_BASE}/exam/objects`, {
                        method: 'POST',
                        body: objForm,
                    }).then(r => r.json()).catch(() => null);
                }

                let audioPromise = Promise.resolve(null);
                if (audioBufferRef.current.length > 0) {
                    const chunks = audioBufferRef.current;
                    audioBufferRef.current = [];
                    const totalLen = chunks.reduce((sum, c) => sum + c.length, 0);
                    const merged = new Float32Array(totalLen);
                    let offset = 0;
                    for (const chunk of chunks) {
                        merged.set(chunk, offset);
                        offset += chunk.length;
                    }
                    const wavBuffer = encodeWav(merged, audioContextRef.current?.sampleRate || 16000);
                    const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                    const audioForm = new FormData();
                    audioForm.append('user_id', userId.trim());
                    audioForm.append('audio', wavBlob, 'chunk.wav');
                    audioPromise = fetch(`${API_BASE}/exam/audio`, {
                        method: 'POST',
                        body: audioForm,
                    }).then(r => r.json()).catch(() => null);
                }

                const [faceData, objData, audioData] = await Promise.all([facePromise, objectsPromise, audioPromise]);

                // Handle face result
                if (faceData) {
                    setStats(prev => ({ ...prev, ...faceData, forbidden_objects: objData?.forbidden_objects || [] }));
                    if (faceData.flagged) {
                        if (faceData.status === 'multiple_faces') {
                            setStatus('flagged');
                            setMessage(`🚨 MULTIPLE FACES DETECTED (${faceData.face_count} faces)`);
                            addFlag(`Multiple faces: ${faceData.face_count}`);
                        } else if (faceData.status === 'camera_blocked') {
                            setStatus('flagged');
                            setMessage('🚨 CAMERA BLOCKED — Your camera appears to be covered or off.');
                            addFlag('Camera blocked / off');
                        } else if (faceData.status === 'no_face') {
                            setStatus('no_face');
                            setMessage('⚠ No face detected. Please look at the camera.');
                            addFlag('No face detected');
                        } else if (faceData.status === 'identity_mismatch') {
                            setStatus('flagged');
                            setMessage('🚨 IDENTITY MISMATCH — Face does not match registered user.');
                            addFlag('Identity mismatch');
                        }
                    } else {
                        setStatus('ok');
                        setMessage('✅ Verified — You are being monitored.');
                    }
                }

                // Handle object detection result (overrides face status if objects found)
                if (objData?.flagged && objData.forbidden_objects?.length > 0) {
                    const objects = objData.forbidden_objects.map((o) => o.class_name).join(', ');
                    setStatus('flagged');
                    setMessage(`🚨 FORBIDDEN OBJECT DETECTED: ${objects}`);
                    addFlag(`Forbidden object: ${objects}`);
                }

                // Handle audio result
                if (audioData) {
                    setAudioStatus(audioData);
                    if (audioData.is_talking) {
                        setStatus('flagged');
                        setMessage(`🚨 SPEECH DETECTED — Talking is not allowed during the exam.`);
                        addFlag(`Speech detected (${(audioData.speech_prob * 100).toFixed(0)}%)`);
                    }
                }
            } finally {
                inFlightRef.current = false;
            }
        }, 1500);

    }, [userId, captureFrame]);

    const addFlag = useCallback((reason) => {
        const entry = {
            time: new Date().toLocaleTimeString(),
            reason,
        };
        setFlagLog((prev) => [entry, ...prev].slice(0, 20));
    }, []);

    // ── Stop Exam ──────────────────────────────────────────────────────────
    const handleStop = () => {
        if (intervalRef.current) clearInterval(intervalRef.current);
        audioBufferRef.current = [];
        setPhase('setup');
        setStatus('idle');
        setStats(null);
        setAudioStatus(null);
        setMessage('Exam ended. Enter your ID to start a new session.');
    };

    // ── Status Helpers ─────────────────────────────────────────────────────
    const borderClass =
        status === 'ok'
            ? 'border-ok'
            : status === 'flagged'
                ? 'border-flagged'
                : status === 'no_face' || status === 'identity_mismatch' || status === 'multiple_faces'
                    ? 'border-warning'
                    : 'border-idle';

    const statusEmoji =
        status === 'ok' ? '🟢' : status === 'flagged' ? '🔴' : status === 'idle' ? '⚪' : '🟡';

    // ── Render ─────────────────────────────────────────────────────────────
    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="header-inner">
                    <div className="logo">
                        <span className="logo-icon">🛡️</span>
                        <h1>AI Exam Proctor</h1>
                    </div>
                    <div className="header-status">
                        <span className="status-dot" data-status={status} />
                        <span className="status-label">
                            {phase === 'monitoring' ? `Monitoring • ${statusEmoji}` : 'Standby'}
                        </span>
                    </div>
                </div>
            </header>

            <main className="main">
                {/* Left Panel — Video & Controls */}
                <section className="panel video-panel">
                    <div className={`video-wrapper ${borderClass}`}>
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            id="webcam-video"
                        />
                        <canvas ref={canvasRef} style={{ display: 'none' }} />
                        {phase === 'monitoring' && (
                            <div className={`video-overlay ${borderClass}`}>
                                <span>{statusEmoji} {status.toUpperCase().replace('_', ' ')}</span>
                            </div>
                        )}
                    </div>

                    {/* Message Banner */}
                    <div className={`message-banner ${status === 'flagged' ? 'flagged' : status === 'ok' ? 'ok' : ''}`}>
                        <p>{message}</p>
                    </div>

                    {/* Controls */}
                    {phase === 'setup' || phase === 'registering' ? (
                        <div className="controls">
                            <div className="input-group">
                                <input
                                    type="text"
                                    id="user-id-input"
                                    placeholder="Enter Student ID"
                                    value={userId}
                                    onChange={(e) => setUserId(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleStart()}
                                    disabled={phase === 'registering'}
                                />
                                <button
                                    id="start-btn"
                                    className="btn btn-primary"
                                    onClick={handleStart}
                                    disabled={phase === 'registering' || !cameraReady}
                                >
                                    {phase === 'registering' ? (
                                        <><span className="spinner" /> Registering…</>
                                    ) : (
                                        '🚀 Start Exam'
                                    )}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="controls">
                            <button id="stop-btn" className="btn btn-danger" onClick={handleStop}>
                                ⏹ End Exam
                            </button>
                        </div>
                    )}
                </section>

                {/* Right Panel — Stats & Log */}
                <aside className="panel info-panel">
                    {/* Live Stats */}
                    <div className="card stats-card">
                        <h2>📊 Live Analysis</h2>
                        {stats ? (
                            <div className="stats-grid">
                                <div className="stat">
                                    <span className="stat-label">Identity</span>
                                    <span className={`stat-value ${stats.identity_match ? 'good' : 'bad'}`}>
                                        {stats.identity_match ? '✓ Match' : '✗ Mismatch'}
                                    </span>
                                </div>
                                <div className="stat">
                                    <span className="stat-label">Faces</span>
                                    <span className={`stat-value ${stats.face_count === 1 ? 'good' : 'bad'}`}>
                                        {stats.face_count}
                                    </span>
                                </div>
                                <div className="stat">
                                    <span className="stat-label">Similarity</span>
                                    <span className="stat-value mono">
                                        {(stats.similarity_score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="stat">
                                    <span className="stat-label">Objects</span>
                                    <span className={`stat-value ${stats.forbidden_objects?.length > 0 ? 'bad' : 'good'}`}>
                                        {stats.forbidden_objects?.length > 0
                                            ? stats.forbidden_objects.map((o) => o.class_name).join(', ')
                                            : 'None'}
                                    </span>
                                </div>
                                <div className="stat">
                                    <span className="stat-label">🎙️ Audio</span>
                                    <span className={`stat-value ${audioStatus?.is_talking ? 'bad' : 'good'}`}>
                                        {audioStatus
                                            ? audioStatus.is_talking
                                                ? `🚨 Speech (${(audioStatus.speech_prob * 100).toFixed(0)}%)`
                                                : `Silent (${(audioStatus.speech_prob * 100).toFixed(0)}%)`
                                            : 'Listening...'}
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <p className="placeholder-text">Start exam to see live stats</p>
                        )}
                    </div>

                    {/* Flag Log */}
                    <div className="card log-card">
                        <h2>🚩 Flag History</h2>
                        {flagLog.length > 0 ? (
                            <ul className="flag-list">
                                {flagLog.map((entry, i) => (
                                    <li key={i} className="flag-item">
                                        <span className="flag-time">{entry.time}</span>
                                        <span className="flag-reason">{entry.reason}</span>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="placeholder-text">No flags recorded</p>
                        )}
                    </div>

                    {/* System Info */}
                    <div className="card info-card">
                        <h2>ℹ️ How It Works</h2>
                        <ul className="info-list">
                            <li><strong>🟢 Green</strong> — Identity verified, all clear</li>
                            <li><strong>🟡 Yellow</strong> — Warning (no face / looking away)</li>
                            <li><strong>🔴 Red</strong> — Flagged (wrong person / device / multiple faces)</li>
                        </ul>
                    </div>
                </aside>
            </main>
        </div>
    );
}

export default App;
