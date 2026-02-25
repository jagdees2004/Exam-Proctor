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
    const [tabSwitchCount, setTabSwitchCount] = useState(0);

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const intervalRef = useRef(null);
    const wsRef = useRef(null); // WebSocket connection for real-time AI
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

    const addFlag = useCallback((reason) => {
        const entry = {
            time: new Date().toLocaleTimeString(),
            reason,
        };
        setFlagLog((prev) => [entry, ...prev].slice(0, 20));
    }, []);

    // ── Tab / Window Switch Detection ───────────────────────────────────
    useEffect(() => {
        let leaveTime = null;

        const handleVisibilityChange = () => {
            if (document.hidden && phase === 'monitoring') {
                leaveTime = Date.now();
                console.log('[Proctor] ⚠ Tab switched away');
            } else if (!document.hidden && phase === 'monitoring' && leaveTime) {
                const awaySeconds = ((Date.now() - leaveTime) / 1000).toFixed(1);
                leaveTime = null;
                setTabSwitchCount(prev => prev + 1);
                setStatus('flagged');
                setMessage(`🚨 TAB SWITCH DETECTED — You left for ${awaySeconds}s`);
                addFlag(`Tab/window switch (away ${awaySeconds}s)`);
            }
        };

        const handleWindowBlur = () => {
            if (phase === 'monitoring' && !document.hidden) {
                // Window lost focus but tab didn't change (e.g., alt-tab to another app)
                leaveTime = Date.now();
                console.log('[Proctor] ⚠ Window lost focus');
            }
        };

        const handleWindowFocus = () => {
            if (phase === 'monitoring' && leaveTime) {
                const awaySeconds = ((Date.now() - leaveTime) / 1000).toFixed(1);
                leaveTime = null;
                setTabSwitchCount(prev => prev + 1);
                setStatus('flagged');
                setMessage(`🚨 WINDOW SWITCH DETECTED — You left for ${awaySeconds}s`);
                addFlag(`Window switch (away ${awaySeconds}s)`);
            }
        };

        document.addEventListener('visibilitychange', handleVisibilityChange);
        window.addEventListener('blur', handleWindowBlur);
        window.addEventListener('focus', handleWindowFocus);

        return () => {
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            window.removeEventListener('blur', handleWindowBlur);
            window.removeEventListener('focus', handleWindowFocus);
        };
    }, [phase, addFlag]);

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

        return canvas.toDataURL('image/jpeg', 0.85);
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

        const b64 = await captureFrame();

        let blob = null;
        if (b64 && b64 !== 'dark') {
            const res = await fetch(b64);
            blob = await res.blob();
        }

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
    // ── Monitoring Loop (WebSocket) ────────────────────────────────────────
    const startMonitoring = useCallback(() => {
        // Build WS URL robustly for both local (Vite proxy) 
        let wsUrl;
        if (API_BASE) {
            // Production: convert https://... to wss://...
            const wsProtocol = API_BASE.startsWith('https') ? 'wss' : 'ws';
            const cleanBase = API_BASE.replace(/^https?:\/\//, '');
            wsUrl = `${wsProtocol}://${cleanBase}/exam/ws/${userId.trim()}`;
        } else {
            // Local dev: connect to same host/port (Vite proxy will forward it to 8000)
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            wsUrl = `${wsProtocol}//${window.location.host}/exam/ws/${userId.trim()}`;
        }

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        let isActive = true;
        let isProcessingFrame = false;

        ws.onopen = () => {
            console.log('[WS] Connected for real-time AI');
            // Trigger first frame
            sendFrame();
        };

        ws.onmessage = (event) => {
            if (!isActive) return;
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'video_result') {
                    // Handle face/object result
                    setStats(prev => ({
                        ...prev,
                        ...data
                    }));

                    if (data.flagged) {
                        if (data.status === 'multiple_faces') {
                            setStatus('flagged');
                            setMessage(`🚨 MULTIPLE FACES DETECTED (${data.face_count} faces)`);
                            addFlag(`Multiple faces: ${data.face_count}`);
                        } else if (data.status === 'camera_blocked') {
                            setStatus('flagged');
                            setMessage('🚨 CAMERA BLOCKED — Your camera appears to be covered or off.');
                            addFlag('Camera blocked / off');
                        } else if (data.status === 'no_face') {
                            setStatus('no_face');
                            setMessage('⚠ No face detected. Please look at the camera.');
                            addFlag('No face detected');
                        } else if (data.status === 'identity_mismatch') {
                            setStatus('flagged');
                            setMessage('🚨 IDENTITY MISMATCH — Face does not match registered user.');
                            addFlag('Identity mismatch');
                        } else if (data.forbidden_objects?.length > 0) {
                            const objects = data.forbidden_objects.map((o) => o.class_name).join(', ');
                            setStatus('flagged');
                            setMessage(`🚨 FORBIDDEN OBJECT DETECTED: ${objects}`);
                            addFlag(`Forbidden object: ${objects}`);
                        }
                    } else {
                        setStatus('ok');
                        setMessage('✅ Verified — You are being monitored.');
                    }

                    // Frame processed, wait a bit then send next
                    isProcessingFrame = false;
                    setTimeout(sendFrame, 1500); // 1.5-second self-pacing delay
                }
                else if (data.type === 'audio_result') {
                    setAudioStatus(data);
                    if (data.is_talking) {
                        setStatus('flagged');
                        setMessage(`🚨 SPEECH DETECTED — Talking is not allowed during the exam.`);
                        addFlag(`Speech detected (${(data.speech_prob * 100).toFixed(0)}%)`);
                    }
                }
            } catch (err) {
                console.error('[WS] parse error', err);
                isProcessingFrame = false;
            }
        };

        ws.onerror = (err) => {
            console.error('[WS] Error', err);
            isProcessingFrame = false;
        };

        ws.onclose = () => {
            console.log('[WS] Disconnected');
            isActive = false;
        };

        const sendFrame = async () => {
            if (!isActive || ws.readyState !== WebSocket.OPEN) return;
            if (isProcessingFrame) return;
            isProcessingFrame = true;

            try {
                const b64 = await captureFrame();

                if (!b64) {
                    isProcessingFrame = false;
                    setTimeout(sendFrame, 1000);
                    return;
                }

                if (b64 === 'dark') {
                    setStatus('flagged');
                    setMessage('🚨 CAMERA BLOCKED — Your camera appears to be covered or off.');
                    setStats(prev => prev ? { ...prev, status: 'camera_blocked', flagged: true } : prev);
                    addFlag('Camera blocked / off');
                    isProcessingFrame = false;
                    setTimeout(sendFrame, 1500);
                    return;
                }

                ws.send(JSON.stringify({ type: 'frame', data: b64 }));

            } catch (err) {
                console.error('[WS] send frame error', err);
                isProcessingFrame = false;
                setTimeout(sendFrame, 1500);
            }
        };

        // Audio interval (every 3s)
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = setInterval(() => {
            if (!isActive || ws.readyState !== WebSocket.OPEN) return;

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

                // Convert ArrayBuffer to base64
                let binary = '';
                const bytes = new Uint8Array(wavBuffer);
                const len = bytes.byteLength;
                for (let i = 0; i < len; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                const b64Audio = window.btoa(binary);

                ws.send(JSON.stringify({ type: 'audio', data: b64Audio }));
            }
        }, 3000);

    }, [userId, captureFrame]);

    // ── Stop Exam ──────────────────────────────────────────────────────────
    const handleStop = () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (intervalRef.current) clearInterval(intervalRef.current);
        audioBufferRef.current = [];
        setPhase('setup');
        setStatus('idle');
        setStats(null);
        setAudioStatus(null);
        setTabSwitchCount(0);
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
                                <div className="stat">
                                    <span className="stat-label">🔀 Tab Switches</span>
                                    <span className={`stat-value ${tabSwitchCount > 0 ? 'bad' : 'good'}`}>
                                        {tabSwitchCount}
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
