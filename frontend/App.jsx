import { useState, useRef, useCallback } from 'react';
import ModelViewer from './components/ModelViewer';
import MeasurementsPanel from './components/MeasurementsPanel';
import ScanStage from './components/ScanStage';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

// idle -> ready -> processing -> done
//                             \-> error
export default function App() {
  const [stage, setStage] = useState('idle');
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [height, setHeight] = useState('');
  const [jobId, setJobId] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [modelUrl, setModelUrl] = useState(null);
  const [measurements, setMeasurements] = useState(null);
  const pollRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoFile(file);
    setVideoPreviewUrl(URL.createObjectURL(file));
    setStage('ready');
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const pollStatus = useCallback((id) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/scan/${id}/status`);
        if (!res.ok) throw new Error(`Status check failed (${res.status})`);
        const data = await res.json();
        setStatusMessage(data.message || data.status);

        if (data.status === 'done') {
          stopPolling();
          setModelUrl(`${API_BASE}/api/scan/${id}/model.obj`);
          const measRes = await fetch(`${API_BASE}/api/scan/${id}/measurements`);
          if (measRes.ok) setMeasurements(await measRes.json());
          setStage('done');
        } else if (data.status === 'error') {
          stopPolling();
          setErrorMessage(data.message || 'Something went wrong while processing your scan.');
          setStage('error');
        }
      } catch (err) {
        stopPolling();
        setErrorMessage(err.message || 'Lost connection to the server.');
        setStage('error');
      }
    }, 2500);
  }, []);

  const startScan = async () => {
    if (!videoFile || !height) return;
    setStage('processing');
    setErrorMessage('');
    setStatusMessage('Uploading your video…');

    const form = new FormData();
    form.append('video', videoFile);
    form.append('height_cm', height);

    try {
      const res = await fetch(`${API_BASE}/api/scan`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Upload failed (${res.status})`);
      const data = await res.json();
      setJobId(data.job_id);
      setStatusMessage('Reading your video…');
      pollStatus(data.job_id);
    } catch (err) {
      setErrorMessage(err.message || 'Could not reach the server.');
      setStage('error');
    }
  };

  const reset = () => {
    stopPolling();
    setStage('idle');
    setVideoFile(null);
    setVideoPreviewUrl(null);
    setHeight('');
    setJobId(null);
    setStatusMessage('');
    setErrorMessage('');
    setModelUrl(null);
    setMeasurements(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <span className="eyebrow">virtual fitting room</span>
        <h1>Turn a video into your digital twin</h1>
        <p className="subhead">Upload one clip of yourself turning around. We'll build a 3D model sized to you.</p>
      </header>

      {(stage === 'idle' || stage === 'ready') && (
        <div className="upload-card">
          <label className={`dropzone ${videoFile ? 'has-file' : ''}`}>
            <input type="file" accept="video/*" onChange={handleFileChange} hidden />
            {videoPreviewUrl ? (
              <video src={videoPreviewUrl} className="preview-video" muted loop autoPlay playsInline />
            ) : (
              <>
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M12 16V4M12 4l-4 4M12 4l4 4" stroke="var(--cyan)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="var(--cyan)" strokeWidth="1.6" strokeLinecap="round"/>
                </svg>
                <span className="dropzone-label">Drop a video, or click to choose one</span>
                <span className="dropzone-hint">Turn slowly, full body in frame — front, side, and back</span>
              </>
            )}
          </label>

          <div className="field-row">
            <label htmlFor="height" className="field-label">Your height</label>
            <div className="height-input-wrap">
              <input
                id="height"
                type="number"
                inputMode="decimal"
                min="100"
                max="230"
                placeholder="170"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
              />
              <span className="unit">cm</span>
            </div>
          </div>

          <button className="cta" disabled={!videoFile || !height} onClick={startScan}>
            Build my model
          </button>
        </div>
      )}

      {stage === 'processing' && (
        <ScanStage videoUrl={videoPreviewUrl} statusMessage={statusMessage} />
      )}

      {stage === 'error' && (
        <div className="error-card">
          <span className="error-label">Scan failed</span>
          <p>{errorMessage}</p>
          <button className="cta ghost" onClick={reset}>Try again</button>
        </div>
      )}

      {stage === 'done' && (
        <div className="results">
          <div className="viewer-panel">
            <ModelViewer objUrl={modelUrl} />
          </div>
          <MeasurementsPanel measurements={measurements} onReset={reset} />
        </div>
      )}
    </div>
  );
}
