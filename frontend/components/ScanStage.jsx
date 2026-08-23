export default function ScanStage({ videoUrl, statusMessage }) {
  return (
    <div className="scan-stage">
      <div className="scan-frame">
        {videoUrl && <video src={videoUrl} className="scan-video" muted loop autoPlay playsInline />}
        <div className="scan-line" aria-hidden="true" />
        <div className="scan-grid" aria-hidden="true" />
      </div>
      <div className="scan-status">
        <span className="scan-dot" aria-hidden="true" />
        <span className="mono">{statusMessage}</span>
      </div>
    </div>
  );
}
