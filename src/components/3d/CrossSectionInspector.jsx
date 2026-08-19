import React from 'react';

export function CrossSectionInspector({ bodyParams, garmentData, sizeKey, activeZone = 'waist' }) {
  const garmentSize = garmentData.sizes[sizeKey] || garmentData.sizes['M'];
  
  let bodyCirc = bodyParams[activeZone] || 68;
  let garmentCirc = garmentSize[activeZone] || 70;

  const clearanceCm = (garmentCirc - bodyCirc) / 2; // radius gap in cm
  const clearanceMm = Math.round(clearanceCm * 10);

  // SVG radii mapping
  const baseRadius = 60;
  const garmentRadius = baseRadius + clearanceMm * 2.2;

  let zoneColor = '#10B981';
  let zoneLabel = 'Optimal Clearance';
  if (clearanceMm < 0) {
    zoneColor = '#EF4444';
    zoneLabel = `Stretch Compression (${Math.abs(clearanceMm)} mm tight)`;
  } else if (clearanceMm <= 5) {
    zoneColor = '#F59E0B';
    zoneLabel = `Precision Fit (${clearanceMm} mm clearance)`;
  } else if (clearanceMm > 25) {
    zoneColor = '#3B82F6';
    zoneLabel = `Loose Drape (${clearanceMm} mm gap)`;
  }

  return (
    <div className="glass-panel p-4 w-full">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-heading font-semibold text-sm text-slate-200 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: zoneColor }}></span>
          Cross-Section Slice: <span className="capitalize text-cyan-400">{activeZone}</span> Zone
        </h4>
        <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
          {clearanceMm >= 0 ? `+${clearanceMm} mm` : `${clearanceMm} mm`}
        </span>
      </div>

      {/* SVG Radial Clearance Plot */}
      <div className="flex justify-center items-center my-2 relative">
        <svg width="180" height="180" viewBox="0 0 200 200">
          <defs>
            <radialGradient id="bodyGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#334155" />
              <stop offset="100%" stopColor="#1E293B" />
            </radialGradient>
          </defs>

          {/* Background Grid */}
          <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
          <circle cx="100" cy="100" r="40" fill="none" stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />

          {/* Garment Mesh Circumference Outer Ring */}
          <circle
            cx="100"
            cy="100"
            r={Math.min(90, Math.max(35, garmentRadius))}
            fill="none"
            stroke={zoneColor}
            strokeWidth="3"
            strokeDasharray={clearanceMm < 0 ? "4,4" : "none"}
          />

          {/* Body Mesh Circumference Inner Ellipse */}
          <ellipse
            cx="100"
            cy="100"
            rx={baseRadius}
            ry={baseRadius * 0.75}
            fill="url(#bodyGrad)"
            stroke="#00F0FF"
            strokeWidth="2"
          />

          {/* Center Point */}
          <circle cx="100" cy="100" r="3" fill="#00F0FF" />

          {/* Metric Annotations */}
          <text x="100" y="96" textAnchor="middle" fill="#F1F5F9" fontSize="12" fontWeight="bold">
            Body: {bodyCirc} cm
          </text>
          <text x="100" y="112" textAnchor="middle" fill={zoneColor} fontSize="11" fontWeight="500">
            Dress: {garmentCirc} cm
          </text>
        </svg>
      </div>

      <div className="text-center mt-1">
        <span className="text-xs text-slate-300 font-medium">{zoneLabel}</span>
      </div>
    </div>
  );
}
