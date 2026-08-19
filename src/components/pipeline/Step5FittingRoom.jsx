import React, { useState, useMemo } from 'react';
import { Eye, Layers, Sliders, Sparkles, CheckCircle2, AlertTriangle, Info, RotateCcw, Maximize2, Download, ChevronRight } from 'lucide-react';
import { FittingCanvas } from '../3d/FittingCanvas';
import { CrossSectionInspector } from '../3d/CrossSectionInspector';
import { computeFitClearance } from '../../services/measurementEngine';
import { GARMENT_CATALOG } from '../../data/garmentsData';
import confetti from 'canvas-confetti';

export function Step5FittingRoom({
  bodyParams,
  setBodyParams,
  selectedGarment,
  setSelectedGarment,
  selectedSize,
  setSelectedSize
}) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showMarkers, setShowMarkers] = useState(true);
  const [activeSliceZone, setActiveSliceZone] = useState('waist');
  const [showReportModal, setShowReportModal] = useState(false);

  // Compute live clearance & fit analytics
  const fitMetrics = useMemo(() => {
    return computeFitClearance(bodyParams, selectedGarment, selectedSize);
  }, [bodyParams, selectedGarment, selectedSize]);

  // Trigger celebration confetti when user hits optimal fit score > 90
  const handleExportReport = () => {
    confetti({
      particleCount: 70,
      spread: 60,
      origin: { y: 0.6 }
    });
    setShowReportModal(true);
  };

  return (
    <div className="w-full max-w-7xl mx-auto space-y-4 animate-fadeIn">
      {/* Studio Header Bar */}
      <div className="glass-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Eye className="w-5 h-5" />
          </div>
          <div>
            <h2 className="font-heading font-extrabold text-lg text-white flex items-center gap-2">
              Virtual Fitting Studio
              <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                1:1 Scale Active
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Interactive 3D Fit Analysis • SAM 3D Reconstruction
            </p>
          </div>
        </div>

        {/* Garment Quick Switcher & Size Selector */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Garment Select */}
          <select
            value={selectedGarment.id}
            onChange={(e) => {
              const found = GARMENT_CATALOG.find(g => g.id === e.target.value);
              if (found) setSelectedGarment(found);
            }}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs font-heading font-semibold rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-400"
          >
            {GARMENT_CATALOG.map(g => (
              <option key={g.id} value={g.id}>{g.name}</option>
            ))}
          </select>

          {/* Size Pills */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
            {['XS', 'S', 'M', 'L', 'XL'].map((sz) => (
              <button
                key={sz}
                onClick={() => setSelectedSize(sz)}
                className={`px-2.5 py-1 rounded-lg text-xs font-heading font-bold transition-all ${
                  selectedSize === sz
                    ? 'bg-cyan-400 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {sz}
              </button>
            ))}
          </div>

          <button
            onClick={handleExportReport}
            className="btn-neon text-xs px-3.5 py-2 flex items-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            Fit Report
          </button>
        </div>
      </div>

      {/* Main Studio Viewport Grid: 3D Canvas (8 cols) + Analysis Dashboard (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* 3D Viewport Column */}
        <div className="lg:col-span-8 h-[550px] relative glass-panel p-2 flex flex-col">
          {/* Viewport Overlay Controls */}
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold flex items-center gap-2 backdrop-blur-md transition-all ${
                showHeatmap
                  ? 'bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-400/20'
                  : 'bg-slate-900/80 text-slate-300 border border-slate-700'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              {showHeatmap ? 'Heatmap Mode ON' : 'Texture Mode'}
            </button>

            <button
              onClick={() => setShowMarkers(!showMarkers)}
              className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold backdrop-blur-md transition-all ${
                showMarkers
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                  : 'bg-slate-900/80 text-slate-400 border border-slate-700'
              }`}
            >
              {showMarkers ? 'Anatomical Markers ON' : 'Markers OFF'}
            </button>
          </div>

          {/* Heatmap Tension Legend Overlay */}
          {showHeatmap && (
            <div className="absolute bottom-4 left-4 z-10 glass-panel p-2.5 flex items-center gap-3 text-[10px] font-mono">
              <span className="text-slate-300 font-semibold">Tension Map:</span>
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span>
                <span className="text-slate-400">Tight / Stretch</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                <span className="text-slate-400">Optimal Fit</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                <span className="text-slate-400">Loose / Drape</span>
              </div>
            </div>
          )}

          {/* R3F 3D Canvas */}
          <FittingCanvas
            bodyParams={bodyParams}
            garmentData={selectedGarment}
            sizeKey={selectedSize}
            showHeatmap={showHeatmap}
            showMarkers={showMarkers}
          />
        </div>

        {/* Analysis Dashboard Column (4 cols) */}
        <div className="lg:col-span-4 space-y-4 flex flex-col justify-between">
          {/* Fit Match Score Card */}
          <div className="glass-panel p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Overall Fit Score</span>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-slate-900 text-cyan-400 border border-slate-800">
                {selectedSize} Tag Size
              </span>
            </div>

            <div className="flex items-baseline gap-3">
              <span className={`text-4xl font-extrabold font-heading ${
                fitMetrics.overallScore > 85 ? 'text-emerald-400' : fitMetrics.overallScore > 65 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {fitMetrics.overallScore}%
              </span>
              <span className="text-xs text-slate-300 font-medium">Fit Compatibility</span>
            </div>

            <p className="text-xs text-slate-300 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
              {fitMetrics.recommendation}
            </p>

            {/* Zone Clearance Table */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              {[
                { zone: 'Bust / Chest', data: fitMetrics.chest, key: 'chest' },
                { zone: 'Waist Line', data: fitMetrics.waist, key: 'waist' },
                { zone: 'Hip Circumference', data: fitMetrics.hip, key: 'hip' }
              ].map((z) => (
                <div
                  key={z.key}
                  onClick={() => setActiveSliceZone(z.key)}
                  className={`p-2 rounded-lg flex items-center justify-between text-xs cursor-pointer transition-all ${
                    activeSliceZone === z.key
                      ? 'bg-cyan-500/10 border border-cyan-500/30'
                      : 'bg-slate-900/40 hover:bg-slate-900/80 border border-transparent'
                  }`}
                >
                  <span className="text-slate-300 font-medium">{z.zone}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px]" style={{ color: z.data.color }}>
                      {z.data.delta > 0 ? `+${z.data.delta}cm` : `${z.data.delta}cm`} ({z.data.status})
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 2D Slice Clearance Inspector */}
          <CrossSectionInspector
            bodyParams={bodyParams}
            garmentData={selectedGarment}
            sizeKey={selectedSize}
            activeZone={activeSliceZone}
          />
        </div>
      </div>

      {/* Fit Report Modal */}
      {showReportModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-panel max-w-lg w-full p-6 space-y-4 border-cyan-500/40 animate-fadeIn">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-cyan-400" />
                <h3 className="font-heading font-extrabold text-lg text-white">Garment Fit Report</h3>
              </div>
              <button
                onClick={() => setShowReportModal(false)}
                className="text-slate-400 hover:text-white text-sm"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-slate-400 block">Garment Model</span>
                  <span className="font-bold text-white text-sm">{selectedGarment.name}</span>
                </div>
                <span className="font-mono font-bold text-cyan-400 text-sm">Size {selectedSize}</span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-slate-300">
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400 block font-mono text-[10px]">User Height</span>
                  <span className="font-bold text-white">{bodyParams.height} cm</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <span className="text-slate-400 block font-mono text-[10px]">Bust / Waist / Hip</span>
                  <span className="font-bold text-white">{bodyParams.chest} / {bodyParams.waist} / {bodyParams.hip} cm</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                <span className="font-bold block mb-1">Tailor Recommendation:</span>
                {fitMetrics.recommendation}
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowReportModal(false)}
                className="btn-neon text-xs px-5 py-2.5"
              >
                Close Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
