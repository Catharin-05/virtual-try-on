import React from 'react';
import { Ruler, Sparkles, Sliders, ArrowRight, RotateCcw, Check } from 'lucide-react';
import { calculateDerivedMeasurements } from '../../services/measurementEngine';

export function Step3Measurements({ bodyParams, setBodyParams, onNext }) {
  const resetToDerived = () => {
    const derived = calculateDerivedMeasurements(bodyParams.height);
    setBodyParams(derived);
  };

  const updateParam = (key, val) => {
    setBodyParams((prev) => ({
      ...prev,
      [key]: Number(val)
    }));
  };

  const metricsList = [
    { key: 'chest', label: 'Bust / Chest', desc: 'Fullest point of bust', unit: 'cm', color: 'text-pink-400' },
    { key: 'waist', label: 'Waist Circumference', desc: 'Narrowest waist point', unit: 'cm', color: 'text-cyan-400' },
    { key: 'hip', label: 'Hip Circumference', desc: 'Fullest hip / seat point', unit: 'cm', color: 'text-emerald-400' },
    { key: 'shoulderWidth', label: 'Shoulder Width', desc: 'Across shoulder bone tips', unit: 'cm', color: 'text-purple-400' },
    { key: 'inseam', label: 'Leg Inseam', desc: 'Inner crotch to ankle', unit: 'cm', color: 'text-amber-400' },
    { key: 'torsoLength', label: 'Torso Length', desc: 'Nape of neck to waist', unit: 'cm', color: 'text-blue-400' }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fadeIn">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-xs font-semibold mb-2">
          <Ruler className="w-3.5 h-3.5" /> STAGE 3: ANTHROPOMETRIC SCALING & MEASUREMENTS
        </div>
        <h2 className="text-2xl md:text-3xl font-extrabold font-heading text-white">
          Extracted 3D Body <span className="gradient-text-primary">Circumferences</span>
        </h2>
        <p className="text-slate-400 text-sm max-w-xl mx-auto">
          SAM 3D calculated your anatomical circumferences based on your {bodyParams.height} cm height calibration. You can fine-tune any measurement below.
        </p>
      </div>

      <div className="glass-panel p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2 text-slate-200 font-heading font-semibold text-base">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span>Interactive Measurement Fine-Tuner</span>
          </div>
          <button
            onClick={resetToDerived}
            className="text-xs text-slate-400 hover:text-cyan-400 flex items-center gap-1 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset to SAM 3D Auto
          </button>
        </div>

        {/* Sliders Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {metricsList.map((m) => {
            const val = bodyParams[m.key] || 70;
            return (
              <div key={m.key} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 hover:border-slate-700 transition-colors">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-xs font-semibold text-white">{m.label}</h4>
                    <p className="text-[10px] text-slate-400">{m.desc}</p>
                  </div>
                  <span className={`font-mono font-bold text-lg ${m.color}`}>
                    {val} <span className="text-xs font-normal text-slate-400">{m.unit}</span>
                  </span>
                </div>

                <input
                  type="range"
                  min={m.key === 'shoulderWidth' ? 30 : 50}
                  max={m.key === 'shoulderWidth' ? 55 : 130}
                  value={val}
                  onChange={(e) => updateParam(m.key, e.target.value)}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>
            );
          })}
        </div>

        {/* Summary Card */}
        <div className="p-4 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-400 text-slate-950 flex items-center justify-center font-bold">
              <Check className="w-6 h-6" />
            </div>
            <div>
              <h4 className="text-xs font-bold font-heading text-white">3D Mesh Calibration Locked</h4>
              <p className="text-[11px] text-slate-300">
                Height: {bodyParams.height} cm | Bust: {bodyParams.chest} cm | Waist: {bodyParams.waist} cm | Hips: {bodyParams.hip} cm
              </p>
            </div>
          </div>

          <button onClick={onNext} className="btn-neon text-xs px-5 py-2.5">
            <span>Proceed to Garment Catalog</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
