import React from 'react';
import { Sparkles, Shirt, Layers, Cpu, Server } from 'lucide-react';

export function Navbar({ activeStep, onReset }) {
  return (
    <header className="w-full glass-panel rounded-none border-x-0 border-t-0 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3 cursor-pointer" onClick={onReset}>
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-400 via-blue-500 to-purple-600 p-[1px] shadow-lg shadow-cyan-500/20">
          <div className="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
            <Shirt className="w-5 h-5 text-cyan-400" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-heading font-extrabold text-xl tracking-tight text-white">
              AURA <span className="gradient-text-primary">3D</span>
            </h1>
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              SAM 3D ENGINE
            </span>
          </div>
          <p className="text-xs text-slate-400">AI Human Reconstruction & Virtual Fitting Studio</p>
        </div>
      </div>

      {/* Center Badges */}
      <div className="hidden md:flex items-center gap-4 text-xs font-mono text-slate-400">
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>SAM 3D Mesh Recovery</span>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800">
          <Layers className="w-3.5 h-3.5 text-purple-400" />
          <span>Tension Heatmap Shader</span>
        </div>
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={onReset}
          className="btn-outline text-xs px-3 py-1.5 flex items-center gap-1.5"
        >
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          New Fitting Session
        </button>
      </div>
    </header>
  );
}
