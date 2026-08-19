import React from 'react';
import { Shirt, Sparkles, Check, ArrowRight, ShieldCheck, Info } from 'lucide-react';
import { GARMENT_CATALOG } from '../../data/garmentsData';

export function Step4GarmentSelect({
  selectedGarment,
  setSelectedGarment,
  selectedSize,
  setSelectedSize,
  onEnterFittingRoom
}) {
  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fadeIn">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 font-mono text-xs font-semibold mb-2">
          <Shirt className="w-3.5 h-3.5" /> STAGE 4: GARMENT SELECTION & SIZE SPECIFICATION
        </div>
        <h2 className="text-2xl md:text-3xl font-extrabold font-heading text-white">
          Choose a Dress & <span className="gradient-text-primary">Size Variant</span>
        </h2>
        <p className="text-slate-400 text-sm max-w-xl mx-auto">
          Select any dress from the studio collection to simulate fabric drape, stretch tension, and clearance against your SAM 3D body avatar.
        </p>
      </div>

      {/* Garments Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {GARMENT_CATALOG.map((item) => {
          const isSelected = selectedGarment.id === item.id;
          return (
            <div
              key={item.id}
              onClick={() => setSelectedGarment(item)}
              className={`glass-panel p-5 space-y-4 cursor-pointer transition-all ${
                isSelected
                  ? 'border-cyan-400 bg-cyan-500/10 shadow-lg shadow-cyan-400/10'
                  : 'hover:border-slate-700'
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[10px] font-mono font-semibold text-cyan-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                    {item.category}
                  </span>
                  <h3 className="font-heading font-bold text-lg text-white mt-1">{item.name}</h3>
                </div>
                <div
                  className="w-5 h-5 rounded-full border flex items-center justify-center transition-all"
                  style={{
                    borderColor: isSelected ? '#00F0FF' : '#475569',
                    backgroundColor: isSelected ? '#00F0FF' : 'transparent'
                  }}
                >
                  {isSelected && <Check className="w-3.5 h-3.5 text-slate-950 stroke-[3]" />}
                </div>
              </div>

              <p className="text-xs text-slate-300 line-clamp-2">{item.description}</p>

              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-2 border-t border-slate-800/80">
                <div className="text-slate-400">
                  Fabric: <span className="text-slate-200">{item.fabric}</span>
                </div>
                <div className="text-slate-400">
                  Elasticity: <span className="text-cyan-300">{item.elasticity}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Size Picker & Fitting Studio CTA */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h4 className="text-xs font-mono font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Select Garment Tag Size:
            </h4>
            <div className="flex items-center gap-2">
              {['XS', 'S', 'M', 'L', 'XL'].map((sz) => {
                const active = selectedSize === sz;
                return (
                  <button
                    key={sz}
                    onClick={() => setSelectedSize(sz)}
                    className={`w-12 h-10 rounded-xl font-heading font-bold text-sm transition-all ${
                      active
                        ? 'bg-cyan-400 text-slate-950 shadow-md shadow-cyan-400/30'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                    }`}
                  >
                    {sz}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right hidden md:block">
              <span className="text-xs text-slate-400 font-mono block">Selected Garment</span>
              <span className="text-sm font-bold font-heading text-white">{selectedGarment.name} ({selectedSize})</span>
            </div>
            <button
              onClick={onEnterFittingRoom}
              className="btn-neon px-6 py-3 text-sm"
            >
              <span>Launch 3D Fitting Room</span>
              <Sparkles className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
