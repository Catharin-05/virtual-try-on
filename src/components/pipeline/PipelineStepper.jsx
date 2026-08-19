import React from 'react';
import { Video, Film, Box, Ruler, CheckCircle2, Eye } from 'lucide-react';

export const PIPELINE_STEPS = [
  { id: 1, title: 'Video & Height', desc: 'Upload 360° Video', icon: Video },
  { id: 2, title: 'Frame Extraction', desc: 'SAM 2D Keyframes', icon: Film },
  { id: 3, title: 'SAM 3D Mesh', desc: 'Body Reconstruction', icon: Box },
  { id: 4, title: 'Scale & Measure', desc: 'Anthropometrics', icon: Ruler },
  { id: 5, title: 'Garment Fit & 3D', desc: 'Fitting Room Studio', icon: Eye }
];

export function PipelineStepper({ currentStep, setStep }) {
  return (
    <div className="w-full max-w-5xl mx-auto my-6 px-4">
      <div className="glass-panel p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        {PIPELINE_STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isActive = currentStep === step.id;
          const isCompleted = currentStep > step.id;

          return (
            <React.Fragment key={step.id}>
              {/* Step Item */}
              <div
                onClick={() => isCompleted && setStep(step.id)}
                className={`flex items-center gap-3 p-2 rounded-xl transition-all cursor-pointer ${
                  isActive
                    ? 'bg-cyan-500/10 border border-cyan-500/30 text-white'
                    : isCompleted
                    ? 'text-slate-300 hover:bg-slate-800/50'
                    : 'text-slate-500 opacity-60 cursor-not-allowed'
                }`}
              >
                <div
                  className={`w-9 h-9 rounded-lg flex items-center justify-center font-heading font-bold text-sm transition-all ${
                    isActive
                      ? 'bg-cyan-400 text-slate-950 shadow-md shadow-cyan-400/30'
                      : isCompleted
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {isCompleted ? <CheckCircle2 className="w-5 h-5" /> : <Icon className="w-4 h-4" />}
                </div>

                <div className="text-left">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] font-mono text-cyan-400 font-semibold">
                      0{step.id}
                    </span>
                    <h3 className="text-xs font-semibold font-heading leading-tight">{step.title}</h3>
                  </div>
                  <p className="text-[10px] text-slate-400 hidden lg:block">{step.desc}</p>
                </div>
              </div>

              {/* Arrow Connector */}
              {idx < PIPELINE_STEPS.length - 1 && (
                <div className="hidden md:block w-6 h-[2px] bg-slate-800 relative">
                  <div
                    className="h-full bg-cyan-400 transition-all duration-500"
                    style={{ width: currentStep > step.id ? '100%' : '0%' }}
                  />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
