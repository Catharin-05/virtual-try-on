import React, { useEffect, useState } from 'react';
import { Cpu, CheckCircle2, Loader2, Sparkles, Layers, Box, ArrowRight } from 'lucide-react';
import { processVideoRotation } from '../../services/sam3dSimulator';

export function Step2SAM3D({ video, heightCm, onComplete }) {
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('Initializing SAM 3D neural pipeline...');
  const [result, setResult] = useState(null);

  useEffect(() => {
    let isMounted = true;
    processVideoRotation(video, heightCm, (step) => {
      if (isMounted) {
        setProgress(step.progress);
        setStatusMsg(step.status);
      }
    }).then((res) => {
      if (isMounted) {
        setResult(res);
      }
    });

    return () => { isMounted = false; };
  }, [video, heightCm]);

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fadeIn">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono text-xs font-semibold mb-2">
          <Cpu className="w-3.5 h-3.5" /> STAGE 2: SAM 3D BODY MESH RECOVERY
        </div>
        <h2 className="text-2xl md:text-3xl font-extrabold font-heading text-white">
          Multi-Angle Keyframe Extraction & <span className="gradient-text-primary">SAM 3D Lifting</span>
        </h2>
        <p className="text-slate-400 text-sm max-w-xl mx-auto">
          SAM (Segment Anything Model) isolates body contours from 360° rotation frames and reconstructs a high-density 3D spatial vertex mesh.
        </p>
      </div>

      <div className="glass-panel p-6 space-y-6">
        {/* Progress Bar & Status */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-cyan-400 font-semibold flex items-center gap-2">
              {progress < 100 ? <Loader2 className="w-4 h-4 animate-spin text-cyan-400" /> : <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              {statusMsg}
            </span>
            <span className="text-white font-bold">{progress}%</span>
          </div>

          <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800 p-0.5">
            <div
              className="h-full bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 rounded-full transition-all duration-500 shadow-md shadow-cyan-400/30"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Multi-View Extracted Frames */}
        <div className="space-y-3">
          <h3 className="text-xs font-mono font-semibold text-slate-400 uppercase tracking-wider">
            Extracted Keyframe Angles (360° Rotational Tracking):
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { angle: '0° Front', img: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80' },
              { angle: '45° Quarter', img: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=300&q=80' },
              { angle: '90° Profile', img: 'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=300&q=80' },
              { angle: '180° Back', img: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?auto=format&fit=crop&w=300&q=80' },
              { angle: '270° Side', img: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=300&q=80' }
            ].map((f, i) => (
              <div key={i} className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-950 group">
                <img src={f.img} alt={f.angle} className="w-full h-28 object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent opacity-90" />
                <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
                  <span className="text-[10px] font-mono font-semibold text-cyan-300 bg-slate-900/80 px-1.5 py-0.5 rounded">
                    {f.angle}
                  </span>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Reconstruction Mesh Diagnostics */}
        {result && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-slate-800 animate-fadeIn">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[11px] text-slate-400 block font-mono">SAM Confidence</span>
              <span className="text-xl font-bold font-heading text-emerald-400">{result.confidenceScore}%</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[11px] text-slate-400 block font-mono">Spatial Vertices</span>
              <span className="text-xl font-bold font-heading text-cyan-400">{result.vertexCount.toLocaleString()}</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[11px] text-slate-400 block font-mono">Polygon Faces</span>
              <span className="text-xl font-bold font-heading text-purple-400">{result.faceCount.toLocaleString()}</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[11px] text-slate-400 block font-mono">Scale Calibration</span>
              <span className="text-xl font-bold font-heading text-amber-400">1.000 ({heightCm}cm)</span>
            </div>
          </div>
        )}

        {/* CTA */}
        {result && (
          <div className="pt-2 flex justify-end">
            <button
              onClick={() => onComplete(result)}
              className="btn-neon px-6 py-3"
            >
              <span>View Extracted Measurements</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
