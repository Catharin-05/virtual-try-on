import React, { useState } from 'react';
import { Upload, Video, Ruler, Sparkles, Check, Play, Info } from 'lucide-react';
import { SAMPLE_VIDEOS } from '../../services/sam3dSimulator';

export function Step1VideoUpload({ onNext, heightCm, setHeightCm, selectedVideo, setSelectedVideo }) {
  const [dragActive, setDragActive] = useState(false);
  const [unit, setUnit] = useState('cm'); // 'cm' or 'in'

  const handleHeightChange = (val) => {
    if (unit === 'in') {
      const cmVal = Math.round(Number(val) * 2.54);
      setHeightCm(cmVal);
    } else {
      setHeightCm(Number(val));
    }
  };

  const currentHeightInputVal = unit === 'in' ? Math.round(heightCm / 2.54) : heightCm;

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fadeIn">
      {/* Intro Header */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl md:text-3xl font-extrabold font-heading text-white">
          Upload Your <span className="gradient-text-primary">360° Body Rotation Video</span>
        </h2>
        <p className="text-slate-400 text-sm max-w-xl mx-auto">
          SAM 3D extracts multi-angle keyframes from your turn video and uses your exact height to generate a photorealistic 1:1 scaled 3D body model.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Left Column: Video Dropzone & Sample Picker (7 cols) */}
        <div className="md:col-span-7 space-y-4">
          <div
            className={`glass-panel p-8 text-center border-2 border-dashed transition-all cursor-pointer ${
              dragActive ? 'border-cyan-400 bg-cyan-500/10' : 'border-slate-700 hover:border-slate-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                setSelectedVideo({
                  id: 'uploaded-custom',
                  name: e.dataTransfer.files[0].name,
                  duration: '00:10',
                  quality: 'User Upload',
                  thumbnail: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=400&q=80'
                });
              }
            }}
          >
            <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-700 flex items-center justify-center mx-auto mb-4 text-cyan-400">
              <Upload className="w-8 h-8" />
            </div>
            <h3 className="font-heading font-bold text-lg text-white mb-1">
              Drag & Drop your 360° Video
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Supported formats: MP4, MOV, WebM (Full body visible, 360° rotation)
            </p>
            <label className="btn-outline text-xs px-4 py-2 cursor-pointer inline-flex items-center gap-2">
              <Video className="w-4 h-4 text-cyan-400" />
              Browse Local Video File
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setSelectedVideo({
                      id: 'uploaded-custom',
                      name: e.target.files[0].name,
                      duration: '00:10',
                      quality: 'User Upload',
                      thumbnail: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=400&q=80'
                    });
                  }
                }}
              />
            </label>
          </div>

          {/* Sample Videos */}
          <div className="glass-panel p-4">
            <h4 className="text-xs font-mono font-semibold text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" /> Or select a pre-loaded studio demo video:
            </h4>
            <div className="grid grid-cols-2 gap-3">
              {SAMPLE_VIDEOS.map((vid) => {
                const isSel = selectedVideo?.id === vid.id;
                return (
                  <div
                    key={vid.id}
                    onClick={() => setSelectedVideo(vid)}
                    className={`p-2.5 rounded-xl border transition-all cursor-pointer flex items-center gap-3 ${
                      isSel
                        ? 'bg-cyan-500/10 border-cyan-400 shadow-md shadow-cyan-400/10'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="w-12 h-12 rounded-lg bg-slate-800 overflow-hidden relative flex-shrink-0">
                      <img src={vid.thumbnail} alt={vid.name} className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                        <Play className="w-4 h-4 text-white fill-white" />
                      </div>
                    </div>
                    <div className="overflow-hidden">
                      <h5 className="text-xs font-semibold text-white truncate">{vid.name}</h5>
                      <span className="text-[10px] text-slate-400 font-mono">{vid.quality} • {vid.duration}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Height Input & Calibration Specs (5 cols) */}
        <div className="md:col-span-5 space-y-4">
          <div className="glass-panel p-6 space-y-5">
            <div className="flex items-center gap-2 text-cyan-400">
              <Ruler className="w-5 h-5" />
              <h3 className="font-heading font-bold text-lg text-white">Body Height Calibration</h3>
            </div>

            <p className="text-xs text-slate-400">
              Your height acts as the ground truth scale anchor for SAM 3D to convert pixel mesh geometry into real physical centimeters.
            </p>

            {/* Height Selector Slider + Input */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono text-slate-300">Target Height:</label>
                <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
                  <button
                    onClick={() => setUnit('cm')}
                    className={`px-2.5 py-0.5 rounded text-xs font-mono font-semibold transition-all ${
                      unit === 'cm' ? 'bg-cyan-400 text-slate-950' : 'text-slate-400'
                    }`}
                  >
                    CM
                  </button>
                  <button
                    onClick={() => setUnit('in')}
                    className={`px-2.5 py-0.5 rounded text-xs font-mono font-semibold transition-all ${
                      unit === 'in' ? 'bg-cyan-400 text-slate-950' : 'text-slate-400'
                    }`}
                  >
                    INCH
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <input
                  type="number"
                  value={currentHeightInputVal}
                  onChange={(e) => handleHeightChange(e.target.value)}
                  className="w-28 px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-cyan-400 font-mono font-bold text-xl text-center focus:outline-none focus:border-cyan-400"
                />
                <span className="font-heading font-semibold text-slate-300 text-sm">{unit.toUpperCase()}</span>
              </div>

              <input
                type="range"
                min={unit === 'cm' ? 140 : 55}
                max={unit === 'cm' ? 205 : 81}
                value={currentHeightInputVal}
                onChange={(e) => handleHeightChange(e.target.value)}
                className="w-full accent-cyan-400 cursor-pointer"
              />
            </div>

            {/* Best Practice Tips */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-2">
              <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                <Info className="w-4 h-4" />
                <span>Optimal Video Tips</span>
              </div>
              <ul className="text-[11px] text-slate-400 space-y-1 list-disc list-inside">
                <li>Wear fitted clothing or activewear for best silhouette accuracy.</li>
                <li>Rotate slowly 360° over 8 to 12 seconds.</li>
                <li>Ensure full body from head to feet is inside the camera frame.</li>
              </ul>
            </div>

            {/* Next CTA */}
            <button
              onClick={onNext}
              disabled={!selectedVideo}
              className="w-full btn-neon justify-center py-3.5 mt-2"
            >
              <span>Process Video with SAM 3D</span>
              <Sparkles className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
