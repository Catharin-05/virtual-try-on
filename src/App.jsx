import React, { useState } from 'react';
import { Shirt, Cpu } from 'lucide-react';
import { FittingCanvas } from './components/3d/FittingCanvas';
import { ControlPanel } from './components/ui/ControlPanel';
import { GARMENT_CATALOG } from './data/garmentsData';

export default function App() {
  // Base measurements that drive the 3D twin
  const [measurements, setMeasurements] = useState({
    height: 172,
    bust: 88,
    waist: 68,
    hips: 94
  });

  // Wardrobe selection
  const [selectedGarment, setSelectedGarment] = useState(GARMENT_CATALOG[0]);
  const [selectedSize, setSelectedSize] = useState('M');
  
  // View options
  const [viewMode, setViewMode] = useState('heatmap'); // 'heatmap' or 'realistic'

  return (
    <div className="dashboard-layout">
      {/* Header */}
      <header className="dashboard-header glass-panel">
        <div className="flex-between" style={{ gap: '12px' }}>
          <div style={{
            width: '36px', height: '36px', borderRadius: '10px',
            background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Cpu size={20} color="var(--bg-deep)" />
          </div>
          <div>
            <h1 className="text-heading" style={{ fontSize: '1.1rem', margin: 0 }}>
              AURA <span className="gradient-text">DIGITAL TWIN</span>
            </h1>
            <p className="text-xs text-muted" style={{ margin: 0 }}>Meta MHR / SAM3D Visualizer</p>
          </div>
        </div>
        <div>
          <button className="btn-secondary">
            <Shirt size={16} /> 3D Wardrobe Active
          </button>
        </div>
      </header>

      {/* 3D Viewport (Left) */}
      <main className="dashboard-3d-view glass-panel">
        <FittingCanvas 
          measurements={measurements}
          selectedGarment={selectedGarment}
          selectedSize={selectedSize}
          viewMode={viewMode}
        />
      </main>

      {/* Control Panel (Right) */}
      <aside className="dashboard-controls glass-panel" style={{ padding: '20px' }}>
        <ControlPanel 
          measurements={measurements}
          setMeasurements={setMeasurements}
          selectedGarment={selectedGarment}
          setSelectedGarment={setSelectedGarment}
          selectedSize={selectedSize}
          setSelectedSize={setSelectedSize}
          viewMode={viewMode}
          setViewMode={setViewMode}
        />
      </aside>
    </div>
  );
}
