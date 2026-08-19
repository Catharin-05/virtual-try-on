import React, { useState } from 'react';
import { Ruler, Shirt, Activity, Info } from 'lucide-react';
import { GARMENT_CATALOG } from '../../data/garmentsData';

export function ControlPanel({ 
  measurements, 
  setMeasurements, 
  selectedGarment, 
  setSelectedGarment, 
  selectedSize, 
  setSelectedSize,
  viewMode,
  setViewMode
}) {
  const [activeTab, setActiveTab] = useState('measurements');

  const handleMeasurementChange = (key, value) => {
    setMeasurements(prev => ({ ...prev, [key]: Number(value) }));
  };

  const calculateClearance = () => {
    // Simple mock calculation based on garment size minus body size
    const gSize = selectedGarment.sizes[selectedSize];
    if (!gSize) return { bust: 0, waist: 0, hips: 0 };
    return {
      bust: gSize.chest - measurements.bust,
      waist: gSize.waist - measurements.waist,
      hips: gSize.hip - measurements.hips
    };
  };

  const clearance = calculateClearance();

  const getFitStatus = (gap) => {
    if (gap < 0) return { label: 'Tight', color: '#FF007F' };
    if (gap <= 5) return { label: 'Optimal', color: '#00FF88' };
    return { label: 'Loose', color: '#00F0FF' };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Tabs */}
      <div className="tabs-header">
        <button 
          className={`tab-btn ${activeTab === 'measurements' ? 'active' : ''}`}
          onClick={() => setActiveTab('measurements')}
        >
          <Ruler size={16} style={{ marginBottom: '-3px', marginRight: '6px' }} /> Config
        </button>
        <button 
          className={`tab-btn ${activeTab === 'wardrobe' ? 'active' : ''}`}
          onClick={() => setActiveTab('wardrobe')}
        >
          <Shirt size={16} style={{ marginBottom: '-3px', marginRight: '6px' }} /> Wardrobe
        </button>
        <button 
          className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          <Activity size={16} style={{ marginBottom: '-3px', marginRight: '6px' }} /> Analysis
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        
        {/* MEASUREMENTS TAB */}
        {activeTab === 'measurements' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ padding: '12px', background: 'rgba(0, 240, 255, 0.05)', borderRadius: '12px', border: '1px solid rgba(0, 240, 255, 0.2)' }}>
              <p className="text-xs text-muted" style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                <Info size={14} color="var(--primary)" /> 
                Adjusting sliders morphs the 3D twin in real-time.
              </p>
            </div>

            <div className="form-group">
              <div className="form-label">
                <span>Height</span>
                <span className="form-value">{measurements.height} cm</span>
              </div>
              <input 
                type="range" min="140" max="210" 
                value={measurements.height} 
                onChange={(e) => handleMeasurementChange('height', e.target.value)} 
              />
            </div>
            <div className="form-group">
              <div className="form-label">
                <span>Bust</span>
                <span className="form-value">{measurements.bust} cm</span>
              </div>
              <input 
                type="range" min="70" max="130" 
                value={measurements.bust} 
                onChange={(e) => handleMeasurementChange('bust', e.target.value)} 
              />
            </div>
            <div className="form-group">
              <div className="form-label">
                <span>Waist</span>
                <span className="form-value">{measurements.waist} cm</span>
              </div>
              <input 
                type="range" min="50" max="110" 
                value={measurements.waist} 
                onChange={(e) => handleMeasurementChange('waist', e.target.value)} 
              />
            </div>
            <div className="form-group">
              <div className="form-label">
                <span>Hips</span>
                <span className="form-value">{measurements.hips} cm</span>
              </div>
              <input 
                type="range" min="70" max="140" 
                value={measurements.hips} 
                onChange={(e) => handleMeasurementChange('hips', e.target.value)} 
              />
            </div>
          </div>
        )}

        {/* WARDROBE TAB */}
        {activeTab === 'wardrobe' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div>
              <label className="form-label">Select Garment</label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {GARMENT_CATALOG.map(garment => (
                  <button
                    key={garment.id}
                    onClick={() => setSelectedGarment(garment)}
                    style={{
                      background: selectedGarment.id === garment.id ? 'rgba(0, 240, 255, 0.1)' : 'transparent',
                      border: `1px solid ${selectedGarment.id === garment.id ? 'var(--primary)' : 'var(--border-light)'}`,
                      padding: '12px',
                      borderRadius: '12px',
                      textAlign: 'left',
                      color: 'var(--text-main)',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    <div style={{ fontSize: '0.9rem', fontWeight: 600, fontFamily: 'var(--font-heading)' }}>{garment.name}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{garment.fabric}</div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="form-label">Select Size</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {['XS', 'S', 'M', 'L', 'XL'].map(size => (
                  <button
                    key={size}
                    onClick={() => setSelectedSize(size)}
                    style={{
                      flex: 1,
                      padding: '10px 0',
                      background: selectedSize === size ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                      color: selectedSize === size ? 'var(--bg-deep)' : 'var(--text-main)',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: 600,
                      fontFamily: 'var(--font-heading)'
                    }}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ANALYSIS TAB */}
        {activeTab === 'analysis' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '4px' }}>
              <button 
                onClick={() => setViewMode('heatmap')}
                style={{
                  flex: 1, padding: '8px', border: 'none', borderRadius: '8px', cursor: 'pointer',
                  background: viewMode === 'heatmap' ? 'rgba(255,255,255,0.1)' : 'transparent',
                  color: viewMode === 'heatmap' ? 'white' : 'var(--text-muted)'
                }}
              >Tension Heatmap</button>
              <button 
                onClick={() => setViewMode('realistic')}
                style={{
                  flex: 1, padding: '8px', border: 'none', borderRadius: '8px', cursor: 'pointer',
                  background: viewMode === 'realistic' ? 'rgba(255,255,255,0.1)' : 'transparent',
                  color: viewMode === 'realistic' ? 'white' : 'var(--text-muted)'
                }}
              >Fabric Render</button>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-light)' }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '0.9rem' }}>Fit Clearance Report</h4>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {['bust', 'waist', 'hips'].map(zone => {
                  const gap = clearance[zone];
                  const status = getFitStatus(gap);
                  return (
                    <div key={zone} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <div style={{ textTransform: 'capitalize', fontSize: '0.85rem' }}>{zone}</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span className="text-mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                          {gap > 0 ? `+${gap}` : gap} cm
                        </span>
                        <span style={{ 
                          fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px', borderRadius: '4px',
                          background: `${status.color}20`, color: status.color
                        }}>
                          {status.label}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}
