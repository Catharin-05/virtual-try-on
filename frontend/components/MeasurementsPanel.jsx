const ROWS = [
  { key: 'shoulder_width_cm', label: 'Shoulder width', kind: 'length' },
  { key: 'sleeve_length_cm', label: 'Sleeve length', kind: 'length' },
  { key: 'inseam_cm', label: 'Inseam', kind: 'length' },
  { key: 'torso_length_cm', label: 'Torso length', kind: 'length' },
  { key: 'chest_bust', label: 'Chest / bust', kind: 'circumference' },
  { key: 'waist', label: 'Waist', kind: 'circumference' },
  { key: 'hips', label: 'Hips', kind: 'circumference' },
  { key: 'neck', label: 'Neck', kind: 'circumference' },
  { key: 'thigh', label: 'Thigh', kind: 'circumference' },
];

function readValue(measurements, row) {
  if (!measurements) return null;
  const raw = measurements[row.key];
  if (row.kind === 'length') return typeof raw === 'number' ? raw : null;
  if (raw && typeof raw === 'object') return raw.circumference_cm ?? null;
  return null;
}

export default function MeasurementsPanel({ measurements, onReset }) {
  return (
    <aside className="measurements-panel">
      <div className="panel-heading">
        <span className="eyebrow">your measurements</span>
        <h2>Digital twin, ready</h2>
      </div>

      <div className="measurement-list">
        {ROWS.map((row) => {
          const value = readValue(measurements, row);
          return (
            <div className="measurement-row" key={row.key}>
              <span className="m-label">{row.label}</span>
              <span className={`m-value mono ${value === null ? 'unavailable' : ''}`}>
                {value === null ? '—' : `${value.toFixed(1)} cm`}
              </span>
            </div>
          );
        })}
      </div>

      <button className="cta ghost full-width" onClick={onReset}>Scan again</button>
    </aside>
  );
}
