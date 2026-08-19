import * as THREE from 'three';

/**
 * Calculates anthropometric proportions from base user height (cm)
 */
export function calculateDerivedMeasurements(heightCm = 172) {
  const h = Number(heightCm);
  return {
    height: h,
    chest: Math.round(h * 0.512),       // ~88cm for 172cm height
    waist: Math.round(h * 0.395),       // ~68cm
    hip: Math.round(h * 0.546),         // ~94cm
    shoulderWidth: Math.round(h * 0.228),// ~39cm
    inseam: Math.round(h * 0.448),      // ~77cm
    torsoLength: Math.round(h * 0.31),  // ~53cm
    neckCircumference: Math.round(h * 0.20), // ~34cm
    thighCircumference: Math.round(h * 0.31) // ~53cm
  };
}

/**
 * Generates a parametric 3D Human Body BufferGeometry for Three.js
 */
export function createParametricHumanGeometry(params) {
  const { height = 172, chest = 88, waist = 68, hip = 94, shoulderWidth = 39 } = params;
  
  // Height scaling factor: Standard base height model is 1.75 meters in Three.js units
  const heightScale = height / 172;
  const radiusScale = (val, baseVal) => (val / baseVal);

  const chestScale = radiusScale(chest, 88);
  const waistScale = radiusScale(waist, 68);
  const hipScale = radiusScale(hip, 94);
  const shoulderScale = radiusScale(shoulderWidth, 39);

  // Resolution parameters
  const radialSegments = 32;
  const heightSegments = 64;

  const geometry = new THREE.CylinderGeometry(
    0.15 * shoulderScale, // top radius (neck/shoulders)
    0.12 * hipScale,      // bottom radius (legs)
    1.65 * heightScale,
    radialSegments,
    heightSegments,
    true
  );

  const posAttr = geometry.attributes.position;
  const count = posAttr.count;

  // Deform cylinder to realistic anatomical silhouette (Bust, Waist, Hips, Legs)
  for (let i = 0; i < count; i++) {
    let x = posAttr.getX(i);
    let y = posAttr.getY(i);
    let z = posAttr.getZ(i);

    // Normalized height yNorm from -1 (feet/hips) to +1 (neck/head)
    const yNorm = y / (0.825 * heightScale);

    let currentScale = 1.0;

    if (yNorm > 0.4) {
      // Chest / Bust region (yNorm ~ 0.4 to 0.7)
      const bustFactor = Math.sin((yNorm - 0.4) / 0.3 * Math.PI);
      currentScale = 0.85 + 0.35 * chestScale * (1 + 0.2 * bustFactor);
      z *= (1 + 0.25 * chestScale * bustFactor); // front projection for bust
    } else if (yNorm >= -0.1 && yNorm <= 0.4) {
      // Waist region (yNorm ~ -0.1 to 0.4)
      const waistFactor = Math.cos(((yNorm - 0.15) / 0.25) * (Math.PI / 2));
      currentScale = 0.65 + 0.35 * waistScale * (1 - 0.15 * (1 - waistFactor));
    } else if (yNorm >= -0.6 && yNorm < -0.1) {
      // Hips region (yNorm ~ -0.6 to -0.1)
      const hipFactor = Math.sin(((-yNorm - 0.1) / 0.5) * Math.PI);
      currentScale = 0.85 + 0.35 * hipScale * (1 + 0.2 * hipFactor);
      x *= (1 + 0.15 * hipScale * hipFactor); // wider hip curve
    } else {
      // Legs / Thigh region
      const legFactor = Math.max(0, (yNorm + 1.0) / 0.4);
      currentScale = 0.6 + 0.25 * legFactor;
    }

    // Apply anatomic radial displacement
    posAttr.setX(i, x * currentScale);
    posAttr.setY(i, y);
    posAttr.setZ(i, z * currentScale);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Computes exact clearance delta and vertex colors for Garment Tension Heatmap
 * Red: Tight (Stretch), Green: Perfect, Blue: Loose
 */
export function computeFitClearance(bodyParams, garmentData, sizeKey) {
  const garmentSize = garmentData.sizes[sizeKey] || garmentData.sizes['M'];
  
  const chestDelta = garmentSize.chest - bodyParams.chest;
  const waistDelta = garmentSize.waist - bodyParams.waist;
  const hipDelta = garmentSize.hip - bodyParams.hip;

  const assessZone = (delta, elasticityFactor = 1.0) => {
    if (delta < -2 * elasticityFactor) return { status: 'Very Tight', color: '#EF4444', score: 20 };
    if (delta < 1 * elasticityFactor) return { status: 'Slightly Tight', color: '#F59E0B', score: 65 };
    if (delta <= 6 * elasticityFactor) return { status: 'Optimal Fit', color: '#10B981', score: 98 };
    if (delta <= 12 * elasticityFactor) return { status: 'Relaxed Fit', color: '#3B82F6', score: 85 };
    return { status: 'Loose / Oversized', color: '#6366F1', score: 50 };
  };

  const isHighStretch = garmentData.elasticity.includes('High');
  const elast = isHighStretch ? 1.8 : 1.0;

  const chestFit = assessZone(chestDelta, elast);
  const waistFit = assessZone(waistDelta, elast);
  const hipFit = assessZone(hipDelta, elast);

  const overallScore = Math.round((chestFit.score + waistFit.score + hipFit.score) / 3);

  let recommendation = 'Perfect fit! Matches silhouette comfortably.';
  if (overallScore < 60) {
    recommendation = 'Consider sizing up for a less restrictive fit across key tension points.';
  } else if (chestFit.status.includes('Tight') || hipFit.status.includes('Tight')) {
    recommendation = `Fits well overall, but tight at the ${chestFit.status.includes('Tight') ? 'Bust' : 'Hips'}.`;
  }

  return {
    chest: { delta: chestDelta, ...chestFit },
    waist: { delta: waistDelta, ...waistFit },
    hip: { delta: hipDelta, ...hipFit },
    overallScore,
    recommendation
  };
}
