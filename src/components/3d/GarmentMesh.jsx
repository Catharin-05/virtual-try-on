import React, { useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { FitHeatmapMaterial } from './FitHeatmapShader';
import { computeFitClearance } from '../../services/measurementEngine';

export function GarmentMesh({ garmentData, sizeKey, bodyParams, showHeatmap = true }) {
  const fitMetrics = useMemo(() => {
    return computeFitClearance(bodyParams, garmentData, sizeKey);
  }, [bodyParams, garmentData, sizeKey]);

  // Compute garment shape geometry matching body scale + dress drape profile
  const garmentGeometry = useMemo(() => {
    const garmentSize = garmentData.sizes[sizeKey] || garmentData.sizes['M'];
    const heightScale = bodyParams.height / 172;
    
    // Scale radii based on dress size specifications vs body scale
    const chestRadius = (garmentSize.chest / (2 * Math.PI * 10)) * 1.18;
    const waistRadius = (garmentSize.waist / (2 * Math.PI * 10)) * 1.18;
    const hipRadius = (garmentSize.hip / (2 * Math.PI * 10)) * 1.25 * (garmentData.drapeFactor || 1.1);

    const lengthScale = (garmentSize.length / 110) * 1.25 * heightScale;

    // Create 3D Dress Silhouette
    const geometry = new THREE.CylinderGeometry(
      chestRadius,
      hipRadius,
      lengthScale,
      36,
      48,
      true
    );

    const posAttr = geometry.attributes.position;
    const count = posAttr.count;

    for (let i = 0; i < count; i++) {
      let x = posAttr.getX(i);
      let y = posAttr.getY(i);
      let z = posAttr.getZ(i);

      const yNorm = y / (lengthScale * 0.5);

      if (yNorm > 0.2) {
        // Upper bodice section
        const factor = (yNorm - 0.2) / 0.8;
        x *= (1 - 0.05 * factor);
        z *= (1 + 0.1 * factor);
      } else if (yNorm >= -0.3 && yNorm <= 0.2) {
        // Waist cinching zone
        const waistFactor = Math.cos(((yNorm + 0.05) / 0.25) * (Math.PI / 2));
        const waistRatio = waistRadius / chestRadius;
        x *= mix(1.0, waistRatio, waistFactor);
        z *= mix(1.0, waistRatio, waistFactor);
      } else {
        // Skirt flare / drape zone
        const skirtFactor = Math.abs(yNorm + 0.3) / 0.7;
        const flare = 1.0 + (garmentData.drapeFactor - 1.0) * skirtFactor;
        x *= flare;
        z *= flare;
      }

      posAttr.setX(i, x);
      posAttr.setY(i, y);
      posAttr.setZ(i, z);
    }

    geometry.computeVertexNormals();
    return geometry;
  }, [garmentData, sizeKey, bodyParams.height]);

  // Helper linear interpolation
  function mix(a, b, t) {
    return a * (1 - t) + b * t;
  }

  // Update shader uniforms when fit clearances change
  useEffect(() => {
    if (FitHeatmapMaterial) {
      FitHeatmapMaterial.uniforms.uChestDelta.value = fitMetrics.chest.delta;
      FitHeatmapMaterial.uniforms.uWaistDelta.value = fitMetrics.waist.delta;
      FitHeatmapMaterial.uniforms.uHipDelta.value = fitMetrics.hip.delta;
      FitHeatmapMaterial.uniforms.uHeatmapActive.value = showHeatmap ? 1.0 : 0.0;
      FitHeatmapMaterial.uniforms.uBaseColor.value = new THREE.Color(garmentData.baseColor || '#00F0FF');
    }
  }, [fitMetrics, showHeatmap, garmentData.baseColor]);

  const heightScale = bodyParams.height / 172;
  const dressCenterY = 0.1 * heightScale;

  return (
    <group position={[0, dressCenterY, 0]}>
      <mesh geometry={garmentGeometry} material={FitHeatmapMaterial} castShadow receiveShadow />
    </group>
  );
}
