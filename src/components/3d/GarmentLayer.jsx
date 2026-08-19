import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

export function GarmentLayer({ measurements, garment, size, viewMode }) {
  const materialRef = useRef();

  // Procedurally generate a garment mesh that drapes over the mannequin
  const geometry = useMemo(() => {
    const points = [];
    const segments = 100;
    
    // Get target garment sizes
    const targetSize = garment.sizes[size];
    if (!targetSize) return new THREE.BufferGeometry();

    const heightFactor = measurements.height / 170;
    
    // The garment radius at key points
    const bustR = (targetSize.chest / (2 * Math.PI)) * 0.1;
    const waistR = (targetSize.waist / (2 * Math.PI)) * 0.1;
    const hipsR = (targetSize.hip / (2 * Math.PI)) * 0.1;

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const y = (t * 2 - 1) * 0.8 * heightFactor;
      let r = 0;

      if (t < 0.2) {
        // Skirt flare (garments drape down)
        const localT = t / 0.2;
        r = THREE.MathUtils.lerp(hipsR * garment.drapeFactor, hipsR, Math.sin(localT * Math.PI / 2));
      } else if (t < 0.5) {
        const localT = (t - 0.2) / 0.3;
        r = THREE.MathUtils.lerp(hipsR, waistR, Math.sin(localT * Math.PI / 2));
      } else if (t < 0.8) {
        const localT = (t - 0.5) / 0.3;
        r = THREE.MathUtils.lerp(waistR, bustR, Math.sin(localT * Math.PI / 2));
      } else {
        const localT = (t - 0.8) / 0.2;
        r = THREE.MathUtils.lerp(bustR, bustR * 0.4, Math.pow(localT, 2));
      }

      points.push(new THREE.Vector2(r, y));
    }

    const geo = new THREE.LatheGeometry(points, 64);
    
    // Flatten Z slightly to match the body
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const z = positions.getZ(i);
      positions.setZ(i, z * 0.68); // Slightly thicker than the body (0.65)
    }
    
    geo.computeVertexNormals();
    return geo;
  }, [measurements, garment, size]);

  // Heatmap calculation
  const shaderUniforms = useMemo(() => {
    const targetSize = garment.sizes[size];
    const bustDelta = targetSize ? targetSize.chest - measurements.bust : 0;
    const waistDelta = targetSize ? targetSize.waist - measurements.waist : 0;
    const hipsDelta = targetSize ? targetSize.hip - measurements.hips : 0;

    return {
      bustDelta: { value: bustDelta },
      waistDelta: { value: waistDelta },
      hipsDelta: { value: hipsDelta },
      isHeatmap: { value: viewMode === 'heatmap' ? 1.0 : 0.0 },
      baseColor: { value: new THREE.Color(garment.baseColor || '#9D4EDD') }
    };
  }, [measurements, garment, size, viewMode]);

  // Update uniforms without re-mounting
  useFrame(() => {
    if (materialRef.current) {
      const targetSize = garment.sizes[size];
      materialRef.current.uniforms.bustDelta.value = targetSize ? targetSize.chest - measurements.bust : 0;
      materialRef.current.uniforms.waistDelta.value = targetSize ? targetSize.waist - measurements.waist : 0;
      materialRef.current.uniforms.hipsDelta.value = targetSize ? targetSize.hip - measurements.hips : 0;
      materialRef.current.uniforms.isHeatmap.value = viewMode === 'heatmap' ? 1.0 : 0.0;
    }
  });

  const vertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    void main() {
      vUv = uv;
      vPosition = position;
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const fragmentShader = `
    uniform float bustDelta;
    uniform float waistDelta;
    uniform float hipsDelta;
    uniform float isHeatmap;
    uniform vec3 baseColor;

    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    void main() {
      vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
      float diff = max(dot(vNormal, lightDir), 0.2);
      
      if (isHeatmap < 0.5) {
        // Realistic rendering mode (Glassy/Fabric)
        vec3 finalCol = baseColor * (diff + 0.5);
        gl_FragColor = vec4(finalCol, 0.85);
        return;
      }

      // Heatmap rendering mode
      float yNorm = vPosition.y; // Roughly -0.8 to 0.8
      float delta = 0.0;
      
      // Interpolate the delta based on Y position (rough zones)
      if (yNorm > 0.3) {
        delta = bustDelta;
      } else if (yNorm > -0.2) {
        float t = (yNorm + 0.2) / 0.5;
        delta = mix(waistDelta, bustDelta, t);
      } else if (yNorm > -0.6) {
        float t = (yNorm + 0.6) / 0.4;
        delta = mix(hipsDelta, waistDelta, t);
      } else {
        delta = hipsDelta + 5.0; // Skirt flare is naturally looser
      }

      vec3 heatColor;
      if (delta < 0.0) {
        heatColor = mix(vec3(1.0, 0.0, 0.3), vec3(1.0, 0.5, 0.0), clamp(1.0 + delta/10.0, 0.0, 1.0)); // Red/Orange (Tight)
      } else if (delta <= 5.0) {
        heatColor = mix(vec3(0.0, 1.0, 0.5), vec3(0.0, 0.8, 1.0), delta / 5.0); // Green/Cyan (Optimal)
      } else {
        heatColor = vec3(0.0, 0.5, 1.0); // Blue (Loose)
      }

      gl_FragColor = vec4(heatColor * (diff + 0.5), 0.85);
    }
  `;

  return (
    <group position={[0, 0.8, 0]}>
      <mesh geometry={geometry}>
        <shaderMaterial
          ref={materialRef}
          uniforms={shaderUniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          transparent={true}
          side={THREE.DoubleSide}
          depthWrite={false}
          blending={THREE.NormalBlending}
        />
      </mesh>
    </group>
  );
}
