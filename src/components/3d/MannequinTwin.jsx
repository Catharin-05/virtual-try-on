import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

export function MannequinTwin({ measurements }) {
  const materialRef = useRef();

  // Procedurally generate a highly detailed, smooth mannequin torso based on measurements
  const geometry = useMemo(() => {
    const points = [];
    const segments = 100;
    
    // Normalize measurements
    const heightFactor = measurements.height / 170;
    const bustR = (measurements.bust / (2 * Math.PI)) * 0.1;
    const waistR = (measurements.waist / (2 * Math.PI)) * 0.1;
    const hipsR = (measurements.hips / (2 * Math.PI)) * 0.1;

    // Create a smooth spline for the body profile
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const y = (t * 2 - 1) * 0.8 * heightFactor; // From -0.8 to 0.8
      let r = 0;

      // Mathematical shaping for a realistic torso silhouette
      if (t < 0.2) {
        // Legs to Hips
        const localT = t / 0.2;
        r = THREE.MathUtils.lerp(hipsR * 0.7, hipsR, Math.sin(localT * Math.PI / 2));
      } else if (t < 0.5) {
        // Hips to Waist
        const localT = (t - 0.2) / 0.3;
        r = THREE.MathUtils.lerp(hipsR, waistR, Math.sin(localT * Math.PI / 2));
      } else if (t < 0.8) {
        // Waist to Bust
        const localT = (t - 0.5) / 0.3;
        r = THREE.MathUtils.lerp(waistR, bustR, Math.sin(localT * Math.PI / 2));
        // Add bust prominence on the front (handled in shader or by scaling, Lathe is radially symmetric.
        // We will make it slightly wider to compensate).
      } else {
        // Bust to Neck
        const localT = (t - 0.8) / 0.2;
        r = THREE.MathUtils.lerp(bustR, bustR * 0.4, Math.pow(localT, 2));
      }

      points.push(new THREE.Vector2(r, y));
    }

    const geo = new THREE.LatheGeometry(points, 64);
    
    // Slightly flatten the Z axis (depth) to look more human, less cylindrical
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const z = positions.getZ(i);
      positions.setZ(i, z * 0.65); // Humans are wider than they are deep
    }
    
    geo.computeVertexNormals();
    return geo;
  }, [measurements]);

  // Animate the holographic scanline
  useFrame(({ clock }) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = clock.getElapsedTime();
    }
  });

  // Premium Holographic Shader
  const shaderUniforms = useMemo(() => ({
    time: { value: 0 },
    color: { value: new THREE.Color('#00F0FF') }
  }), []);

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
    uniform float time;
    uniform vec3 color;
    varying vec2 vUv;
    varying vec3 vPosition;
    varying vec3 vNormal;
    
    void main() {
      // Edge glow (Fresnel effect)
      vec3 viewDir = normalize(-vPosition);
      float fresnel = dot(viewDir, vNormal);
      fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
      fresnel = pow(fresnel, 3.0);
      
      // Scanning lines
      float scanline = sin(vPosition.y * 50.0 - time * 2.0) * 0.5 + 0.5;
      scanline = pow(scanline, 5.0) * 0.5;
      
      // Base glass tint
      float alpha = 0.15 + fresnel * 0.8 + scanline;
      vec3 finalColor = color * (fresnel + 0.2) + (scanline * vec3(1.0));
      
      gl_FragColor = vec4(finalColor, alpha);
    }
  `;

  return (
    <group position={[0, 0.8, 0]}>
      {/* Inner glowing core */}
      <mesh geometry={geometry}>
        <meshBasicMaterial color="#001122" transparent opacity={0.8} />
      </mesh>
      
      {/* Outer Holographic Shell */}
      <mesh geometry={geometry}>
        <shaderMaterial
          ref={materialRef}
          uniforms={shaderUniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          transparent={true}
          side={THREE.DoubleSide}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}
