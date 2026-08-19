import React, { useMemo } from 'react';
import * as THREE from 'three';
import { createParametricHumanGeometry } from '../../services/measurementEngine';

export function DynamicHumanMesh({ bodyParams, showMarkers = true }) {
  // Regenerate geometry whenever height, chest, waist, or hip parameters change
  const geometry = useMemo(() => {
    return createParametricHumanGeometry(bodyParams);
  }, [bodyParams.height, bodyParams.chest, bodyParams.waist, bodyParams.hip, bodyParams.shoulderWidth]);

  // Marker ring Y positions relative to body height scale
  const heightScale = bodyParams.height / 172;

  const bustY = 0.45 * heightScale;
  const waistY = 0.12 * heightScale;
  const hipY = -0.30 * heightScale;

  const bustRadius = (bodyParams.chest / (2 * Math.PI * 10)) * 1.15;
  const waistRadius = (bodyParams.waist / (2 * Math.PI * 10)) * 1.15;
  const hipRadius = (bodyParams.hip / (2 * Math.PI * 10)) * 1.15;

  return (
    <group position={[0, 0, 0]}>
      {/* 3D Human Avatar Body */}
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          color="#334155"
          metalness={0.6}
          roughness={0.3}
          wireframe={false}
        />
      </mesh>

      {/* Subtle Avatar Wireframe Outline for Tech Aesthetic */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          color="#00F0FF"
          wireframe={true}
          transparent={true}
          opacity={0.06}
        />
      </mesh>

      {/* Anatomical Marker Rings */}
      {showMarkers && (
        <group>
          {/* Bust Ring */}
          <mesh position={[0, bustY, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[bustRadius, 0.006, 16, 64]} />
            <meshBasicMaterial color="#FF2E93" transparent opacity={0.8} />
          </mesh>

          {/* Waist Ring */}
          <mesh position={[0, waistY, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[waistRadius, 0.006, 16, 64]} />
            <meshBasicMaterial color="#00F0FF" transparent opacity={0.8} />
          </mesh>

          {/* Hip Ring */}
          <mesh position={[0, hipY, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[hipRadius, 0.006, 16, 64]} />
            <meshBasicMaterial color="#10B981" transparent opacity={0.8} />
          </mesh>
        </group>
      )}
    </group>
  );
}
