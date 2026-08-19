import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import { MannequinTwin } from './MannequinTwin';
import { GarmentLayer } from './GarmentLayer';

export function FittingCanvas({ measurements, selectedGarment, selectedSize, viewMode }) {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ position: [0, 1, 4], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={['#05080F']} />
        
        {/* Cinematic Lighting */}
        <ambientLight intensity={0.2} />
        <spotLight position={[5, 5, 5]} intensity={2} color="#00F0FF" angle={0.5} penumbra={1} castShadow />
        <spotLight position={[-5, 5, -5]} intensity={2} color="#FF007F" angle={0.5} penumbra={1} />
        <directionalLight position={[0, 2, 2]} intensity={1} color="#ffffff" />

        <Suspense fallback={null}>
          <group position={[0, -1, 0]}>
            {/* The base Digital Twin */}
            <MannequinTwin measurements={measurements} />
            
            {/* The Garment Layer overlaid */}
            <GarmentLayer 
              measurements={measurements} 
              garment={selectedGarment} 
              size={selectedSize} 
              viewMode={viewMode} 
            />

            <ContactShadows position={[0, 0, 0]} opacity={0.8} scale={10} blur={2} far={4} color="#00F0FF" />
          </group>
          <Environment preset="city" />
        </Suspense>

        <OrbitControls 
          enablePan={false} 
          minDistance={2} 
          maxDistance={6}
          maxPolarAngle={Math.PI / 2 + 0.1}
          target={[0, 0.8, 0]}
        />

        {/* Post Processing for Premium Glow */}
        <EffectComposer disableNormalPass>
          <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} />
          <Vignette eskil={false} offset={0.1} darkness={1.1} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
