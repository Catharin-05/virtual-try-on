import * as THREE from 'three';

export const FitHeatmapMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uChestDelta: { value: 0.0 },
    uWaistDelta: { value: 0.0 },
    uHipDelta: { value: 0.0 },
    uHeatmapActive: { value: 1.0 }, // 1.0 = Heatmap, 0.0 = Real Fabric Shader
    uBaseColor: { value: new THREE.Color('#00F0FF') },
    uRoughness: { value: 0.4 }
  },
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vUv;

    void main() {
      vNormal = normalize(normalMatrix * normal);
      vPosition = position;
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float uChestDelta;
    uniform float uWaistDelta;
    uniform float uHipDelta;
    uniform float uHeatmapActive;
    uniform vec3 uBaseColor;

    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vUv;

    void main() {
      // Light directional angle
      vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
      float diff = max(dot(vNormal, lightDir), 0.25);

      if (uHeatmapActive < 0.5) {
        // Standard fabric render
        vec3 finalCol = uBaseColor * (diff + 0.3);
        gl_FragColor = vec4(finalCol, 0.92);
        return;
      }

      // Height-based zone estimation (vPosition.y ranges from -1.0 to 1.0)
      float yNorm = vPosition.y;
      float clearance = 0.0;

      if (yNorm > 0.3) {
        clearance = uChestDelta;
      } else if (yNorm > -0.1) {
        float mixRatio = (yNorm - (-0.1)) / 0.4;
        clearance = mix(uWaistDelta, uChestDelta, mixRatio);
      } else if (yNorm > -0.5) {
        float mixRatio = (yNorm - (-0.5)) / 0.4;
        clearance = mix(uHipDelta, uWaistDelta, mixRatio);
      } else {
        clearance = uHipDelta + 2.0; // skirt drape loose zone
      }

      // Color mapping:
      // clearance < 0.0  -> Red (Tight / Stretch)
      // clearance 0 to 4 -> Green (Optimal Fit)
      // clearance > 4.0  -> Blue (Loose / Drape)
      vec3 heatmapColor;

      if (clearance < 0.0) {
        float t = clamp(-clearance / 5.0, 0.0, 1.0);
        heatmapColor = mix(vec3(0.93, 0.26, 0.26), vec3(0.96, 0.62, 0.04), 1.0 - t); // Red to Orange
      } else if (clearance <= 5.0) {
        float t = clearance / 5.0;
        heatmapColor = mix(vec3(0.06, 0.72, 0.50), vec3(0.23, 0.51, 0.96), t); // Green to Cyan-Blue
      } else {
        float t = clamp((clearance - 5.0) / 10.0, 0.0, 1.0);
        heatmapColor = mix(vec3(0.23, 0.51, 0.96), vec3(0.38, 0.30, 0.86), t); // Blue to Indigo
      }

      // Blend with subtle ambient lighting
      vec3 colorWithLight = heatmapColor * (diff * 0.75 + 0.35);
      gl_FragColor = vec4(colorWithLight, 0.92);
    }
  `,
  transparent: true,
  side: THREE.DoubleSide
});
