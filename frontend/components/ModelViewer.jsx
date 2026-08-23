import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export default function ModelViewer({ objUrl }) {
  const mountRef = useRef(null);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    if (!objUrl || !mountRef.current) return;

    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = null;

    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 5000);
    camera.position.set(0, 100, 350);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 50;
    controls.maxDistance = 1000;

    const key = new THREE.DirectionalLight(0xbfe9ff, 1.1);
    key.position.set(150, 300, 200);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9b7bff, 0.5);
    fill.position.set(-200, 100, -100);
    scene.add(fill);
    scene.add(new THREE.AmbientLight(0x404060, 0.6));

    // Faint reference grid, matches the app's "scan" motif
    const grid = new THREE.GridHelper(600, 24, 0x2a3142, 0x1a1f2b);
    scene.add(grid);

    let mesh;
    let animationId;
    const loader = new OBJLoader();

    loader.load(
      objUrl,
      (obj) => {
        const material = new THREE.MeshStandardMaterial({
          color: 0x5ee6ec,
          metalness: 0.15,
          roughness: 0.55,
          flatShading: false,
        });
        obj.traverse((child) => {
          if (child.isMesh) child.material = material;
        });

        // Center the mesh and sit it on the grid, regardless of the
        // coordinate convention it was exported with.
        const box = new THREE.Box3().setFromObject(obj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        obj.position.x -= center.x;
        obj.position.z -= center.z;
        obj.position.y -= box.min.y;

        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 180 / maxDim; // normalize to a consistent on-screen size
        obj.scale.setScalar(scale);

        scene.add(obj);
        mesh = obj;

        camera.position.set(0, size.y * scale * 0.55, maxDim * scale * 1.3);
        controls.target.set(0, size.y * scale * 0.4, 0);
        controls.update();
      },
      undefined,
      (err) => {
        console.error('Failed to load OBJ model:', err);
        setLoadError('Could not load the 3D model. It may still be processing, or the file is malformed.');
      }
    );

    const animate = () => {
      animationId = requestAnimationFrame(animate);
      if (mesh) mesh.rotation.y += 0.0025;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, [objUrl]);

  return (
    <div className="model-viewer">
      <div ref={mountRef} className="model-canvas" />
      {loadError && <div className="viewer-error">{loadError}</div>}
      <div className="viewer-hint mono">drag to rotate · scroll to zoom</div>
    </div>
  );
}
