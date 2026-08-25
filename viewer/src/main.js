import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

let scene, camera, renderer, controls;
let pointCloud = null;
let measureMode = false;
let measurePoints = [];
let measureLine = null;

init();

function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(30, 30, 30);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const grid = new THREE.GridHelper(100, 50, 0x333355, 0x222244);
    scene.add(grid);

    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);

    setupUpload();
    setupToolbar();
    setupMeasurement();

    window.addEventListener('resize', onResize);
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function setupUpload() {
    const overlay = document.getElementById('upload-overlay');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) loadFile(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
    });
}

function loadFile(file) {
    const loader = new PLYLoader();
    const reader = new FileReader();

    reader.onload = (e) => {
        const geometry = loader.parse(e.target.result);
        displayPointCloud(geometry);
        document.getElementById('upload-overlay').style.display = 'none';
        document.getElementById('toolbar').style.display = 'flex';
        document.getElementById('info-panel').style.display = 'block';
    };

    reader.readAsArrayBuffer(file);
}

function displayPointCloud(geometry) {
    if (pointCloud) scene.remove(pointCloud);

    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    let material;
    if (geometry.hasAttribute('color')) {
        material = new THREE.PointsMaterial({
            size: 0.08,
            vertexColors: true,
            sizeAttenuation: true,
        });
    } else {
        material = new THREE.PointsMaterial({
            size: 0.08,
            color: 0x6c9cff,
            sizeAttenuation: true,
        });
    }

    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    const size = new THREE.Vector3();
    bbox.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(maxDim, maxDim * 0.8, maxDim);
    controls.target.set(0, 0, 0);
    controls.update();

    updateInfoPanel(geometry, size);
}

function updateInfoPanel(geometry, size) {
    const numPoints = geometry.attributes.position.count;
    document.getElementById('val-points').textContent = numPoints.toLocaleString();
    document.getElementById('val-width').textContent = size.x.toFixed(2) + ' m';
    document.getElementById('val-depth').textContent = size.y.toFixed(2) + ' m';
    document.getElementById('val-height').textContent = size.z.toFixed(2) + ' m';
}

function setupToolbar() {
    document.querySelectorAll('.tool-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tool = btn.dataset.tool;
            if (tool === 'reset') {
                resetView();
                return;
            }
            if (tool === 'measure') {
                toggleMeasure();
            } else if (tool === 'confidence') {
                toggleConfidence();
            } else {
                measureMode = false;
                controls.enabled = true;
            }

            document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function resetView() {
    if (!pointCloud) return;
    const bbox = new THREE.Box3().setFromObject(pointCloud);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(maxDim, maxDim * 0.8, maxDim);
    controls.target.set(0, 0, 0);
    controls.update();
}

function setupMeasurement() {
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.2;
    const mouse = new THREE.Vector2();

    renderer.domElement.addEventListener('click', (e) => {
        if (!measureMode || !pointCloud) return;

        mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(pointCloud);

        if (intersects.length > 0) {
            const point = intersects[0].point.clone();
            measurePoints.push(point);

            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.15),
                new THREE.MeshBasicMaterial({ color: 0xff4444 })
            );
            sphere.position.copy(point);
            scene.add(sphere);

            if (measurePoints.length === 2) {
                const dist = measurePoints[0].distanceTo(measurePoints[1]);
                showMeasurement(dist);
                drawMeasureLine(measurePoints[0], measurePoints[1]);
                measurePoints = [];
            }
        }
    });
}

function toggleMeasure() {
    measureMode = !measureMode;
    controls.enabled = !measureMode;
    document.getElementById('measurement-display').style.display = measureMode ? 'block' : 'none';
    if (measureMode) {
        document.getElementById('measurement-display').textContent = 'Click two points to measure distance';
    }
}

function toggleConfidence() {
    const legend = document.getElementById('confidence-legend');
    legend.style.display = legend.style.display === 'none' ? 'block' : 'none';
}

function showMeasurement(distance) {
    const display = document.getElementById('measurement-display');
    display.textContent = `Distance: ${distance.toFixed(3)} m`;
    display.style.display = 'block';
}

function drawMeasureLine(p1, p2) {
    if (measureLine) scene.remove(measureLine);
    const geometry = new THREE.BufferGeometry().setFromPoints([p1, p2]);
    const material = new THREE.LineBasicMaterial({ color: 0xff4444, linewidth: 2 });
    measureLine = new THREE.Line(geometry, material);
    scene.add(measureLine);
}
