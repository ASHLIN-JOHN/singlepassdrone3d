"""V23D — Video to 3D Model Generator

A Qt application that:
1. Lets the user select a video file
2. Generates a 3D model using multi-frame fusion with MiDaS depth
3. Detects and models buildings, roads, and turbines (only if present)
4. Displays the PLY result in a built-in OpenGL 3D viewer
5. Shows preview images (segmentation, depth map)
"""

import sys
import os
import time
import struct
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTabWidget,
    QGroupBox, QGridLayout, QStatusBar, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtOpenGL import QGLWidget

from OpenGL.GL import *
from OpenGL.GLU import *


class PLYData:
    """Holds loaded PLY mesh data."""
    def __init__(self):
        self.vertices = None
        self.colors = None
        self.faces = None
        self.center = np.zeros(3)
        self.scale = 1.0

    def load(self, path):
        with open(path, 'rb') as f:
            n_verts = 0
            n_faces = 0
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_verts = int(line.split()[-1])
                elif line.startswith('element face'):
                    n_faces = int(line.split()[-1])
                elif line == 'end_header':
                    break

            vdt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            raw = np.frombuffer(f.read(n_verts * vdt.itemsize), dtype=vdt)
            self.vertices = np.column_stack([raw['x'], raw['y'], raw['z']]).astype(np.float32)
            self.colors = np.column_stack([raw['r'], raw['g'], raw['b']]).astype(np.float32) / 255.0

            if n_faces > 0:
                fdt = np.dtype([('n', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
                fraw = np.frombuffer(f.read(n_faces * fdt.itemsize), dtype=fdt)
                self.faces = np.column_stack([fraw['v0'], fraw['v1'], fraw['v2']]).astype(np.uint32)

        self.center = self.vertices.mean(axis=0)
        extent = self.vertices.max(axis=0) - self.vertices.min(axis=0)
        self.scale = max(extent) if max(extent) > 0 else 1.0


class GLViewer(QGLWidget):
    """OpenGL 3D viewer using VBO arrays for fast rendering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ply = None
        self.rot_x = 20.0
        self.rot_y = 0.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pos = None
        self.display_list = None
        self.setMinimumSize(700, 500)

    def set_ply(self, ply_data):
        self.ply = ply_data
        self.rot_x = 55.0  # top-down isometric angle
        self.rot_y = -30.0
        self.zoom = 1.2
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._build_display_list()
        self.updateGL()

    def _build_display_list(self):
        """Compile geometry into a display list for instant replay."""
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None

        if self.ply is None:
            return

        self.makeCurrent()
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)

        verts = self.ply.vertices
        colors = self.ply.colors
        faces = self.ply.faces

        if faces is not None and len(faces) > 0:
            # Use vertex arrays for fast rendering
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glDisable(GL_LIGHTING)

            # Flatten face indices to get triangle vertex data
            max_faces = min(len(faces), 150000)
            step = max(1, len(faces) // max_faces)
            subset = faces[::step]

            # Build interleaved arrays
            tri_verts = verts[subset.flatten()].astype(np.float32)
            tri_colors = colors[subset.flatten()].astype(np.float32)

            glVertexPointer(3, GL_FLOAT, 0, tri_verts)
            glColorPointer(3, GL_FLOAT, 0, tri_colors)
            glDrawArrays(GL_TRIANGLES, 0, len(tri_verts))

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)
            glPointSize(2.0)
            step = max(1, len(verts) // 200000)
            subset_v = verts[::step].astype(np.float32)
            subset_c = colors[::step].astype(np.float32)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, subset_v)
            glColorPointer(3, GL_FLOAT, 0, subset_c)
            glDrawArrays(GL_POINTS, 0, len(subset_v))
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glEnable(GL_LIGHTING)

        glEndList()

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / max(h, 1)
        gluPerspective(45.0, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        if self.ply is None or self.display_list is None:
            return

        dist = 2.0 / self.zoom
        glTranslatef(self.pan_x, self.pan_y, -dist)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        s = 1.5 / self.ply.scale
        glScalef(s, s, s)
        glTranslatef(-self.ply.center[0], -self.ply.center[1], -self.ply.center[2])

        glCallList(self.display_list)

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.LeftButton:
            self.rot_y += dx * 0.5
            self.rot_x += dy * 0.5
        elif event.buttons() & Qt.RightButton:
            self.pan_x += dx * 0.005
            self.pan_y -= dy * 0.005

        self.last_pos = event.pos()
        self.updateGL()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        self.zoom = max(0.1, min(self.zoom, 50.0))
        self.updateGL()


class ReconstructionWorker(QThread):
    """Runs the reconstruction in a background thread."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    image_ready = pyqtSignal(str, str)  # (image_path, label)

    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir

    def run(self):
        try:
            model_path = str(Path(__file__).parent / "models" / "midas_small.onnx")
            if not Path(model_path).exists():
                self.error.emit(f"MiDaS model not found: {model_path}")
                return

            # Extract frames
            self.progress.emit("Extracting frames from video...")
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            n_sample = min(12, max(3, total_frames // 30))
            frame_indices = np.linspace(
                total_frames * 0.1, total_frames * 0.9, n_sample
            ).astype(int)

            proc_w = min(768, vid_w)
            proc_h = int(proc_w * vid_h / vid_w)

            frames_bgr = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, fr = cap.read()
                if ret:
                    frames_bgr.append(cv2.resize(fr, (proc_w, proc_h)))
            cap.release()

            if len(frames_bgr) < 1:
                self.error.emit("Failed to read video frames")
                return

            self.progress.emit(f"Processing {len(frames_bgr)} frames...")

            ref_idx = len(frames_bgr) // 2
            frame = frames_bgr[ref_idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            preview_path = str(Path(self.output_dir) / "source_frame.png")
            cv2.imwrite(preview_path, frame)
            self.image_ready.emit(preview_path, "Source Frame")

            # MiDaS depth (multi-frame median)
            self.progress.emit("Running MiDaS depth estimation...")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            depth_maps = [self._run_midas(session, input_name, fr, proc_w, proc_h)
                          for fr in frames_bgr]
            depth_norm = np.median(np.stack(depth_maps), axis=0).astype(np.float32)

            depth_vis = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            depth_path = str(Path(self.output_dir) / "depth_map.png")
            cv2.imwrite(depth_path, depth_color)
            self.image_ready.emit(depth_path, "MiDaS Depth")

            # Fused color
            frames_rgb_all = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
            fused_rgb = np.median(np.stack(frames_rgb_all), axis=0).astype(np.uint8)

            # Segmentation
            self.progress.emit("Segmenting scene...")
            sky_mask, ground_mask = self._segment_flat(frame, gray, hsv, depth_norm, proc_w, proc_h)
            road_mask = self._detect_roads(frame, gray, hsv, depth_norm, ground_mask, proc_w, proc_h)
            building_rects = self._detect_buildings(frame, gray, ground_mask, depth_norm, proc_w, proc_h)
            turbines = self._detect_turbines_strict(frames_bgr, depth_maps, proc_w, proc_h)
            trees, bushes = self._detect_vegetation(frame, gray, hsv, ground_mask, road_mask, depth_norm, proc_w, proc_h)

            # Segmentation preview
            seg_vis = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
            seg_vis[sky_mask] = [135, 206, 235]
            seg_vis[ground_mask] = [34, 139, 34]
            seg_vis[road_mask] = [60, 60, 60]
            for rect in building_rects:
                x, y, bw, bh = rect
                seg_vis[y:y+bh, x:x+bw] = [200, 160, 60]
            for tx, ty in trees:
                cv2.circle(seg_vis, (int(tx), int(ty)), 8, (0, 100, 0), -1)
            for bx, by in bushes:
                cv2.circle(seg_vis, (int(bx), int(by)), 5, (0, 150, 50), -1)
            seg_path = str(Path(self.output_dir) / "segmentation.png")
            cv2.imwrite(seg_path, seg_vis)
            self.image_ready.emit(seg_path, "Segmentation")

            # === Build clean flat 3D model (like Blender reference) ===
            self.progress.emit("Building flat terrain...")
            scene_size = 100.0

            all_v, all_c, all_f = [], [], []
            offset = 0

            # 1. Full square ground plane with gentle hills
            gv, gc, gf = self._build_flat_ground(
                fused_rgb, ground_mask, road_mask, depth_norm,
                proc_w, proc_h, scene_size, offset
            )
            all_v.append(gv); all_c.append(gc); all_f.append(gf)
            offset += len(gv)

            # 2. Roads as flat dark strips with white edge lines
            if road_mask.any():
                self.progress.emit("Building roads...")
                rv, rc, rf = self._build_road(
                    road_mask, proc_w, proc_h, scene_size, offset
                )
                all_v.append(rv); all_c.append(rc); all_f.append(rf)
                offset += len(rv)

            # 3. Buildings as clean boxes
            if building_rects:
                self.progress.emit(f"Building {len(building_rects)} structures...")
                for rect in building_rects:
                    bv, bc, bf = self._build_box(
                        rect, fused_rgb, proc_w, proc_h, scene_size, offset
                    )
                    all_v.append(bv); all_c.append(bc); all_f.append(bf)
                    offset += len(bv)

            # 4. Trees and bushes (only if detected in video)
            if trees or bushes:
                veg_label = []
                if trees:
                    veg_label.append(f"{len(trees)} trees")
                if bushes:
                    veg_label.append(f"{len(bushes)} bushes")
                self.progress.emit(f"Building {', '.join(veg_label)}...")
                for tx, ty in trees:
                    tv, tc, tf = self._build_tree(
                        tx, ty, proc_w, proc_h, scene_size, offset
                    )
                    all_v.append(tv); all_c.append(tc); all_f.append(tf)
                    offset += len(tv)
                for bx, by in bushes:
                    bv, bc, bf = self._build_bush(
                        bx, by, proc_w, proc_h, scene_size, offset
                    )
                    all_v.append(bv); all_c.append(bc); all_f.append(bf)
                    offset += len(bv)

            # 5. Sky/clouds/mountains as horizontal backdrop panel
            self.progress.emit("Building sky backdrop...")
            skv, skc, skf = self._build_sky_backdrop(
                fused_rgb, sky_mask, proc_w, proc_h, scene_size, offset
            )
            all_v.append(skv); all_c.append(skc); all_f.append(skf)
            offset += len(skv)

            # 6. Turbines (only if detected in video)
            if turbines:
                self.progress.emit(f"Building {len(turbines)} turbine(s)...")
                for t in turbines:
                    tv, tc, tf = self._build_turbine(
                        t, proc_w, proc_h, scene_size, offset
                    )
                    all_v.append(tv); all_c.append(tc); all_f.append(tf)
                    offset += len(tv)

            # Merge
            self.progress.emit("Saving PLY mesh...")
            all_v = [v for v in all_v if len(v) > 0]
            all_c = [c for c in all_c if len(c) > 0]
            all_f = [f for f in all_f if len(f) > 0]

            if not all_v:
                self.error.emit("No geometry generated")
                return

            vertices = np.concatenate(all_v)
            colors = np.concatenate(all_c)
            faces = np.concatenate(all_f)

            ply_path = str(Path(self.output_dir) / "terrain.ply")
            self._save_ply(vertices, colors, faces, ply_path)

            parts = f"{len(vertices):,} verts, {len(faces):,} faces"
            if turbines:
                parts += f", {len(turbines)} turbine(s)"
            if building_rects:
                parts += f", {len(building_rects)} building(s)"
            if trees:
                parts += f", {len(trees)} trees"
            if bushes:
                parts += f", {len(bushes)} bushes"
            self.progress.emit(f"Done! {parts}")
            self.finished.emit(ply_path)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def _run_midas(self, session, input_name, frame, w, h):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_input = ((img_resized - mean) / std).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        result = session.run(None, {input_name: img_input})
        depth_raw = result[0].squeeze()
        depth_map = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        d_min, d_max = np.percentile(depth_map, [2, 98])
        return np.clip((depth_map - d_min) / (d_max - d_min + 1e-6), 0, 1)

    def _segment_flat(self, frame, gray, hsv, depth_norm, w, h):
        """Simple sky/ground segmentation for flat model."""
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        sky = np.zeros((h, w), dtype=bool)
        sky |= (depth_norm < 0.15)
        upper = np.zeros((h, w), dtype=bool)
        upper[:int(h * 0.55)] = True
        sky |= upper & (depth_norm < 0.30)
        sky |= (gray > 170) & (sat < 50) & upper
        sky |= (hue < 30) & (sat > 60) & (val > 120) & upper
        sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((11, 11), np.uint8)).astype(bool)
        sky[:int(h * 0.25)] = True
        ground = ~sky
        return sky, ground

    def _detect_roads(self, frame, gray, hsv, depth_norm, ground_mask, w, h):
        """Detect road regions using multiple cues: color, texture, edges.
        Skip for synthetic/low-texture scenes."""
        # Check if scene is synthetic (very low overall texture)
        texture = cv2.Laplacian(gray, cv2.CV_32F)
        avg_texture = np.abs(texture).mean()
        if avg_texture < 4.0:  # Very low texture = synthetic scene, no roads
            return np.zeros((h, w), dtype=bool)

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Method 1: Low saturation + darker than surroundings + smooth
        road1 = ground_mask.copy()
        road1 &= (sat < 80)
        road1 &= (gray > 20) & (gray < 160)
        # Smooth areas (low texture)
        texture_smooth = cv2.GaussianBlur(np.abs(texture), (15, 15), 0)
        road1 &= (texture_smooth < 12.0)

        # Method 2: Detect as darker linear features in ground area
        # Local contrast: road is darker than neighboring grass
        blur_large = cv2.GaussianBlur(gray.astype(np.float32), (51, 51), 0)
        darker_than_surround = (gray.astype(np.float32) < blur_large - 20)
        road2 = ground_mask & darker_than_surround & (sat < 100)

        # Combine
        road = road1 | road2
        road[:int(h * 0.3)] = False  # no roads in sky area

        # Morphology: roads are elongated, connect nearby segments
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
        road_h = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, kernel_h)
        road_v = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, kernel_v)
        road = (road_h | road_v).astype(bool)
        # Remove small blobs
        road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((7, 7), np.uint8)).astype(bool)

        # Must have minimum area
        if road.sum() < w * h * 0.005:
            return np.zeros((h, w), dtype=bool)

        # Keep only large elongated components (actual roads)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            road.astype(np.uint8), connectivity=8)
        min_road_size = w * h * 0.003
        road_final = np.zeros((h, w), dtype=bool)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_road_size:
                continue
            # Roads must be elongated (not blobs)
            rw = stats[i, cv2.CC_STAT_WIDTH]
            rh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(rw, rh) / (min(rw, rh) + 1)
            if aspect > 2.5:  # Must be elongated
                road_final |= (labels == i)

        return road_final

    def _detect_buildings(self, frame, gray, ground_mask, depth_norm, w, h):
        """Detect rectangular building structures including synthetic/clean buildings.
        Different rules for synthetic vs real scenes."""
        # Check if scene is synthetic (low texture AND low depth variation = CG scene)
        texture = cv2.Laplacian(gray, cv2.CV_32F)
        avg_texture = np.abs(texture).mean()
        depth_std = depth_norm.std()
        # Synthetic: BOTH low texture AND low depth variation
        is_synthetic = (avg_texture < 4.0) and (depth_std < 0.27)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        buildings = []
        min_area = w * h * 0.002
        max_area = w * h * 0.25

        if is_synthetic:
            # SYNTHETIC SCENE: Look for uniform color regions (like gray cube)
            # Focus on middle area of frame
            search_area = ground_mask.copy()
            search_area[:int(h*0.15)] = False  # Skip very top
            search_area[int(h*0.85):] = False  # Skip very bottom

            # Low saturation regions (gray/neutral buildings)
            gray_regions = search_area & (sat < 50) & (gray > 70) & (gray < 200)
            gray_regions = cv2.morphologyEx(gray_regions.astype(np.uint8), cv2.MORPH_CLOSE,
                                           np.ones((21, 21), np.uint8)).astype(bool)

            if gray_regions.sum() > min_area:
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    gray_regions.astype(np.uint8), connectivity=8)
                for i in range(1, n_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < min_area or area > max_area:
                        continue
                    rx = stats[i, cv2.CC_STAT_LEFT]
                    ry = stats[i, cv2.CC_STAT_TOP]
                    rw = stats[i, cv2.CC_STAT_WIDTH]
                    rh = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect = rw / max(rh, 1)
                    # Synthetic buildings can be any reasonable aspect
                    if 0.3 < aspect < 4.0:
                        # Check if uniform color (low std)
                        region_std = gray[ry:ry+rh, rx:rx+rw].std()
                        if region_std < 45:  # Very uniform
                            buildings.append((rx, ry, rw, rh))
        else:
            # REAL SCENE: No building detection
            # Real drone footage captures landscapes, not distinct buildings to model
            pass

        if not buildings:
            return []

        # Deduplicate
        buildings.sort(key=lambda b: b[2]*b[3], reverse=True)
        kept = []
        for b in buildings:
            overlap = False
            for k in kept:
                ox = max(0, min(b[0]+b[2], k[0]+k[2]) - max(b[0], k[0]))
                oy = max(0, min(b[1]+b[3], k[1]+k[3]) - max(b[1], k[1]))
                if ox * oy > 0.5 * min(b[2]*b[3], k[2]*k[3]):
                    overlap = True; break
            if not overlap:
                kept.append(b)
        return kept[:8]

    def _detect_turbines_strict(self, frames_bgr, depth_maps, w, h):
        """Detect turbines ONLY if very confident - must be tall, narrow, consistent across frames."""
        # Collect poles from ALL frames
        all_poles = []
        frames_with_poles = 0
        for i, fr in enumerate(frames_bgr):
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            poles = self._find_vertical_poles(gray, depth_maps[i], w, h)
            if poles:
                frames_with_poles += 1
            for p in poles:
                p['frame_idx'] = i
                all_poles.append(p)

        # STRICT: Turbines must appear in at least 50% of frames
        if frames_with_poles < len(frames_bgr) * 0.5:
            return []

        if not all_poles:
            return []

        # STRICT filtering: only keep poles that are VERY TALL and VERY NARROW
        turbine_candidates = []
        for pole in all_poles:
            height_frac = pole['height_px'] / h
            aspect = pole['width_px'] / max(pole['height_px'], 1)

            # Must be at least 12% of frame height (turbines are dominant features)
            if height_frac < 0.12:
                continue

            # Must be very narrow (aspect < 0.25)
            if aspect > 0.25:
                continue

            # Must extend into upper 40% of frame (against sky)
            if pole['top_y'] > h * 0.4:
                continue

            # Must be in middle depth range (not sky-level, not extreme foreground)
            if pole['depth'] < 0.15 or pole['depth'] > 0.7:
                continue

            turbine_candidates.append(pole)

        # If very few candidates, likely not turbines
        if len(turbine_candidates) < 3:
            return []

        # Group by depth levels (different distances = different turbines)
        depth_groups = {}
        for t in turbine_candidates:
            d_key = round(t['depth'] * 10) / 10
            if d_key not in depth_groups:
                depth_groups[d_key] = []
            depth_groups[d_key].append(t)

        # Take the best from each depth group
        final_turbines = []
        for d_key, group in sorted(depth_groups.items(), reverse=True):
            group.sort(key=lambda t: t['center_x'])
            # Cluster by x position
            clusters = []
            cur = [group[0]]
            for t in group[1:]:
                if abs(t['center_x'] - cur[-1]['center_x']) < w * 0.08:
                    cur.append(t)
                else:
                    clusters.append(cur)
                    cur = [t]
            clusters.append(cur)
            for cl in clusters:
                best = max(cl, key=lambda t: t['height_px'])
                final_turbines.append(best)

        # STRICT: Must have at least 2 turbines to be confident (turbines come in farms)
        if len(final_turbines) < 2:
            return []

        final_turbines.sort(key=lambda t: -t['depth'])
        return final_turbines[:8]

    def _find_vertical_poles(self, gray, depth_norm, w, h):
        """Find vertical pole-like structures that could be turbines."""
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=50, maxLineGap=20)
        if lines is None or len(lines) == 0:
            return []
        vsegs = []
        for line in lines:
            pts = line.flatten()
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            # STRICT: must be long (>60px) and very vertical (dx < 15% of length)
            if length > 60 and abs(dx) < length * 0.15:
                vsegs.append((x1, y1, x2, y2, length))
        if not vsegs:
            return []
        vsegs.sort(key=lambda s: s[0])
        clusters, cur = [], [vsegs[0]]
        for seg in vsegs[1:]:
            if abs(seg[0] - cur[-1][0]) < 35:
                cur.append(seg)
            else:
                clusters.append(cur); cur = [seg]
        clusters.append(cur)
        poles = []
        for cl in clusters:
            all_x = [s[0] for s in cl] + [s[2] for s in cl]
            all_y = [s[1] for s in cl] + [s[3] for s in cl]
            cx = int(np.mean(all_x))
            top_y, base_y = min(all_y), max(all_y)
            height_px = base_y - top_y
            width_px = max(all_x) - min(all_x) + 1
            total_len = sum(s[4] for s in cl)

            # STRICT: minimum height 8% of frame
            if height_px < h * 0.08:
                continue
            # STRICT: must have substantial line length
            if total_len < 100:
                continue
            # STRICT: must be narrow (width < 30% of height)
            if width_px > height_px * 0.3:
                continue

            ys, ye = max(0, top_y), min(h, base_y)
            xs, xe = max(0, cx-10), min(w, cx+10)
            avg_depth = depth_norm[ys:ye, xs:xe].mean()

            # STRICT: depth must be in valid range
            if avg_depth < 0.15 or avg_depth > 0.7:
                continue

            poles.append({
                'base_x': float(cx), 'base_y': float(base_y),
                'top_y': float(top_y), 'center_x': float(cx),
                'height_px': float(height_px), 'width_px': float(width_px),
                'depth': float(avg_depth),
            })
        return poles

    # ==================== FLAT MODEL BUILDERS ====================

    def _build_flat_ground(self, frame_rgb, ground_mask, road_mask, depth_norm,
                           w, h, size, offset):
        """Build a FULL square ground plane with gentle hills from depth.
        The entire ground is filled (no gaps). Video colors mapped on top."""
        # Use finer stride for quality
        grid_res = 200  # 200x200 grid = 40K quads
        xs = np.linspace(0, w-1, grid_res).astype(int)
        ys = np.linspace(0, h-1, grid_res).astype(int)
        grid_h, grid_w = grid_res, grid_res
        gx, gy = np.meshgrid(xs, ys)

        # World X/Y: full square plane
        world_x = (np.linspace(-0.5, 0.5, grid_res)[np.newaxis, :] * np.ones((grid_res, 1))).astype(np.float32) * size
        world_y = (np.linspace(-0.5, 0.5, grid_res)[:, np.newaxis] * np.ones((1, grid_res))).astype(np.float32) * size

        # Z: gentle hills from depth (only on ground areas, clamped small)
        depth_at_grid = depth_norm[gy, gx]
        ground_at_grid = ground_mask[gy, gx]
        road_at_grid = road_mask[gy, gx]

        # Check if scene is synthetic (low depth variation = flat CG scene)
        depth_std = depth_norm.std()
        is_synthetic = depth_std < 0.27  # Very low depth variation

        # Height: ground gets slight bumps (max ~3 units), road stays flat at 0
        world_z = np.zeros((grid_res, grid_res), dtype=np.float32)

        # Identify sky areas (for both synthetic and real scenes)
        sky_at_grid = ~ground_at_grid

        if not is_synthetic:
            # Real scene: add gentle hills from depth
            ground_height = depth_at_grid * 3.0
            ground_height = cv2.GaussianBlur(ground_height, (15, 15), 0)
            world_z[ground_at_grid & ~road_at_grid] = ground_height[ground_at_grid & ~road_at_grid]
            # Sky areas (background) get slight negative height
            world_z[sky_at_grid] = -0.3
        # else: synthetic scene stays completely flat (Z=0 everywhere)

        # Force road areas completely flat
        world_z[road_at_grid] = 0.01

        # ALL vertices are valid (full square plane)
        verts = np.column_stack([
            world_x.flatten(), world_y.flatten(), world_z.flatten()
        ]).astype(np.float32)

        # Colors: use actual video colors everywhere
        cols = frame_rgb[gy.flatten(), gx.flatten()].copy()

        # For areas marked as non-ground (sky pixels in image), boost green slightly
        # to make them grass-like, but keep some original color
        sky_flat = sky_at_grid.flatten()
        if sky_flat.any():
            sky_cols = cols[sky_flat].astype(np.float32)
            # Blend with grass green (70% original, 30% grass)
            grass_green = np.array([45, 85, 30], dtype=np.float32)
            cols[sky_flat] = (sky_cols * 0.7 + grass_green * 0.3).astype(np.uint8)

        # Road areas get dark gray
        road_flat = road_at_grid.flatten()
        cols[road_flat] = [40, 40, 40]

        # Build faces (full grid, no gaps)
        idx = np.arange(grid_res * grid_res).reshape(grid_res, grid_res)
        v00 = idx[:-1, :-1].flatten()
        v01 = idx[:-1, 1:].flatten()
        v10 = idx[1:, :-1].flatten()
        v11 = idx[1:, 1:].flatten()
        f1 = np.column_stack([v00+offset, v10+offset, v01+offset])
        f2 = np.column_stack([v10+offset, v11+offset, v01+offset])
        faces = np.concatenate([f1, f2]).astype(np.int32)

        return verts, cols.astype(np.uint8), faces

    def _build_road(self, road_mask, w, h, size, offset):
        """Build road as raised surface (Z=0.05) with white edge markings."""
        # Road edges for white line markings
        road_eroded = cv2.erode(road_mask.astype(np.uint8),
                                np.ones((9, 9), np.uint8)).astype(bool)
        edge_mask = road_mask & ~road_eroded

        stride = 2  # high quality for roads
        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        grid_h, grid_w = len(ys), len(xs)
        gx, gy = np.meshgrid(xs, ys)

        road_grid = road_mask[gy, gx]
        edge_grid = edge_mask[gy, gx]

        world_x = (gx.astype(np.float32) / w - 0.5) * size
        world_y = (gy.astype(np.float32) / h - 0.5) * size
        world_z = np.full_like(world_x, 0.05)

        valid = road_grid.flatten()
        if valid.sum() < 20:
            return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8), np.zeros((0,3), np.int32)

        verts = np.column_stack([
            world_x.flatten()[valid],
            world_y.flatten()[valid],
            world_z.flatten()[valid]
        ]).astype(np.float32)

        # Dark road surface + white edges
        cols = np.full((valid.sum(), 3), 35, dtype=np.uint8)
        edge_flat = edge_grid.flatten()[valid]
        cols[edge_flat] = [245, 245, 245]

        vmap = np.full(grid_h * grid_w, -1, dtype=np.int32)
        vmap[valid] = np.arange(valid.sum())
        vmap = vmap.reshape(grid_h, grid_w)

        v00 = vmap[:-1, :-1].flatten()
        v01 = vmap[:-1, 1:].flatten()
        v10 = vmap[1:, :-1].flatten()
        v11 = vmap[1:, 1:].flatten()

        m1 = (v00 >= 0) & (v10 >= 0) & (v01 >= 0)
        f1 = np.column_stack([v00[m1]+offset, v10[m1]+offset, v01[m1]+offset])
        m2 = (v10 >= 0) & (v11 >= 0) & (v01 >= 0)
        f2 = np.column_stack([v10[m2]+offset, v11[m2]+offset, v01[m2]+offset])

        faces = np.concatenate([f1, f2]) if len(f1) and len(f2) else (
            f1 if len(f1) else f2 if len(f2) else np.zeros((0,3), np.int32))
        return verts, cols, faces.astype(np.int32)

    def _build_sky_backdrop(self, frame_rgb, sky_mask, w, h, size, offset):
        """Build sky/clouds/mountains as a vertical backdrop panel behind terrain."""
        # Extract and clean sky region
        sky_rgb = frame_rgb.copy()
        gray = cv2.cvtColor(cv2.cvtColor(sky_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        sky_blurred = cv2.GaussianBlur(sky_rgb, (0, 0), sigmaX=40)
        dark = sky_mask & (gray < 100)
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        sky_rgb[dark] = sky_blurred[dark]

        # Higher resolution for better gradient representation
        panel_res_x = 200
        panel_res_y = 100

        # Sample sky from top portion of image
        sky_rows = int(h * 0.6)
        xs = np.linspace(0, w-1, panel_res_x).astype(int)
        ys = np.linspace(0, sky_rows-1, panel_res_y).astype(int)
        gx, gy = np.meshgrid(xs, ys)

        # Backdrop is a large vertical wall at the back
        panel_width = size * 1.4
        panel_height = size * 0.6

        world_x = np.linspace(-panel_width/2, panel_width/2, panel_res_x)[np.newaxis, :] * np.ones((panel_res_y, 1))
        world_y = np.full((panel_res_y, panel_res_x), -size/2 - 5.0)  # far behind terrain
        world_z = np.linspace(panel_height, -2.0, panel_res_y)[:, np.newaxis] * np.ones((1, panel_res_x))

        verts = np.column_stack([
            world_x.flatten().astype(np.float32),
            world_y.flatten().astype(np.float32),
            world_z.flatten().astype(np.float32),
        ])

        # Get colors from video with smooth sampling
        cols = sky_rgb[gy.flatten(), gx.flatten()].astype(np.uint8)

        # Build faces (smooth shaded mesh)
        idx = np.arange(panel_res_y * panel_res_x).reshape(panel_res_y, panel_res_x)
        v00 = idx[:-1, :-1].flatten()
        v01 = idx[:-1, 1:].flatten()
        v10 = idx[1:, :-1].flatten()
        v11 = idx[1:, 1:].flatten()
        f1 = np.column_stack([v00+offset, v01+offset, v10+offset])
        f2 = np.column_stack([v01+offset, v11+offset, v10+offset])
        faces = np.concatenate([f1, f2]).astype(np.int32)

        return verts, cols, faces

    def _detect_vegetation(self, frame, gray, hsv, ground_mask, road_mask, depth_norm, w, h):
        """Detect trees and bushes from green vegetation with texture."""
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Trees/bushes are: green hue, moderate-high saturation, darker than flat grass
        veg_mask = ground_mask & ~road_mask
        veg_mask &= (hue > 25) & (hue < 90)  # green range
        veg_mask &= (sat > 40)
        veg_mask &= (val > 20) & (val < 200)

        # Trees are darker green with high texture (canopy detail)
        texture = cv2.Laplacian(gray, cv2.CV_32F)
        texture_mag = cv2.GaussianBlur(np.abs(texture), (11, 11), 0)

        # Trees: darker, more textured
        tree_mask = veg_mask & (gray < 120) & (texture_mag > 5.0)
        tree_mask[:int(h * 0.3)] = False  # no trees in sky area

        # Bushes: medium green, some texture but less than trees, smaller
        bush_mask = veg_mask & (gray >= 60) & (gray < 150) & (texture_mag > 3.0) & (texture_mag < 10.0)
        bush_mask[:int(h * 0.4)] = False
        bush_mask &= ~tree_mask

        # Clean with morphology
        tree_mask = cv2.morphologyEx(tree_mask.astype(np.uint8), cv2.MORPH_OPEN,
                                     np.ones((7, 7), np.uint8)).astype(bool)
        bush_mask = cv2.morphologyEx(bush_mask.astype(np.uint8), cv2.MORPH_OPEN,
                                     np.ones((5, 5), np.uint8)).astype(bool)

        # Find tree cluster centers
        trees = []
        if tree_mask.sum() > w * h * 0.005:
            # Find connected components for trees
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                tree_mask.astype(np.uint8), connectivity=8)
            min_tree_area = w * h * 0.001
            for i in range(1, n_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_tree_area:
                    cx, cy = centroids[i]
                    trees.append((cx, cy))
                    # Large clusters get multiple trees
                    if area > min_tree_area * 5:
                        bx = stats[i, cv2.CC_STAT_LEFT]
                        bw = stats[i, cv2.CC_STAT_WIDTH]
                        by = stats[i, cv2.CC_STAT_TOP]
                        bh_stat = stats[i, cv2.CC_STAT_HEIGHT]
                        for _ in range(min(3, int(area / (min_tree_area * 3)))):
                            tx = bx + np.random.randint(0, max(1, bw))
                            ty = by + np.random.randint(0, max(1, bh_stat))
                            if tree_mask[min(ty, h-1), min(tx, w-1)]:
                                trees.append((float(tx), float(ty)))

        # Find bush cluster centers
        bushes = []
        if bush_mask.sum() > w * h * 0.003:
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bush_mask.astype(np.uint8), connectivity=8)
            min_bush_area = w * h * 0.0005
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] > min_bush_area:
                    cx, cy = centroids[i]
                    bushes.append((cx, cy))

        # Limit counts
        trees = trees[:20]
        bushes = bushes[:30]
        return trees, bushes

    def _build_tree(self, img_x, img_y, img_w, img_h, size, offset):
        """Build a simple tree: brown cylinder trunk + green cone canopy."""
        wx = (img_x / img_w - 0.5) * size
        wy = (img_y / img_h - 0.5) * size

        trunk_h = 4.0 + np.random.random() * 2.0
        trunk_r = 0.4
        canopy_h = 5.0 + np.random.random() * 3.0
        canopy_r = 2.5 + np.random.random() * 1.5

        trunk_col = np.array([70, 45, 20], dtype=np.uint8)
        canopy_col = np.array([30, 80 + int(np.random.random() * 40), 15], dtype=np.uint8)

        verts, colors, faces = [], [], []
        n_sides = 8

        # Trunk (cylinder)
        for sec in range(3):
            z = sec / 2.0 * trunk_h
            for side in range(n_sides):
                a = 2 * np.pi * side / n_sides
                verts.append([wx + trunk_r * np.cos(a), wy + trunk_r * np.sin(a), z])
                colors.append(trunk_col)
        for sec in range(2):
            for side in range(n_sides):
                ns = (side + 1) % n_sides
                i0 = sec * n_sides + side
                i1 = sec * n_sides + ns
                i2 = (sec+1) * n_sides + side
                i3 = (sec+1) * n_sides + ns
                faces.append([i0+offset, i2+offset, i1+offset])
                faces.append([i1+offset, i2+offset, i3+offset])

        # Canopy (cone)
        base_idx = len(verts)
        base_z = trunk_h
        # Base circle
        for side in range(n_sides):
            a = 2 * np.pi * side / n_sides
            verts.append([wx + canopy_r * np.cos(a), wy + canopy_r * np.sin(a), base_z])
            colors.append(canopy_col)
        # Tip
        tip_idx = len(verts)
        verts.append([wx, wy, base_z + canopy_h])
        colors.append(canopy_col)
        # Cone faces
        for side in range(n_sides):
            ns = (side + 1) % n_sides
            faces.append([base_idx + side + offset, base_idx + ns + offset, tip_idx + offset])
        # Bottom cap
        center_idx = len(verts)
        verts.append([wx, wy, base_z])
        colors.append(canopy_col)
        for side in range(n_sides):
            ns = (side + 1) % n_sides
            faces.append([center_idx + offset, base_idx + side + offset, base_idx + ns + offset])

        return (np.array(verts, np.float32), np.array(colors, np.uint8),
                np.array(faces, np.int32))

    def _build_bush(self, img_x, img_y, img_w, img_h, size, offset):
        """Build a simple bush: low green hemisphere."""
        wx = (img_x / img_w - 0.5) * size
        wy = (img_y / img_h - 0.5) * size

        radius = 1.2 + np.random.random() * 0.8
        bush_col = np.array([35, 90 + int(np.random.random() * 30), 20], dtype=np.uint8)

        verts, colors, faces = [], [], []
        n_rings = 4
        n_sides = 8

        # Hemisphere vertices
        for ring in range(n_rings + 1):
            phi = (np.pi / 2) * ring / n_rings  # 0 to pi/2
            z = radius * np.cos(phi)
            r = radius * np.sin(phi)
            for side in range(n_sides):
                a = 2 * np.pi * side / n_sides
                verts.append([wx + r * np.cos(a), wy + r * np.sin(a), z])
                shade = 0.7 + 0.3 * (1.0 - ring / n_rings)
                colors.append(np.clip(bush_col * shade, 0, 255).astype(np.uint8))

        # Faces
        for ring in range(n_rings):
            for side in range(n_sides):
                ns = (side + 1) % n_sides
                i0 = ring * n_sides + side
                i1 = ring * n_sides + ns
                i2 = (ring+1) * n_sides + side
                i3 = (ring+1) * n_sides + ns
                faces.append([i0+offset, i1+offset, i2+offset])
                faces.append([i1+offset, i3+offset, i2+offset])

        return (np.array(verts, np.float32), np.array(colors, np.uint8),
                np.array(faces, np.int32))

    def _build_box(self, rect, frame_rgb, img_w, img_h, size, offset):
        """Build a clean box (building) standing on the ground plane."""
        rx, ry, rw, rh = rect
        # Position on ground plane
        cx = (rx + rw/2) / img_w
        cy = (ry + rh/2) / img_h
        wx = (cx - 0.5) * size
        wy = (cy - 0.5) * size

        # Size proportional to image footprint
        bw = (rw / img_w) * size * 0.8
        bd = (rh / img_h) * size * 0.5
        bh = max(bw, bd) * 0.6  # height from footprint

        # Color from video
        region = frame_rgb[ry:ry+rh, rx:rx+rw]
        wall_col = region.mean(axis=(0,1)).astype(np.uint8)
        roof_col = np.clip(wall_col.astype(int) - 40, 0, 255).astype(np.uint8)

        x0, x1 = wx - bw/2, wx + bw/2
        y0, y1 = wy - bd/2, wy + bd/2
        z0, z1 = 0.0, bh

        verts = np.array([
            [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
        ], dtype=np.float32)
        colors = np.array([
            wall_col, wall_col, wall_col, wall_col,
            roof_col, roof_col, roof_col, roof_col,
        ], dtype=np.uint8)
        face_idx = np.array([
            [0,2,1],[0,3,2],[4,5,6],[4,6,7],
            [0,1,5],[0,5,4],[2,3,7],[2,7,6],
            [0,4,7],[0,7,3],[1,2,6],[1,6,5],
        ], dtype=np.int32) + offset
        return verts, colors, face_idx

    def _build_turbine(self, info, img_w, img_h, size, offset):
        """Build a clean wind turbine: dark pole + white nacelle + 3 white blades."""
        # Position on ground — use depth for Y (distance into scene)
        wx = (info['center_x'] / img_w - 0.5) * size
        # Y position: deeper turbines further back
        wy = (1.0 - info['depth']) * size * 0.4 - size * 0.2

        # Scale by depth: closer turbines bigger
        scale = 0.4 + info['depth'] * 0.8
        tower_h = 30.0 * scale
        tower_r = 0.8 * scale
        blade_len = 14.0 * scale
        nacelle_size = 2.0 * scale
        pole_color = np.array([50, 45, 40], dtype=np.uint8)
        blade_color = np.array([240, 240, 235], dtype=np.uint8)
        nacelle_color = np.array([60, 55, 50], dtype=np.uint8)

        verts, colors, faces = [], [], []

        # Tower: cylinder (8 sides, 6 sections)
        n_sides, n_sec = 8, 6
        for sec in range(n_sec + 1):
            frac = sec / n_sec
            z = frac * tower_h
            r = tower_r * (1.0 - frac * 0.2)
            for side in range(n_sides):
                a = 2 * np.pi * side / n_sides
                verts.append([wx + r * np.cos(a), wy + r * np.sin(a), z])
                colors.append(pole_color)
        for sec in range(n_sec):
            for side in range(n_sides):
                ns = (side + 1) % n_sides
                i0 = sec * n_sides + side
                i1 = sec * n_sides + ns
                i2 = (sec+1) * n_sides + side
                i3 = (sec+1) * n_sides + ns
                faces.append([i0+offset, i2+offset, i1+offset])
                faces.append([i1+offset, i2+offset, i3+offset])

        # Nacelle box at top
        nb = len(verts)
        ns = nacelle_size
        nz = tower_h
        for corner in [
            [wx-ns, wy-ns*0.4, nz-ns*0.3], [wx+ns, wy-ns*0.4, nz-ns*0.3],
            [wx+ns, wy+ns*0.4, nz-ns*0.3], [wx-ns, wy+ns*0.4, nz-ns*0.3],
            [wx-ns, wy-ns*0.4, nz+ns*0.3], [wx+ns, wy-ns*0.4, nz+ns*0.3],
            [wx+ns, wy+ns*0.4, nz+ns*0.3], [wx-ns, wy+ns*0.4, nz+ns*0.3],
        ]:
            verts.append(corner)
            colors.append(nacelle_color)
        nbo = nb + offset
        for bf in [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                   [2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]]:
            faces.append([nbo+bf[0], nbo+bf[1], nbo+bf[2]])

        # 3 blades from top of nacelle
        hub_z = nz + ns * 0.3
        for bi in range(3):
            a = bi * 2 * np.pi / 3 + 0.3
            tip_x = wx + blade_len * np.cos(a)
            tip_z = hub_z + blade_len * np.sin(a)
            vi = len(verts)
            # Blade as a thin triangle (two-sided)
            verts.extend([
                [wx, wy - 0.3, hub_z],
                [wx, wy + 0.3, hub_z],
                [tip_x, wy, tip_z],
            ])
            colors.extend([blade_color, blade_color, blade_color])
            faces.append([vi+offset, vi+1+offset, vi+2+offset])
            # Back face
            verts.extend([
                [wx, wy + 0.3, hub_z],
                [wx, wy - 0.3, hub_z],
                [tip_x, wy, tip_z],
            ])
            colors.extend([blade_color, blade_color, blade_color])
            faces.append([vi+3+offset, vi+4+offset, vi+5+offset])

        return (np.array(verts, np.float32), np.array(colors, np.uint8),
                np.array(faces, np.int32))

    def _save_ply(self, vertices, colors, faces, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        n_v, n_f = len(vertices), len(faces)
        with open(path, 'wb') as f:
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n_v}\n"
            header += "property float x\nproperty float y\nproperty float z\n"
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            header += f"element face {n_f}\nproperty list uchar int vertex_indices\nend_header\n"
            f.write(header.encode('ascii'))
            vd = np.zeros(n_v, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                       ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            vd['x'], vd['y'], vd['z'] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            vd['r'], vd['g'], vd['b'] = colors[:, 0], colors[:, 1], colors[:, 2]
            f.write(vd.tobytes())
            fd = np.zeros(n_f, dtype=[('n', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
            fd['n'] = 3
            fd['v0'], fd['v1'], fd['v2'] = faces[:, 0], faces[:, 1], faces[:, 2]
            f.write(fd.tobytes())


class V23DMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("V23D — Video to 3D Model Generator")
        self.setMinimumSize(1400, 850)
        self.worker = None
        self.output_dir = str(Path(__file__).parent / "output")
        Path(self.output_dir).mkdir(exist_ok=True)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel("V23D")
        header.setFont(QFont("Segoe UI", 22, QFont.Bold))
        header.setStyleSheet("color: #2196F3;")
        sub = QLabel("Video to 3D Model Generator")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color: #666;")
        header_layout = QVBoxLayout()
        header_layout.addWidget(header)
        header_layout.addWidget(sub)
        header_layout.setSpacing(2)
        main_layout.addLayout(header_layout)

        # Control bar
        ctrl_group = QGroupBox("Input")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.file_label = QLabel("No video selected")
        self.file_label.setStyleSheet("color: #888; font-style: italic;")
        self.btn_browse = QPushButton("Choose Video")
        self.btn_browse.setMinimumWidth(120)
        self.btn_browse.clicked.connect(self._browse_video)
        self.btn_generate = QPushButton("Generate 3D Model")
        self.btn_generate.setMinimumWidth(150)
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self._start_generation)
        self.btn_generate.setStyleSheet(
            "QPushButton:enabled { background-color: #2196F3; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 6px 16px; }"
        )
        ctrl_layout.addWidget(self.file_label, 1)
        ctrl_layout.addWidget(self.btn_browse)
        ctrl_layout.addWidget(self.btn_generate)
        main_layout.addWidget(ctrl_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #444;")
        prog_layout = QHBoxLayout()
        prog_layout.addWidget(self.progress_label, 1)
        prog_layout.addWidget(self.progress_bar, 2)
        main_layout.addLayout(prog_layout)

        # Main content: splitter with 3D viewer and image tabs
        splitter = QSplitter(Qt.Horizontal)

        # Left: 3D Viewer
        viewer_group = QGroupBox("3D Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        self.gl_viewer = GLViewer()
        viewer_layout.addWidget(self.gl_viewer)
        self.viewer_info = QLabel("Load a model to view")
        self.viewer_info.setStyleSheet("color: #666; font-size: 11px;")
        viewer_layout.addWidget(self.viewer_info)
        splitter.addWidget(viewer_group)

        # Right: Image previews
        img_group = QGroupBox("Preview Images")
        img_layout = QVBoxLayout(img_group)
        self.img_tabs = QTabWidget()
        img_layout.addWidget(self.img_tabs)
        splitter.addWidget(img_group)

        splitter.setSizes([900, 300])
        main_layout.addWidget(splitter, 1)

        # Status bar
        self.statusBar().showMessage("Ready — Select a video to begin")

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if path:
            self.video_path = path
            name = Path(path).name
            self.file_label.setText(f"Selected: {name}")
            self.file_label.setStyleSheet("color: #333; font-weight: bold;")
            self.btn_generate.setEnabled(True)
            self.statusBar().showMessage(f"Video loaded: {name}")

    def _start_generation(self):
        self.btn_generate.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Starting...")

        # Clear old tabs
        while self.img_tabs.count() > 0:
            self.img_tabs.removeTab(0)

        self.worker = ReconstructionWorker(self.video_path, self.output_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.image_ready.connect(self._on_image)
        self.worker.start()

    def _on_progress(self, msg):
        self.progress_label.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_image(self, path, label):
        img = QImage(path)
        if img.isNull():
            return
        pixmap = QPixmap.fromImage(img)
        lbl = QLabel()
        lbl.setPixmap(pixmap.scaled(500, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        self.img_tabs.addTab(lbl, label)

    def _on_finished(self, ply_path):
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.btn_generate.setEnabled(True)
        self.btn_browse.setEnabled(True)

        # Load PLY into 3D viewer
        ply = PLYData()
        ply.load(ply_path)
        self.gl_viewer.set_ply(ply)
        n_v = len(ply.vertices)
        n_f = len(ply.faces) if ply.faces is not None else 0
        self.viewer_info.setText(
            f"{n_v:,} vertices | {n_f:,} faces | "
            f"Drag to rotate, scroll to zoom, right-drag to pan"
        )
        self.statusBar().showMessage(f"Model generated: {ply_path}")

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.progress_label.setText(f"Error: {msg}")
        self.progress_label.setStyleSheet("color: red;")
        self.btn_generate.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.statusBar().showMessage(f"Error: {msg}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("V23D")
    app.setStyle("Fusion")
    window = V23DMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
