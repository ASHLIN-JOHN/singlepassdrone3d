"""3D Scene reconstruction: horizontal terrain + sky backdrop + turbines.

- Terrain: horizontal ground with roads visible (fine grid)
- Sky/clouds/sun: separate backdrop plane behind terrain
- Wind turbines: procedural 3D models (cylinder + blades)
- Orientation: X = wide, Y = depth (into scene), Z = up
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main():
    video_path = "data/synthetic/v1.mp4"
    model_path = "models/midas_small.onnx"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  3D SCENE — Horizontal Terrain + Sky + Turbines")
    print("=" * 60)
    print()

    FRAME_POS = 350

    t0 = time.time()
    print("[1/6] Loading frame + depth...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_POS)
    ret, frame_full = cap.read()
    cap.release()

    # Process at decent resolution
    proc_w, proc_h = 768, 1344
    frame = cv2.resize(frame_full, (proc_w, proc_h))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    depth_norm = run_midas(session, input_name, frame, proc_w, proc_h)
    print(f"       ({time.time()-t0:.1f}s)")
    print()

    # Segment scene into sky, ground, mountains
    t0 = time.time()
    print("[2/6] Segmenting scene...")
    sky_mask, ground_mask, mountain_mask = segment_scene(
        frame, gray, hsv, depth_norm, proc_w, proc_h
    )
    sky_pct = sky_mask.sum() / sky_mask.size * 100
    ground_pct = ground_mask.sum() / ground_mask.size * 100
    mountain_pct = mountain_mask.sum() / mountain_mask.size * 100
    print(f"       Sky: {sky_pct:.0f}%  Ground: {ground_pct:.0f}%  Mountains: {mountain_pct:.0f}%")
    print(f"       ({time.time()-t0:.1f}s)")
    print()

    # Detect turbines
    t0 = time.time()
    print("[3/6] Detecting wind turbines...")
    turbines = detect_turbines(frame, gray, depth_norm, proc_w, proc_h)
    print(f"       Found {len(turbines)} turbines")
    for i, t in enumerate(turbines):
        print(f"         #{i+1}: x={t['base_x']:.0f} height={t['height_px']:.0f}px depth={t['depth']:.2f}")
    print(f"       ({time.time()-t0:.1f}s)")
    print()

    # Build terrain (ground + mountains as continuous surface)
    t0 = time.time()
    print("[4/6] Building terrain mesh...")
    terrain_mask = ground_mask | mountain_mask

    # Remove turbine silhouettes from terrain in the SKY ZONE ONLY.
    # Keep ground-level pixels intact for roads/grass.
    gray_t = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    horizon_row = int(proc_h * 0.55)  # below this = ground, keep it
    for t in turbines:
        margin_x = max(t['width_px'] * 3, 30)
        x0 = max(0, int(t['center_x'] - margin_x))
        x1 = min(proc_w, int(t['center_x'] + margin_x))
        y0 = max(0, int(t['top_y'] - 20))
        y1 = min(horizon_row, int(t['base_y']))  # only above horizon
        if y1 <= y0:
            continue
        # Remove dark turbine pixels from terrain (above horizon)
        dark_here = gray_t[y0:y1, x0:x1] < 140
        terrain_mask[y0:y1, x0:x1][dark_here] = False
        sky_mask[y0:y1, x0:x1][dark_here] = True

    terrain_rgb = frame_rgb.copy()

    terrain_v, terrain_c, terrain_f = build_terrain_horizontal(
        terrain_rgb, depth_norm, terrain_mask, mountain_mask, proc_w, proc_h
    )
    print(f"       Terrain: {len(terrain_v):,} verts, {len(terrain_f):,} faces")
    print(f"       ({time.time()-t0:.1f}s)")
    print()

    # Build sky backdrop (flat plane behind terrain)
    t0 = time.time()
    print("[5/6] Building sky backdrop...")
    sky_v, sky_c, sky_f = build_sky_backdrop(
        frame_rgb, sky_mask, proc_w, proc_h, len(terrain_v)
    )
    print(f"       Sky: {len(sky_v):,} verts, {len(sky_f):,} faces")
    print(f"       ({time.time()-t0:.1f}s)")
    print()

    # Build turbine models
    t0 = time.time()
    print("[6/6] Generating turbines + merging...")
    offset = len(terrain_v) + len(sky_v)
    turbine_verts = []
    turbine_colors = []
    turbine_faces = []

    for t in turbines:
        tv, tc, tf = create_turbine_3d(t, frame_rgb, proc_w, proc_h, offset)
        if len(tv) > 0:
            turbine_verts.append(tv)
            turbine_colors.append(tc)
            turbine_faces.append(tf)
            offset += len(tv)

    # Merge everything
    all_v = [terrain_v, sky_v] + turbine_verts
    all_c = [terrain_c, sky_c] + turbine_colors
    all_f = [terrain_f, sky_f] + turbine_faces

    all_v = [v for v in all_v if len(v) > 0]
    all_c = [c for c in all_c if len(c) > 0]
    all_f = [f for f in all_f if len(f) > 0]

    vertices = np.concatenate(all_v, axis=0)
    colors = np.concatenate(all_c, axis=0)
    faces = np.concatenate(all_f, axis=0)

    ply_path = output_dir / "terrain.ply"
    save_mesh_ply(vertices, colors, faces, ply_path)

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    print(f"       Total: {len(vertices):,} verts, {len(faces):,} faces")
    print(f"       Scene: {extent[0]:.0f}m wide x {extent[1]:.0f}m deep x {extent[2]:.0f}m tall")
    print(f"       Saved: {ply_path}")
    print(f"       ({time.time()-t0:.1f}s)")
    print()
    print("=" * 60)
    print("  DONE —", ply_path.absolute())
    print("  Open in Blender: Import PLY, view from front (Numpad 1)")
    print("=" * 60)


def run_midas(session, input_name, frame, proc_w, proc_h):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_input = ((img_resized - mean) / std).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    result = session.run(None, {input_name: img_input})
    depth_raw = result[0].squeeze()
    depth_map = cv2.resize(depth_raw, (proc_w, proc_h), interpolation=cv2.INTER_CUBIC)
    d_min, d_max = np.percentile(depth_map, [2, 98])
    return np.clip((depth_map - d_min) / (d_max - d_min + 1e-6), 0, 1)


def segment_scene(frame, gray, hsv, depth_norm, w, h):
    """Segment into sky (with sun/clouds), ground, mountains.

    Sky and terrain together must cover the entire frame (no gaps).
    """
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Sky: upper portion of frame — everything above the ground horizon
    sky = np.zeros((h, w), dtype=bool)

    # Far background (very low depth)
    sky |= (depth_norm < 0.12)

    # Upper region with low depth
    upper_mask = np.zeros((h, w), dtype=bool)
    upper_mask[:int(h * 0.55)] = True
    sky |= upper_mask & (depth_norm < 0.30)

    # Sunset/sun colors anywhere in upper half
    sunset = (hue < 30) & (sat > 60) & (val > 120) & upper_mask
    sky |= sunset

    # Clouds
    clouds = (gray > 170) & (sat < 50) & upper_mask
    sky |= clouds

    # Bright horizon glow
    horizon_band = np.zeros((h, w), dtype=bool)
    horizon_band[int(h * 0.35):int(h * 0.6)] = True
    sky |= horizon_band & (depth_norm < 0.15)

    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((11, 11), np.uint8)).astype(bool)
    # Force top 25% to always be sky (eliminates black gaps at top)
    sky[:int(h * 0.25)] = True

    # Mountains: mid-depth in the transition zone between sky and ground
    mountain_band = np.zeros((h, w), dtype=bool)
    mountain_band[int(h * 0.35):int(h * 0.7)] = True
    mountains = mountain_band & ~sky & (depth_norm > 0.08) & (depth_norm < 0.35)
    mountains &= (sat < 120) | (gray < 100)
    mountains = cv2.morphologyEx(mountains.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((7, 7), np.uint8)).astype(bool)

    # Ground: everything that's not sky (ensures full coverage)
    ground = ~sky & ~mountains

    return sky, ground, mountains


def build_terrain_horizontal(frame_rgb, depth_norm, terrain_mask, mountain_mask, w, h):
    """Build terrain as a tilted surface viewable from the front.

    Unified Z = (1 - row/h) * scene_height — same formula used for sky backdrop,
    so sky and terrain meet seamlessly at the boundary row.
    """
    stride = 3
    ys_img = np.arange(0, h, stride)
    xs_img = np.arange(0, w, stride)
    grid_h, grid_w = len(ys_img), len(xs_img)

    gx, gy = np.meshgrid(xs_img, ys_img)
    mask_grid = terrain_mask[gy, gx]
    depth_grid = depth_norm[gy, gx]
    mtn_grid = mountain_mask[gy, gx]

    scene_width = 250.0
    scene_depth = 80.0
    scene_height = 100.0  # total Z range for the whole image (shared with sky)
    mountain_extra = 12.0

    # X: image column → horizontal
    world_x = (gx.astype(np.float32) / w - 0.5) * scene_width

    # Y: depth from MiDaS (far = large Y)
    world_y = (1.0 - depth_grid) * scene_depth

    # Z: row-based height (top of image = high Z, bottom = Z≈0)
    row_frac = gy.astype(np.float32) / h
    world_z = (1.0 - row_frac) * scene_height
    # Mountains protrude forward slightly
    world_z[mtn_grid] += mountain_extra * (1.0 - depth_grid[mtn_grid])

    # Build vertices
    vertex_map = np.full((grid_h, grid_w), -1, dtype=np.int32)
    verts = []
    cols = []
    count = 0
    for i in range(grid_h):
        for j in range(grid_w):
            if mask_grid[i, j]:
                vertex_map[i, j] = count
                verts.append([world_x[i, j], world_y[i, j], world_z[i, j]])
                cols.append(frame_rgb[ys_img[i], xs_img[j]])
                count += 1

    if count < 100:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.int32)

    verts = np.array(verts, dtype=np.float32)
    cols = np.array(cols, dtype=np.uint8)

    # Build faces — skip triangles that span depth discontinuities
    faces = []
    max_edge = 8.0
    for i in range(grid_h - 1):
        for j in range(grid_w - 1):
            v00 = vertex_map[i, j]
            v01 = vertex_map[i, j + 1]
            v10 = vertex_map[i + 1, j]
            v11 = vertex_map[i + 1, j + 1]
            if v00 >= 0 and v10 >= 0 and v01 >= 0:
                if _edge_ok(verts, v00, v10, v01, max_edge):
                    faces.append([v00, v10, v01])
            if v10 >= 0 and v11 >= 0 and v01 >= 0:
                if _edge_ok(verts, v10, v11, v01, max_edge):
                    faces.append([v10, v11, v01])

    faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), np.int32)
    return verts, cols, faces


def build_sky_backdrop(frame_rgb, sky_mask, w, h, vertex_offset):
    """Build sky as a vertical flat plane BEHIND the terrain.

    The sky plane sits at a fixed large Y (far away) and extends
    in X (wide) and Z (tall). Turbine dark pixels are removed and
    filled with surrounding sky colors.
    """
    # Clean the sky: replace dark foreground objects (turbine silhouettes)
    # with smoothly interpolated sky color using heavy blur
    sky_rgb = frame_rgb.copy()
    gray_sky = cv2.cvtColor(cv2.cvtColor(sky_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    # Replace dark sky pixels with row-average sky color
    # (computed only from valid/bright sky pixels in each row)
    local_mean = cv2.blur(gray_sky.astype(np.float32), (81, 81))
    dark_in_sky = sky_mask & (gray_sky.astype(np.float32) < local_mean - 15)
    dark_in_sky |= sky_mask & (gray_sky < 120)
    dark_in_sky = cv2.dilate(dark_in_sky.astype(np.uint8),
                             np.ones((15, 15), np.uint8)).astype(bool)

    # For each row, compute average color of non-dark sky pixels
    for row in range(h):
        valid = sky_mask[row] & ~dark_in_sky[row]
        if valid.sum() > 10:
            avg_color = sky_rgb[row][valid].mean(axis=0).astype(np.uint8)
        else:
            avg_color = sky_rgb[row].mean(axis=0).astype(np.uint8)
        # Replace dark pixels in this row with the row's average sky color
        sky_rgb[row][dark_in_sky[row]] = avg_color

    stride = 4
    ys_img = np.arange(0, h, stride)
    xs_img = np.arange(0, w, stride)
    grid_h, grid_w = len(ys_img), len(xs_img)

    gx, gy = np.meshgrid(xs_img, ys_img)
    mask_grid = sky_mask[gy, gx]

    scene_width = 250.0
    scene_height = 100.0  # Same as terrain — unified Z
    sky_y = 90.0  # Behind the terrain (terrain Y goes to ~80)

    # Same Z formula as terrain: Z = (1 - row/h) * scene_height
    world_x = (gx.astype(np.float32) / w - 0.5) * scene_width * 1.2
    world_z = (1.0 - gy.astype(np.float32) / h) * scene_height

    vertex_map = np.full((grid_h, grid_w), -1, dtype=np.int32)
    verts = []
    cols = []
    count = 0
    for i in range(grid_h):
        for j in range(grid_w):
            if mask_grid[i, j]:
                vertex_map[i, j] = count
                verts.append([world_x[i, j], sky_y, world_z[i, j]])
                cols.append(sky_rgb[ys_img[i], xs_img[j]])
                count += 1

    if count < 50:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.int32)

    verts = np.array(verts, dtype=np.float32)
    cols = np.array(cols, dtype=np.uint8)

    faces = []
    for i in range(grid_h - 1):
        for j in range(grid_w - 1):
            v00 = vertex_map[i, j]
            v01 = vertex_map[i, j + 1]
            v10 = vertex_map[i + 1, j]
            v11 = vertex_map[i + 1, j + 1]
            if v00 >= 0 and v10 >= 0 and v01 >= 0:
                faces.append([v00 + vertex_offset, v10 + vertex_offset, v01 + vertex_offset])
            if v10 >= 0 and v11 >= 0 and v01 >= 0:
                faces.append([v10 + vertex_offset, v11 + vertex_offset, v01 + vertex_offset])

    faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), np.int32)
    return verts, cols, faces


def detect_turbines(frame, gray, depth_norm, w, h):
    """Detect turbines via Hough vertical lines + clustering."""
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=60, maxLineGap=20)

    if lines is None or len(lines) == 0:
        return []

    vertical_segs = []
    for i in range(len(lines)):
        x1, y1, x2, y2 = int(lines[i][0]), int(lines[i][1]), int(lines[i][2]), int(lines[i][3])
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length > 80 and abs(dx) < length * 0.25:
            vertical_segs.append((x1, y1, x2, y2, length))

    if not vertical_segs:
        return []

    vertical_segs.sort(key=lambda s: s[0])
    clusters = []
    current_cluster = [vertical_segs[0]]

    for seg in vertical_segs[1:]:
        if abs(seg[0] - current_cluster[-1][0]) < 40:
            current_cluster.append(seg)
        else:
            clusters.append(current_cluster)
            current_cluster = [seg]
    clusters.append(current_cluster)

    turbines = []
    for cluster in clusters:
        all_x = [s[0] for s in cluster] + [s[2] for s in cluster]
        all_y = [s[1] for s in cluster] + [s[3] for s in cluster]
        cx = int(np.mean(all_x))
        top_y = min(all_y)
        base_y = max(all_y)
        height_px = base_y - top_y
        width_px = max(all_x) - min(all_x) + 1
        total_len = sum(s[4] for s in cluster)

        if height_px < h * 0.08 or total_len < 100:
            continue

        y_start = max(0, top_y)
        y_end = min(h, base_y)
        x_start = max(0, cx - 10)
        x_end = min(w, cx + 10)
        avg_depth = depth_norm[y_start:y_end, x_start:x_end].mean()

        if avg_depth < 0.2:
            continue

        turbines.append({
            'base_x': float(cx),
            'base_y': float(base_y),
            'top_y': float(top_y),
            'center_x': float(cx),
            'height_px': float(height_px),
            'width_px': float(width_px),
            'depth': float(avg_depth),
        })

    turbines.sort(key=lambda t: -t['depth'])
    return turbines[:8]


def create_turbine_3d(turbine_info, frame_rgb, img_w, img_h, vertex_offset):
    """Create 3D turbine: cylinder tower + nacelle + 3 blades.

    Positioned in the same coordinate system as terrain:
      X = left-right, Y = depth (far=large), Z = up
    """
    scene_width = 250.0
    scene_depth = 80.0
    scene_height = 100.0

    # X position from image column
    world_x = (turbine_info['base_x'] / img_w - 0.5) * scene_width

    # Y position from depth (closer turbines = smaller Y)
    world_y = (1.0 - turbine_info['depth']) * scene_depth

    # Tower base on the terrain surface: Z = (1 - row/h) * scene_height
    row_frac = turbine_info['base_y'] / img_h
    world_z_base = (1.0 - row_frac) * scene_height

    # Scale turbine by its apparent size
    apparent_scale = turbine_info['depth']
    tower_height = 35.0 * apparent_scale + 15.0
    tower_radius = 1.2 * apparent_scale + 0.4
    blade_length = 18.0 * apparent_scale + 8.0
    nacelle_size = 2.5 * apparent_scale + 1.0

    # Sample color from the pole
    py_mid = int((turbine_info['base_y'] + turbine_info['top_y']) / 2)
    px = int(turbine_info['center_x'])
    py_mid = min(py_mid, img_h - 1)
    px = min(px, img_w - 1)
    samples = []
    for off in [-30, -10, 0, 10, 30]:
        sy = min(max(0, py_mid + off), img_h - 1)
        samples.append(frame_rgb[sy, px])
    pole_color = np.median(samples, axis=0).astype(np.uint8)

    verts = []
    colors = []
    faces = []

    # --- Tower (cylinder) ---
    n_sides = 12
    n_sections = 8
    for sec in range(n_sections + 1):
        frac = sec / n_sections
        z = world_z_base + frac * tower_height
        r = tower_radius * (1.0 - frac * 0.3)
        for side in range(n_sides):
            angle = 2 * np.pi * side / n_sides
            x = world_x + r * np.cos(angle)
            y = world_y + r * np.sin(angle)
            verts.append([x, y, z])
            shade = 0.7 + 0.3 * abs(np.cos(angle))
            c = np.clip(pole_color * shade, 0, 255).astype(np.uint8)
            colors.append(c)

    for sec in range(n_sections):
        for side in range(n_sides):
            ns = (side + 1) % n_sides
            i0 = sec * n_sides + side
            i1 = sec * n_sides + ns
            i2 = (sec + 1) * n_sides + side
            i3 = (sec + 1) * n_sides + ns
            faces.append([i0 + vertex_offset, i2 + vertex_offset, i1 + vertex_offset])
            faces.append([i1 + vertex_offset, i2 + vertex_offset, i3 + vertex_offset])

    # --- Nacelle (box) ---
    nacelle_base = len(verts)
    nz = world_z_base + tower_height
    ns = nacelle_size
    corners = [
        [world_x - ns, world_y - ns / 2, nz - ns / 3],
        [world_x + ns, world_y - ns / 2, nz - ns / 3],
        [world_x + ns, world_y + ns / 2, nz - ns / 3],
        [world_x - ns, world_y + ns / 2, nz - ns / 3],
        [world_x - ns, world_y - ns / 2, nz + ns / 3],
        [world_x + ns, world_y - ns / 2, nz + ns / 3],
        [world_x + ns, world_y + ns / 2, nz + ns / 3],
        [world_x - ns, world_y + ns / 2, nz + ns / 3],
    ]
    nacelle_color = np.array([190, 190, 190], dtype=np.uint8)
    for corner in corners:
        verts.append(corner)
        colors.append(nacelle_color)

    nb = nacelle_base + vertex_offset
    for bf in [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
               [2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]]:
        faces.append([nb + bf[0], nb + bf[1], nb + bf[2]])

    # --- Blades (3 triangular) ---
    hub_z = nz + ns / 3
    blade_color = np.array([220, 220, 215], dtype=np.uint8)

    for blade_idx in range(3):
        angle = blade_idx * 2 * np.pi / 3 + 0.4
        tip_x = world_x + blade_length * np.cos(angle)
        tip_z = hub_z + blade_length * np.sin(angle)
        blade_w = 1.2

        bi = len(verts)
        verts.append([world_x, world_y - blade_w / 3, hub_z])
        verts.append([world_x, world_y + blade_w / 3, hub_z])
        verts.append([tip_x, world_y, tip_z])
        colors.extend([blade_color, blade_color, blade_color])
        faces.append([bi + vertex_offset, bi + 1 + vertex_offset, bi + 2 + vertex_offset])

        # Back face
        verts.append([world_x, world_y + blade_w / 3, hub_z])
        verts.append([world_x, world_y - blade_w / 3, hub_z])
        verts.append([tip_x, world_y, tip_z])
        colors.extend([blade_color, blade_color, blade_color])
        faces.append([bi + 3 + vertex_offset, bi + 4 + vertex_offset, bi + 5 + vertex_offset])

    return (np.array(verts, np.float32), np.array(colors, np.uint8),
            np.array(faces, np.int32))


def _edge_ok(verts, a, b, c, max_len):
    pa, pb, pc = verts[a], verts[b], verts[c]
    return max(np.linalg.norm(pa - pb), np.linalg.norm(pb - pc),
               np.linalg.norm(pc - pa)) < max_len


def save_mesh_ply(vertices, colors, faces, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
