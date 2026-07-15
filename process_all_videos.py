"""
PROCESADOR DE MÚLTIPLES VIDEOS - POSE DETECTION
Procesa Video 2, 3 y 4 con los mismos parámetros que Video 1
"""

import os
import sys
import cv2
import numpy as np
import math
from collections import deque

print("\n" + "=" * 80)
print("PROCESADOR DE MÚLTIPLES VIDEOS - POSE DETECTION")
print("=" * 80 + "\n")

try:
    from ultralytics import YOLO
    print("✓ YOLOv8 importado\n")
except ImportError:
    print("✗ Error: instala ultralytics")
    sys.exit(1)

print("=" * 80)
print("Cargando modelo YOLOv8 Pose...")
print("=" * 80 + "\n")

try:
    model = YOLO('yolov8n-pose.pt')
    print("✓ Modelo YOLOv8 Pose cargado\n")
except:
    print("✗ Error cargando modelo")
    sys.exit(1)

# Parámetros
ALPHA = 0.6  # Suavizado exponencial
VISIBILITY_THRESHOLD = 0.15  # Umbral bajo
CONFIDENCE_MIN = 0.35  # Confianza mínima

# Tamaño de vértices (puntos)
POINT_RADIUS = 4
POINT_BORDER = 1

# Overlay de ángulos (marcha)
ANGLE_VISIBILITY_THRESHOLD = 0.25
ANGLE_SMOOTH_ALPHA = 0.45
ANGLE_FONT = cv2.FONT_HERSHEY_SIMPLEX
ANGLE_FONT_SCALE = 0.55
ANGLE_TEXT_THICKNESS = 1
ANGLE_OUTLINE_THICKNESS = 2
ANGLE_TEXT_COLOR = (255, 255, 255)
ANGLE_OUTLINE_COLOR = (0, 0, 0)

ANGLE_PANEL_ALPHA = 0.35
ANGLE_PANEL_BG_COLOR = (0, 0, 0)
ANGLE_PANEL_PADDING = 5
ANGLE_PANEL_FONT_SCALE = 0.46
ANGLE_JOINT_FONT_SCALE = 0.38

UI_ACCENT = (255, 255, 0)       # cian (BGR)
UI_WARN = (0, 165, 255)         # naranja
UI_BAD = (0, 0, 255)            # rojo
UI_GOOD = (0, 220, 0)           # verde
UI_MUTED = (200, 200, 200)

ASYM_WARN_DEG = 10
PELVIS_WARN_DEG = 8
TRUNK_WARN_DEG = 10
KNEE_ROM_LOW_DEG = 15


def _angle_degrees(a_xy: np.ndarray, b_xy: np.ndarray, c_xy: np.ndarray) -> float | None:
    ba = a_xy - b_xy
    bc = c_xy - b_xy

    ba_norm = float(np.linalg.norm(ba))
    bc_norm = float(np.linalg.norm(bc))
    if ba_norm == 0.0 or bc_norm == 0.0:
        return None

    cos_angle = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _vector_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float | None:
    v1_norm = float(np.linalg.norm(v1))
    v2_norm = float(np.linalg.norm(v2))
    if v1_norm == 0.0 or v2_norm == 0.0:
        return None

    cos_angle = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _put_text_with_outline(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    text_color: tuple[int, int, int],
    text_thickness: int,
    outline_color: tuple[int, int, int],
    outline_thickness: int,
) -> None:
    cv2.putText(frame, text, org, ANGLE_FONT, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, ANGLE_FONT, font_scale, text_color, text_thickness, cv2.LINE_AA)


def _draw_angle_panel(
    frame: np.ndarray,
    lines: list[str] | list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (10, 10),
) -> tuple[int, int, int, int] | None:
    if not lines:
        return None

    x0, y0 = origin
    h, w = frame.shape[:2]

    # Evitar que los textos se recorten por ancho: envolver a 2 líneas o truncar.
    max_text_w = max(80, (w - 1) - (x0 + ANGLE_PANEL_PADDING) - ANGLE_PANEL_PADDING)

    def _wrap_to_width(text: str) -> list[str]:
        text = " ".join(str(text).split())
        if not text:
            return [""]

        def _fits(s: str) -> bool:
            (tw, _), _b = cv2.getTextSize(s, ANGLE_FONT, ANGLE_PANEL_FONT_SCALE, ANGLE_TEXT_THICKNESS)
            return tw <= max_text_w

        if _fits(text):
            return [text]

        # Envolver en 2 líneas (preferible) usando espacios.
        words = text.split(" ")
        if len(words) >= 2:
            line1 = ""
            for i, word in enumerate(words):
                cand = (line1 + " " + word).strip()
                if _fits(cand):
                    line1 = cand
                else:
                    # Todo lo demás va a la línea 2
                    line2 = " ".join(words[i:]).strip()
                    if _fits(line2):
                        return [line1, line2]
                    # Si la línea 2 sigue larga, truncar con '...'
                    while line2 and not _fits(line2 + "..."):
                        line2 = line2[:-1]
                    return [line1, (line2 + "...") if line2 else "..."]

        # Si no hay espacios útiles, truncar directo.
        s = text
        while s and not _fits(s + "..."):
            s = s[:-1]
        return [(s + "...") if s else "..."]

    expanded_lines: list[tuple[str, tuple[int, int, int]]] = []
    for ln in lines:
        if isinstance(ln, tuple):
            text, color = ln
        else:
            text, color = ln, ANGLE_TEXT_COLOR

        wrapped = _wrap_to_width(text)
        for j, part in enumerate(wrapped):
            # Mantener color; si se parte en 2 líneas, segunda un poco más "muted" si era texto de UI_MUTED
            part_color = color
            if j > 0 and color == UI_MUTED:
                part_color = UI_MUTED
            expanded_lines.append((part, part_color))

    line_texts = [t for (t, _c) in expanded_lines]
    sizes = [cv2.getTextSize(line, ANGLE_FONT, ANGLE_PANEL_FONT_SCALE, ANGLE_TEXT_THICKNESS) for line in line_texts]
    widths = [s[0][0] for s in sizes]
    heights = [s[0][1] for s in sizes]
    baselines = [s[1] for s in sizes]

    text_h = max(heights) if heights else 0
    base_h = max(baselines) if baselines else 0
    line_h = text_h + base_h + 3

    panel_w = int(max(widths) + 2 * ANGLE_PANEL_PADDING)
    panel_h = int(len(expanded_lines) * line_h + 2 * ANGLE_PANEL_PADDING)

    x1 = max(0, min(w - 1, x0 + panel_w))
    y1 = max(0, min(h - 1, y0 + panel_h))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), ANGLE_PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, ANGLE_PANEL_ALPHA, frame, 1 - ANGLE_PANEL_ALPHA, 0, frame)

    y = y0 + ANGLE_PANEL_PADDING + text_h
    for line, color in expanded_lines:

        _put_text_with_outline(
            frame,
            line,
            (x0 + ANGLE_PANEL_PADDING, y),
            ANGLE_PANEL_FONT_SCALE,
            color,
            ANGLE_TEXT_THICKNESS,
            ANGLE_OUTLINE_COLOR,
            ANGLE_OUTLINE_THICKNESS,
        )
        y += line_h

    return (x0, y0, x1, y1)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _draw_tag(
    frame: np.ndarray,
    occupied: list[tuple[int, int, int, int]],
    text: str,
    anchor_xy: tuple[int, int],
    prefer: str,
    text_color: tuple[int, int, int] = UI_ACCENT,
) -> None:
    h, w = frame.shape[:2]
    (tw, th), base = cv2.getTextSize(text, ANGLE_FONT, ANGLE_JOINT_FONT_SCALE, ANGLE_TEXT_THICKNESS)
    pad = 4

    ax, ay = anchor_xy
    # Candidatos (dx, dy) para evitar solapes
    if prefer == "L":
        candidates = [(-tw - 14, -th - 8), (-tw - 14, 6), (-tw - 14, -2 * th - 10), (-tw - 14, 14)]
    else:
        candidates = [(14, -th - 8), (14, 6), (14, -2 * th - 10), (14, 14)]

    for dx, dy in candidates:
        x = ax + dx
        y = ay + dy
        x1 = int(max(0, min(w - 1, x)))
        y1 = int(max(0, min(h - 1, y)))
        x2 = int(max(0, min(w - 1, x1 + tw + 2 * pad)))
        y2 = int(max(0, min(h - 1, y1 + th + base + 2 * pad)))

        box = (x1, y1, x2, y2)
        if any(_boxes_intersect(box, other) for other in occupied):
            continue

        # Fondo sólido (legible "clínica")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        org = (x1 + pad, y1 + pad + th)
        cv2.putText(frame, text, org, ANGLE_FONT, ANGLE_JOINT_FONT_SCALE, text_color, ANGLE_TEXT_THICKNESS, cv2.LINE_AA)
        occupied.append(box)
        return


def _smooth_value(prev: float | None, new: float | None, alpha: float) -> float | None:
    if new is None:
        return prev
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev

# Esqueleto (sin conexiones de cabeza)
SKELETON_CONNECTIONS = [
    (5, 6),                   # Hombros
    (5, 7), (7, 9),           # Brazo izq
    (6, 8), (8, 10),          # Brazo der
    (5, 11), (6, 12),         # Hombros a caderas
    (11, 12),                 # Cintura
    (11, 13), (13, 15),       # Pierna izq
    (12, 14), (14, 16)        # Pierna der
]

# Videos a procesar (auto-detecta en la carpeta del script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_videos_to_process(video_dir: str) -> list[tuple[str, str]]:
    videos: list[tuple[str, str]] = []
    for name in sorted(os.listdir(video_dir)):
        if not name.lower().endswith(".mp4"):
            continue
        if name.lower().endswith("_with_landmarks.mp4"):
            continue

        in_path = os.path.join(video_dir, name)
        stem, _ext = os.path.splitext(name)
        out_name = f"{stem}_with_landmarks.mp4"
        out_path = os.path.join(video_dir, out_name)
        videos.append((in_path, out_path))

    return videos


videos_to_process = _build_videos_to_process(SCRIPT_DIR)

for input_video, output_video in videos_to_process:
    if not os.path.exists(input_video):
        print(f"✗ {input_video} no encontrado\n")
        continue
    
    print("=" * 80)
    print(f"Procesando: {input_video}")
    print("=" * 80 + "\n")
    
    # Leer video
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30
    
    print(f"Resolución: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Fotogramas: {total_frames}\n")
    
    # Crear writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    # Historia para suavizado
    smooth_history = {}
    # Estado de métricas (para mantener último valor válido)
    gait_state = {
        'L': {'hip': None, 'knee': None, 'tibia': None},
        'R': {'hip': None, 'knee': None, 'tibia': None},
        'pelvis_obliq': None,
        'trunk_lean': None,
        'quality': None,
        'face_q': None,
        'view': None,  # 'FRONT' | 'BACK'
        'hip_shift': None,
        'step_width': None,
        'knee_dev_L': None,
        'knee_dev_R': None,
    }

    # Historial corto (ROM aproximado último ~1s)
    knee_hist_L = deque(maxlen=max(1, fps))
    knee_hist_R = deque(maxlen=max(1, fps))
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % max(1, total_frames // 20) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  {progress:.1f}% ({frame_count}/{total_frames})")
            
            h, w = frame.shape[:2]

            # Datos de articulaciones del frame (para etiquetas); si no hay detección, quedan None
            joints_xy = None
            joints_conf = None
            
            # Detección
            results = model(frame, verbose=False, conf=0.3)
            
            if results and len(results) > 0:
                result = results[0]

                if (result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None):
                    # Seleccionar la persona con mayor confianza (más estable para marcha)
                    try:
                        confs = result.boxes.conf.detach().cpu().numpy()
                        best_idx = int(np.argmax(confs))
                    except Exception:
                        best_idx = 0

                    if float(result.boxes.conf[best_idx]) > CONFIDENCE_MIN:
                        keypoints = result.keypoints.xy[best_idx].cpu().numpy()
                        confidences = (
                            result.keypoints.conf[best_idx].cpu().numpy()
                            if result.keypoints.conf is not None
                            else np.ones(17)
                        )
                    
                    # Suavizado
                    smoothed_points = []
                    for i, (pt, conf) in enumerate(zip(keypoints, confidences)):
                        if i not in smooth_history:
                            smooth_history[i] = {'x': pt[0], 'y': pt[1], 'conf': float(conf)}
                        
                        if float(conf) > 0:
                            smooth_history[i]['x'] = ALPHA * pt[0] + (1 - ALPHA) * smooth_history[i]['x']
                            smooth_history[i]['y'] = ALPHA * pt[1] + (1 - ALPHA) * smooth_history[i]['y']
                            smooth_history[i]['conf'] = float(conf)
                        
                        smoothed_points.append(smooth_history[i])
                    
                    # Dibujar esqueleto
                    for start_idx, end_idx in SKELETON_CONNECTIONS:
                        start_pt = smoothed_points[start_idx]
                        end_pt = smoothed_points[end_idx]
                        
                        if start_pt['conf'] > VISIBILITY_THRESHOLD and end_pt['conf'] > VISIBILITY_THRESHOLD:
                            pt1 = (int(start_pt['x']), int(start_pt['y']))
                            pt2 = (int(end_pt['x']), int(end_pt['y']))
                            
                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                conf_avg = (start_pt['conf'] + end_pt['conf']) / 2
                                intensity = int(min(255, conf_avg * 500))
                                color = (0, intensity, 100)
                                cv2.line(frame, pt1, pt2, color, 3)
                    
                    # Dibujar puntos
                    for i, pt_data in enumerate(smoothed_points):
                        # Punto 0: solo de frente
                        if i == 0:
                            if pt_data['conf'] > 0.4:
                                x = int(pt_data['x'])
                                y = int(pt_data['y'])
                                conf = pt_data['conf']
                                
                                if 0 <= x < w and 0 <= y < h:
                                    intensity = int(min(255, conf * 500))
                                    color = (0, intensity, 255 - intensity)
                                    
                                    cv2.circle(frame, (x, y), POINT_RADIUS, color, -1)
                                    cv2.circle(frame, (x, y), POINT_RADIUS, (255, 255, 255), POINT_BORDER)
                        
                        # Puntos 1-16: cara y cuerpo
                        elif i >= 1:
                            if pt_data['conf'] > VISIBILITY_THRESHOLD:
                                x = int(pt_data['x'])
                                y = int(pt_data['y'])
                                conf = pt_data['conf']
                                
                                if 0 <= x < w and 0 <= y < h:
                                    intensity = int(min(255, conf * 500))
                                    color = (0, intensity, 255 - intensity)
                                    
                                    cv2.circle(frame, (x, y), POINT_RADIUS, color, -1)
                                    cv2.circle(frame, (x, y), POINT_RADIUS, (255, 255, 255), POINT_BORDER)

                        # Guardar joints del frame para etiquetas
                        joints_xy = np.array([[float(p['x']), float(p['y'])] for p in smoothed_points], dtype=np.float32)
                        joints_conf = np.array([float(p['conf']) for p in smoothed_points], dtype=np.float32)

                        # Calidad (conf media de puntos relevantes)
                        relevant = [5, 6, 11, 12, 13, 14, 15, 16]
                        try:
                            q = float(np.mean([float(joints_conf[i]) for i in relevant]))
                        except Exception:
                            q = None

                        # Confianza de cara (para inferir si va de frente)
                        face_idxs = [0, 1, 2, 3, 4]  # nariz, ojos, orejas
                        try:
                            face_q = float(np.mean([float(joints_conf[i]) for i in face_idxs]))
                        except Exception:
                            face_q = None

                        # Métricas de marcha
                        def _pt(idx: int) -> tuple[np.ndarray, float]:
                            return joints_xy[idx], float(joints_conf[idx])

                        l_shoulder, l_shoulder_c = _pt(5)
                        r_shoulder, r_shoulder_c = _pt(6)
                        l_hip, l_hip_c = _pt(11)
                        r_hip, r_hip_c = _pt(12)
                        l_knee, l_knee_c = _pt(13)
                        r_knee, r_knee_c = _pt(14)
                        l_ankle, l_ankle_c = _pt(15)
                        r_ankle, r_ankle_c = _pt(16)

                        # Cadera: (hombro, cadera, rodilla)
                        left_hip = None
                        right_hip = None
                        if min(l_shoulder_c, l_hip_c, l_knee_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            left_hip = _angle_degrees(l_shoulder, l_hip, l_knee)
                        if min(r_shoulder_c, r_hip_c, r_knee_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            right_hip = _angle_degrees(r_shoulder, r_hip, r_knee)

                        # Rodilla: (cadera, rodilla, tobillo)
                        left_knee = None
                        right_knee = None
                        if min(l_hip_c, l_knee_c, l_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            left_knee = _angle_degrees(l_hip, l_knee, l_ankle)
                        if min(r_hip_c, r_knee_c, r_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            right_knee = _angle_degrees(r_hip, r_knee, r_ankle)

                        # Tobillo (proxy): inclinación de tibia vs vertical
                        vertical_up = np.array([0.0, -1.0], dtype=np.float32)
                        left_tibia = None
                        right_tibia = None
                        if min(l_knee_c, l_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            left_tibia = _vector_angle_degrees(l_knee - l_ankle, vertical_up)
                        if min(r_knee_c, r_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            right_tibia = _vector_angle_degrees(r_knee - r_ankle, vertical_up)

                        # Pelvis oblicuidad: línea cadera-cadera vs horizontal
                        pelvis_obliq = None
                        if min(l_hip_c, r_hip_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            v = r_hip - l_hip
                            pelvis_obliq = _vector_angle_degrees(v, np.array([1.0, 0.0], dtype=np.float32))

                        # Tronco: vector medio caderas -> medio hombros vs vertical
                        trunk_lean = None
                        if min(l_hip_c, r_hip_c, l_shoulder_c, r_shoulder_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            mid_hip = (l_hip + r_hip) / 2.0
                            mid_sh = (l_shoulder + r_shoulder) / 2.0
                            trunk_lean = _vector_angle_degrees(mid_sh - mid_hip, vertical_up)

                        # Métricas útiles para vista frontal (2D normalizadas)
                        hip_width = None
                        if min(l_hip_c, r_hip_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            hip_width = float(abs(r_hip[0] - l_hip[0]))

                        hip_shift = None
                        step_width = None
                        if hip_width is not None and hip_width > 1.0:
                            mid_hip = (l_hip + r_hip) / 2.0
                            hip_shift = float((mid_hip[0] - (w / 2.0)) / hip_width)
                            if min(l_ankle_c, r_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                                step_width = float(abs(r_ankle[0] - l_ankle[0]) / hip_width)

                        def _point_line_distance_norm(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
                            ab = b - a
                            ab_len = float(np.linalg.norm(ab))
                            if ab_len == 0.0:
                                return None
                            # Distancia de p a línea ab
                            ap = p - a
                            cross_2d = float(ab[0] * ap[1] - ab[1] * ap[0])
                            dist = float(abs(cross_2d) / ab_len)
                            return dist / ab_len

                        knee_dev_L = None
                        knee_dev_R = None
                        if min(l_hip_c, l_knee_c, l_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            knee_dev_L = _point_line_distance_norm(l_knee, l_hip, l_ankle)
                        if min(r_hip_c, r_knee_c, r_ankle_c) >= ANGLE_VISIBILITY_THRESHOLD:
                            knee_dev_R = _point_line_distance_norm(r_knee, r_hip, r_ankle)

                        # Actualizar estado (suavizado y retención)
                        gait_state['L']['hip'] = _smooth_value(gait_state['L']['hip'], left_hip, ANGLE_SMOOTH_ALPHA)
                        gait_state['R']['hip'] = _smooth_value(gait_state['R']['hip'], right_hip, ANGLE_SMOOTH_ALPHA)
                        gait_state['L']['knee'] = _smooth_value(gait_state['L']['knee'], left_knee, ANGLE_SMOOTH_ALPHA)
                        gait_state['R']['knee'] = _smooth_value(gait_state['R']['knee'], right_knee, ANGLE_SMOOTH_ALPHA)
                        gait_state['L']['tibia'] = _smooth_value(gait_state['L']['tibia'], left_tibia, ANGLE_SMOOTH_ALPHA)
                        gait_state['R']['tibia'] = _smooth_value(gait_state['R']['tibia'], right_tibia, ANGLE_SMOOTH_ALPHA)
                        gait_state['pelvis_obliq'] = _smooth_value(gait_state['pelvis_obliq'], pelvis_obliq, ANGLE_SMOOTH_ALPHA)
                        gait_state['trunk_lean'] = _smooth_value(gait_state['trunk_lean'], trunk_lean, ANGLE_SMOOTH_ALPHA)
                        gait_state['quality'] = _smooth_value(gait_state['quality'], q, 0.25)
                        gait_state['face_q'] = _smooth_value(gait_state['face_q'], face_q, 0.25)
                        gait_state['hip_shift'] = _smooth_value(gait_state['hip_shift'], hip_shift, 0.25)
                        gait_state['step_width'] = _smooth_value(gait_state['step_width'], step_width, 0.25)
                        gait_state['knee_dev_L'] = _smooth_value(gait_state['knee_dev_L'], knee_dev_L, 0.25)
                        gait_state['knee_dev_R'] = _smooth_value(gait_state['knee_dev_R'], knee_dev_R, 0.25)

                        # Vista (histeresis simple)
                        fq = gait_state['face_q']
                        if fq is not None:
                            if gait_state['view'] != 'FRONT' and fq >= 0.35:
                                gait_state['view'] = 'FRONT'
                            elif gait_state['view'] != 'BACK' and fq <= 0.20:
                                gait_state['view'] = 'BACK'

                        if gait_state['L']['knee'] is not None:
                            knee_hist_L.append(float(gait_state['L']['knee']))
                        if gait_state['R']['knee'] is not None:
                            knee_hist_R.append(float(gait_state['R']['knee']))

            # Dibujar overlay clínico SIEMPRE (usa último valor válido)
            def _fmt_int(v: float | None) -> str:
                return "--" if v is None else f"{v:.0f}"

            def _delta(a: float | None, b: float | None) -> float | None:
                if a is None or b is None:
                    return None
                return abs(float(a) - float(b))

            hip_L = gait_state['L']['hip']
            hip_R = gait_state['R']['hip']
            knee_L = gait_state['L']['knee']
            knee_R = gait_state['R']['knee']
            tib_L = gait_state['L']['tibia']
            tib_R = gait_state['R']['tibia']
            pelvis = gait_state['pelvis_obliq']
            trunk = gait_state['trunk_lean']
            quality = gait_state['quality']
            face_q = gait_state['face_q']
            view = gait_state['view']
            hip_shift = gait_state['hip_shift']
            step_width = gait_state['step_width']
            knee_dev_L = gait_state['knee_dev_L']
            knee_dev_R = gait_state['knee_dev_R']

            d_knee = _delta(knee_L, knee_R)
            d_hip = _delta(hip_L, hip_R)
            d_tib = _delta(tib_L, tib_R)

            # ROM rodilla último ~1s
            rom_L = (max(knee_hist_L) - min(knee_hist_L)) if len(knee_hist_L) >= 5 else None
            rom_R = (max(knee_hist_R) - min(knee_hist_R)) if len(knee_hist_R) >= 5 else None

            t_sec = frame_count / float(fps) if fps else 0.0

            # Indicadores (abreviados para no tapar el video)
            flags: list[str] = []
            if d_knee is not None and d_knee >= ASYM_WARN_DEG:
                flags.append("asimet R")
            if d_hip is not None and d_hip >= ASYM_WARN_DEG:
                flags.append("asimet C")
            if pelvis is not None and float(pelvis) >= PELVIS_WARN_DEG:
                flags.append("pelvis")
            if trunk is not None and float(trunk) >= TRUNK_WARN_DEG:
                flags.append("tronco")
            if rom_L is not None and float(rom_L) <= KNEE_ROM_LOW_DEG:
                flags.append("ROM I")
            if rom_R is not None and float(rom_R) <= KNEE_ROM_LOW_DEG:
                flags.append("ROM D")

            # Solo útil/fiable sobre todo cuando va de frente
            if view == 'FRONT':
                if knee_dev_L is not None and float(knee_dev_L) >= 0.12:
                    flags.append("alin I")
                if knee_dev_R is not None and float(knee_dev_R) >= 0.12:
                    flags.append("alin D")

            if not flags:
                indicator_text = "IND: OK"
                indicator_color = UI_GOOD
            elif len(flags) <= 2:
                indicator_text = "IND: REVISAR (" + ", ".join(flags[:2]) + ")"
                indicator_color = UI_WARN
            else:
                indicator_text = "IND: ALERTA (" + ", ".join(flags[:3]) + ")"
                indicator_color = UI_BAD

            q_txt = "--" if quality is None else f"{int(max(0, min(100, quality * 100)))}%"
            view_txt = "AUTO" if view is None else ("FRONTAL" if view == 'FRONT' else "ESPALDA")
            fq_txt = "--" if face_q is None else f"{int(max(0, min(100, face_q * 100)))}%"

            panel_lines: list[tuple[str, tuple[int, int, int]]] = [
                ("REPORTE DE MARCHA", UI_ACCENT),
                (f"t={t_sec:05.1f}s  f={frame_count}/{total_frames}  q {q_txt}  v {view_txt}", UI_ACCENT),
                (indicator_text, indicator_color),
                (f"Cadera   I {_fmt_int(hip_L):>3}   D {_fmt_int(hip_R):>3}", ANGLE_TEXT_COLOR),
                (f"Rodilla  I {_fmt_int(knee_L):>3}   D {_fmt_int(knee_R):>3}", ANGLE_TEXT_COLOR),
                (f"Tibia incl  I {_fmt_int(tib_L):>3}   D {_fmt_int(tib_R):>3}", ANGLE_TEXT_COLOR),
                (f"Pelvis {_fmt_int(pelvis):>3}   Tronco {_fmt_int(trunk):>3}", ANGLE_TEXT_COLOR),
            ]

            if view == 'FRONT':
                devL = "--" if knee_dev_L is None else f"{knee_dev_L*100:.0f}%"
                devR = "--" if knee_dev_R is None else f"{knee_dev_R*100:.0f}%"
                sw = "--" if step_width is None else f"{step_width:.2f}"
                hs = "--" if hip_shift is None else f"{hip_shift:+.2f}"
                panel_lines.append((f"Alineac rodilla  I {devL:>3}  D {devR:>3}", UI_MUTED))
                panel_lines.append((f"Paso {sw}   PelvisX {hs}", UI_MUTED))
            else:
                pass

            if d_knee is not None or d_hip is not None or d_tib is not None:
                panel_lines.append((f"Asimetria  C {_fmt_int(d_hip):>3}  R {_fmt_int(d_knee):>3}  T {_fmt_int(d_tib):>3}", UI_MUTED))

            if rom_L is not None or rom_R is not None:
                panel_lines.append((f"ROM rodilla (1s)  I {_fmt_int(rom_L):>3}  D {_fmt_int(rom_R):>3}", UI_MUTED))

            panel_box = _draw_angle_panel(frame, panel_lines, origin=(10, 10))

            # Etiquetas en articulaciones (solo si hay joints del frame)
            if joints_xy is not None and joints_conf is not None:
                occupied: list[tuple[int, int, int, int]] = []
                if panel_box is not None:
                    occupied.append(panel_box)

                def _tag(idx: int, prefer: str, text: str) -> None:
                    if float(joints_conf[idx]) < ANGLE_VISIBILITY_THRESHOLD:
                        return
                    xj, yj = int(joints_xy[idx][0]), int(joints_xy[idx][1])
                    if 0 <= xj < w and 0 <= yj < h:
                        _draw_tag(frame, occupied, text, (xj, yj), prefer)

                # Tags compactos para no pisarse (sin símbolo °)
                if hip_L is not None:
                    _tag(11, "L", f"C I{hip_L:.0f}")
                if hip_R is not None:
                    _tag(12, "R", f"C D{hip_R:.0f}")
                if knee_L is not None:
                    _tag(13, "L", f"R I{knee_L:.0f}")
                if knee_R is not None:
                    _tag(14, "R", f"R D{knee_R:.0f}")
                if tib_L is not None:
                    _tag(15, "L", f"T I{tib_L:.0f}")
                if tib_R is not None:
                    _tag(16, "R", f"T D{tib_R:.0f}")
            
            out.write(frame)
        
        print(f"\n✓ Completado: {frame_count} fotogramas")
        print(f"  Guardado: {output_video}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
    
    finally:
        cap.release()
        out.release()

print("=" * 80)
print("✓ TODOS LOS VIDEOS PROCESADOS")
print("=" * 80 + "\n")
