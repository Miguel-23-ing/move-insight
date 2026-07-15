"""
EXTRACTOR DE POSE FINAL - ESQUELETO COMPLETO
- 17 puntos YOLOv8 Pose
- Suavizado fluido (ALPHA=0.4)
- Líneas de brazos, piernas y torso
"""

import os
import sys
import cv2
import numpy as np

print("\n" + "=" * 80)
print("EXTRACTOR DE POSE - ESQUELETO COMPLETO")
print("=" * 80 + "\n")

VIDEO_PATH = "Video 1.mp4"
OUTPUT_PATH = "Video_with_landmarks.mp4"

if not os.path.exists(VIDEO_PATH):
    print(f"✗ Error: {VIDEO_PATH} no encontrado")
    sys.exit(1)

print(f"✓ Video encontrado: {VIDEO_PATH}\n")

print("=" * 80)
print("Importando librerías...")
print("=" * 80 + "\n")

try:
    from ultralytics import YOLO
    print("✓ YOLOv8 importado")
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

# Leer video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 80)
print("Info del video")
print("=" * 80 + "\n")
print(f"Resolución: {frame_width}x{frame_height}")
print(f"FPS: {fps}")
print(f"Fotogramas: {total_frames}\n")

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Parámetros
ALPHA = 0.25  # Suavizado muy bajo = EXCELENTE seguimiento de movimientos rápidos
VISIBILITY_THRESHOLD = 0.08  # Umbral MÁS bajo para ver articulaciones débiles
CONFIDENCE_MIN = 0.3  # Confianza mínima de detección
POINT_RADIUS = 7  # Radio de los puntos
LINE_THICKNESS = 5  # Grosor de las líneas
JOINT_ANGLES_RADIUS = 8  # Puntos ligeramente más grandes para articulaciones clave

# Esqueleto YOLOv8 (17 puntos, solo conexiones del cuerpo sin cabeza)
SKELETON_CONNECTIONS = [
    (5, 6),                   # Hombros
    (5, 7), (7, 9),           # Brazo izq
    (6, 8), (8, 10),          # Brazo der
    (5, 11), (6, 12),         # Hombros a caderas
    (11, 12),                 # Cintura
    (11, 13), (13, 15),       # Pierna izq
    (12, 14), (14, 16)        # Pierna der
]

# Historia para suavizado por punto
smooth_history = {}
frame_count = 0

print("=" * 80)
print("Procesando video...")
print("=" * 80 + "\n")

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
        
        # Detección de pose
        results = model(frame, verbose=False, conf=0.3)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Validar que hay detección con confianza suficiente
            if (result.boxes is not None and len(result.boxes) > 0 and 
                result.boxes.conf[0] > CONFIDENCE_MIN and result.keypoints is not None):
                
                keypoints = result.keypoints.xy[0].cpu().numpy()
                confidences = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(17)
                
                # Suavizado de todos los puntos
                smoothed_points = []
                for i, (pt, conf) in enumerate(zip(keypoints, confidences)):
                    if i not in smooth_history:
                        smooth_history[i] = {'x': pt[0], 'y': pt[1], 'conf': float(conf)}
                    
                    if float(conf) > 0:
                        smooth_history[i]['x'] = ALPHA * pt[0] + (1 - ALPHA) * smooth_history[i]['x']
                        smooth_history[i]['y'] = ALPHA * pt[1] + (1 - ALPHA) * smooth_history[i]['y']
                        smooth_history[i]['conf'] = float(conf)
                    
                    smoothed_points.append(smooth_history[i])
                
                # Dibujar esqueleto PRIMERO (detrás de los puntos)
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    start_pt = smoothed_points[start_idx]
                    end_pt = smoothed_points[end_idx]
                    
                    if start_pt['conf'] > VISIBILITY_THRESHOLD and end_pt['conf'] > VISIBILITY_THRESHOLD:
                        pt1 = (int(start_pt['x']), int(start_pt['y']))
                        pt2 = (int(end_pt['x']), int(end_pt['y']))
                        
                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                            0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            # Color verde para todas las aristas
                            cv2.line(frame, pt1, pt2, (0, 255, 0), LINE_THICKNESS)
                
                # Dibujar puntos ENCIMA
                for i, pt_data in enumerate(smoothed_points):
                    # Punto de cabeza (0): solo si está de frente (alta confianza)
                    if i == 0:
                        if pt_data['conf'] > 0.4:  # Confanza alta = de frente
                            x = int(pt_data['x'])
                            y = int(pt_data['y'])
                            conf = pt_data['conf']
                            
                            if 0 <= x < w and 0 <= y < h:
                                # Contorno oscuro
                                cv2.circle(frame, (x, y), POINT_RADIUS + 1, (0, 0, 0), -1)
                                # Círculo verde
                                cv2.circle(frame, (x, y), POINT_RADIUS, (0, 255, 0), -1)
                                # Pequeño resalte blanco
                                cv2.circle(frame, (x, y), POINT_RADIUS - 2, (255, 255, 255), 1)
                    
                    # Puntos de cara y cuerpo (1-16)
                    elif i >= 1:
                        if pt_data['conf'] > VISIBILITY_THRESHOLD:
                            x = int(pt_data['x'])
                            y = int(pt_data['y'])
                            conf = pt_data['conf']
                            
                            if 0 <= x < w and 0 <= y < h:
                                # Rodillas y tobillos: MÁS VISIBLES para análisis de marcha
                                is_knee = i in [13, 14]  # Rodilla izq y der
                                is_ankle = i in [15, 16]  # Tobillo izq y der
                                is_key_joint = is_knee or is_ankle
                                
                                point_size = JOINT_ANGLES_RADIUS if is_key_joint else POINT_RADIUS
                                
                                # Contorno oscuro
                                cv2.circle(frame, (x, y), point_size + 1, (0, 0, 0), -1)
                                # Círculo verde
                                cv2.circle(frame, (x, y), point_size, (0, 255, 0), -1)
                                # Resalte blanco
                                cv2.circle(frame, (x, y), point_size - 2, (255, 255, 255), 1)

        
        out.write(frame)
    
    print(f"\n✓ Completado: {frame_count} fotogramas\n")
    
except Exception as e:
    print(f"\n✗ Error: {e}\n")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    out.release()
    
    print("=" * 80)
    print(f"✓ VIDEO GUARDADO: {OUTPUT_PATH}")
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  Tamaño: {size_mb:.1f} MB")
    print("=" * 80 + "\n")
