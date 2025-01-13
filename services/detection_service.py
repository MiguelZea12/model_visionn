import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Cargar modelo YOLO
model = YOLO('yolov8s.pt')

# Leer clases desde un archivo COCO
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Clases relevantes para detección
TARGET_CLASSES = {"car", "truck", "bus", "motorcycle"}

def detect_vehicles(frame):
    """
    Detecta vehículos en un frame y devuelve el frame anotado con detecciones.
    """
    frame_resized = cv2.resize(frame, (1020, 500))
    results = model.predict(frame_resized)
    detections = results[0].boxes.data

    if detections is not None:
        df = pd.DataFrame(detections).astype("float")
        for _, row in df.iterrows():
            x1, y1, x2, y2, conf, cls_idx = map(int, row[:6])
            class_name = class_list[cls_idx]

            if class_name in TARGET_CLASSES:
                # Dibujar la detección en el frame
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
    return frame_resized
