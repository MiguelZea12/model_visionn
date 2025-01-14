import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class DetectionService:
    def __init__(self):
        self.model = YOLO('app/views/static/yolov8s.pt')
        
        # Leer clases desde archivo COCO
        with open("app/views/static/coco.txt", "r") as f:
            self.class_list = f.read().splitlines()
            
        # Clases relevantes para detección
        self.TARGET_CLASSES = {"car", "truck", "bus", "motorcycle"}
        
        # Almacenamiento de detecciones
        self.detections_history = []
        self.video_finished = False
        
        # Mapeo de clases a números
        self.class_mapping = {
            'car': 0,
            'truck': 1,
            'bus': 2,
            'motorcycle': 3
        }
        
    #Detecta vehículos en un frame y almacena los resultados.    
    def detect_vehicles(self, frame):
        
        frame_resized = cv2.resize(frame, (1020, 500))
        results = self.model.predict(frame_resized)
        detections = results[0].boxes.data
        
        frame_detections = []
        
        if detections is not None:
            df = pd.DataFrame(detections).astype("float")
            
            for _, row in df.iterrows():
                x1, y1, x2, y2, conf, cls_idx = map(float, row[:6])
                class_name = self.class_list[int(cls_idx)]
                
                if class_name in self.TARGET_CLASSES:
                    # Almacenar la detección
                    detection_info = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    }
                    frame_detections.append(detection_info)
                    
                    # Dibujar la detección
                    cv2.rectangle(frame_resized, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 
                                2)
                    cv2.putText(
                        frame_resized,
                        f"{class_name} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
        
        # Almacenar detecciones del frame
        if frame_detections:
            self.detections_history.extend(frame_detections)
        
        return frame_resized
    
    #Procesa el historial de detecciones para generar métricas.
    def get_detection_metrics(self):
        if not self.detections_history:
            raise ValueError("No hay detecciones almacenadas para generar métricas")
            
        # Procesar detecciones para crear y_pred
        y_pred = []
        y_true = []
        
        # Convertir detecciones a etiquetas numéricas
        for detection in self.detections_history:
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Incluir detecciones con alta confianza
            if confidence > 0.5:
                if class_name in self.class_mapping:
                    y_pred.append(self.class_mapping[class_name])
                    y_true.append(self.class_mapping[class_name])
        
        print(f"Total de detecciones procesadas: {len(y_pred)}")
        return y_true, y_pred

    def set_video_finished(self):
        """
        Marca el video como terminado
        """
        self.video_finished = True