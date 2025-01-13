import cv2
import time
from services.detection_service import detect_vehicles
from utils.pdf_report import PDFReport

video_state = {"play": True}

def generate_frames():
    """
    Genera frames procesados con detecciones de vehículos en un video.
    """
    cap = cv2.VideoCapture("calle2.mp4")  # Video de entrada
    delay = 0  # Delay entre frames
    detection_data = []  # Almacena datos de detecciones para el PDF

    while True:
        if video_state["play"]:
            ret, frame = cap.read()
            if not ret:
                # El video terminó
                cap.release()
                
                # Generar el reporte PDF
                y_true = [1, 0, 1]  # Simulando etiquetas reales (debes reemplazarlas con datos reales)
                y_pred = [1, 0, 0]  # Simulando predicciones (debes reemplazarlas con datos reales)
                labels = ["Background", "Car", "SUV", "Truck", "Motorcycle"]

                pdf = PDFReport("final_report.pdf")
                pdf.add_title("Vehicle Detection Report")
                pdf.add_metrics_visualization(y_true, y_pred, labels)
                pdf.generate()

                print("PDF generado con éxito.")
                break

            # Procesar frame con detecciones
            detections = detect_vehicles(frame)  # Supongamos que retorna datos útiles
            detection_data.append(detections)  # Guardar detecciones
            frame = detect_vehicles(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Enviar frame como flujo de bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

            time.sleep(delay)
        else:
            time.sleep(0)