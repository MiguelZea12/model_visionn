from flask import Blueprint, Response, jsonify, render_template, send_from_directory
from app.services.detection_service import DetectionService
from app.services.video_service import VideoService

detection_bp = Blueprint('detection', __name__)

# Inicializar servicios
detection_service = DetectionService()
video_service = VideoService(detection_service)

@detection_bp.route('/')
def index():
    #Renderiza la página principal

    return render_template('index.html')

@detection_bp.route('/video_feed')
def video_feed():
    #Transmite el video procesado

    return Response(
        video_service.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@detection_bp.route('/play', methods=['POST'])
def play():
    #Inicia la reproducción del video
    video_service.video_state["play"] = True
    return jsonify({"status": "success"})

@detection_bp.route('/pause', methods=['POST'])
def pause():
    #Pausa la reproducción del video
    video_service.video_state["play"] = False
    return jsonify({"status": "success"})

@detection_bp.route('/generate_report', methods=['POST'])
def generate_report():
    #Genera el reporte de detecciones
    try:
        message = video_service.generate_final_report()
        if message.startswith("Error"):
            return jsonify({"error": message}), 400
        return jsonify({
            "message": message, 
            "pdf_url": "/static/final_report.pdf"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@detection_bp.route('/static/<path:filename>')
def serve_static(filename):
    #Sirve archivos estáticos

    return send_from_directory('app/views/static', filename)