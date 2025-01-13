from flask import Flask, Response, jsonify, request, render_template
from services.video_service import generate_frames
from utils.pdf_report import PDFReport
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # Obtener los datos del cliente
        y_true = request.json.get('true_labels')
        y_pred = request.json.get('predicted_labels')
        
        if not y_true or not y_pred:
            return jsonify({"error": "Missing true_labels or predicted_labels in the request."}), 400

        labels = ["Background", "Car", "SUV", "Truck", "Motorcycle"]

        # Verificar que las longitudes coincidan
        if len(y_true) != len(y_pred):
            return jsonify({"error": "Length of true_labels and predicted_labels must match."}), 400

        # Crear PDF con visualizaciones
        pdf = PDFReport("report.pdf")
        pdf.add_title("Vehicle Recognition Report")
        
        # Agregar visualizaciones y métricas
        pdf.add_metrics_visualization(y_true, y_pred, labels)
        
        # Generar el PDF
        pdf.generate()

        return jsonify({"message": "Report generated successfully!"})
    except Exception as e:
        print("Error generating report:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Failed to generate report.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)