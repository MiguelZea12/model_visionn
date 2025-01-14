from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import tempfile
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class PDFReport:
    def __init__(self, filename):
        self.pdf = FPDF()
        self.filename = filename
        # Establecer márgenes para mejor presentación
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def add_title(self, title):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", size=16)
        self.pdf.cell(200, 10, txt=title, ln=True, align="C")
        
    def add_section(self, title, content):
        self.pdf.set_font("Arial", "B", size=12)
        self.pdf.cell(200, 10, txt=title, ln=True)
        self.pdf.set_font("Arial", size=10)
        self.pdf.multi_cell(0, 10, content)

    def add_metrics_visualization(self, y_true, y_pred, labels):
        try:
            # Asegurarse de que las etiquetas coincidan con las clases presentes en los datos
            unique_classes = sorted(set(y_true) | set(y_pred))
            labels = [labels[i] for i in unique_classes] if labels else [str(cls) for cls in unique_classes]

            # Calcular métricas
            report_dict = classification_report(y_true, y_pred, target_names=labels, 
                                             output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_true, y_pred)

            # 1. Gráfico de barras para precisión, recall y f1-score
            metrics_df = pd.DataFrame(report_dict).drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            metrics_df = metrics_df.T[['precision', 'recall', 'f1-score']]

            plt.figure(figsize=(10, 6))
            metrics_df.plot(kind='bar', width=0.8)
            plt.title('Métricas por Clase')
            plt.xlabel('Clase')
            plt.ylabel('Puntuación')
            plt.legend(loc='lower right')
            plt.tight_layout()

            # Guardar gráfico temporalmente en disco
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
                tmp_file.close()  # Cerrar archivo antes de usarlo
                self.pdf.add_page()
                self.pdf.image(tmp_file.name, x=10, w=190)
            os.unlink(tmp_file.name)  # Eliminar archivo temporal
            plt.close()

            # 2. Matriz de confusión como heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Matriz de Confusión')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            plt.tight_layout()

            # Guardar heatmap temporalmente en disco
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
                tmp_file.close()  # Cerrar archivo antes de usarlo
                self.pdf.add_page()
                self.pdf.image(tmp_file.name, x=10, w=190)
            os.unlink(tmp_file.name)  # Eliminar archivo temporal
            plt.close()

            # Agregar resumen de métricas
            self.pdf.add_page()
            self.pdf.set_font("Arial", "B", size=14)
            self.pdf.cell(200, 10, txt="Resumen de Métricas", ln=True)
            
            # Agregar accuracy global
            self.pdf.set_font("Arial", size=12)
            accuracy = report_dict['accuracy']
            self.pdf.cell(200, 10, f"Accuracy Global: {accuracy:.2%}", ln=True)
            
            # Tabla de métricas por clase
            self.pdf.set_font("Arial", "B", size=12)
            self.pdf.cell(200, 10, txt="Métricas por Clase:", ln=True)
            self.pdf.set_font("Arial", size=10)
            
            # Encabezados de la tabla
            col_width = 40
            self.pdf.cell(col_width, 10, "Clase", 1)
            self.pdf.cell(col_width, 10, "Precisión", 1)
            self.pdf.cell(col_width, 10, "Recall", 1)
            self.pdf.cell(col_width, 10, "F1-Score", 1)
            self.pdf.ln()
            
            # Datos de la tabla
            for label in labels:
                metrics = report_dict[label]
                self.pdf.cell(col_width, 10, label, 1)
                self.pdf.cell(col_width, 10, f"{metrics['precision']:.3f}", 1)
                self.pdf.cell(col_width, 10, f"{metrics['recall']:.3f}", 1)
                self.pdf.cell(col_width, 10, f"{metrics['f1-score']:.3f}", 1)
                self.pdf.ln()

        except Exception as e:
            print(f"Error while generating metrics visualization: {e}")
            raise

    def generate(self):
        self.pdf.output(self.filename)
        print(f"PDF generated successfully: {self.filename}")