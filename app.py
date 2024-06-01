from flask import Flask, jsonify, request, send_file, after_this_request
from video_manager.upload_video import upload_video
from video_manager.delete_video import delete_video
from modelo import process_video, model  # Asegúrate de que iaModel.py está en el mismo directorio que app.py
import os
import threading
import time

app = Flask(__name__)

# Configuración de la carpeta de subida y resultados
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def delayed_remove_file(path, delay=5):
    """Elimina un archivo después de un retraso."""
    def remove():
        time.sleep(delay)
        try:
            if os.path.exists(path):
                os.remove(path)
                app.logger.info(f"Archivo {path} eliminado exitosamente.")
        except Exception as error:
            app.logger.error("Error removing file: %s", error)

    threading.Thread(target=remove).start()

def process_and_respond(filename, is_hd):
    input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")

    model_input_size = [720, 1280] if is_hd else [480, 854]  # Tamaños para HD y no HD
    process_video(input_video_path, output_video_path, model, model_input_size)

    @after_this_request
    def remove_files(response):
        delayed_remove_file(input_video_path)
        delayed_remove_file(output_video_path)
        return response

    return send_file(output_video_path, as_attachment=True, download_name=f"processed_{filename}")

@app.route('/upload_hd', methods=['POST'])
def upload_hd():
    filename, response = upload_video()
    if filename:
        return process_and_respond(filename, True)
    return response

@app.route('/upload_non_hd', methods=['POST'])
def upload_non_hd():
    filename, response = upload_video()
    if filename:
        return process_and_respond(filename, False)
    return response

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=3000)
