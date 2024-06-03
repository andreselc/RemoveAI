from flask import Flask, jsonify, request, send_file, after_this_request, render_template, redirect, url_for
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

@app.route('/upload_hd', methods=['POST'])
def upload_hd():
    file = request.files['video']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Inicia el procesamiento del video
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    model_input_size = [720, 1280]  # Tamaños para HD
    threading.Thread(target=process_video, args=(filepath, output_video_path, model, model_input_size)).start()

    # Redirige a la página de carga mientras se procesa el video
    return redirect(url_for('carga', filename=filename))

@app.route('/upload_non_hd', methods=['POST'])
def upload_non_hd():
    file = request.files['video']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Inicia el procesamiento del video
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    model_input_size = [480, 854]  # Tamaños para no HD
    threading.Thread(target=process_video, args=(filepath, output_video_path, model, model_input_size)).start()

    # Redirige a la página de carga mientras se procesa el video
    return redirect(url_for('carga', filename=filename))

@app.route('/carga')
def carga():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('home'))
    return render_template('processing.html', filename=filename)

@app.route('/check_processing')
def check_processing():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'processing_complete': False})
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    flag_file_path = output_video_path + ".done"
    
    # Verificación del archivo de marcador y del tamaño del archivo procesado
    processing_complete = os.path.exists(flag_file_path) and os.path.getsize(output_video_path) > 0

    return jsonify({'processing_complete': processing_complete})

@app.route('/results')
def results():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('home'))
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    if not os.path.exists(output_video_path):
        return redirect(url_for('carga', filename=filename))
    return render_template('results.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    
    @after_this_request
    def remove_files(response):
        delayed_remove_file(output_video_path)
        delayed_remove_file(output_video_path + ".done")
        delayed_remove_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return response
    
    return send_file(output_video_path, as_attachment=True, download_name=f"processed_{filename}")

@app.route('/delete_and_home/<filename>', methods=['POST'])
def delete_and_home(filename):
    output_video_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{filename}")
    flag_file_path = output_video_path + ".done"
    upload_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Verificar si ambos archivos existen antes de intentar eliminarlos
    if os.path.exists(output_video_path) and os.path.exists(flag_file_path):
        delayed_remove_file(output_video_path,0)
        delayed_remove_file(flag_file_path,0)
        delayed_remove_file(upload_file_path,0)
    
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
