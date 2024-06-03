import os
from flask import current_app, request, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_video():
    if 'file' not in request.files:
        return None, jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return None, jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
        return filename, jsonify({'message': 'File uploaded successfully', 'filename': filename})
    else:
        return None, jsonify({'error': 'File type not allowed'}), 400
