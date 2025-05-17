from flask import Flask, request, jsonify
import os
import uuid
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_type = request.form.get('type') 
    filename = f"{uuid.uuid4()}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    if file_type == 'pdf':
        # Blur işlemi
        blurred_path = os.path.join(RESULTS_FOLDER, f"blurred_{filename}")
        subprocess.run(['python', 'blur.py', save_path, blurred_path], check=True)
        # Gemini ile yorumlat
        from explain_with_gemini import explain_pdf
        yorum = explain_pdf(blurred_path)
        return jsonify({'result': yorum})
    elif file_type == 'image':
        # Görsel modeline gönder, sonucu dön (örnek)
        return jsonify({'result': 'Görsel modeli sonucu (örnek)'})
    else:
        return jsonify({'error': 'Geçersiz dosya türü'}), 400

if __name__ == '__main__':
    app.run(debug=True)
