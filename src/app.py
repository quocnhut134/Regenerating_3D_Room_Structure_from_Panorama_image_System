from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from pipeline import run_3d_reconstruction_pipeline
from PIL import Image

app = Flask(__name__)

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(SRC_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'panorama_image' not in request.files:
        return jsonify({"error": "No panorama image provided!"}), 400
        
    file = request.files['panorama_image']
    model_choice = request.form.get('model_choice', 'LGT-Net')
    
    if file.filename == '':
        return jsonify({"error": "No file selected!"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            with Image.open(filepath) as img:
                img = img.convert('RGB')
                img = img.resize((1024, 512), Image.Resampling.LANCZOS)
                img.save(filepath, format='JPEG', quality=95)
        except Exception as e:
            print(e)
        
        try:
            result_dir_relative = run_3d_reconstruction_pipeline(filepath, model_type=model_choice)
            
            obj_url = url_for('static', filename=f"{result_dir_relative}/3d_model.obj")
            mtl_url = url_for('static', filename=f"{result_dir_relative}/3d_model.mtl")
            
            return jsonify({
                "success": True, 
                "obj_url": obj_url,
                "mtl_url": mtl_url
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Web Demo server...")
    app.run(host='0.0.0.0', port=7860, debug=False)