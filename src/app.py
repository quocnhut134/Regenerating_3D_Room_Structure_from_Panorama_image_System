from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from pipeline import run_3d_reconstruction_pipeline
from PIL import Image
import shutil
import glob 

app = Flask(__name__)

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(SRC_DIR, 'static', 'uploads')
SAMPLE_FOLDER = os.path.join(SRC_DIR, 'static', 'samples') 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    model_choice = request.form.get('model_choice', 'LGT-Net')
    file = request.files.get('panorama_image')
    sample_name = request.form.get('selected_sample')
    
    filepath = None

    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
    elif sample_name and sample_name != '':
        sample_filename = secure_filename(sample_name)
        src_path = os.path.join(SAMPLE_FOLDER, sample_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, filepath)
        else:
            return jsonify({"error": f"Cannot find sample image: {sample_filename}"}), 400
            
    else:
        return jsonify({"error": "Please select an image to upload or choose a sample image!"}), 400

    try:
        outputs_dir = os.path.abspath(os.path.join(SRC_DIR, '..', 'outputs')) 
        raw_json_files = glob.glob(os.path.join(outputs_dir, '1_raw_json', '*'))
        layout3d_files = glob.glob(os.path.join(outputs_dir, '2_layouts3d', '*'))
        
        for f in raw_json_files + layout3d_files:
            if os.path.isfile(f):
                os.remove(f)
    except Exception as e:
        print(f"Error occurred while cleaning up outputs: {e}")

    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')
            img = img.resize((1024, 512), Image.Resampling.LANCZOS)
            img.save(filepath, format='JPEG', quality=95)
    except Exception as e:
        print(f"Error occurred while resizing image: {e}")
        return jsonify({"error": "Cannot process image size!"}), 500
    
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
        print(f"Error occurred while running 3D reconstruction pipeline: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Web Demo server...")
    app.run(host='0.0.0.0', port=7860, debug=False)