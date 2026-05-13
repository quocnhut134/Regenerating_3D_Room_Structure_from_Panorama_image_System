import os
import subprocess
import sys

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

PYTHON_LGT = r"D:\Workspace\Working\Computer_Graphics\Project\src\LGT-Net\env\cs105\python.exe"        
PYTHON_DOP = sys.executable 
PYTHON_WEB = sys.executable                                 

def run_3d_reconstruction_pipeline(image_path, model_type="LGT-Net"):
    image_name = os.path.basename(image_path)
    image_base = os.path.splitext(image_name)[0]
    
    raw_json_dir = os.path.join(PROJECT_DIR, 'outputs', '1_raw_json')
    layout3d_dir = os.path.join(PROJECT_DIR, 'outputs', '2_layouts3d')
    dummy_dir = os.path.join(PROJECT_DIR, 'outputs', 'dummy')
    
    final_render_dir = os.path.join(SRC_DIR, 'static', 'results', image_base)
    
    for d in [raw_json_dir, layout3d_dir, dummy_dir, final_render_dir]:
        os.makedirs(d, exist_ok=True)
        
    raw_json_path = os.path.join(raw_json_dir, f"{image_base}_raw.json")
    layout3d_path = os.path.join(layout3d_dir, f"{image_base}_layout3d.json")

    if model_type == "LGT-Net":
        print("[*] Running Model: LGT-Net...")
        lgt_dir = os.path.join(SRC_DIR, 'LGT-Net')
        cmd = f'"{PYTHON_LGT}" infer_all.py --cfg src/config/mp3d.yaml --img_glob "{image_path}" --json_dir "{raw_json_dir}" --img_dir "{dummy_dir}" --txt_dir "{dummy_dir}" --post_processing manhattan'
        subprocess.run(cmd, shell=True, check=True, cwd=lgt_dir)
    else:
        print("[*] Running Model: DOPNet...")
        dopnet_dir = os.path.join(SRC_DIR, 'DOPNet')
        
        cmd = f'"{PYTHON_DOP}" inference.py --cfg "src/my_config/mp3d.yaml" --img_glob "{image_path}" --output_dir "{raw_json_dir}"'
        
        subprocess.run(cmd, shell=True, check=True, cwd=dopnet_dir)

    print("[*] Running Post-processing: Manhattan Alignment...")
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    
    n3_script = os.path.join(SRC_DIR, 'boundary_json_to_layout3d.py')
    cmd_n3 = f'"{PYTHON_WEB}" "{n3_script}" "{raw_json_path}" -o "{layout3d_dir}"'
    subprocess.run(cmd_n3, shell=True, check=True, env=env, cwd=SRC_DIR)

    print("[*] Running 3D Rendering...")
    n4_script = os.path.join(SRC_DIR, '3d_render.py')
    cmd_n4 = f'"{PYTHON_WEB}" "{n4_script}" "{layout3d_path}" "{image_path}" "{final_render_dir}"'
    subprocess.run(cmd_n4, shell=True, check=True, cwd=SRC_DIR)

    return f"results/{image_base}"