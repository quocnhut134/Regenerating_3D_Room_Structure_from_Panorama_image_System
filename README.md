# 3D Room Layout Reconstruction from Panoramic Images

## Overview

This project presents an end-to-end web application for reconstructing 3D room layouts from single 2D panoramic images (360-degree). By integrating two state-of-the-art deep learning architectures—**LGT-Net** and **DOPNet**—the system extracts structural boundaries from panoramas and generates navigable 3D meshes (`.obj`) with corresponding materials (`.mtl`).

The project features a Flask-based backend and a custom Three.js frontend, offering an immersive first-person perspective to explore the generated 3D environments directly within the browser.

## Authors

This project was managed and developed by:
-   **Duong Quoc Nhut - 23521132 - Leader**
-   **Dang Hoai Nam - 23520967 - Member**
-   **Tran Bao Tran - 23521623 - Member**
-   **Nguyen Hoang Kha - 23520667 - Member**

## Demo

You can find and enjoy our deployed web system at [3D Room Reconstruction from Panorama Image](https://huggingface.co/spaces/SaitoHoujou/3d-room-reconstruction-from-panorama-image)

Or, you can watch our demo video on Youtube at [Demo Web of 3D Reconstruction from Panorama Image System](https://www.youtube.com/watch?v=wkcJDULa31w)

|**Panorama Room Image**| **3D Outside** | **3D Inside** | **3D Anatomy View** | 
| :---: | :---: | :---: | :---: |
|<img width="1024" height="512" alt="Image" src="https://github.com/user-attachments/assets/90b47f1e-7555-466c-ba10-d60e633c084e" width="80%"/>| <img width="839" height="550" alt="Image" src="https://github.com/user-attachments/assets/9eb95fda-dd5d-4c9d-b2f9-e221dfb90e97" width="80%"/> | <img width="1399" height="797" alt="Image" src="https://github.com/user-attachments/assets/0f67c2d0-6c8c-430d-84ac-9b2cc10c3fbe" width="60%"/> | <img width="704" height="502" alt="Image" src="https://github.com/user-attachments/assets/a5fa6074-43f0-463a-bc1d-8b08835e5f10" width="80%"/> | 

## System Architecture

1. **Frontend:** HTML5, CSS3, JavaScript (Three.js). Handles user input, displays the 3D scene, and manages API requests.
2. **Backend:** Flask (Python). Manages file routing, data cleansing, and orchestrates the inference pipelines.
3. **Inference Pipeline:** PyTorch-based neural networks process the panorama, extract corners/boundaries, and utilize geometry processing scripts to assemble the final `.obj` and `.mtl` files.

## Installation & Local Setup

### Prerequisites

* Python 3.10
* Git

### 1. Clone the repository

```bash
git clone https://github.com/quocnhut134/Regenerating_3D_Room_Structure_from_Panorama_image_System
cd Regenerating_3D_Room_Structure_from_Panorama_image_System/src

```

### 2. Set up the Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate # On Windows
source .venv/bin/activate # On Linux/Mac

```

### 3. Install Dependencies

Due to strict version requirements for specific ML operations, install PyTorch and MMCV-full manually before installing the rest of the requirements:

```bash
# Install PyTorch (CPU version for broad compatibility)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# Install MMCV-full (Pre-compiled wheel to avoid C++ build errors)
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.0/index.html

# Install the remaining dependencies
pip install "numpy<2" termcolor
pip install -r requirements.txt

```

### 4. Download Model Weights (Checkpoints)

For local development, you must download the pre-trained weights from [DOPNet](https://github.com/zhijieshen-bjtu/DOPNet) and [LGT-Net](https://github.com/zhigangjiang/LGT-Net), and place them in the correct directories:

* **DOPNet:** Place `model_best_mp3d.pkl` into `DOPNet/checkpoints/My_Layout_Net/mp3d/`
* **LGT-Net:** Place the respective checkpoint into the LGT-Net checkpoint directory (LGT-Net/checkpoints/SWG_Transformer_LGT_Net/mp3d/).

*(Note: If deploying via Docker, the `Dockerfile` will automatically `wget` these files from your external model repository).*

### 5. Run the Application

```bash
python app.py

```

The web interface will be available at `http://localhost:7860`.


## Acknowledgments

This project heavily relies on the foundational research and open-source implementations of:

* **LGT-Net:** [Indoor Panoramic Room Layout Estimation with Geometry-Aware
Transformer Network](https://arxiv.org/pdf/2203.01824)
* **DOPNet:** [Disentangling Orthogonal Planes for Indoor Panoramic Room Layout
Estimation with Cross-Scale Distortion Awareness](https://arxiv.org/pdf/2303.00971)

Special thanks to the open-source community for providing tools like Three.js, Flask, and PyTorch.

---
