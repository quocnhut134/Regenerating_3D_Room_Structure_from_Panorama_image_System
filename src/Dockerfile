FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    wget \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.0/index.html

RUN pip install "numpy<2" termcolor

COPY . .

RUN mkdir -p /app/DOPNet/checkpoints/My_Layout_Net/mp3d/
RUN wget -O /app/DOPNet/checkpoints/My_Layout_Net/mp3d/model_best_mp3d.pkl "https://huggingface.co/SaitoHoujou/dopnet-lgtnet-weights/resolve/main/model_best_mp3d.pkl"

RUN mkdir -p /app/LGT-Net/checkpoints/SWG_Transformer_LGT_Net/mp3d
RUN wget -O /app/LGT-Net/checkpoints/SWG_Transformer_LGT_Net/mp3d/best.pkl "https://huggingface.co/SaitoHoujou/dopnet-lgtnet-weights/resolve/main/best.pkl"

RUN chmod -R 777 /app

CMD ["python", "app.py"]