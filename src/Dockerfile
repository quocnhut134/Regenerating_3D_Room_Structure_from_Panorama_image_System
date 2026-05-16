FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.0/index.html

COPY . .

RUN chmod -R 777 /app

CMD ["python", "app.py"]