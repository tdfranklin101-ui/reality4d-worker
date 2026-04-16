FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN mkdir -p /models/ltx-video

COPY src/rp_handler.py /src/rp_handler.py

RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); import runpod; print('RunPod OK')"

CMD ["python3", "-u", "/src/rp_handler.py"]