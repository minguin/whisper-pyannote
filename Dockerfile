# CUDAとcuDNNが含まれるNVIDIAのベースイメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# システムパッケージのインストール時にタイムゾーンの選択が必要なため
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    TZ=Asia/Tokyo

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
