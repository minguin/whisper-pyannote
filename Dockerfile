# CUDAとcuDNNが含まれるNVIDIAのベースイメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# pipのアップグレード
RUN pip3 install --upgrade pip

# 作業ディレクトリの設定
WORKDIR /app

# requirements.txtをコンテナ内にコピーしてインストール
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# アプリケーションコードをコンテナ内にコピー
COPY . /app/

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    TZ=Asia/Tokyo

# Streamlitのデフォルトポートを公開
EXPOSE 8501

# コンテナ起動時にStreamlitアプリを実行
CMD ["streamlit", "run", "streamlit_app.py"]
