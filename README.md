# Whisper-Pyannote Speech Recognition App

音声ファイルから文字起こしと話者分離を行うStreamlitアプリケーション。

## 機能

- 音声ファイルのアップロード（対応形式: WAV, MP3, M4A）
- Whisperによる音声文字起こし
- Pyannoteによる話者分離
- 結果のテキストファイルダウンロード

## EC2 GPU環境での実行手順

### 前提条件

- AWS EC2インスタンス
  * タイプ: g4dn.xlarge以上
  * AMI: Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2)
    * NVIDIA DriverとDocker-GPUが事前インストール済み
- Hugging Face API Token
- AWS ECRリポジトリ

### デプロイ手順

1. リポジトリのクローン
```bash
git clone [repository-url]
cd whisper-pyannote
```

2. Dockerイメージのビルドとプッシュ
```bash
# イメージのビルド
docker build -t whisper-pyannote .

# ECRにプッシュ
docker tag whisper-pyannote:latest $ECR_REGISTRY/whisper-pyannote:latest
docker push $ECR_REGISTRY/whisper-pyannote:latest
```

3. EC2での実行
```bash
# Hugging Face APIトークンの設定
export HF_TOKEN=your_hugging_face_token

# コンテナの実行（GPUサポート有効）
docker run --gpus all -p 8501:8501 -e HF_TOKEN=$HF_TOKEN $ECR_REGISTRY/whisper-pyannote:latest
```

4. アプリケーションへのアクセス
- ブラウザで `http://[EC2-インスタンスのパブリックIP]:8501` にアクセス

## 注意事項

- EC2インスタンスのセキュリティグループで、ポート8501への接続を許可してください
- 大きな音声ファイルを処理する場合は、十分なGPUメモリを持つインスタンスタイプを選択してください