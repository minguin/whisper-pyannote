import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
import librosa
import os
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
# https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb
def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"

@st.cache_resource(show_spinner=False)
def get_speaker_for_segment(seg, diar_segments):
    best_overlap = 0
    assigned_speaker = None
    for d in diar_segments:
        overlap = max(0, min(seg['end'], d['end']) - max(seg['start'], d['start']))
        if overlap > best_overlap:
            best_overlap = overlap
            assigned_speaker = d['speaker']
    return assigned_speaker if assigned_speaker is not None else "UNKNOWN"

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pyannoteの読込
    HF_TOKEN = os.getenv("HF_TOKEN", "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")
    diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    diar_pipeline.to(device)
    # whisperの読込
    whisper_model = whisper.load_model("turbo", device=device)
    st.info(f"{device}環境にて実行")
    return {"device": device, "diar_pipeline": diar_pipeline, "whisper_model": whisper_model}

@st.cache_resource(show_spinner=False)
def load_audio_file(audio_path, sr=16000):
    return librosa.load(audio_path, sr=sr)

@st.cache_resource(show_spinner=False)
def predict(audio_data, _model):
    # 元音声を一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        audio_path = tmp_file.name

    # 音声ファイルを読み込み
    y, sr = load_audio_file(audio_path, sr=16000)
        
    # 長さが足りない場合はパディング（pyannoteの要件対応）
    expected_chunk = 160000
    remainder = len(y) % expected_chunk
    if remainder != 0:
        pad_width = expected_chunk - remainder
        padded_y = np.pad(y, (0, pad_width), mode='constant')
    else:
        padded_y = y
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as padded_tmp_file:
        sf.write(padded_tmp_file.name, padded_y, sr)
        padded_audio_path = padded_tmp_file.name

    with st.spinner("音声文字起こし（whisper）処理中..."):
        result = _model["whisper_model"].transcribe(padded_audio_path, language="ja")
    
    with st.spinner("話者分離（pyannote）処理中..."):
        with ProgressHook() as hook:
            diarization = _model["diar_pipeline"](padded_audio_path, hook=hook)
        diarization_segments = [
            {'start': turn.start, 'end': turn.end, 'speaker': speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    # 話者分離結果を利用して各セグメントに話者情報を付与
    segments = [
        {
            "start": format_time(seg['start']),
            "end": format_time(seg['end']),
            "speaker": get_speaker_for_segment(seg, diarization_segments),
            "text": seg['text'].strip()
        }
        for seg in result.get('segments', [])
    ]
    # 一時ファイルの削除
    try:
        os.remove(audio_path)
        os.remove(padded_audio_path)
    except Exception:
        pass
    return segments

def main():
    st.markdown("### 音声文字起こし（whisper）・話者分離（pyannote）")
    
    # セッション状態に結果を保持するための初期化
    if 'segments' not in st.session_state:
        st.session_state['segments'] = None

    uploaded_file = st.file_uploader("音声ファイルのアップロード", type=["wav", "mp3", "m4a"])
    
    # 「処理開始」ボタンが押されたら新規処理を実行して結果を上書き
    if uploaded_file is not None:
        if st.button("処理開始"):
            with st.spinner("モデルを読み込み中..."):
                model = load_model()
            with st.spinner("音声ファイルを読み込み中..."):
                audio_data = uploaded_file.read()
            segments = predict(audio_data, model)
            if segments:
                st.session_state['segments'] = segments
                st.success("処理が完了しました。")
            else:
                st.error("処理が失敗しました。")
    
    # 前回の結果がセッションに保持されていれば表示する
    if st.session_state['segments'] is not None:
        st.markdown("#### 結果")
        result_text = ""
        for seg in st.session_state['segments']:
            line = f"[{seg['start']} - {seg['end']}] {seg['speaker']}: {seg['text']}\n"
            st.write(line)
            result_text += line
        
        # ダウンロードボタンの設置（テキスト形式）
        st.download_button(
            label="結果をテキストファイルとしてダウンロード",
            data=result_text,
            file_name="transcription_result.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
