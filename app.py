
import os
import copy
import tempfile
from dotenv import load_dotenv
import numpy as np
import streamlit as st
import librosa
import torch
import whisper
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv(verbose=True)
PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"

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
def load_model(model_name="turbo"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pyannoteの読込
    # diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_TOKEN"))
    diar_pipeline = Pipeline.from_pretrained(PATH_TO_CONFIG)
    diar_pipeline.to(device)
    # whisperの読込
    whisper_model = whisper.load_model(model_name, device=device)
    st.info(f"{device}にて実行")
    return {"device": device, "diar_pipeline": diar_pipeline, "whisper_model": whisper_model}

@st.cache_resource(show_spinner=False)
def load_audio_file(audio_path, sr=16000):
    return librosa.load(audio_path, sr=sr)

# model_nameはcacheのための識別
@st.cache_resource(show_spinner=False)
def predict(audio_data, model_name, _model, initial_prompt=None, language="ja"):
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
        result = _model["whisper_model"].transcribe(padded_audio_path, initial_prompt=initial_prompt, language=language)
    
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

    uploaded_file = st.file_uploader("音声ファイルのアップロード", type=["wav", "mp3", "m4a"])
    model_name = st.radio("モデルの選択", ["tiny", "base", "small", "medium", "large", "turbo"], index=4, horizontal=True)
    language = st.radio("言語の選択", ["ja", "en"], index=0, horizontal=True)
    initial_prompt = st.text_area("initial_promptの設定（句読点含む文章や単語スペースなど与える）", value="")
    # 「処理開始」ボタンが押されたら新規処理を実行して結果を上書き
    if uploaded_file is not None:
        if st.button("処理開始"):
            with st.spinner("モデルを読み込み中..."):
                model = load_model(model_name)
            with st.spinner("音声ファイルを読み込み中..."):
                audio_data = uploaded_file.read()
            segments = predict(audio_data, model_name, model, initial_prompt, language)
            if segments:
                st.session_state['segments'] = segments
                st.success("処理が完了しました。")
            else:
                st.error("処理が失敗しました。")
    
    # 前回の結果がセッションに保持されていれば表示する
    if 'segments' in st.session_state:
        segments = copy.deepcopy(st.session_state['segments'])
        # データ中のユニークな speaker を取得
        unique_speakers = sorted(list({entry["speaker"] for entry in st.session_state['segments']}))
        st.markdown("#### 置換先名を設定")
        # 置換用のマッピングを動的に入力できるようにテキストボックスを用意
        replacement_mapping = {}
        for speaker in unique_speakers:
            # 初期値は元のスピーカー名とする（変更がなければそのまま）
            replacement = st.text_input(f"{speaker} の置換先名", value=speaker)
            replacement_mapping[speaker] = replacement
        # 各エントリーの speaker を、入力されたマッピングに従って更新
        for seg in segments:
            orig = seg.get("speaker")
            # 入力が空でない、かつ元と異なれば置換（入力が空の場合はそのまま）
            if orig in replacement_mapping and replacement_mapping[orig] and replacement_mapping[orig] != orig:
                seg["speaker"] = replacement_mapping[orig]

        # st.markdown("#### 結果")
        # result_text = ""
        # for seg in segments:
        #     line = f"[{seg['start']} - {seg['end']}] {seg['speaker']}: {seg['text']}\n"
        #     st.write(line)
        #     result_text += line
        st.markdown("#### 結果")
        result_text = ""

        if segments:
            # 最初のセグメントでグループの初期化
            group_start = segments[0]['start']
            group_end = segments[0]['end']
            group_speaker = segments[0]['speaker']
            group_text = segments[0]['text']
            
            # 2番目以降のセグメントを処理
            for seg in segments[1:]:
                if seg['speaker'] == group_speaker:
                    # 同じ speaker の場合は、終了時刻を更新しテキストを連結（間に空白を入れる）
                    group_end = seg['end']
                    group_text += " " + seg['text']
                else:
                    # speaker が変わったら、前のグループを出力
                    line = f"[{group_start} - {group_end}] {group_speaker}: {group_text}\n"
                    st.write(line)
                    result_text += line
                    
                    # 新しいグループとして初期化
                    group_start = seg['start']
                    group_end = seg['end']
                    group_speaker = seg['speaker']
                    group_text = seg['text']
            
            # 最後のグループも出力
            line = f"[{group_start} - {group_end}] {group_speaker}: {group_text}\n"
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
