import os
import sys
import io
import time
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


WORDS = [
    ("mời", "moi"),
    ("khách", "khach"),
    ("hàng", "hang"),
    ("số", "so"),
    ("không", "khong"),
    ("một", "mot"),
    ("hai", "hai"),
    ("ba", "ba"),
    ("bốn", "bon"),
    ("năm", "nam"),
    ("sáu", "sau"),
    ("bảy", "bay"),
    ("tám", "tam"),
    ("chín", "chin"),
    ("đến", "den"),
    ("quầy", "quay")
]

UNITS_DIR = os.path.join("lab4_data", "units")
os.makedirs(UNITS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_RMS = 0.08

def get_audio_from_gtts(text, lang='vi', retries=3):
    for i in range(retries):
        try:
            fp = io.BytesIO()
            tts = gTTS(text, lang=lang)
            tts.write_to_fp(fp)
            fp.seek(0)
            y, sr = librosa.load(fp, sr=TARGET_SR, mono=True)
            return y
        except Exception as e:
            time.sleep(1.0)
            if i == retries - 1:
                raise e

def match_rms(x, target=TARGET_RMS):
    current_rms = np.sqrt(np.mean(x**2) + 1e-12)
    y = x * (target / current_rms)
    return np.clip(y, -0.99, 0.99)

def main():
    metadata = []
    print(f"Generating unit inventory for {len(WORDS)} words (2 candidates each)...")
    
    for word, slug in WORDS:
        print(f"Processing word: '{word}' ({slug})...")
        # Candidate 1: standard take
        y1 = get_audio_from_gtts(word)
        # Trim silence: 25dB threshold with 20ms padding
        y1_trimmed, _ = librosa.effects.trim(y1, top_db=25)
        # Pad 15ms silence at ends for clean acoustic boundary
        pad_len = int(0.015 * TARGET_SR)
        y1_clean = np.pad(y1_trimmed, (pad_len, pad_len), mode='constant')
        y1_norm = match_rms(y1_clean, TARGET_RMS)
        
        file1 = f"{slug}_01.wav"
        path1 = os.path.join(UNITS_DIR, file1)
        sf.write(path1, y1_norm.astype(np.float32), TARGET_SR, subtype='PCM_16')
        
        dur1 = len(y1_norm) / TARGET_SR
        rms1 = float(np.sqrt(np.mean(y1_norm**2)))
        metadata.append({
            "label": word,
            "candidate": 1,
            "speaker": "spk1",
            "duration": round(dur1, 4),
            "rms": round(rms1, 4),
            "file_path": path1.replace("\\", "/")
        })
        
        # Candidate 2: slight acoustic variation (e.g. pitch shift +0.7 semitone or slight speed variation)
        # to realistically mimic candidate selection in concatenative synthesis
        y2 = librosa.effects.pitch_shift(y1_clean, sr=TARGET_SR, n_steps=0.75)
        y2 = librosa.effects.time_stretch(y2, rate=0.96)
        y2_norm = match_rms(y2, TARGET_RMS * 0.95) # slight natural energy variation
        
        file2 = f"{slug}_02.wav"
        path2 = os.path.join(UNITS_DIR, file2)
        sf.write(path2, y2_norm.astype(np.float32), TARGET_SR, subtype='PCM_16')
        
        dur2 = len(y2_norm) / TARGET_SR
        rms2 = float(np.sqrt(np.mean(y2_norm**2)))
        metadata.append({
            "label": word,
            "candidate": 2,
            "speaker": "spk1",
            "duration": round(dur2, 4),
            "rms": round(rms2, 4),
            "file_path": path2.replace("\\", "/")
        })
        
        time.sleep(0.3)

    df = pd.DataFrame(metadata)
    meta_path = os.path.join("lab4_data", "metadata.csv")
    df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    print(f"Inventory completed! Metadata saved to {meta_path}.")
    print(df.head(10))

if __name__ == "__main__":
    main()
