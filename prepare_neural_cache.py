import os
import sys
import io
import time
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


CACHE_DIR = os.path.join("lab4_data", "neural_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SENTENCES = [
    ("Mời khách hàng số 23 đến quầy 5", "sent1.wav"),
    ("Mời khách hàng số 104 đến quầy 2", "sent2.wav"),
    ("Mời khách hàng số 89 đến quầy 3", "sent3.wav")
]

print("Caching pretrained neural audio for offline inference...")
for text, filename in SENTENCES:
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"Generating neural sample for: '{text}' -> {filename}")
        fp = io.BytesIO()
        tts = gTTS(text, lang='vi')
        tts.write_to_fp(fp)
        fp.seek(0)
        y, sr = librosa.load(fp, sr=16000, mono=True)
        # normalize
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.9
        sf.write(path, y.astype(np.float32), 16000, subtype='PCM_16')
    else:
        print(f"Already exists: {filename}")
print("Pretrained neural cache ready.")
