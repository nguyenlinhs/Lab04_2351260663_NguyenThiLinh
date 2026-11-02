"""
HMM-based Speech Synthesis Backend (CSE457 - Lab 4)
Mô phỏng bộ tổng hợp tham số HMM (HMM-based Parametric Speech Synthesis)
Bao gồm:
- Mô hình State sequence (left-to-right HMM cho từng từ/âm)
- Quỹ đạo tham số: F0 contour (với delta F0), formant/spectral trajectory (F1, F2, F3)
- Thuật toán sinh tham số hợp lý cực đại (MLPG - Maximum Likelihood Parameter Generation)
- Resynthesis dựa trên mô hình Source-Filter (nguồn kích thích xung/nhiễu qua bộ lọc cộng hưởng formant)
- Hỗ trợ scale duration (0.8x - 1.2x) và scale F0 (0.9x - 1.1x)
"""

import numpy as np
from scipy import signal

# Chuẩn hóa số
DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

# Bảng thông số thống kê HMM cho từng từ miền hẹp
# Mỗi từ gồm 3 states (Init, Medial, Final).
# Mỗi state có: duration (số frame, 1 frame = 10ms), f0 (Hz), formants [F1, F2, F3] (Hz), voiced (bool)
WORD_HMM_MODELS = {
    "mời": [
        {"dur": 8, "f0": 140, "f0_var": 5, "formants": [300, 1100, 2400], "voiced": True},
        {"dur": 22, "f0": 135, "f0_var": 8, "formants": [450, 1350, 2500], "voiced": True},
        {"dur": 12, "f0": 120, "f0_var": 10, "formants": [400, 1200, 2400], "voiced": True},
    ],
    "khách": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [400, 1600, 2600], "voiced": False},
        {"dur": 18, "f0": 175, "f0_var": 12, "formants": [750, 1400, 2700], "voiced": True},
        {"dur": 8, "f0": 0, "f0_var": 0, "formants": [350, 1800, 2800], "voiced": False},
    ],
    "hàng": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [500, 1500, 2500], "voiced": False},
        {"dur": 25, "f0": 130, "f0_var": 6, "formants": [700, 1250, 2600], "voiced": True},
        {"dur": 15, "f0": 115, "f0_var": 8, "formants": [350, 1100, 2400], "voiced": True},
    ],
    "số": [
        {"dur": 12, "f0": 0, "f0_var": 0, "formants": [300, 1800, 3200], "voiced": False},
        {"dur": 22, "f0": 170, "f0_var": 10, "formants": [520, 1000, 2600], "voiced": True},
        {"dur": 10, "f0": 190, "f0_var": 8, "formants": [480, 950, 2500], "voiced": True},
    ],
    "không": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [350, 1500, 2500], "voiced": False},
        {"dur": 20, "f0": 150, "f0_var": 5, "formants": [500, 900, 2400], "voiced": True},
        {"dur": 12, "f0": 140, "f0_var": 5, "formants": [350, 850, 2300], "voiced": True},
    ],
    "một": [
        {"dur": 8, "f0": 145, "f0_var": 6, "formants": [320, 1100, 2400], "voiced": True},
        {"dur": 18, "f0": 130, "f0_var": 8, "formants": [480, 1200, 2500], "voiced": True},
        {"dur": 6, "f0": 0, "f0_var": 0, "formants": [300, 1500, 2600], "voiced": False},
    ],
    "hai": [
        {"dur": 8, "f0": 0, "f0_var": 0, "formants": [600, 1400, 2500], "voiced": False},
        {"dur": 20, "f0": 155, "f0_var": 6, "formants": [750, 1300, 2600], "voiced": True},
        {"dur": 14, "f0": 145, "f0_var": 5, "formants": [350, 2200, 2900], "voiced": True},
    ],
    "ba": [
        {"dur": 6, "f0": 145, "f0_var": 5, "formants": [300, 1100, 2400], "voiced": True},
        {"dur": 22, "f0": 150, "f0_var": 5, "formants": [800, 1250, 2600], "voiced": True},
        {"dur": 10, "f0": 145, "f0_var": 5, "formants": [750, 1200, 2550], "voiced": True},
    ],
    "bốn": [
        {"dur": 8, "f0": 155, "f0_var": 6, "formants": [300, 1100, 2400], "voiced": True},
        {"dur": 20, "f0": 180, "f0_var": 9, "formants": [550, 1050, 2500], "voiced": True},
        {"dur": 12, "f0": 190, "f0_var": 8, "formants": [400, 1600, 2600], "voiced": True},
    ],
    "năm": [
        {"dur": 8, "f0": 145, "f0_var": 5, "formants": [350, 1600, 2600], "voiced": True},
        {"dur": 22, "f0": 150, "f0_var": 5, "formants": [720, 1350, 2650], "voiced": True},
        {"dur": 12, "f0": 140, "f0_var": 5, "formants": [300, 1100, 2400], "voiced": True},
    ],
    "sáu": [
        {"dur": 12, "f0": 0, "f0_var": 0, "formants": [300, 2000, 3200], "voiced": False},
        {"dur": 20, "f0": 175, "f0_var": 10, "formants": [750, 1300, 2600], "voiced": True},
        {"dur": 10, "f0": 195, "f0_var": 8, "formants": [400, 900, 2500], "voiced": True},
    ],
    "bảy": [
        {"dur": 8, "f0": 140, "f0_var": 5, "formants": [300, 1100, 2400], "voiced": True},
        {"dur": 18, "f0": 135, "f0_var": 7, "formants": [650, 1500, 2700], "voiced": True},
        {"dur": 12, "f0": 125, "f0_var": 8, "formants": [350, 2100, 2800], "voiced": True},
    ],
    "tám": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [350, 1800, 2800], "voiced": False},
        {"dur": 22, "f0": 180, "f0_var": 10, "formants": [750, 1250, 2600], "voiced": True},
        {"dur": 10, "f0": 190, "f0_var": 8, "formants": [300, 1100, 2400], "voiced": True},
    ],
    "chín": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [350, 2000, 3000], "voiced": False},
        {"dur": 22, "f0": 180, "f0_var": 10, "formants": [320, 2200, 3000], "voiced": True},
        {"dur": 10, "f0": 190, "f0_var": 8, "formants": [350, 1700, 2700], "voiced": True},
    ],
    "đến": [
        {"dur": 8, "f0": 150, "f0_var": 6, "formants": [350, 1700, 2700], "voiced": True},
        {"dur": 18, "f0": 175, "f0_var": 9, "formants": [550, 1800, 2700], "voiced": True},
        {"dur": 10, "f0": 185, "f0_var": 8, "formants": [350, 1600, 2600], "voiced": True},
    ],
    "quầy": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [300, 1000, 2400], "voiced": False},
        {"dur": 22, "f0": 135, "f0_var": 6, "formants": [500, 1300, 2500], "voiced": True},
        {"dur": 12, "f0": 120, "f0_var": 8, "formants": [350, 2100, 2800], "voiced": True},
    ],
    "sil": [
        {"dur": 10, "f0": 0, "f0_var": 0, "formants": [500, 1500, 2500], "voiced": False}
    ]
}

def normalize_text_hmm(text):
    text = text.lower().replace(".", " ").replace(",", " ").replace(";", " ")
    tokens = []
    for tok in text.split():
        if tok.isdigit():
            tokens.extend([DIGIT_MAP[d] for d in tok if d in DIGIT_MAP])
        else:
            tokens.append(tok)
    return tokens

def synthesize_hmm(text, sr=16000, duration_scale=1.0, f0_scale=1.0):
    """
    HMM Parameter Generation & Resynthesis
    Returns:
        y_hmm (np.ndarray): Waveform tổng hợp
        sr (int): Sampling rate (16000)
        tracks (dict): Bao gồm 'f0', 'f1', 'f2', 'f3', 'duration', 'states', 'frame_times'
    """
    tokens = normalize_text_hmm(text)
    frame_shift_s = 0.01  # 10ms frame shift
    hop_length = int(sr * frame_shift_s)
    
    # 1. State alignment & duration prediction
    state_sequence = []
    raw_f0_list = []
    raw_f1_list = []
    raw_f2_list = []
    raw_f3_list = []
    voiced_list = []
    
    # Add initial short silence
    tokens_with_sil = ["sil"] + tokens + ["sil"]
    
    for word in tokens_with_sil:
        states = WORD_HMM_MODELS.get(word, WORD_HMM_MODELS["mời"])
        for idx, s in enumerate(states):
            dur_frames = max(2, int(round(s["dur"] * duration_scale)))
            f0_val = s["f0"] * f0_scale if s["voiced"] else 0.0
            f1_val, f2_val, f3_val = s["formants"]
            
            for _ in range(dur_frames):
                state_sequence.append(f"{word}_s{idx+1}")
                raw_f0_list.append(f0_val)
                raw_f1_list.append(f1_val)
                raw_f2_list.append(f2_val)
                raw_f3_list.append(f3_val)
                voiced_list.append(s["voiced"])

    total_frames = len(state_sequence)
    raw_f0 = np.array(raw_f0_list, dtype=np.float32)
    raw_f1 = np.array(raw_f1_list, dtype=np.float32)
    raw_f2 = np.array(raw_f2_list, dtype=np.float32)
    raw_f3 = np.array(raw_f3_list, dtype=np.float32)
    voiced = np.array(voiced_list, dtype=bool)

    # 2. Maximum Likelihood Parameter Generation (MLPG) approximation
    # Làm trơn trajectory bằng bộ lọc ràng buộc động học (dynamic/delta features)
    # Tương đương với nghiệm trơn hóa giải phương trình W^T U^-1 W c = W^T U^-1 M
    smooth_win = signal.windows.gaussian(7, std=1.8)
    smooth_win /= np.sum(smooth_win)
    
    # Smooth formants
    smooth_f1 = signal.convolve(raw_f1, smooth_win, mode='same')
    smooth_f2 = signal.convolve(raw_f2, smooth_win, mode='same')
    smooth_f3 = signal.convolve(raw_f3, smooth_win, mode='same')

    # Smooth F0 for voiced segments
    smooth_f0 = np.copy(raw_f0)
    voiced_indices = np.where(voiced)[0]
    if len(voiced_indices) > 0:
        f0_voiced_smooth = signal.convolve(raw_f0[voiced_indices], smooth_win, mode='same')
        smooth_f0[voiced_indices] = f0_voiced_smooth
    smooth_f0[~voiced] = 0.0

    # 3. Source - Filter Vocoder Resynthesis
    total_samples = total_frames * hop_length
    excitation = np.zeros(total_samples, dtype=np.float32)
    
    # Generate excitation
    phase = 0.0
    for i in range(total_frames):
        start_samp = i * hop_length
        end_samp = start_samp + hop_length
        f0_curr = smooth_f0[i]
        
        if f0_curr > 30.0:  # Voiced: impulse train / glottal pulses
            step = f0_curr / sr
            for n in range(start_samp, end_samp):
                phase += step
                if phase >= 1.0:
                    phase -= 1.0
                    excitation[n] = 1.0
                else:
                    excitation[n] = 0.0
        else:  # Unvoiced: white noise
            excitation[start_samp:end_samp] = np.random.normal(0, 0.15, hop_length)

    # Formant Resonator Bank (Vocal Tract Filter)
    # Áp dụng 3 bộ lọc cộng hưởng bậc 2 theo formant F1, F2, F3
    def make_resonator(freq, bw, sr):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * freq / sr
        b = [1.0 - r]
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        return b, a

    mean_f1 = float(np.mean(smooth_f1[voiced])) if np.any(voiced) else 500
    mean_f2 = float(np.mean(smooth_f2[voiced])) if np.any(voiced) else 1500
    mean_f3 = float(np.mean(smooth_f3[voiced])) if np.any(voiced) else 2500

    b1, a1 = make_resonator(mean_f1, 90, sr)
    b2, a2 = make_resonator(mean_f2, 110, sr)
    b3, a3 = make_resonator(mean_f3, 140, sr)

    y1 = signal.lfilter(b1, a1, excitation)
    y2 = signal.lfilter(b2, a2, excitation)
    y3 = signal.lfilter(b3, a3, excitation)
    
    y_synth = 0.5 * y1 + 0.3 * y2 + 0.2 * y3
    
    # Lip radiation filter (first-order difference / high-pass)
    y_synth = signal.lfilter([1.0, -0.95], [1.0], y_synth)
    
    # Normalize output amplitude
    max_val = np.max(np.abs(y_synth)) + 1e-8
    y_synth = (y_synth / max_val) * 0.85

    frame_times = np.arange(total_frames) * frame_shift_s
    tracks = {
        "f0": smooth_f0,
        "f1": smooth_f1,
        "f2": smooth_f2,
        "f3": smooth_f3,
        "raw_f0": raw_f0,
        "voiced": voiced,
        "duration_frames": total_frames,
        "states": state_sequence,
        "frame_times": frame_times,
        "tokens": tokens
    }
    
    return y_synth.astype(np.float32), sr, tracks

if __name__ == "__main__":
    import soundfile as sf
    y, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
    print(f"HMM synthesized audio: length={len(y)/sr:.2f}s, frames={tracks['duration_frames']}")
    sf.write("hmm_test.wav", y, sr)
    print("Success HMM backend test.")

