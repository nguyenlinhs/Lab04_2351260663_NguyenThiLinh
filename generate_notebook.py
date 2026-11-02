import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# Cell 1: Header
add_md("""# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
## LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH - TTS)
**Sinh viên thực hiện:** Nguyễn Thị Linh  
**MSSV:** 2351260663  
**Bộ môn:** Trí tuệ Nhân tạo - Khoa CNTT - Trường Đại học Thủy Lợi  

---
### Mục tiêu bài thực hành:
1. Xây dựng pipeline xử lý Text-to-Speech: *Text Normalization $\\rightarrow$ Phonetic/Word Unit Mapping $\\rightarrow$ Prosody $\\rightarrow$ Synthesis Backend*.
2. Xây dựng và tối ưu hệ thống **Concatenative Synthesis** miền hẹp thông qua **Unit Selection** (Dynamic Programming) và làm mượt biên **Cross-fade**.
3. Phân tích mô hình **Source-Filter** và thực nghiệm biến đổi Prosody ($F_0$, duration).
4. Quan sát và điều khiển tổng hợp tham số thống kê **HMM-based Synthesis** (State sequence, trajectory $F_0$ và Formant).
5. Thực hiện suy luận mô hình **Neural TTS Pretrained**, đo đạc độ trễ Latency.
6. Đánh giá chất lượng âm thanh dựa trên kiểm thử nghe thẩm định chủ quan (**MOS / Listening Test**) và so sánh 3 hướng tiếp cận.""")

# Cell 2: Imports & Environment
add_code("""import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display

DATA_DIR = "lab4_data"
UNITS_DIR = os.path.join(DATA_DIR, "units")
AUDIO_OUT_DIR = "audio_output"
FIG_DIR = "figures"

os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

from hmm_tts_backend import synthesize_hmm, WORD_HMM_MODELS
from neural_tts_backend import synthesize as synthesize_neural

print("Đã nạp thành công các thư viện và module TTS backend!")""")

# Cell 3: Section 1 Markdown
add_md("""---
## 1. Chuẩn bị Dữ liệu & Text Normalization (Phần A)
Kho âm thanh (Unit Inventory) bao gồm 16 từ thuộc miền hẹp *'Thông báo xếp hàng'*: `mời`, `khách`, `hàng`, `số`, `không`–`chín`, `đến`, `quầy`. Mỗi từ có 2 candidates được thu âm/chuẩn hóa ở tần số lấy mẫu 16 kHz, mono, 16-bit PCM.""")

# Cell 4: Section 1 Code
add_code("""meta_path = os.path.join(DATA_DIR, "metadata.csv")
df_meta = pd.read_csv(meta_path)
display(df_meta.head(10))

# Hiển thị Hình 1: Histogram Duration và RMS
fig1_path = os.path.join(FIG_DIR, "fig1_inventory_stats.png")
if os.path.exists(fig1_path):
    plt.figure(figsize=(12, 5))
    img = plt.imread(fig1_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Hình 1: Phân phối thời lượng (Duration) và năng lượng hiệu dụng (RMS) trong Inventory", fontsize=12, fontweight="bold")
    plt.show()""")

# Cell 5: Section 2 Markdown
add_md("""---
## 2. Baseline Concatenation (Phần B)
Ghép trực tiếp các candidate cố định (Candidate 1) của từng token không qua cross-fade.
Câu thử nghiệm: **'Mời khách hàng số 23 đến quầy 5.'**""")

# Cell 6: Section 2 Code
add_code("""from Lab04 import normalize_text, baseline_concatenate

test_sentence = "Mời khách hàng số 23 đến quầy 5."
tokens = normalize_text(test_sentence)
print("Câu văn bản gốc:", test_sentence)
print("Chuỗi token đã chuẩn hóa:", tokens)

y_raw, b_times = baseline_concatenate(tokens, candidate_idx=1, sr=16000)
raw_path = os.path.join(AUDIO_OUT_DIR, "raw_concat.wav")
sf.write(raw_path, y_raw, 16000, subtype='PCM_16')

print(f"Đã xuất file âm thanh raw: {raw_path}")
print("Phát âm thanh Baseline (Chưa làm mượt):")
display(Audio(y_raw, rate=16000))""")

# Cell 7: Section 3 Markdown
add_md("""---
## 3. Unit Selection & Cross-fade Smoothing (Phần C)
Sử dụng thuật toán **Dynamic Programming (Viterbi)** để tìm chuỗi candidate có tổng chi phí nhỏ nhất:
$$\\theta^* = \\arg\\min_{\\theta_1 \\dots \\theta_J} \\left[ \\sum_{j} C_{target}(\\theta_j, t_j) + \\lambda \\sum_{j} C_{join}(\\theta_{j-1}, \\theta_j) \\right]$$
Sau đó làm mượt các biên nối bằng kỹ thuật **Cross-fade** (so sánh 0ms, 5ms, 15ms, 30ms).""")

# Cell 8: Section 3 Code
add_code("""from Lab04 import unit_selection_dp, concat_units

selected_paths, selected_indices, best_cost = unit_selection_dp(tokens, sr=16000, lambda_join=1.0)
print(f"Tổng chi phí tối ưu (Total Cost): {best_cost:.2f}")

cost_table = []
for tok, p, idx in zip(tokens, selected_paths, selected_indices):
    cost_table.append({"Token": tok, "Candidate": f"Candidate {idx+1:02d}", "File": os.path.basename(p)})
display(pd.DataFrame(cost_table))

# Ghép với Cross-fade tối ưu 15ms
y_best = concat_units(selected_paths, sr=16000, xfade_ms=15)
best_path = os.path.join(AUDIO_OUT_DIR, "best_concat.wav")
sf.write(best_path, y_best, 16000, subtype='PCM_16')

print("Phát âm thanh sau Unit Selection + Cross-fade 15ms:")
display(Audio(y_best, rate=16000))

# Hiển thị Hình 2: So sánh Waveform và Spectrogram trước/sau Cross-fade
fig2_path = os.path.join(FIG_DIR, "fig2_baseline_vs_crossfade.png")
if os.path.exists(fig2_path):
    plt.figure(figsize=(14, 10))
    img = plt.imread(fig2_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()""")

# Cell 9: Section 4 Markdown
add_md("""---
## 4. Prosody & Mô hình Nguồn - Lọc (Source - Filter) (Phần D)
- Ước lượng tần số cơ bản $F_0$ trên đoạn âm hữu thanh bằng hàm tự tương quan (Autocorrelation): $F_0 \\approx F_s / N_p$.
- Thực nghiệm thay đổi tốc độ (Time-stretch $0.8\\times, 1.2\\times$) và dịch cao độ (Pitch-shift $\\pm 2$ semitones).""")

# Cell 10: Section 4 Code
add_code("""from Lab04 import analyze_f0, load_unit

unit_voiced_path = os.path.join(UNITS_DIR, "hai_01.wav")
y_voiced = load_unit(unit_voiced_path, sr=16000)
f0_est, peak_lag, seg, corr = analyze_f0(y_voiced, sr=16000)

print(f"Tần số lấy mẫu: Fs = 16000 Hz")
print(f"Chu kỳ pitch tìm được: Np = {peak_lag} mẫu (khoảng {peak_lag/16000*1000:.2f} ms)")
print(f"Tần số cơ bản ước lượng: F0 = Fs / Np = {f0_est:.2f} Hz")

# Hiển thị Hình 3: Phân tích Prosody và Biến đổi Pitch/Duration
fig3_path = os.path.join(FIG_DIR, "fig3_prosody_pitch_duration.png")
if os.path.exists(fig3_path):
    plt.figure(figsize=(14, 10))
    img = plt.imread(fig3_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()""")

# Cell 11: Section 5 Markdown
add_md("""---
## 5. HMM-based Speech Synthesis (Phần E)
Quan sát quá trình sinh tham số âm học (F0 contour, Formant/phổ trajectories) qua chuỗi các trạng thái HMM và tái tạo âm thanh qua bộ lọc thanh đạo (Source-Filter Vocoder).""")

# Cell 12: Section 5 Code
add_code("""y_hmm, sr, tracks = synthesize_hmm(test_sentence, sr=16000)
print(f"Tổng số frame sinh bởi HMM: {tracks['duration_frames']} frames ({tracks['duration_frames']*0.01:.2f}s)")
print("Phát âm thanh HMM Synthesis:")
display(Audio(y_hmm, rate=16000))

# Thí nghiệm điều khiển tham số HMM
y_hmm_fast, _, _ = synthesize_hmm(test_sentence, sr=16000, duration_scale=0.8)
y_hmm_pitch_high, _, _ = synthesize_hmm(test_sentence, sr=16000, f0_scale=1.1)

print("Phát âm thanh HMM (Scale Duration 0.8x - Nói nhanh hơn):")
display(Audio(y_hmm_fast, rate=16000))

print("Phát âm thanh HMM (Scale F0 1.1x - Tông giọng cao hơn):")
display(Audio(y_hmm_pitch_high, rate=16000))

# Hiển thị Hình 4: Quỹ đạo tham số HMM và State Alignment
fig4_path = os.path.join(FIG_DIR, "fig4_hmm_parameter_trajectories.png")
if os.path.exists(fig4_path):
    plt.figure(figsize=(14, 9))
    img = plt.imread(fig4_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()""")

# Cell 13: Section 6 Markdown
add_md("""---
## 6. Pretrained Neural TTS Inference (Phần F)
Thực hiện suy luận bằng mô hình học sâu Pretrained TTS (chạy offline), trích xuất Mel-spectrogram và đo đạc độ trễ (latency).""")

# Cell 14: Section 6 Code
add_code("""sentences_test = [
    "Mời khách hàng số 23 đến quầy 5",
    "Mời khách hàng số 104 đến quầy 2",
    "Mời khách hàng số 89 đến quầy 3"
]

neural_results = []
for i, sent in enumerate(sentences_test, 1):
    y_nn, sr, extra = synthesize_neural(sent, sr=16000)
    neural_results.append({
        "Câu": f"Câu {i}",
        "Nội dung": sent,
        "Thời lượng (s)": round(extra["duration"], 2),
        "Độ trễ Latency (ms)": round(extra["latency"] * 1000, 2)
    })
    if i == 1:
        y_neural_main = y_nn

display(pd.DataFrame(neural_results))
print("Phát âm thanh Pretrained Neural TTS:")
display(Audio(y_neural_main, rate=16000))""")

# Cell 15: Section 7 Markdown
add_md("""---
## 7. Đánh giá, Thẩm định (MOS Test) & So sánh Đối chiếu (Phần G)
So sánh 3 phương pháp tổng hợp tiếng nói dựa trên kiểm tra nghe thẩm định chủ quan (MOS Listening Test trên 10 người nghe) và biểu đồ đối chiếu Waveform/Spectrogram.""")

# Cell 16: Section 7 Code
add_code("""# Hiển thị Hình 5: Đối chiếu Waveform và Spectrogram 3 hệ thống
fig5_path = os.path.join(FIG_DIR, "fig5_backend_comparison.png")
if os.path.exists(fig5_path):
    plt.figure(figsize=(15, 9))
    img = plt.imread(fig5_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Đọc kết quả khảo sát Listening Test
df_ratings = pd.read_csv("ratings.csv")
mos_summary = df_ratings.groupby("method").agg({
    "naturalness": ["mean", "std"],
    "intelligibility": ["mean", "std"],
    "continuity": ["mean", "std"],
    "overall_mos": ["mean", "std"]
})
display(mos_summary)""")

# Cell 17: Summary Markdown
add_md("""---
## 8. Kết luận Thực nghiệm
1. **Concatenative TTS (Ghép nối)**:
   - *Ưu điểm:* Chất lượng âm sắc rất tự nhiên ở từng từ đơn lẻ vì phát trực tiếp từ bản thu người thật.
   - *Hạn chế:* Dễ bị đứt gãy năng lượng/cao độ tại các biên nối; cần inventory rất lớn để bao phủ ngữ cảnh.
2. **HMM-based Synthesis (Tham số HMM)**:
   - *Ưu điểm:* Độ liền mạch (Continuity) rất cao, không bị click ở biên, dung lượng mô hình rất nhẹ, khả năng điều khiển ngữ điệu ($F_0$, duration) vô cùng linh hoạt.
   - *Hạn chế:* Giọng nghe có tính 'máy móc' (*robotic/buzzy*) do giả định nguồn kích thích đơn giản hóa và hiệu ứng làm mịn quá mức (*over-smoothing*).
3. **Neural TTS (Học sâu)**:
   - *Ưu điểm:* Đạt điểm tự nhiên (*Naturalness*) và độ rõ (*Intelligibility*) vượt trội, tái tạo ngữ điệu giàu cảm xúc và chất giọng mượt mà.
   - *Hạn chế:* Đòi hỏi tài nguyên tính toán (GPU/CPU) và độ trễ sinh âm (*latency*) cao hơn các phương pháp tham số truyền thống.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Lab04.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Đã tạo thành công Lab04.ipynb!")
