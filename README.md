# CSE457 - Xử lý Âm thanh và Tiếng nói | Lab 4: Tổng hợp Tiếng nói (TTS)

* **Sinh viên:** Nguyễn Thị Linh
* **MSSV:** 2351260663
* **Đề tài:** Lab 4 - Tổng hợp Tiếng nói (Text-To-Speech - TTS)

---

## 1. Cấu trúc thư mục

```
Lab04_2351260663_NguyenThiLinh/
├── audio_output/                   # File âm thanh xuất ra từ các thực nghiệm
│   ├── raw_concat.wav              # Ghép nối thô (baseline, xfade = 0ms)
│   ├── best_concat.wav             # Ghép nối tối ưu (unit selection + xfade 15ms)
│   ├── concat_xfade_0ms.wav, 5ms, 15ms, 30ms
│   ├── hmm_tts.wav                 # Âm thanh tổng hợp HMM baseline
│   ├── hmm_tts_dur0.8.wav          # HMM thay đổi tốc độ (scale 0.8x)
│   ├── hmm_tts_f0_1.1.wav          # HMM thay đổi cao độ (scale 1.1x)
│   └── neural_tts.wav              # Âm thanh tổng hợp Neural TTS
├── figures/                        # 5 biểu đồ trực quan hóa kết quả
│   ├── fig1_inventory_stats.png
│   ├── fig2_baseline_vs_crossfade.png
│   ├── fig3_prosody_pitch_duration.png
│   ├── fig4_hmm_parameter_trajectories.png
│   └── fig5_backend_comparison.png
├── lab4_data/                      # Kho âm thanh và dữ liệu mô hình
│   ├── metadata.csv                # Quản lý 32 unit âm thanh
│   ├── units/                      # 16 từ vựng x 2 candidates (16kHz mono)
│   ├── hmm_demo/                   # Dữ liệu mô hình tham số HMM
│   └── neural_cache/               # Dữ liệu phục vụ Neural TTS offline
├── Lab04.py                        # Script thực thi toàn bộ luồng bài Lab
├── Lab04.ipynb                     # Jupyter Notebook tương tác (đã execute đầy đủ)
├── hmm_tts_backend.py              # Backend tổng hợp HMM-based
├── neural_tts_backend.py           # Backend tổng hợp Neural TTS
├── ratings.csv                     # Dữ liệu khảo sát thẩm định MOS 10 người nghe
├── report_Lab04.md                 # Báo cáo chi tiết kết quả thực hành
└── README.md                       # Hướng dẫn chạy và tổng quan
```

---

## 2. Hướng dẫn chạy chương trình

### Cách 1: Chạy trực tiếp qua Python Script
```bash
python Lab04.py
```
Toàn bộ quy trình từ chuẩn hóa văn bản, ghép nối âm thanh, tính chi phí unit selection, mô phỏng HMM, suy luận neural TTS và sinh các biểu đồ sẽ được thực hiện tự động.

### Cách 2: Sử dụng Jupyter Notebook tương tác
Mở file `Lab04.ipynb` trong VS Code hoặc khởi chạy Jupyter Server:
```bash
jupyter notebook Lab04.ipynb
```
Notebook đã được thực thi sẵn và nhúng đầy đủ các biểu đồ, bảng dữ liệu và trình phát âm thanh audio widget tương tác.

---

## 3. Tóm tắt kết quả chính
1. **Inventory**: 16 từ miền hẹp "thông báo xếp hàng", mỗi từ 2 bản ghi, chuẩn hóa $16\text{ kHz}$ mono, $\text{RMS} = 0.08$.
2. **Concatenation Baseline vs Best**: Cross-fade $15\text{ ms}$ triệt tiêu hoàn toàn tiếng click ở biên nối; thuật toán Unit Selection Viterbi giảm tổng chi phí xuống $4055.99$.
3. **Prosody & Source-Filter**: Đo chính xác $F_0 \approx 155.34\text{ Hz}$ ($N_p = 103$ mẫu), kiểm chứng bảo toàn bao phổ Formant khi biến đổi Pitch $\pm 2$ semitones và Duration $0.8\times, 1.2\times$.
4. **HMM-based TTS**: Minh họa thuật toán MLPG làm trơn quỹ đạo $F_0$ và Formant qua các state; điều khiển linh hoạt thời lượng và cao độ.
5. **Neural TTS**: Đạt điểm chất lượng MOS cao nhất ($4.82/5$) với độ trễ suy luận thời gian thực $\approx 280\text{ ms}$.

