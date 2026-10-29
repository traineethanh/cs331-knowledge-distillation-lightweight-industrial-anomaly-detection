# Cập Nhật Quyết Định Chiến Lược Nghiên Cứu (Phase 1)

**Ngày cập nhật:** 29/10/2026  
**Nhóm sinh viên:** Nguyễn Văn Thành (Researcher A) & Trần Minh Luân (Researcher B)  
**Môn học:** CS331 — Thị giác Máy tính Nâng cao  

---

## 1. Cấu Trúc Ba Tầng Chiến Lược

Để thỏa mãn yêu cầu của giảng viên về tài liệu 2025–2026 trong khi vẫn đảm bảo khả năng tái hiện thực tế trên Kaggle:

```
TẦNG 1: FOUNDATIONAL BASELINE (Mỏ neo kiểm chứng thực nghiệm)
└── STFPM (Wang et al., BMVC 2021)
    • Mạng học sinh ResNet-18 rút gọn chỉ 2.78M tham số, VRAM 580 MB, train 3-4 phút trên T4.
    • Đóng vai trò kiểm chứng toàn bộ pipeline dữ liệu, hàm loss, code tính AUROC/PRO và tạo ngưỡng sàn thực nghiệm.

TẦNG 2: PRIMARY RECENT EXPERIMENTAL COMPARATOR (Bộ so sánh hiện đại chính)
└── SK-RD4AD (Park et al., CVPRW 2025)
    • Toàn văn mở trên CVF Open Access, mã nguồn PyTorch chuẩn trên GitHub (pej0918/SK-RD4AD).
    • Đại diện cho nhóm Reverse Distillation thế hệ mới, giải quyết mất mát đặc trưng sâu qua skip-connections.

TẦNG 3: LITERATURE REVIEW & SECONDARY PRIOR ART (Ca nghiên cứu cấu trúc)
└── CDA (Hu et al., Pattern Recognition 2026)
    • Đóng vai trò ca nghiên cứu lý thuyết về chưng cất chéo tỉ lệ và phát hiện "Nghịch lý tham số 16.52M".
```

---

## 2. Bảng So Sánh và Lý Do Lựa Chọn Các Ứng Viên 2025–2026

| Ứng viên | Hội nghị / Tạp chí | Toàn văn | Mã nguồn | Đánh giá khả thi trên Kaggle | Quyết định chiến lược |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SK-RD4AD** | CVPRW 2025 | **Mở (CVF)** | **Có sẵn trên GitHub** | **Khả thi cao**. Chạy PyTorch thuần, không lỗi compile CUDA. | **CHỌN LÀM BỘ SO SÁNH 2025 CHÍNH**. |
| **STFPM** | BMVC 2021 | **Mở (arXiv)** | **Có sẵn (Anomalib)** | **Khả thi cực cao (<4 phút/run)**. | **CHỌN LÀM BASELINE NỀN TẢNG**. |
| **CDA** | Pattern Rec. 2026 | **Trả phí (Elsevier)**| Đã kiểm toán lỗi | Khả thi qua bản clean-room PyTorch của nhóm (`src/`). | **GIỮ LÀM ĐỐI CHỨNG CẤU TRÚC**. |
| **CPKD** | AAAI 2025 | Mở (AAAI OJS) | **Không tìm thấy code** | Rủi ro tự viết lại từ đầu quá lớn trong 3 tuần. | **LOẠI KHỎI THỰC NGHIỆM** (chỉ trích dẫn lý thuyết). |
| **DHKD** | ESWA 2026 | Trả phí (Elsevier) | Không chính thức | Rủi ro biên dịch `mamba-ssm` C++ CUDA kernel crash trên Kaggle. | **LOẠI KHỎI THỰC NGHIỆM**. |
| **LKDF-AD** | Sci. Reports 2026 | Mở (Nature) | Code phụ lục | Lệch giao thức: bài toán **Few-Shot (1–10 ảnh)**. | **LOẠI KHỎI BẢNG SO SÁNH CHÍNH**. |

---

## 3. Quy Chuẩn Không Rò Rỉ Dữ Liệu Kiểm Thử

1. **Cấm tuyệt đối việc chọn checkpoint dựa trên điểm test:** Mọi thử nghiệm đều đánh giá trên **checkpoint epoch cố định cuối cùng (`last.pth`)**.
2. **Không rò rỉ khi hiệu chuẩn:** Các hệ số thống kê $\mu, \sigma$ của Thành chỉ được tính trên tập ảnh chuẩn huấn luyện (`train/good`).
3. **Đo đạc phần cứng chuẩn mực:** Luân đo Parameters, FLOPs, Latency ($\text{ms}$), và Peak VRAM trên cùng một GPU Kaggle với 50 lần warmup và 200 lần đo có `torch.cuda.synchronize()`.
