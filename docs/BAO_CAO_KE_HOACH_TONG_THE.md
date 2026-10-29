# BÁO CÁO KẾ HOẠCH TỔNG THỂ VÀ CHI TIẾT ĐỒ ÁN NGHIÊN CỨU CS331

> **Đề tài:** Phát hiện và Định vị Khuyết tật Bề mặt Công nghiệp Trọng lượng nhẹ Dựa trên Chưng cất Tri thức (Knowledge Distillation-Based Lightweight Industrial Anomaly Detection)  
> **Nhóm sinh viên thực hiện:**  
> - **Trần Quang Thành** (Researcher A — Hướng Định vị Khuyết tật & Hiệu chuẩn Độ tin cậy)  
> - **Phạm Mạnh Luân** (Researcher B — Hướng Tối ưu Kiến trúc & Nén Mô hình Phần cứng)  
> **Môi trường nghiên cứu:** Kaggle GPU (Tesla T4 / P100, 16 GB VRAM) & GitHub  
> **Nền tảng Demo cuối khóa:** Hugging Face Spaces (Gradio Web UI chạy trên CPU)  
> **Thời điểm cập nhật:** 29/10/2026 — Đã hoàn thành nghiệm thu thực tế Stage A & Stage B trên Kaggle  

---

## 1. TỔNG QUAN ĐỀ TÀI VÀ CƠ SỞ KHOA HỌC (2025–2026)

### 1.1 Thách thức Thực tế trong Công nghiệp
Trong sản xuất công nghiệp hiện đại, việc kiểm định tự động bề mặt sản phẩm (như vải dệt, cổ chai, ren bu-lông kim loại) đòi hỏi mô hình phải đáp ứng 4 tiêu chuẩn khắt khe:
1. **Học không giám sát (Unsupervised):** Chỉ học trên ảnh sản phẩm chuẩn (defect-free), không dựa vào dữ liệu lỗi nhân tạo vì khuyết tật thực tế muôn hình vạn trạng.
2. **Nhiệm vụ kép:** Vừa phát hiện ở mức ảnh (Image-level: PASS hay DEFECT), vừa khoanh vùng chính xác từng pixel lỗi (Pixel-level Localization) để công nhân kiểm tra vị trí.
3. **Độ nhạy khuyết tật vi mô:** Nhận diện được các lỗi rất nhỏ như sợi chỉ đứt, vết châm kim, xước viền ren.
4. **Trọng lượng nhẹ & Tốc độ thời gian thực:** Phù hợp triển khai trên máy tính nhúng/cạnh dây chuyền sản xuất với VRAM nhỏ và độ trễ tính bằng mili-giây.

### 1.2 Yêu cầu Học thuật và Cơ sở Khoa học Mới (2025–2026)
Giảng viên hướng dẫn yêu cầu: *Phần tổng quan tài liệu và luận cứ khoa học phải dựa chủ đạo trên các công bố mới nhất (2025–2026).* Nhóm xây dựng cơ sở học thuật từ các công trình đỉnh cao:
- **MVTec AD 2 Benchmark (IJCV / arXiv 2025, Heckler-Kram et al.):** Chứng minh rằng các bộ dữ liệu cũ như MVTec AD kinh điển đã bị bão hòa ($>98–99\%$ AUROC). Điểm phân định thực sự giữa các mô hình nằm ở **độ nhạy khuyết tật vi mô, khả năng chống báo động giả trên bề mặt vân phức tạp, và độ chính xác ở vùng false-positive cực thấp ($\text{AUPRO}_{0.05}$ và Pixel-AP)** $\to$ Củng cố trực tiếp động lực của Thành.
- **SK-RD4AD (CVPRW 2025, Park et al.):** Chỉ ra việc nén qua bottleneck của Reverse Distillation (RD4AD) gây mất mát đặc trưng sâu, đề xuất kết nối tắt liên cấp (skip-connections) $\to$ Trở thành bộ so sánh 2025 mở và minh bạch của nhóm.
- **CDA (Pattern Recognition 2026, Hu et al.):** Phân tích mã nguồn bài báo này làm lộ ra **Nghịch lý tham số**: 6 khối tích chập adapter $3\times3$ ngốn tới **16.52M tham số** ($38.25\%$ mạng học sinh), tạo nên sự lãng phí tài nguyên phần cứng lớn $\to$ Củng cố trực tiếp động lực của Luân.
- **Dinomaly (CVPR 2025):** Đại diện cho xu hướng dùng Foundation Models (DINOv2) đạt độ chính xác cao nhưng rất nặng nề ($>80\text{M}$ tham số), là đối trọng lớn mà các mô hình nhẹ cần cạnh tranh về tốc độ và chi phí tính toán.

---

## 2. CHIẾN LƯỢC BASELINE HAI TẦNG (TWO-LAYER STRATEGY)

Để dung hòa giữa tính hiện đại của tài liệu và khả năng tái hiện thực tế trên hạ tầng Kaggle miễn phí:

```
TẦNG 1: FOUNDATIONAL BASELINE (Mỏ neo thực nghiệm vững chắc)
└── STFPM (Wang et al., BMVC 2021)
    • Mạng học sinh ResNet-18 rút gọn chỉ có 2.78M tham số.
    • Thực nghiệm thực tế trên Kaggle Tesla T4: Ăn đúng 579.6 MB VRAM, train 100 epochs chỉ mất ~3-4 phút.
    • Vai trò: Mỏ neo kiểm chứng pipeline nạp dữ liệu, tính loss cosine/L2, và tính AUROC/AUPRO.

TẦNG 2: RECENT COMPARATORS & PRIOR ART (Cơ sở so sánh 2025–2026)
├── SK-RD4AD (CVPRW 2025 - Park et al.): Bộ so sánh chính cho nhóm Reverse Distillation hiện đại (toàn văn mở trên CVF, code PyTorch chuẩn).
├── MVTec AD 2 (IJCV 2025): Bộ tiêu chuẩn đánh giá khuyết tật siêu nhỏ và AUPRO 0.05.
└── CDA (Pattern Recognition 2026): Ca nghiên cứu chỉ ra nghịch lý tham số và bài toán nén mô hình.
```

---

## 3. PHÂN CHIA TRÁCH NHIỆM & HAI HƯỚNG ĐÓNG GÓP ĐỘC LẬP

Thành và Luân đảm nhận hai hướng đi trực giao (Orthogonal), không phụ thuộc hay xung đột mã nguồn:

| Tiêu chí | **TRẦN QUANG THÀNH (Researcher A)**<br/>*Hướng Định vị & Hiệu chuẩn Độ tin cậy* | **PHẠM MẠNH LUÂN (Researcher B)**<br/>*Hướng Tối ưu Kiến trúc & Nén Mô hình* |
| :--- | :--- | :--- |
| **Vấn đề cốt lõi** | Bản đồ sai lệch bị nhòe khuyết tật nhỏ; bề mặt vân lặp (`carpet`) bị báo động giả cao do phương sai nền. | 6 khối tích chập adapter $3\times3$ chiếm tới 16.52M tham số (gấp 2.37 lần bộ giải mã GSU 6.96M). |
| **Tầng can thiệp** | Tầng **hậu xử lý / suy luận** (giữ nguyên trọng số mạng nơ-ron). | Tầng **cấu trúc mạng nơ-ron** (huấn luyện các biến thể kiến trúc nén). |
| **Giải pháp đề xuất** | **`A1`:** Chuẩn hóa Z-Score theo phương sai ảnh sạch.<br>**`A2`:** Cổng tương thích nhất quán đa tỉ lệ (Concordance Gating).<br>**`A3`:** Làm mờ thích ứng và pooling Top-$q\%$. | **`B1`:** Cắt tỉa adapter lúc suy luận (Aux-CSD: -38% params).<br>**`B2`:** Phân rã tích chập Depthwise-Separable (-88% adapter).<br>**`B3`:** Kiến trúc siêu nhẹ CDA-Lite (-55% params toàn mạng). |
| **Mục tiêu khoa học** | Tăng Pixel-AP $\ge +1.5\%$, tăng $\text{AUPRO}_{0.05} \ge +2.0\%$ trên bề mặt vân phức tạp. | Giảm $\ge 35–50\%$ tham số, tăng FPS $\ge 25\%$, bảo toàn $\ge 99.5\%$ AUROC. |
| **Chi phí phần cứng** | **Thêm 0 tham số**, độ trễ thêm $<2\text{ ms}$. | Giảm mạnh VRAM và FLOPs, tăng tốc độ trên thiết bị biên. |

---

## 4. GIAO THỨC THỰC NGHIỆM VÀ NGUYÊN TẮC KHÔNG RÒ RỈ DỮ LIỆU

1. **Nguyên tắc "Zero-Claim":** Tuyệt đối không đưa số liệu thời gian, bộ nhớ hay AUROC vào báo cáo nếu chưa có log thực nghiệm GPU thực tế.
2. **Cấm triệt để rò rỉ tập kiểm thử (No Test-Set Leakage):** Không đánh giá test set sau mỗi epoch để nhặt checkpoint đẹp nhất. Mọi đánh giá chỉ dùng checkpoint epoch cố định cuối cùng (`last.pth`).
3. **Hiệu chuẩn Không Rò rỉ (Zero-Leakage Calibration):** Nhánh A của Thành chỉ đo $\mu_k, \sigma_k$ trên tập huấn luyện sạch (`train/good`).
4. **Bộ ba danh mục đại diện (Core Triad):** `carpet` (vân phức tạp), `bottle` (vật thể cứng phản xạ), `screw` (khuyết tật ren vi mô).
5. **Hệ thống 6 chỉ số đo đạc:** Image-AUROC, Image-AP, Pixel-AUROC, Pixel-AP, $\text{AUPRO}_{0.30}$, $\text{AUPRO}_{0.05}$.

---

## 5. LỘ TRÌNH 8 GIAI ĐOẠN TRIỂN KHAI (ROADMAP)

- **Giai đoạn 1 (Xác thực dữ liệu MVTec AD):** **ĐÃ ĐẠT (100%)** — Đủ 280 train và 117 test `carpet`, mặt nạ $256 \times 256$.
- **Giai đoạn 2 (STFPM Smoke Test 2 Epochs):** **ĐÃ ĐẠT (100%)** — Chạy trên Kaggle Tesla T4, loss giảm từ `1.502` $\to$ `1.140`, VRAM đo được chỉ **`579.6 MB`**.
- **Giai đoạn 3 (Huấn luyện Full STFPM 100 Epochs):** Đang thực hiện trên Kaggle GPU (khoảng 3–4 phút).
- **Giai đoạn 4 (Đánh giá Chỉ số & Heatmap):** Đo đủ 6 chỉ số chất lượng định vị và xuất ảnh trực quan.
- **Giai đoạn 5 (Thực nghiệm SK-RD4AD):** Huấn luyện mô hình đối thủ 2025 theo quy chuẩn epoch cố định.
- **Giai đoạn 6 (Triển khai Nhánh A & Nhánh B):** Thành triển khai bộ lọc thích ứng; Luân triển khai các biến thể nén mạng.
- **Giai đoạn 7 (Hợp nhất C1 & Phân tích Đánh đổi):** Kết hợp mô hình nén của Luân với bộ lọc của Thành để tạo ra mô hình tối ưu toàn diện.
- **Giai đoạn 8 (Đóng gói Demo & Khóa luận):** Triển khai ứng dụng Gradio lên Hugging Face Spaces và hoàn thiện báo cáo bảo vệ trước hội đồng.

---

## 6. THIẾT KẾ BẢN DEMO TƯƠNG TÁC (GRADIO TRÊN HUGGING FACE SPACES)

- **Nền tảng:** Hugging Face Spaces (CPU Basic: 2 vCPU, 16 GB RAM).
- **Nguyên tắc độc lập:** Mã giao diện UI tách biệt với mã suy luận. Không tải dataset MVTec nặng lên Space; chỉ nạp trọng số đã train 1 lần duy nhất lúc khởi động.
- **Tính năng người dùng:** Tải ảnh lên $\to$ Chọn danh mục $\to$ Chỉnh ngưỡng nhạy cảm $\tau \to$ Hiển thị tức thì:
  1. Ảnh gốc $256 \times 256$.
  2. Bản đồ nhiệt khuyết tật (Jet Colormap).
  3. Ảnh phủ khuyết tật (Alpha Overlay).
  4. Quyết định tự động: **PASS (Bình thường)** hoặc **DEFECT (Có lỗi)** kèm điểm số dị thường và độ trễ tính bằng mili-giây.
- **Hiện trạng:** Đã hoàn thành file `demo/app.py`, `demo/requirements.txt`, và `demo/README.md`. Đã chạy thử nghiệm suy luận trên CPU trơn tru không lỗi.

---

## 7. QUẢN TRỊ RỦI RO & BƯỚC ĐI TIẾP THEO

- **Rủi ro hết giờ Kaggle (30h/tuần):** STFPM chỉ mất 3–4 phút/run; mã nguồn tự động lưu checkpoint `last.pth` và metadata để có thể tiếp tục bất cứ lúc nào.
- **Rủi ro mô hình SK-RD4AD lỗi thư viện:** Đã có phương án dự phòng là phiên bản sạch `B0` (CDA sạch không Mamba, thuần PyTorch) đã được kiểm thử 16/16 unit test.
- **Khuyến nghị tức thì:** Bạn hãy theo dõi ô code 100 Epochs (Giai đoạn 3 & 4) trên Kaggle để ghi nhận kết quả AUROC đầu tiên của đề tài!
