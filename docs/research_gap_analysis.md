# Phân Tích Khoảng Trống Nghiên Cứu và Câu Hỏi Nghiên Cứu Điều Chỉnh

**Đề tài:** Phát hiện và Định vị Khuyết tật Bề mặt Công nghiệp Trọng lượng nhẹ Bằng Chưng cất Tri thức  
**Nhóm sinh viên:** Nguyễn Văn Thành (Researcher A) & Trần Minh Luân (Researcher B)  
**Môn học:** CS331 — Thị giác Máy tính Nâng cao  

---

## 1. Bối Cảnh và Điểm Tựa Từ Yăn Bản Mới Nhất (2025–2026)

Nghiên cứu phát hiện khuyết tật công nghiệp (Industrial Anomaly Detection - IAD) giai đoạn 2021–2026 đã phát triển qua các giai đoạn:
1. **Chưng cất Xuôi Căn bản (2021–2022):** STFPM (BMVC 2021) khởi xướng so khớp kim tự tháp đặc trưng đa tỉ lệ, nhưng gặp hạn chế do học sinh nhìn trực tiếp ảnh gốc, dễ khái quát hóa xuyên qua khuyết tật (identity over-generalization).
2. **Chưng cất Ngược & Mã hóa Nút thắt (2022–2024):** RD4AD (CVPR 2022) giới thiệu nút thắt One-Class Bottleneck Embedding (OCBE); EfficientAD (WACV 2024) đưa ra chuẩn hóa quantile; MambaAD (NeurIPS 2024) đưa ra bộ giải mã chuỗi State-Space tuyến tính.
3. **Chưng cất Dị thể & Đa Tỉ lệ Nâng cao (2025–2026):**
   - **MVTec AD 2 Benchmark (IJCV 2025, Heckler-Kram et al.):** Chứng minh các bộ dữ liệu cũ đã bão hòa ($>98–99\%$ AUROC). Thách thức thực sự nằm ở khuyết tật siêu nhỏ, lỗi biên viền, và độ chính xác ở vùng tỷ lệ báo động giả cực thấp ($\text{AUPRO}_{0.05}$, Pixel-AP).
   - **SK-RD4AD (CVPRW 2025, Park et al.):** Khắc phục hiện tượng mất mát đặc trưng sâu do nén quá mức qua nút thắt trong Reverse Distillation bằng các kết nối tắt liên cấp (skip-connections).
   - **Dinomaly (CVPR 2025):** Xu hướng dùng Foundation Model (DINOv2) đạt độ chính xác cao nhưng rất nặng nề ($>80\text{M}$ tham số), không phù hợp cho biên công nghiệp.
   - **CDA (Pattern Recognition 2026, Hu et al.):** Đưa ra chưng cất chéo tỉ lệ (Cross-Scale Distillation) và bộ giải mã GSU, nhưng để lộ "Nghịch lý tham số 16.52M" và phép cộng dồn bản đồ lỗi tĩnh không được hiệu chuẩn.

---

## 2. Ranh Giới Đỏ: Những Gì TUYỆT ĐỐI KHÔNG Tuyên Bố Là Tính Mới

Để bảo đảm tính liêm chính học thuật, nhóm xác lập các ranh giới không được nhận vơ:
- **Không nhận phát minh ra chưng cất chéo tỉ lệ (Cross-Scale Distillation):** Đây là đóng góp của CDA (Hu et al., 2026).
- **Không nhận phát minh ra khối giải mã Mamba/GSU:** MambaAD (2024) và CDA (2026) đã đề xuất.
- **Không nhận phát minh ra chuẩn hóa phân phối ảnh sạch:** Nguyên lý này thuộc về EfficientAD (2024).
- **Không tuyên bố tính mới chỉ bằng việc giảm số kênh hoặc thay trọng số scalar:** Đây là tinh chỉnh siêu tham số thông thường, không phải đóng góp khoa học.

---

## 3. Hướng A — Nguyễn Văn Thành: Định Vị Khuyết Tật & Triệt Tiêu Báo Động Giả

### 3.1 Khoảng trống Nghiên cứu Nhắm tới
Trong các mô hình hiện tại, bản đồ sai lệch từ các tầng được cộng dồn tuyến tính với trọng số cố định ($2.0 \sum a_k + 0.1 \sum ac_k$) và bộ lọc Gauss $\sigma=4.0$ đồng nhất:
- **Vấn đề phân kỳ phương sai nền:** Tầng nông ($64 \times 64$) luôn có phương sai tự nhiên cao trên bề mặt vân (`carpet`), gây ra vô số đỉnh báo động giả (False Positives).
- **Vấn đề xói mòn khuyết tật nhỏ:** Tầng sâu ($16 \times 16$) làm mờ các vết rách chỉ, châm kim nhỏ khi bị lọc qua $\sigma=4.0$.

### 3.2 Câu hỏi Nghiên cứu (RQ-A) & Giả thuyết
> **RQ-A:** *Liệu việc chuẩn hóa thống kê Z-Score kết hợp cổng tương thích chéo tỉ lệ có thể triệt tiêu báo động giả trên bề mặt vân lặp và bảo toàn khuyết tật vi mô mà không cần huấn luyện lại mạng nơ-ron?*

- **Giả thuyết A.1 (Chuẩn hóa Z-Score):** $\tilde{a}_k = (a_k - \mu_k^{\text{norm}}) / \sigma_k^{\text{norm}}$ đo trên tập ảnh sạch (`train/good`) kéo phương sai nền về 0, giúp tăng Pixel-AP $\ge +1.5\%$.
- **Giả thuyết A.2 (Cổng Tương thích Concordance Gating):** $a_k^{\text{gated}} = \tilde{a}_k \odot \text{Sigmoid}(ac_k)$ kích hoạt vùng lỗi có tính nhất quán đa tỉ lệ và dập tắt nhiễu đơn tầng, tăng $\text{AUPRO}_{0.05} \ge +2.0\%$.
- **Giả thuyết A.3 (Zero Overhead):** Can thiệp ở tầng suy luận, **thêm 0 tham số**, độ trễ $<2\text{ ms}$.

---

## 4. Hướng B — Trần Minh Luân: Đánh Đổi Độ Chính Xác - Hiệu Năng & Nén Mô Hình

### 4.1 Khoảng trống Nghiên cứu Nhắm tới
- **Nghịch lý tham số Adapter:** 6 khối tích chập $3\times3$ liên tầng (`c11..c32`) ngốn tới **16.52M tham số** ($38.25\%$ mạng học sinh, lớn gấp 2.37 lần bộ giải mã GSU 6.96M).
- **Câu hỏi then chốt:** Nhánh chéo tỉ lệ này là "bộ điều hòa lúc huấn luyện" hay là "bộ dự đoán bắt buộc lúc suy luận"?

### 4.2 Câu hỏi Nghiên cứu (RQ-B) & Giả thuyết
> **RQ-B:** *Liệu các khối adapter chéo tỉ lệ có thể được cắt tỉa lúc suy luận hoặc phân rã thành tích chập tách biệt chiều sâu để giảm mạnh tham số và độ trễ trong khi vẫn bảo toàn $\ge 99.5\%$ chất lượng AUROC?*

- **Giả thuyết B.1 (Cắt tỉa lúc suy luận - Aux-CSD):** Cắt bỏ 6 khối adapter lúc suy luận giúp loại bỏ ngay **16.52M tham số hoạt động (-38.3%)**, tăng tốc $\ge 20\%$ mà độ suy giảm AUROC $\le 0.5\%$.
- **Giả thuyết B.2 (Phân rã Separable):** Thay thế bằng tích chập Depthwise-Separable ($1\times1$ pointwise $+ 3\times3$ depthwise) giảm tham số adapter từ **16.52M xuống 1.86M (-88.7%)**.
- **Giả thuyết B.3 (Kiến trúc Siêu nhẹ CDA-Lite):** Phân rã cả khối nút thắt `GCM.res` tạo ra mạng học sinh chỉ **19.46M tham số** (giảm $-54.9\%$ tổng thể), tăng vọt tốc độ xử lý (FPS) trên phần cứng biên.

---

## 5. Hợp Nhất Hệ Thống (Giai đoạn 7 — C1)

Kết hợp mô hình nén của Luân (`B2` hoặc `B3`) với bộ lọc hiệu chuẩn của Thành (`A3`) để kiểm chứng xem bộ lọc thích ứng có bù đắp được phần suy giảm độ chính xác nhỏ do nén mô hình hay không.
