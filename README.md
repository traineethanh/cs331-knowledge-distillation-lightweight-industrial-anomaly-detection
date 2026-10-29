# CS331 — Phát Hiện Khuyết Tật Bề Mặt Công Nghiệp Trọng Lượng Nhẹ Bằng Chưng Cất Tri Thức

> **Đồ án môn học:** Thị giác Máy tính Nâng cao (CS331 — Advanced Computer Vision)  
> **Nhóm sinh viên nghiên cứu:**  
> - **Nguyễn Văn Thành** (Researcher A — Hướng Định vị Khuyết tật & Hiệu chuẩn Độ tin cậy)  
> - **Trần Minh Luân** (Researcher B — Hướng Tối ưu Kiến trúc & Nén Mô hình Phần cứng)  
> **Môi trường thực nghiệm:** Kaggle GPU (Tesla T4 / P100) & GitHub  
> **Nền tảng Demo cuối khóa:** Hugging Face Spaces (Gradio Web UI trên CPU)  
> **Chiến lược Baseline Hai Tầng:**  
> 1. **Baseline Nền tảng (Foundational Baseline):** STFPM (*BMVC 2021*, Wang et al., `arXiv:2103.04257`)  
> 2. **Bộ So sánh Hiện đại (Recent Comparator):** SK-RD4AD (*CVPRW 2025*, Park et al., `pej0918/SK-RD4AD`)  
> 3. **Nghiên cứu Cấu trúc & Động lực Lý thuyết:** MVTec AD 2 (*IJCV 2025*), CDA (*Pattern Recognition 2026*)  

---

## 1. Động Lực và Mục Tiêu Nghiên Cứu

Phát hiện bất thường công nghiệp không giám sát (Unsupervised Industrial Anomaly Detection - IAD) hướng tới việc phát hiện sản phẩm lỗi và khoanh vùng chính xác từng pixel khuyết tật chỉ bằng việc học từ ảnh sản phẩm chuẩn (defect-free).

### Chiến lược Nghiên cứu Hai Tầng (Two-Layer Research Strategy):
1. **Tầng 1 — Foundational Baseline (STFPM 2021):** Mạng học sinh ResNet-18 rút gọn siêu nhẹ (**2.78M tham số**, VRAM thực tế chỉ **579.6 MB**, huấn luyện mất **3–4 phút** trên Kaggle GPU). Đóng vai trò mỏ neo thực nghiệm để xác thực 100% bộ đọc dữ liệu MVTec AD, hàm mất mát và bộ công cụ đo lường AUROC/AUPRO.
2. **Tầng 2 — Cơ sở Khoa học 2025–2026:**
   - **MVTec AD 2 (IJCV 2025):** Chứng minh benchmark cũ đã bão hòa; thách thức thực tế nằm ở khuyết tật nhỏ, lỗi biên viền và độ nhạy ở vùng báo động giả cực thấp ($\text{AUPRO}_{0.05}$ và Pixel-AP).
   - **SK-RD4AD (CVPRW 2025):** Giải quyết vấn đề mất mát đặc trưng sâu trong Reverse Distillation bằng kết nối tắt liên cấp (skip-connections).
   - **CDA (Pattern Recognition 2026):** Phát hiện **Nghịch lý tham số adapter** (6 khối tích chập $3\times3$ ngốn tới **16.52M tham số**, chiếm gần 40% mạng học sinh).

### Hai Hướng Đóng Góp Độc Lập:
- **Hướng A — Thành (Researcher A):** Nghiên cứu phương pháp **Hiệu chuẩn Thống kê Phân phối Sạch & Cổng Tương thích Đa tỉ lệ (Concordance Gating)** nhằm triệt tiêu báo động giả trên bề mặt vân phức tạp (`carpet`) và bảo toàn khuyết tật siêu nhỏ mà không phát sinh thêm bất kỳ tham số mạng nào.
- **Hướng B — Luân (Researcher B):** Nghiên cứu **Cắt tỉa Nhánh phụ lúc Suy luận (Aux-CSD Pruning: -38% params)** và **Phân rã Tích chập Depthwise-Separable (-88% params adapter, -55% tổng thể)** nhằm xây dựng mô hình siêu nhẹ có tốc độ xử lý (FPS) cao trên thiết bị biên.

---

## 2. Trạng Thái Thực Nghiệm Hiện Tại (Phase 1)

| Giai đoạn | Trạng thái | Minh chứng Thực nghiệm trên Kaggle / Local |
| :--- | :---: | :--- |
| **Stage A: Xác thực Dữ liệu MVTec AD** | **ĐÃ ĐẠT (100%)** | Xác nhận 280 train và 117 test `carpet`, mặt nạ $256 \times 256$, `zero_train_test_overlap: true`. |
| **Stage B: STFPM Smoke Test 2 Epochs** | **ĐÃ ĐẠT (100%)** | Chạy thành công trên Kaggle Tesla T4: Loss giảm từ `1.502` $\to$ `1.140`, Peak VRAM chỉ **`579.6 MB`**, checkpoint lưu và nạp lại thành công. |
| **Stage C & D: Huấn luyện Full STFPM 100 Epochs** | **ĐANG CHẠY** | Huấn luyện 100 epochs trên `carpet` và đo đạc đầy đủ 6 chỉ số: Image-AUROC, Image-AP, Pixel-AUROC, Pixel-AP, $\text{AUPRO}_{0.30}$, $\text{AUPRO}_{0.05}$. |
| **Stage E: Thực nghiệm Bộ so sánh SK-RD4AD** | **KẾ HOẠCH** | Thực nghiệm trên `carpet` theo giao thức epoch cố định (không rò rỉ tập test). |
| **Stage F & G: Triển khai Cải tiến Nhánh A & B** | **KẾ HOẠCH** | Tách nhánh `feature/thanh-reliability` và `feature/luan-efficiency`. |
| **Stage H: Bản Demo Hugging Face Spaces** | **MÃ NGUỒN SẴN SÀNG** | Đã hoàn thành `demo/app.py` chạy mượt trên CPU, hỗ trợ tải ảnh, hiển thị Heatmap/Overlay và kết luận PASS/DEFECT trong $<60\text{ ms}$. |

---

## 3. Cài Đặt và Môi Trường

```bash
# 1. Clone repository
git clone https://github.com/traineethanh/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection.git
cd cs331-knowledge-distillation-lightweight-industrial-anomaly-detection

# 2. Cài đặt thư viện (Tương thích Python >= 3.10 và Kaggle GPU)
pip install -r requirements.txt
pip install -e .

# 3. Chạy toàn bộ 16 bài kiểm thử tự động (Unit Tests)
python -m unittest discover -s tests -v
```

---

## 4. Chuẩn Bị Dữ Liệu trên Kaggle

1. Trong Kaggle Notebook, chọn **Add Input** $\to$ Tab **Datasets** $\to$ Tìm kiếm `ipythonx/mvtec-ad` (5.27 GB).
2. Chạy script kiểm tra toàn vẹn dữ liệu:
   ```bash
   python scripts/validate_dataset.py --data-root /kaggle/input/datasets/ipythonx/mvtec-ad --categories carpet
   ```

---

## 5. Lệnh Chạy Thực Nghiệm

```bash
# 1. Chạy Smoke Test STFPM 2 epochs (Stage B)
python scripts/train_stfpm.py --data-root /kaggle/input/datasets/ipythonx/mvtec-ad --category carpet --epochs 2 --batch-size 16 --save-dir ./outputs/stfpm_smoke

# 2. Huấn luyện đầy đủ STFPM 100 epochs (Stage C)
python scripts/train_stfpm.py --data-root /kaggle/input/datasets/ipythonx/mvtec-ad --category carpet --epochs 100 --batch-size 32 --save-dir ./outputs/stfpm_carpet

# 3. Đánh giá kiểm thử và tính toán chỉ số (Stage D)
python scripts/evaluate_stfpm.py --data-root /kaggle/input/datasets/ipythonx/mvtec-ad --category carpet --checkpoint ./outputs/stfpm_carpet/last.pth

# 4. Chạy thử nghiệm giao diện Demo nội bộ
python demo/app.py
```

---

## 6. Phân Chia Nhánh Git & Trách Nhiệm Thành Viên

- **`main`:** Nhánh chính, chỉ chứa mã nguồn đã nghiệm thu thực nghiệm và tài liệu tiếng Việt chuẩn mực.
- **`develop`:** Nhánh tích hợp thử nghiệm chung của hai thành viên.
- **`feature/thanh-reliability` (Thành — Researcher A):** Phát triển hiệu chuẩn Z-Score (`A1`), cổng tương thích chéo tầng (`A2`), và làm mờ thích ứng (`A3`).
- **`feature/luan-efficiency` (Luân — Researcher B):** Phát triển cắt tỉa nhánh phụ (`B1`), tích chập phân tách chiều sâu (`B2`), và kiến trúc nén CDA-Lite (`B3`).

---

## 7. Danh Mục Tài Liệu Báo Cáo Chi Tiết (Thư mục `docs/`)

- [docs/BAO_CAO_KE_HOACH_TONG_THE.md](docs/BAO_CAO_KE_HOACH_TONG_THE.md): Bản kế hoạch tổng thể đồ án chi tiết nhất.
- [docs/phase1_audit.md](docs/phase1_audit.md): Báo cáo kiểm toán minh chứng thực nghiệm và phân định xác thực.
- [docs/research_decision_update.md](docs/research_decision_update.md): Cập nhật quyết định chiến lược baseline 2 tầng và bộ so sánh 2025.
- [docs/baseline_reproducibility_plan.md](docs/baseline_reproducibility_plan.md): Quy trình 6 bước tái hiện baseline không rò rỉ dữ liệu.
- [docs/research_gap_analysis.md](docs/research_gap_analysis.md): Phân tích khoảng trống nghiên cứu và giả thuyết khoa học cho Thành & Luân.
- [docs/demo_architecture.md](docs/demo_architecture.md): Kiến trúc ứng dụng Web Demo Gradio trên Hugging Face Spaces.
- [docs/demo_deployment_plan.md](docs/demo_deployment_plan.md): Quy trình đóng gói và triển khai bản demo lên môi trường CPU.
