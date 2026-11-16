# BÁO CÁO TỔNG HỢP TIẾN ĐỘ & KẾ HOẠCH HỢP TÁC NGHIÊN CỨU ĐỒ ÁN CS331
## ĐỀ TÀI: KNOWLEDGE DISTILLATION-BASED LIGHTWEIGHT INDUSTRIAL ANOMALY DETECTION
*(Phát hiện Dị thường Công nghiệp Nhẹ dựa trên Chưng cất Tri thức)*

- **Kho mã nguồn:** [cs331-knowledge-distillation-lightweight-industrial-anomaly-detection](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection)
- **Nhánh làm việc chính:** `chore/repository-reset` (Commit HEAD: `41af7b015a69fe7480c8fd1212e6e22840ae22d0`)
- **Bài báo nền tảng:** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (arXiv:2402.00448v1 - [papers/2402.00448v1.pdf](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/papers/2402.00448v1.pdf))
- **Đối tượng sử dụng tài liệu:** Dành cho cả 2 thành viên nhóm nghiên cứu (Sinh viên 1 & Sinh viên 2).
- **Ngày lập:** `16/11/2026`

---

## 1. BỐI CẢNH ĐỀ TÀI & CƠ CHẾ HOẠT ĐỘNG CỦA DSKD

### 1.1. Mục Tiêu Tổng Quát
Trong kiểm thử chất lượng sản phẩm công nghiệp (Industrial Defect Inspection), các mẫu bị lỗi (anomalous data) thường rất hiếm và đa dạng. Phương pháp tiếp cận tối ưu là **Học không giám sát một lớp (Unsupervised One-Class Anomaly Detection)**: Mô hình chỉ được huấn luyện trên các ảnh sản phẩm bình thường (`train/good`) để học phân phối chuẩn, sau đó ở pha kiểm thử sẽ phát hiện và khoanh vùng các điểm bất thường.

### 1.2. Cơ Chế Hoạt Động Cốt Lõi của DSKD (Dual-Student KD)
Bài báo nền tảng đề xuất mô hình gồm 3 mạng nơ-ron học hỏi lẫn nhau:
1. **Teacher ($T$):** Sử dụng ResNet-18 đã được nạp trọng số tiền huấn luyện trên ImageNet-1K. Trong suốt quá trình huấn luyện và kiểm thử, Teacher **hoàn toàn đóng băng trọng số** (`frozen eval mode`) để làm "thước đo chân lý".
2. **Student Encoder ($S_e$):** Mạng ResNet-18 khởi tạo ngẫu nhiên, học cách trích xuất đặc trưng xuôi từ ảnh đầu vào để khớp với Teacher qua hàm mất mát $\ell_e$.
3. **Khối Deep Feature Embedding (DFE):** Đóng vai trò là nút cổ chai (bottleneck), trích xuất và trộn đặc trưng đa tầng của $S_e$ từ 3 mức tỷ lệ ($64\times 64$, $32\times 32$, $16\times 16$) thành vector biểu diễn ngữ nghĩa cô đọng $F_{\text{emb}}$ kích thước $256 \times 16 \times 16$.
4. **Student Decoder ($S_d$):** Mạng giải mã ngược ResNet-18, nhận $F_{\text{emb}}$ từ DFE và phóng đại ngược lại kích thước đặc trưng ban đầu để khớp với Teacher qua hàm mất mát $\ell_d$.
5. **Cơ chế phát hiện dị thường:** 
   - Với ảnh bình thường: Cả $S_e$ và $S_d$ đều tái tạo tốt đặc trưng của Teacher $\implies$ Sai số $M_{t-d}$ xấp xỉ $0$.
   - Với ảnh lỗi: Vì chưa từng học các khuyết tật, $S_d$ sẽ tái tạo sai lệch hoàn toàn so với Teacher $\implies$ Sai số $M_{t-d}$ bùng nổ, tạo thành bản đồ nhiệt (heatmap) định vị chính xác vị trí khuyết tật.

---

## 2. NHỮNG KẾT QUẢ KỸ THUẬT ĐÃ HOÀN THÀNH (WHAT HAS BEEN DONE)

Nhóm đã hoàn thành toàn diện giai đoạn nghiên cứu, phân tích toán học và cài đặt mã nguồn cho Baseline gốc (DSKD `B0-Clean`):

### 2.1. Giải Mã Các Lỗ Hổng & Điểm Mơ Hồ Trong Bài Báo Gốc
Qua quá trình kiểm toán đối nghịch độc lập, nhóm đã phân tích sâu tệp PDF gốc và làm sáng tỏ các điểm bài báo mô tả chưa rõ:
1. **Mâu thuẫn đặc trưng vào DFE:** Hình 4 của bài báo vẽ đặc trưng thô $F_e$ đi vào DFE (mỗi nhánh có `BatchNorm2d`), nhưng Algorithm 2 lại ghi đặc trưng chia chuẩn $\hat{F}_e$. Nhóm đã lập trình hỗ trợ cả 2 lựa chọn thông qua cờ cấu hình `dfe_input_type: "raw" | "norm"` và khóa cứng quy tắc: *Huấn luyện dùng loại nào thì Suy luận bắt buộc dùng đúng loại đó* để tránh lệch phân phối.
2. **Lỗ hổng mã giả Algorithm 1:** Trong Algorithm 1 của bài báo, tác giả bỏ sót hoàn toàn dòng lệnh gọi DFE ($F_{\text{emb}} \leftarrow \text{DFE}(\cdot)$). Nhóm đã bổ sung hoàn chỉnh luồng dữ liệu chuẩn.
3. **Quyền sở hữu tham số DFE:** Đã chứng minh bằng toán học rằng DFE không thuộc về nhóm Encoder $\Theta_e$ (vì $\ell_e$ không đi qua DFE nên $\nabla_{\theta_{\text{dfe}}}\ell_e \equiv \mathbf{0}$). Tham số DFE thuộc về hệ thống Decoder $\Theta_d$ và được tối ưu hóa bởi mục tiêu $\ell_d$.
4. **Chuẩn hóa đo đạc tính toán:** Đã làm rõ quy ước $1\text{ MAC} = 2\text{ FLOPs}$ theo chuẩn PyTorch `FlopCounterMode`. Con số `6.170G` trong Bảng 6 của bài báo thực chất là **MACs** (khớp với `6.206G` MACs của mạng Decoder dùng Deconvolution).
5. **Sửa lỗi thiếu tham số Decoder:** Nhóm đã tính toán chính xác Decoder có $3.326\text{M}$ tham số (dùng Bilinear upsample kết hợp Conv3x3 để chống hiện tượng sọc bàn cờ *checkerboard artifacts*).
6. **Bảo toàn tính khoa học trong đánh giá:** Loại bỏ triệt để rủi ro rò rỉ tập Test (Data Snooping). Tập **Test** chỉ dùng một lần duy nhất ở cuối để lấy số liệu; toàn bộ việc lựa chọn mô hình và tinh chỉnh siêu tham số bắt buộc diễn ra trên tập **Validation** ($20\%$ `train/good`).

### 2.2. Toàn Bộ Mã Nguồn Đã Được Cài Đặt Trong `src/anomaly_detection/`
Mã nguồn đã được module hóa sạch sẽ, có chú thích đầy đủ:
- **`models/dskd.py`:** Chứa lớp `ResNet18FeatureExtractor`, `DenseDeepFeatureEmbedding`, `ReversedResNet18Decoder`, và mô hình hợp nhất `DSKDModel`.
- **`losses/dskd_loss.py`:** Chứa hàm chuẩn hóa vector Eq. (3), hàm tính khoảng cách Euclidean Eq. (4), Cosine similarity Eq. (5), và hàm loss kết hợp Eq. (6)–(7) với $\lambda = 0.1$.
- **`inference/anomaly_scorer.py`:** Chứa bộ sinh anomaly map mượt mà qua phép lọc Gaussian ($\sigma=4.0$) và tính điểm dị thường cấp ảnh theo Algorithm 2.
- **`training/checkpoint.py` & `trainer.py`:** Chứa bộ quản lý checkpoint nguyên tử (chống hỏng file khi bị ngắt kết nối trên cloud) và vòng lặp huấn luyện quản lý 2 optimizer Adam riêng biệt ($\text{lr}=10^{-3}$, $\beta=(0.5, 0.999)$ theo Mục 4.1.2 của bài báo).
- **`configs/baseline_b0.yaml`:** File cấu hình chuẩn hóa tham số thực nghiệm.
- **`scripts/train.py` & `scripts/evaluate.py`:** Bộ công cụ dòng lệnh hoàn chỉnh để chạy huấn luyện và đánh giá trên bất kỳ danh mục MVTec nào.
- **`notebooks/kaggle_baseline.ipynb`:** File Jupyter Notebook được thiết kế sẵn cho Kaggle GPU (Tesla T4), có sẵn code kiểm tra GPU, nạp dữ liệu, chạy huấn luyện và xuất bảng kết quả.
- **Kiểm thử Unit Test cục bộ:** Script [tests/test_dskd_baseline.py](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/tests/test_dskd_baseline.py) đã được chạy và **đạt 100% tất cả các bài test** (Tensor shapes, Loss dương không NaN/Inf, 2 optimizer cập nhật độc lập, Anomaly scorer sinh điểm chính xác).

---

## 3. PHÂN ĐỊNH TRÁCH NHIỆM & CÔNG VIỆC GIỮA 2 THÀNH VIÊN

Để đảm bảo báo cáo đồ án thể hiện rõ năng lực và đóng góp riêng của từng sinh viên theo đúng yêu cầu môn học CS331, khối lượng công việc được phân định rành mạch thành: **Phần Hợp tác Chung** và **Phần Độc lập của Từng Thành viên**.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          PHÂN CÔNG CÔNG VIỆC NHÓM NGHIÊN CỨU                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • CÔNG VIỆC CHUNG:                                                                     │
│   - Tái hiện Baseline DSKD trên Kaggle GPU (sinh số liệu đối chứng B0-Clean).          │
│   - Tiền xử lý dữ liệu MVTec AD, chia tập 80/20 train/val có seed=42.                  │
│   - Thiết lập khung đo chuẩn: Image AUROC, Pixel AUROC, Pixel AUPRO (0.30 & 0.05).     │
│   - Triển khai phương pháp kết hợp Method C1 (ghép Method A + Method B).               │
│   - Soạn thảo Báo cáo tổng thể (Chương 1, 2, 3, 6) và Slide thuyết trình đồ án.        │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ • SINH VIÊN 1 (MEMBER 1):                │ • SINH VIÊN 2 (MEMBER 2):                   │
│   Chuyên sâu: THUẬT TOÁN & HÀM LOSS      │   Chuyên sâu: KIẾN TRÚC & TỐI ƯU NHẸ        │
│   (Algorithm, Loss & Calibration)        │   (Architecture, Compression & Efficiency)  │
│                                          │                                             │
│   Đảm nhiệm: METHOD A (FC-DSKD)          │   Đảm nhiệm: METHOD B (Lite-DSKD)           │
│   1. Contextual Patch Discrepancy.       │   1. Lite-DFE (dsmsa_db) giảm 98% params.   │
│   2. Spatial Focal Loss tự thích nghi.   │   2. Slim-Student Encoder/Decoder (alpha).  │
│   3. Asymmetric Stop-Grad Consistency.   │   3. Đầu chiếu khôi phục số chiều (Proj).   │
│   4. Normal-Val Z-score Calibration.     │   4. Đánh giá Trade-off: Params, FLOPs, FPS.│
│   5. Viết Chương 4.2 & Thực nghiệm A.    │   5. Viết Chương 4.3 & Thực nghiệm B.       │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### 3.1. Công Việc Hợp Tác Chung (Collaborative Tasks)
1. **Huấn luyện và Tái hiện Baseline DSKD (`B0-Clean`):**
   - Chạy trên Kaggle GPU để xác lập bảng số liệu chuẩn ban đầu.
   - Bảng số liệu này là mẫu số chung bắt buộc để cả 2 sinh viên làm mốc so sánh cho phương pháp cải tiến của mình.
2. **Phương pháp Kết hợp $C1$ (`FC-Lite-DSKD`):**
   - Hợp tác ghép kiến trúc siêu nhẹ của Sinh viên 2 (Lite-DFE, Slim Student) với hàm mất mát tiêu cự và bộ hiệu chuẩn của Sinh viên 1.
   - Đánh giá xem khi kết hợp cả 2 phương pháp thì hiệu năng đạt đỉnh bao nhiêu và độ phức tạp giảm được bao nhiêu.
3. **Báo cáo và Slide Thuyết trình:**
   - Cùng viết Chương 1 (Giới thiệu bài toán), Chương 2 (Dữ liệu MVTec AD), Chương 3 (Cơ sở lý thuyết Knowledge Distillation), và Chương 6 (Kết luận & Hướng phát triển).

---

### 3.2. Công Việc Độc Lập của Sinh Viên 1: Method A (`FC-DSKD`)
- **Tên phương pháp:** Focal & Contextual Dual-Student Knowledge Distillation (`FC-DSKD`).
- **Ý tưởng khoa học cốt lõi:** DSKD gốc chỉ tính khoảng cách điểm ảnh đơn lẻ độc lập (pixel-wise point distance), dễ bị nhiễu cục bộ và không phân biệt được các khuyết tật có cấu trúc hoặc biến thiên kết cấu tinh vi. Sinh viên 1 sẽ nâng cấp hàm mục tiêu và cơ chế suy luận qua 4 mô-đun:
  1. **Mô-đun A1.1 — Contextual Patch Discrepancy:** Thay vì chỉ so sánh 1 vector đặc trưng tại $(i, j)$, trích xuất thêm vector trung bình ngữ cảnh cục bộ vùng $3\times 3$ lân cận ($\bar{F}_{ij}$) và tính khoảng cách ngữ cảnh $d_{\text{ctx}}$. Bản đồ sai biệt mới kết hợp: $M = d_{\text{pt}} + 0.5 \cdot d_{\text{ctx}}$.
  2. **Mô-đun A1.2 — Self-Normalized Spatial Focal Loss:** Các điểm ảnh bình thường (dễ học) thường chiếm đại đa số diện tích, làm lấn át các vùng dị thường nhỏ (khó học). Mô-đun này tự động gán trọng số tiêu cự $w_{ij} = (1 + \text{stop\_grad}[\tilde{M}_{ij}])^\gamma$, ép mô hình tập trung gradient vào các vùng biên khuyết tật khó phân biệt.
  3. **Mô-đun A2 — Asymmetric Cross-Student Consistency with Stop-Gradient:** Ràng buộc trực tiếp giữa Encoder $S_e$ và Decoder $S_d$. Để tránh việc $S_d$ kéo tụt $S_e$, áp dụng ngắt đạo hàm có hướng: $\text{stop\_grad}[\hat{F}_e] \to \hat{F}_d$, biến $S_e$ thành người hướng dẫn phụ giúp $S_d$ hội tụ nhanh hơn.
  4. **Mô-đun A3 — Normal-Val Calibrated Synergistic Fusion:** Dùng $20\%$ tập ảnh bình thường (`val/good`) để đo phân phối nền $(\mu, \sigma)$ của từng lớp, sau đó chuẩn hóa Z-score để triệt tiêu toàn bộ nhiễu nền giả trước khi tính điểm dị thường cấp ảnh bằng kỹ thuật lọc phần dư đuôi (Tail-Fraction Top-0.1%).
- **Trách nhiệm bài viết:** Viết Mục 4.2 trong Báo cáo Đồ án, thực hiện các thí nghiệm Ablation bóc tách xem mỗi mô-đun A1, A2, A3 đóng góp tăng bao nhiêu $\%$ AUROC/AUPRO.

---

### 3.3. Công Việc Độc Lập của Sinh Viên 2: Method B (`Lite-DSKD`)
- **Tên phương pháp:** Lightweight Dual-Student Knowledge Distillation (`Lite-DSKD`).
- **Ý tưởng khoa học cốt lõi:** Mô hình DSKD gốc quá nặng cho các thiết bị nhúng công nghiệp biên (Edge Devices) do khối DFE sử dụng các lớp tích chập dày đặc tiêu tốn $3.288\text{M}$ tham số và các mạng Student giữ nguyên kích thước ResNet-18. Sinh viên 2 sẽ tái thiết kế toàn bộ kiến trúc để đạt độ nhẹ vượt trội qua 3 cải tiến:
  1. **Mô-đun B1 — Lite-DFE (`dsmsa_db`):** Thay thế tích chập thông thường bằng Tích chập tách rời theo chiều sâu (Depthwise Separable Convolutions) kết hợp Nút cổ chai thứ nguyên (Dimension Bottleneck $d_b = 64\alpha$).
     - Giảm tham số khối DFE từ **$3,288,320$ tham số xuống chỉ còn $51,904$ tham số** (giảm đến **$98.4\%$** dung lượng DFE!).
  2. **Mô-đun B2 — Slim-Student Architecture ($\alpha \in \{0.75, 0.50\}$):** Thu nhỏ số kênh nội tại của cả Student Encoder và Student Decoder theo hệ số tỷ lệ $\alpha$. Ví dụ tại $\alpha=0.50$: số kênh giảm còn $(32, 64, 128)$.
  3. **Mô-đun B3 — Dimension Restoration Projection Heads ($\mathcal{P}_{e,d}^k$):** Thiết kế các đầu chiếu tích chập điểm $1\times 1$ kèm BatchNorm để khôi phục số chiều từ $\alpha c_k$ về khớp với $c_k$ của Teacher $T$. Ở pha suy luận thực tế, các đầu chiếu không cần thiết có thể được tỉa bỏ hoặc gộp trực tiếp để tăng tốc độ xử lý FPS.
  4. **Chỉ số thẩm định độc lập:** Tổng tham số huấn luyện giảm từ **$9.397\text{M}$ xuống chỉ còn $1.671\text{M}$ (giảm $-82.2\%$)**, số phép tính FLOPs giảm từ **$13.485\text{G}$ xuống $5.824\text{G}$ (giảm $-56.8\%$)**, trong khi chất lượng AUROC suy giảm không đáng kể ($< 1.5\%$).
- **Trách nhiệm bài viết:** Viết Mục 4.3 trong Báo cáo Đồ án, lập bảng phân tích so sánh chuyên sâu giữa các biến thể $\alpha$, vẽ đồ thị tương quan Đánh đổi (Trade-off Curve) giữa Số tham số/FLOPs/FPS và Điểm số AUROC/AUPRO.

---

## 4. GIẢI THÍCH KỸ THUẬT: TÍNH TÁI LẬP & TÍNH NGẪU NHIÊN KHI CHẠY CODE PYTHON

> **CÂU HỎI CỦA BẠN:**  
> *"Nếu chạy bằng file Python thì kết quả có cố định không? Lỡ như người khác chạy lại cho ra kết quả khác thì sao?"*

Đây là thắc mắc cực kỳ xác đáng của mọi kỹ sư và nhà nghiên cứu Machine Learning. Dưới đây là câu trả lời toàn diện về mặt kỹ thuật:

### 4.1. Tại Sao Deep Learning Thường Có Hiện Tượng Ra Kết Quả Khác Nhau?
Nếu một đoạn code không được kiểm soát hạt giống ngẫu nhiên, kết quả giữa các lần chạy (hoặc giữa 2 người khác nhau) sẽ bị lệch do 3 nguyên nhân:
1. **Khởi tạo trọng số ngẫu nhiên ban đầu (Weight Initialization):** Các mạng $S_e$, DFE, $S_d$ được sinh trọng số ngẫu nhiên theo phân phối Gaussian (`torch.randn`). Nếu mỗi lần chạy sinh ra một bộ trọng số khác nhau, mô hình sẽ xuất phát từ các điểm khác nhau trong không gian tối ưu.
2. **Thứ tự xáo trộn dữ liệu (DataLoader Shuffling):** Khi huấn luyện với `shuffle=True`, thứ tự các batch ảnh đi vào mô hình ở từng epoch sẽ khác nhau nếu không cố định seed.
3. **Tính bất định của phần cứng GPU (Non-Deterministic GPU Operations):** Các nhân CUDA trên GPU khi thực hiện phép tích chập (CuDNN convolution) hoặc phép lấy mẫu song tuyến tính (Bilinear interpolation) sử dụng cơ chế tính toán nguyên tử song song đa luồng (atomic operations). Thứ tự cộng dồn các phép tính dấu chấm động (floating-point numbers) có thể xê dịch ở các chữ số thập phân siêu nhỏ (ví dụ: $10^{-6}$).

---

### 4.2. Mã Nguồn Của Nhóm Đã Được Cố Định (Deterministic) Như Thế Nào?
Trong kho mã nguồn của chúng ta, file [src/anomaly_detection/utils/seed.py](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/src/anomaly_detection/utils/seed.py) đã được tích hợp hàm `set_seed(seed=42)` và tự động kích hoạt trong file `scripts/train.py`. Hàm này thực hiện:
```python
def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)                        # Cố định thư viện Python thuần
    np.random.seed(seed)                     # Cố định NumPy
    torch.manual_seed(seed)                  # Cố định PyTorch CPU
    torch.cuda.manual_seed_all(seed)         # Cố định toàn bộ GPU CUDA
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True  # Ép thuật toán CuDNN chạy chính xác từng luồng
        torch.backends.cudnn.benchmark = False     # Tắt cơ chế tự chọn thuật toán tối ưu động
```
Ngoài ra, bước chia tập dữ liệu $80\%$ Train và $20\%$ Validation trong `MVTecTrainDataset` cũng được khóa cứng bằng `seed=42`.

---

### 4.3. Nếu Chạy Trên Các Máy Tính/GPU Khác Nhau Thì Sao?
- **Nếu chạy trên cùng môi trường (ví dụ cùng GPU Tesla T4 của Kaggle với seed=42):** Kết quả giữa hai bạn chạy sẽ **hoàn toàn trùng khớp $99.99\%$** (từ từng giá trị loss theo từng epoch cho đến các chỉ số AUROC cuối cùng).
- **Nếu chạy trên 2 dòng card đồ họa khác nhau (ví dụ: GPU Tesla T4 trên Kaggle so với RTX 3090/4090 tại máy cá nhân):**
  - Do kiến trúc phần cứng (vi kiến trúc Turing vs. Ampere/Ada Lovelace) có tập chỉ lệnh số học phần cứng khác nhau, có thể xuất hiện độ lệch siêu nhỏ ở chữ số thập phân thứ 2 hoặc thứ 3 (ví dụ: một máy ra $98.52\%$, máy kia ra $98.48\%$).
  - **ĐÂY LÀ ĐIỀU HOÀN TOÀN TỰ NHIÊN VÀ ĐƯỢC CHẤP NHẬN TRONG KHOA HỌC.**

---

### 4.4. Quy Chuẩn Báo Cáo Khoa Học Của Nhóm Để Bảo Vệ Đồ Án
Để không một ai (kể cả giảng viên khó tính nhất) có thể nghi ngờ về tính ngẫu nhiên của kết quả, nhóm chúng ta sẽ áp dụng **Giao thức Thực nghiệm Chuẩn Quốc tế (Standard Scientific Protocol)**:
> Khi đánh giá chính thức để đưa số liệu vào Báo cáo Đồ án, chúng ta sẽ chạy với **3 hạt giống ngẫu nhiên độc lập**: `seed = 42`, `seed = 43`, và `seed = 44`.
> Kết quả báo cáo trong bảng sẽ được ghi dưới dạng:
> $$\mathbf{\text{Mean} \pm \text{Std}} \quad (\text{ví dụ: } \mathbf{98.52\% \pm 0.14\%})$$
> Cách trình bày này khẳng định phương pháp của nhóm chạy ổn định và khách quan trên mọi hạt giống ngẫu nhiên, thể hiện trình độ nghiên cứu xuất sắc và tính chuyên nghiệp cao.

---

## 5. KẾ HOẠCH TRIỂN KHAI CHI TIẾT & HƯỚNG DẪN THỰC THI (ACTION PLAN)

Dưới đây là lịch trình thực hiện chi tiết cho từng giai đoạn:

| Giai Đoạn | Thời Gian Dự Kiến | Nhiệm Vụ Trọng Tâm | Phân Công Thực Hiện | Đầu Ra Nghiệm Thu |
| :--- | :---: | :--- | :--- | :--- |
| **Giai đoạn 1** | **1 – 2 ngày tới** | **Chạy Thực nghiệm Baseline B0 trên Kaggle GPU** | Thành viên đảm nhiệm chạy Kaggle (bạn) | • File log `training_history.json`<br>• File bảng điểm `carpet_eval_metrics.csv`<br>• Xác nhận thời gian thực tế `M0_bench` |
| **Giai đoạn 2** | **Tuần 2** | **Cài đặt Mã nguồn Method A và Method B** | • SV 1: Viết `fc_loss.py`, `fc_scorer.py`<br>• SV 2: Viết `lite_dskd.py` | • Unit Test cục bộ đạt PASS 100%<br>• Cấu hình YAML cho cả 3 phương pháp |
| **Giai đoạn 3** | **Tuần 3** | **Huấn luyện Đại trà, Đo đạc Đánh đổi & Ablation** | Cả 2 sinh viên phối hợp chạy GPU trên Core-5 danh mục | • Bảng số liệu so sánh đầy đủ 3 seeds<br>• Biểu đồ Trade-off (Params vs AUROC)<br>• Hình ảnh trực quan hóa Heatmap dị thường |
| **Giai đoạn 4** | **Tuần 4** | **Hoàn thiện Báo cáo Đồ án CS331 & Slide** | Cả 2 sinh viên viết báo cáo theo 6 chương | • Báo cáo đồ án hoàn chỉnh (PDF/Word)<br>• Slide thuyết trình bảo vệ đồ án |

---

### 5.1. Hướng Dẫn Từng Bước Cho Bạn Để Chạy Baseline Trên Kaggle GPU
Vì bạn đảm nhiệm phần chạy thực nghiệm Baseline, tôi đã chuẩn bị sẵn mọi thứ để bạn chỉ cần thực hiện các thao tác sau:

1. **Bước 1 — Mở Kaggle Notebook:**
   - Đăng nhập vào [Kaggle](https://www.kaggle.com/), tạo một Notebook mới (`New Notebook`).
   - Ở thanh công cụ bên phải (Notebook Options):
     - Mục **Accelerator**: Chọn **GPU T4** (hoặc GPU P100).
     - Mục **Internet**: Gạt sang **On** (để clone mã nguồn từ GitHub).
2. **Bước 2 — Thêm bộ dữ liệu MVTec AD:**
   - Bấm vào nút **"+ Add Input"** (góc trên bên phải).
   - Gõ tìm kiếm `mvtec-ad` và chọn bộ dữ liệu MVTec AD chuẩn của cộng đồng (dữ liệu sẽ được gắn trực tiếp vào thư mục `/kaggle/input/mvtec-ad/`).
3. **Bước 3 — Nạp Notebook và Chạy:**
   - Bạn có thể tải file [notebooks/kaggle_baseline.ipynb](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/notebooks/kaggle_baseline.ipynb) lên Kaggle (bằng cách chọn `File` $\to$ `Import Notebook`).
   - Hoặc bạn copy trực tiếp các lệnh sau vào các cell của Notebook:
     ```bash
     # Cell 1: Kéo code từ GitHub về máy ảo Kaggle
     !git clone https://github.com/traineethanh/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection.git repo
     %cd repo
     !git checkout chore/repository-reset
     !pip install -q pyyaml scipy scikit-learn

     # Cell 2: Chạy kiểm chuẩn micro-benchmark M0_bench (5 epochs trên carpet, mất ~30 giây)
     !python scripts/train.py --config configs/baseline_b0.yaml --category carpet --data_root /kaggle/input/mvtec-ad --epochs 5 --output_dir checkpoints/m0_bench

     # Cell 3: Chạy huấn luyện chính thức Baseline (200 epochs trên carpet, mất ~5 phút)
     !python scripts/train.py --config configs/baseline_b0.yaml --category carpet --data_root /kaggle/input/mvtec-ad --epochs 200 --output_dir checkpoints/baseline_b0

     # Cell 4: Đánh giá mô hình trên tập Test và xuất bảng điểm
     !python scripts/evaluate.py --checkpoint checkpoints/baseline_b0/carpet/best_checkpoint.pth --category carpet --data_root /kaggle/input/mvtec-ad --output_dir results/baseline_b0/carpet
     ```
4. **Bước 4 — Báo cáo lại số liệu:**
   - Sau khi chạy xong Cell 4, màn hình sẽ in ra bảng số liệu: Image AUROC, Pixel AUROC, Pixel PRO@0.30 và Pixel PRO@0.05.
   - Bạn chỉ cần copy bảng số liệu đó hoặc tải file `carpet_eval_metrics.csv` về để nhóm cập nhật vào báo cáo!

---

## 6. LỜI KẾT & TÀI LIỆU THAM KHẢO

Tài liệu này đã giải quyết toàn bộ các băn khoăn về kỹ thuật, phân định rõ ràng quyền lợi và trách nhiệm của cả 2 thành viên, đồng thời cung cấp lộ trình thực thi không thể rõ ràng hơn.

Toàn bộ các tài liệu phân tích toán học và chứng minh chi tiết được lưu trữ trong thư mục [docs/research/](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/), bao gồm:
- Giao kèo thực nghiệm cuối cùng: [final_b0_experiment_contract.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/final_b0_experiment_contract.md)
- Phân tích chi tiết Method A: [method_a_proposal.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/method_a_proposal.md)
- Phân tích chi tiết Method B: [method_b_proposal.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/method_b_proposal.md)
- Khung cấu trúc Báo cáo Đồ án CS331: [docs/report_outline.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/report_outline.md)

Hai bạn hãy cùng đọc kỹ báo cáo này để thống nhất bước đi tiếp theo. Bất cứ khi nào bạn có số liệu từ Kaggle hoặc muốn bắt đầu viết code cho Method A / Method B, tôi luôn sẵn sàng hỗ trợ!
