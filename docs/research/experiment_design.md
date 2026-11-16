# Thiết kế Thực nghiệm & Giao thức Đánh giá Chuẩn (Experiment Design & Evaluation Protocol)

> **Dự án:** CS331 — Chưng cất Tri thức cho Phát hiện Bất thường Công nghiệp Nhẹ (Knowledge Distillation for Lightweight Industrial Anomaly Detection)  
> **Bài báo Gốc (Baseline):** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (Yao & Gao, arXiv:2402.00448v1, 2024)  
> **Phụ trách:** Agent 5 — Experiment & Reproducibility Engineer  
> **Trạng thái Thực nghiệm Hiện tại:** `PENDING` (Thiết kế giao thức tiền thực nghiệm; chưa thực thi huấn luyện mô hình mới)

---

## Hệ thống Nhãn Nhận thức (Epistemic Tagging Legend)

Để đảm bảo tính minh bạch khoa học và khả năng kiểm chứng chéo trong toàn bộ tài liệu nghiên cứu, mọi khẳng định kỹ thuật dưới đây đều được gắn nhãn nguồn gốc rõ ràng:
- `[FACT - Paper]`: Thông tin, công thức toán học, siêu tham số hoặc số liệu trích xuất trực tiếp từ bài báo gốc DSKD (`papers/2402.00448v1.pdf`).
- `[FACT - Repo]`: Cấu trúc mã nguồn, hàm tiện ích, giao thức kiểm tra hoặc thông số đã được cài đặt thực tế trong kho mã nguồn hiện tại (`configs/`, `src/`, `scripts/`, `docs/`).
- `[INTERPRETATION - Agent 5]`: Phân tích kỹ thuật, suy luận giải quyết điểm mơ hồ của bài báo, hoặc quy tắc thiết kế thực nghiệm do Agent 5 xây dựng nhằm đảm bảo tính tái lập và chống rò rỉ dữ liệu.
- `[HYPOTHESIS]`: Giả thuyết khoa học về hành vi của mô hình hoặc hiệu quả kỳ vọng của các cải tiến (Method A, Method B, Combined A+B), cần được kiểm chứng qua thực nghiệm.

---

## 1. Phân tích Yêu cầu Tập dữ liệu MVTec AD & Giao thức Chống Rò rỉ (Zero Test Leakage)

### 1.1. Đặc tả Tập dữ liệu MVTec AD trong Bài toán One-Class Unsupervised AD
- `[FACT - Paper]` Bài báo DSKD (Mục 3, tr. 5 & Mục 4.1.1, tr. 7) định nghĩa bài toán phát hiện bất thường ở chế độ không giám sát một lớp (**Unsupervised / One-Class Anomaly Detection**):
  - Tập dữ liệu **MVTec AD** (Bergmann et al., IJCV 2021) gồm hơn 5,000 ảnh công nghiệp độ phân giải cao thuộc **15 categories** chia làm 2 nhóm chính:
    - **5 Texture Categories (Bề mặt Kết cấu):** `carpet`, `grid`, `leather`, `tile`, `wood`.
    - **10 Object Categories (Vật thể Công nghiệp):** `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`.
  - Tập huấn luyện $\mathcal{I}^t = \{I_1^t, I_2^t, \dots, I_m^t\}$ **chỉ chứa các mẫu bình thường không khuyết tật (`anomaly-free` / `train/good`)** (trung bình khoảng 240 ảnh/category, tổng cộng 3,629 ảnh trên toàn bộ 15 categories).
  - Tập kiểm thử $\mathcal{I}^a = \{I_1^a, I_2^a, \dots, I_m^a\}$ chứa cả ảnh bình thường (`test/good`) và ảnh chứa các loại khuyết tật đa dạng (`test/<defect_type>`), đi kèm mặt nạ phân vùng nhị phân mức pixel (`ground_truth/<defect_type>/<stem>_mask.png`) (trung bình khoảng 100–115 ảnh/category, tổng cộng 1,725 ảnh kiểm thử).

### 1.2. Cơ chế Thực thi Giao thức Chống Rò rỉ Dữ liệu trong Kho Mã nguồn
- `[FACT - Repo]` Cấu hình [`configs/mvtec_ad.yaml`](../../configs/mvtec_ad.yaml) và module [`src/anomaly_detection/data/mvtec.py`](../../src/anomaly_detection/data/mvtec.py) đã cài đặt ba chốt chặn kỹ thuật ở mức mã nguồn để ngăn chặn rò rỉ dữ liệu:
  1. **Kiểm tra Tính Thuần khiết Một Lớp (One-Class Purity Guard):** Khi khởi tạo `MVTecTrainDataset`, hệ thống quét toàn bộ thư mục `<root_dir>/<category>/train/*/*.png`. Nếu phát hiện bất kỳ tệp ảnh nào không nằm trong thư mục con `good/`, hàm khởi tạo lập tức ném ngoại lệ `ValueError("Unsupervised protocol violation: found non-'good' training images...")` ([`mvtec.py`](../../src/anomaly_detection/data/mvtec.py#L112-L118)).
  2. **Chia Tách Tất định 80/20 trên Tập Bình thường (`train/good` Deterministic Split):** Sử dụng bộ sinh số ngẫu nhiên độc lập `random.Random(seed=42)` với tỷ lệ `val_split_ratio: 0.2` ([`mvtec.py`](../../src/anomaly_detection/data/mvtec.py#L140-L150)), tập `train/good` được tách thành hai tập con rời nhau hoàn toàn:
     - `split="train"` ($\approx 80\%$ ảnh `train/good`): Dùng duy nhất để tính gradient và cập nhật tham số mạng học sinh $\theta_e$ (Student Encoder $S_e$), $\theta_{\text{dfe}}$ (Deep Feature Embedding), và $\theta_d$ (Student Decoder $S_d$).
     - `split="val"` ($\approx 20\%$ ảnh `train/good`, gọi tắt là tập `val/good`): Dùng để theo dõi sự hội tụ của hàm mất mát chưng cất trên dữ liệu bình thường chưa từng thấy trong quá trình cập nhật trọng số, chọn checkpoint không rò rỉ, và hiệu chuẩn thống kê bản đồ bất thường.
  3. **Kiểm chứng Rời nhau Tuyệt đối (`validate_mvtec_category`):** Hàm `validate_mvtec_category()` ([`mvtec.py`](../../src/anomaly_detection/data/mvtec.py#L256-L313)) được gọi bởi [`scripts/validate_dataset.py`](../../scripts/validate_dataset.py) đối chiếu đường dẫn tuyệt đối (`Path.resolve()`) để xác nhận hai bất biến: `zero_train_test_overlap == True` ($\text{Train} \cap \text{Test} = \emptyset$) và `zero_train_val_overlap == True` ($\text{Train} \cap \text{Val} = \emptyset$).

### 1.3. Quy tắc Thiết kế Thực nghiệm Không Rò rỉ Tập Kiểm thử (Zero Test-Set Leakage Policy)
- `[INTERPRETATION - Agent 5]` Trong nhiều nghiên cứu phát hiện bất thường công nghiệp, một lỗi phương pháp luận phổ biến là đánh giá mô hình trên tập `test/` sau mỗi epoch và lưu lại checkpoint có `Test AUROC` cao nhất ("test-set oracle checkpoint selection"), hoặc dùng chính giá trị `min/max` của điểm bất thường trên tập `test/` để chuẩn hóa điểm về đoạn $[0, 1]$ trước khi áp ngưỡng $0.5$ như Công thức (1) của bài báo DSKD (tr. 2). Cả hai hành vi này đều làm rò rỉ thông tin phân phối khuyết tật của tập kiểm thử vào quá trình lựa chọn mô hình.
- `[INTERPRETATION - Agent 5]` Để đảm bảo tính nghiêm ngặt học thuật trong đồ án CS331, chúng ta áp dụng **4 Quy tắc Bất khả Xâm phạm về Tập `test/`**:
  1. **Cấm Chọn Checkpoint theo Tập Kiểm thử:** Tập `test/` tuyệt đối không được truyền vào vòng lặp huấn luyện (`Trainer.fit()`). Mô hình dùng để báo cáo kết quả cuối cùng chỉ được chọn theo một trong hai tiêu chuẩn định trước:
     - **Tiêu chuẩn 1 (Paper-Standard Last Epoch):** Checkpoint tại đúng epoch cuối cùng (`epoch = 200`) theo đặc tả của Mục 4.1.2 (tr. 7) trong bài báo DSKD.
     - **Tiêu chuẩn 2 (Normal-Validation Best Loss):** Checkpoint đạt tổng hàm mất mát chưng cất trung bình thấp nhất trên tập `val/good`:
       $$\mathcal{L}_{\text{val}}(\theta_e, \theta_{\text{dfe}}, \theta_d) = \frac{1}{|\mathcal{I}^{\text{val}}|} \sum_{I_m \in \mathcal{I}^{\text{val}}} \left( \ell_e(I_m) + \ell_d(I_m) \right)$$
  2. **Cấm Tinh chỉnh Siêu tham số trên Toàn bộ Tập Kiểm thử:** Việc khảo sát siêu tham số (như hệ số $\lambda$, kiến trúc con của Method A/B) chỉ được dựa trên phân tích lý thuyết, hành vi hội tụ trên `val/good`, và kiểm chứng có kiểm soát trên tập con đại diện ở Stage 1 trước khi khóa cấu hình (freeze config) để chạy Stage 2.
  3. **Hiệu chuẩn Thống kê Điểm Bất thường Độc lập Tập Kiểm thử:** Khi cần chuẩn hóa bản đồ bất thường giữa các tầng đa tỷ lệ $k \in \{1, 2, 3\}$ (trong Method B) hoặc xác định ngưỡng nhị phân $A(\text{score}) \ge \text{threshold}$ (Công thức 1, tr. 2), mọi tham số thống kê ($\mu_k^{\text{val}}, \sigma_k^{\text{val}}$ hoặc bách phân vị $q_{1-\alpha}^{\text{val}}$) phải được ước lượng **duy nhất trên tập `val/good`**.
  4. **Đánh giá Không Phụ thuộc Ngưỡng cho Các Chỉ số Chuẩn:** Các chỉ số xếp hạng chuẩn (`Image-AUROC`, `Image-AP`, `Pixel-AUROC`, `Pixel-AP`, `AUPRO@0.30`, `AUPRO@0.05`) được tính trực tiếp từ bản đồ điểm bất thường liên tục $\widetilde{M}(I_m)$ và điểm mức ảnh $\text{score} = \max(\widetilde{M}(I_m))$ theo đúng cài đặt trong [`src/anomaly_detection/evaluation/metrics.py`](../../src/anomaly_detection/evaluation/metrics.py), không qua bất kỳ phép cắt ngưỡng hậu xử lý nào sử dụng nhãn tập `test/`.

---

## 2. Chiến lược Lựa chọn Category theo Giai đoạn (Staged Category Protocol)

### 2.1. Động cơ Khoa học & Ràng buộc Tài nguyên
- `[INTERPRETATION - Agent 5]` Mỗi lượt chạy đầy đủ toàn bộ 15 categories của MVTec AD đòi hỏi huấn luyện **15 mô hình độc lập** (1 mô hình/category theo chuẩn One-Class AD). Nếu thực hiện toàn bộ các phép kiểm tra kiến trúc (4 biến thể Bảng 7) và nghiên cứu cắt bỏ (ablation study cho `A1..A3`, `B1..B3`) trên cả 15 categories, tổng số mô hình cần huấn luyện sẽ lên tới $>180$ mô hình, gây lãng phí ngân sách GPU Kaggle (giới hạn 30h/tuần/tài khoản) và làm tăng nguy cơ tinh chỉnh quá mức trên toàn bộ benchmark.
- Do đó, quy trình thực nghiệm được chia thành **2 giai đoạn chặt chẽ (2-Stage Execution Protocol)**:
  - **Stage 1 (Core Verification & Ablation Subset — 5 Categories đại diện):** Dùng để kiểm chứng tính đúng đắn của Baseline, tái lập xu hướng Ablation của bài báo gốc, và chọn cấu hình tốt nhất cho Method A và Method B.
  - **Stage 2 (Full 15-Category Benchmark):** Chạy đánh giá toàn diện trên cả 15 categories đối với các cấu hình chính thức đã khóa từ Stage 1 (`B0`, `A_best`, `B_best`, `C1`).

### 2.2. Thiết kế Stage 1: Tập Con 5 Categories Đại diện (2 Textures + 3 Objects)
- `[FACT - Paper]` & `[INTERPRETATION - Agent 5]` Tập con 5 categories trong Stage 1 không được chọn ngẫu nhiên mà được thiết kế để bao phủ cả **2 nhóm cấu trúc (Texture & Object)**, **các quy mô diện tích khuyết tật khác nhau**, và đặc biệt là **nhắm thẳng vào các category điểm yếu đã được chính tác giả bài báo DSKD thừa nhận tại Mục 4.2.1 (tr. 8)**:
  > *"However, the limitation is that our method identifies surrounding anomaly-free pixels as anomalies, especially for the toothbrush, tile and wood."* (`[FACT - Paper]`, Mục 4.2.1, tr. 8).

| STT | Category | Nhóm | Số ảnh `train/good` (Tổng / 80% Train / 20% Val) | Số ảnh `test` (Good / Anomalous) | Kết quả DSKD Gốc (`[FACT - Paper]`, Bảng 1 & 2: Img-AUROC / Pix-AUROC / PRO) | Lý do Khoa học Lựa chọn vào Tập Core-5 (`[INTERPRETATION - Agent 5]`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | `carpet` | Texture | 280 (224 / 56) | 117 (28 / 89) | `100.0%` / `99.2%` / `97.7%` | Kết cấu sợi dệt đồng nhất; đại diện cho nhóm Texture chuẩn nơi DSKD hoạt động lý tưởng; có mặt sẵn trong [`scripts/validate_dataset.py`](../../scripts/validate_dataset.py). |
| **2** | `tile` *(hoặc `wood`)* | Texture | 230 (184 / 46) | 117 (33 / 84) | `98.4%` / `94.5%` / `88.2%` | **Điểm yếu cốt lõi của DSKD trên Texture:** Bài báo (tr. 8) chỉ rõ `tile` và `wood` bị hiện tượng nhận diện nhầm các pixel bình thường quanh vết lỗi thành bất thường (*surrounding false positives*), khiến PRO của `tile` chỉ đạt `88.2%` (thấp hơn cả baseline STPM `92.1%`). Đây là thước đo trực tiếp để kiểm tra xem Method A và Method B có làm sắc nét biên định vị hay không. |
| **3** | `bottle` | Object | 209 (167 / 42) | 83 (20 / 63) | `100.0%` / `98.7%` / `96.4%` | Vật thể cứng, căn chỉnh tâm cố định, khuyết tật diện tích vừa và lớn (`broken_large`, `contamination`); dùng làm mốc kiểm tra hội tụ nhanh (sanity check) cho nhóm Object. |
| **4** | `screw` | Object | 320 (256 / 64) | 160 (41 / 119) | `97.1%` / `99.4%` / `94.2%` | Vật thể kim loại xoay 360 độ không căn chỉnh góc, vùng khuyết tật cực nhỏ ở bước ren/đầu vít; thử thách khả năng bảo toàn chi tiết không gian độ phân giải cao ở tầng $M_{t-d}^1$ và chỉ số nghiêm ngặt `AUPRO@0.05`. |
| **5** | `toothbrush` *(hoặc `cable`)* | Object | 60 (48 / 12) | 42 (12 / 30) | `98.9%` / `99.1%` / `94.5%` | **Điểm yếu cốt lõi của DSKD trên Object & Cỡ mẫu Nhỏ:** (1) Được bài báo nêu đích danh cùng `tile` và `wood` (tr. 8) về lỗi dương tính giả lan ra vùng lân cận sợi lông bàn chải; (2) Chỉ có 60 ảnh `train/good` (48 ảnh sau khi tách 80%), giúp kiểm tra độ ổn định tối ưu hóa khi dữ liệu huấn luyện ít. *(Lưu ý: `cable` với 224 ảnh train và lỗi ngữ nghĩa cấu trúc dây dẫn có thể dùng làm category kiểm tra mở rộng nếu cần).* |

### 2.3. Quy tắc Chống Khái quát hóa Quá mức từ Tập Con (Anti-Overclaiming Rule)
- `[INTERPRETATION - Agent 5]` Để giữ vững tính trung thực học thuật, nhóm thiết lập quy tắc bắt buộc khi trình bày kết quả Stage 1:
  1. **Không Đánh đồng Trung bình 5 Categories với Trung bình MVTec AD:** Trung bình cộng trên 5 categories của Stage 1 phải luôn được đặt tên tường minh là **`Core-5 Subset Mean`** trong mọi bảng biểu và biểu đồ. Tuyệt đối không được đem con số trung bình 5 categories này so sánh trực tiếp với con số trung bình 15 categories của bài báo (`98.5%` Image-AUROC / `97.6%` Pixel-AUROC / `93.5%` PRO).
  2. **Đối chiếu Công bằng theo Đúng Tập Con (Matched-Subset Baseline Comparison):** Khi so sánh mô hình của nhóm ở Stage 1 với số liệu công bố trong bài báo DSKD (Bảng 1 & Bảng 2, tr. 8), giá trị tham chiếu của bài báo phải được tính lại chính xác trên cùng 5 categories (`carpet`, `tile`, `bottle`, `screw`, `toothbrush`):
     - `[FACT - Paper]` Trung bình của bài báo DSKD trên đúng 5 categories này là:
       - $\text{Image-AUROC}_{\text{Core-5}} = \frac{100.0 + 98.4 + 100.0 + 97.1 + 98.9}{5} = \mathbf{98.88\%}$
       - $\text{Pixel-AUROC}_{\text{Core-5}} = \frac{99.2 + 94.5 + 98.7 + 99.4 + 99.1}{5} = \mathbf{98.18\%}$
       - $\text{PRO}_{\text{Core-5}} = \frac{97.7 + 88.2 + 96.4 + 94.2 + 94.5}{5} = \mathbf{94.20\%}$

### 2.4. Thiết kế Stage 2: Đánh giá Toàn diện 15 Categories (Full Benchmark)
- `[INTERPRETATION - Agent 5]` Sau khi hoàn tất Stage 1 và cố định siêu tham số, 4 cấu hình đại diện (`B0` Baseline DSKD, `A_best` DSKD + Method A, `B_best` DSKD + Method B, và `C1` Combined DSKD + A + B) sẽ được huấn luyện và đánh giá trên **10 categories còn lại** (`leather`, `grid`, `wood`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `transistor`, `zipper`) để hợp nhất với 5 categories của Stage 1 thành bảng kết quả **Full 15-Category MVTec AD Benchmark** (báo cáo tách biệt `Texture Mean (5)`, `Object Mean (10)`, và `All-15 Mean`).

---

## 3. Chuẩn hóa Tiền xử lý & Siêu tham số của Baseline DSKD

### 3.1. Phân tích Phản biện Điểm Mơ hồ về Chuẩn hóa $\ell_2$ trong Mục 4.1.2 của Bài báo Gốc
- `[FACT - Paper]` Trong Mục 4.1.2 (*Implement and Environment Details*, tr. 7), tác giả Yao & Gao viết:
  > *"The input images are firstly resized, $256 \times 256$ for MVTec AD and MVTec 3D-AD, and $128 \times 128$ for MT Defects. The resized images are then normalized by $\ell_2$ function."*
- `[FACT - Paper]` Ngược lại, trong phần mô tả toán học tại Mục 3.1 (Công thức 3, tr. 5), Algorithm 1 (dòng 7–9, tr. 4), và Algorithm 2 (dòng 5–8, tr. 6), phép chuẩn hóa $\ell_2$ được áp dụng lên **các đầu ra đặc trưng của mạng** $T(I_j^t)$, $S_e(I_j^t)$ và $S_d(F_{\text{emb}})$ theo chiều kênh tại mỗi vị trí không gian $(i, j)$ cho từng tầng trung gian $k \in \{1, 2, 3\}$:
  $$\hat{F}_t^k(I_m)_{ij} = \frac{F_t^k(I_m)_{ij}}{\|F_t^k(I_m)_{ij}\|_2}, \quad \hat{F}_e^k(I_m)_{ij} = \frac{F_e^k(I_m)_{ij}}{\|F_e^k(I_m)_{ij}\|_2}, \quad \hat{F}_d^k(I_m)_{ij} = \frac{F_d^k(I_m)_{ij}}{\|F_d^k(I_m)_{ij}\|_2}$$
  *(Lưu ý kỹ thuật `[INTERPRETATION - Agent 5]`: Trong Công thức 3 trang 5 của bài báo có một lỗi chế bản hiển nhiên khi ký hiệu mẫu số là $\|F_t^k(I_m)_{ij}\|_{\ell_2}^2$, nhưng sang Công thức 4 và 5 trang 6 thì khoảng cách Euclidean $\frac{1}{2}\|\hat{F}_t - \hat{F}_e\|_2^2$ và khoảng cách Cosine chứng minh mẫu số thực sự là chuẩn Euclidean bậc nhất $\|F_t^k(I_m)_{ij}\|_2$ để đưa vector đặc trưng về vector đơn vị $\|\hat{F}\|_2 = 1$).*
- `[FACT - Repo]` Trong kho mã nguồn hiện tại ([`configs/mvtec_ad.yaml`](../../configs/mvtec_ad.yaml#L7-L9) và [`src/anomaly_detection/data/mvtec.py`](../../src/anomaly_detection/data/mvtec.py#L39-L55)), hàm `build_image_transform()` thực hiện tiền xử lý ảnh đầu vào theo chuẩn ImageNet-1K:
  - `transforms.Resize((256, 256))`
  - `transforms.ToTensor()` (đưa pixel về $[0.0, 1.0]$)
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- `[INTERPRETATION - Agent 5]` **Đánh giá Kỹ thuật về Mâu thuẫn Giữa Mục 4.1.2 và Kiến trúc Pre-trained ResNet18:**
  1. Mạng Teacher $T$ là `ResNet18` đã tiền huấn luyện trên ImageNet-1K với toàn bộ tham số đóng băng (`frozen parameters`, Mục 3.1, tr. 5). Trong `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, các tầng tích chập và đặc biệt là các thống kê `running_mean` / `running_var` của các lớp `BatchNorm2d` được khớp với ảnh đầu vào đã chuẩn hóa Z-score theo `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]`.
  2. Nếu hiểu câu *"The resized images are then normalized by $\ell_2$ function"* ở Mục 4.1.2 theo nghĩa đen (tức chia tensor ảnh $I \in \mathbb{R}^{3 \times 256 \times 256}$ cho chuẩn $\|I\|_2$ trên toàn ảnh hoặc theo từng pixel 3 kênh RGB):
     - Nếu chuẩn hóa $\ell_2$ trên toàn ảnh ($3 \times 256 \times 256 = 196,608$ phần tử): giá trị mỗi pixel sẽ co về xấp xỉ $\sim 2.2 \times 10^{-3}$, làm suy biến hoàn toàn tín hiệu qua các lớp `BatchNorm2d` có `running_std` $\sim 0.22$ của Teacher $T$.
     - Nếu chuẩn hóa $\ell_2$ theo 3 kênh RGB tại mỗi pixel: mọi pixel xám tối $(0.05, 0.05, 0.05)$ và pixel trắng sáng $(0.95, 0.95, 0.95)$ đều bị biến thành cùng một vector $(1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3})$, xóa sổ hoàn toàn thông tin độ sáng/tương phản (luminance) vốn cực kỳ quan trọng để phát hiện vết nứt, vết bẩn hoặc lỗ thủng trên `carpet`, `tile`, `screw`, `metal_nut`.
  3. Do đó, câu văn ở Mục 4.1.2 rất có khả năng là cách diễn đạt tắt của tác giả khi tóm tắt bước chuẩn hóa $\ell_2$ trên bản đồ đặc trưng của ảnh (đã nêu ở dòng 7–9 của Algorithm 1), trong khi ảnh đầu vào vẫn dùng chuẩn hóa ImageNet tiêu chuẩn như các baseline trực tiếp mà bài báo kế thừa là STPM [28] và RD4AD [9].

### 3.2. Giao thức Kiểm chứng Thực nghiệm cho Công thức Chuẩn hóa `Eq. (3)`, Tiền xử lý Mục 4.1.2 và Cỡ mẫu Chia tập (`S0_eq3_norm`, `S0_input_norm`, `S0_split`)
- `[INTERPRETATION - Agent 5 & Technical Audit]` Thay vì suy đoán chủ quan, chúng ta tách biệt rành mạch **3 phép kiểm tra thực nghiệm ngắn gọn (Sanity Checks)** ngay đầu Group 0:
  1. **`S0_eq3_norm` (Kiểm chứng Công thức Chuẩn hóa Đặc trưng `Eq. 3` trang 5):**
     - `N1 (Unit-L2 Norm)`: Chuẩn hóa đơn vị bậc một $\hat{F} = F / (\|F\|_2 + \epsilon)$ ($\|\hat{F}\|_2 = 1$), khi đó `Eq. (4)` $\equiv$ `Eq. (5)` $= 1 - \cos\theta$.
     - `N2 (Raw-L2 + Cosine / MKD-style)`: Tính `Eq. (4)` trên đặc trưng chưa chia chuẩn kết hợp `Eq. (5)` Cosine ($\lambda = 0.1$) để giải thích lý do `Eq. (5)` giữ mẫu số $\|\hat{F}_t\|\|\hat{F}_s\|$.
     - `N3 (Literal Eq. 3 Inv-Sq Norm)`: Chia đúng theo chữ in $\hat{F} = F / (\|F\|_2^2 + \epsilon)$ (với $\epsilon = 10^{-4}$ để tránh bùng nổ đạo hàm bậc $-3$ khi $\|F\|_2 \to 0$).
  2. **`S0_input_norm` (Kiểm chứng Chuẩn hóa Ảnh Đầu vào Mục 4.1.2 trang 7 — Tách biệt hoàn toàn với `Eq. 3`):**
     - `P_imagenet` (Mặc định trong `mvtec.py`): Tiền xử lý ảnh bằng ImageNet `mean`/`std` khớp với BatchNorm của Teacher `ResNet18` đóng băng.
     - `P_img_l2` (Đối chứng theo câu chữ Mục 4.1.2): Thay `transforms.Normalize(mean, std)` bằng chuẩn hóa $\ell_2$ trên tensor ảnh đầu vào.
  3. **`S0_split` (Kiểm chứng Độ nhạy Cỡ mẫu khi Tách 20% `val/good` trên Category Nhỏ):**
     - So sánh `B0 (80% train / 20% val)` (48 ảnh train / 12 ảnh val trên `toothbrush`) với `B0-100% (100% train/good)` trên `toothbrush` và `tile` để xác nhận việc tách 20% `val/good` không làm suy giảm độ chính xác tái lập Baseline.

### 3.3. Bảng Đặc tả Đầy đủ Siêu tham số của Baseline DSKD (`B0`)

| Nhóm Thông số | Tên Siêu tham số | Giá trị Chuẩn hóa | Nguồn Tham chiếu & Ghi chú Kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Kiến trúc Mạng** | Backbone của $T, S_e$ | `ResNet18` (`layer1`, `layer2`, `layer3`) | `[FACT - Paper]` Mục 3.1 (tr. 5) & Mục 4.1.2 (tr. 7). Số tầng trung gian $K = 3$ với số kênh $(64, 128, 256)$ và kích thước không gian $(64\times 64, 32\times 32, 16\times 16)$. |
| **Kiến trúc Mạng** | Trọng số Teacher $T$ | `IMAGENET1K_V1` (Frozen, `eval()` mode) | `[FACT - Paper]` Mục 2 (tr. 2) & Mục 3.1 (tr. 5). Đóng băng cả tham số trọng số lẫn thống kê `running_mean`/`running_var` của BatchNorm trong $T$. |
| **Kiến trúc Mạng** | Trọng số $S_e, \text{DFE}, S_d$ | Khởi tạo ngẫu nhiên (`Kaiming Normal`) | `[FACT - Paper]` Algorithm 1, dòng 1 (tr. 4). |
| **Dữ liệu & Đầu vào** | Độ phân giải ảnh (`img_size`) | $256 \times 256$ | `[FACT - Paper]` Mục 4.1.2 (tr. 7); `[FACT - Repo]` `configs/mvtec_ad.yaml`. |
| **Dữ liệu & Đầu vào** | Chia tập Validation (`val_split_ratio`) | `0.2` (`seed = 42` trên `train/good`) | `[FACT - Repo]` `configs/mvtec_ad.yaml` & `src/anomaly_detection/data/mvtec.py`. |
| **Tối ưu hóa** | Bộ tối ưu (Optimizer) | `Adam` | `[FACT - Paper]` Mục 4.1.2 (tr. 7). Khởi tạo 2 bộ tối ưu độc lập `opt_e` (cho $\theta_e$) và `opt_d` (cho $\theta_{\text{dfe}} \cup \theta_d$) theo Algorithm 1 dòng 16–17. |
| **Tối ưu hóa** | Tốc độ học (`lr`) & $\beta$ | $\text{lr} = 0.001$, $\beta = (0.5, 0.999)$ | `[FACT - Paper]` Mục 4.1.2 (tr. 7). `weight_decay = 0.0`. |
| **Tối ưu hóa** | Số kỷ nguyên (`epochs`) | `200` | `[FACT - Paper]` Mục 4.1.2 (tr. 7). |
| **Tối ưu hóa** | Cỡ lô (`batch_size`) | `16` *(hoặc `8` cho `toothbrush` có 48 ảnh train)* | `[INTERPRETATION - Agent 5]` Bài báo không nêu rõ `batch_size`. Cỡ lô `16` tối ưu tốc độ trên GPU 16GB VRAM (chỉ tốn $\sim 2.2\text{ GB}$ VRAM) và đảm bảo số bước cập nhật ổn định mỗi epoch. |
| **Hàm Mất mát** | Hệ số cân bằng $\lambda$ | $\lambda = 0.1$ | `[FACT - Paper]` Công thức (6) (tr. 6) & Mục 4.1.2 (tr. 7): $M^k(I_m)_{ij} = \lambda \ell_2(I_m)_{ij} + \ell_{\text{cos}}(I_m)_{ij}$. |
| **Suy luận (Inference)** | Nội suy đa tỷ lệ (`Upsample`) | `bilinear` (`align_corners=False`) lên $256\times 256$ | `[FACT - Paper]` Công thức (8) & Algorithm 2 dòng 11 (tr. 6–7). |
| **Suy luận (Inference)** | Bộ lọc khử nhiễu Gaussian | $\sigma = 4.0$ | `[FACT - Paper]` Công thức (9) & Algorithm 2 dòng 13 (tr. 6–7): $\widetilde{M}(I_m) = G_\sigma(M(I_m)), \sigma=4.0$. |
| **Suy luận (Inference)** | Điểm bất thường mức ảnh | $\text{score} = \max(\widetilde{M}(I_m))$ | `[FACT - Paper]` Công thức (10) & Algorithm 2 dòng 14 (tr. 6–7). |

---

## 4. Ma trận Thí nghiệm (Experiment Matrix) & Ablation Study (Đã Đồng Bộ Kiểm Toán Kỹ Thuật)

### 4.1. Tổng quan Cấu trúc 5 Nhóm Thí nghiệm (Group 0 – Group 4)
- `[INTERPRETATION - Agent 5 & Technical Audit]` Ma trận thực nghiệm được tổ chức thành 5 nhóm tuần tự, thống nhất 100% định danh **Method A = `FC-DSKD` (Sinh viên 1)** và **Method B = `Lite-DSKD` (Sinh viên 2)**:
  - **Group 0 (Sanity Checks & DSKD Paper Ablation Verification):** Kiểm chứng `S0_eq3_norm`, `S0_input_norm`, `S0_split` và tái lập các thí nghiệm cắt bỏ cốt lõi của bài báo gốc gồm 4 cấu trúc mạng (`T-E`, `T-D`, `E-D`, `DS` — Bảng 7, tr. 10), vai trò của DFE (Bảng 8, tr. 11), và vai trò của hợp nhất đa tầng $M_1, M_2, M_3, M_{1-3}$ (Bảng 9, tr. 11).
  - **Group 1 (Baseline DSKD — `B0-Clean`):** Huấn luyện và đánh giá chuẩn mô hình DSKD gốc trên Core-5 Subset (Stage 1) và toàn bộ 15 categories (Stage 2).
  - **Group 2 (DSKD + Method A `FC-DSKD` — Sinh viên 1):** Khảo sát cải tiến về **Hàm Mất mát Chưng cất Tiêu cự - Ngữ cảnh (`A1a, A1`), Đồng thuận Học sinh Kép Bất đối xứng có Cô lập Đạo hàm (`A2`), và Hợp nhất Bản đồ Hiệu chuẩn trên `val/good` (`A3`: `Linear` vs. `Synergistic` vs. `SoftMaxUnion` + Top-$\rho_{\text{tail}}$ Scoring với $\rho_{\text{tail}}=0.001$ tức $0.1\%$ số pixel)**.
  - **Group 3 (DSKD + Method B `Lite-DSKD` — Sinh viên 2):** Khảo sát cải tiến về **Phân rã Nút cổ chai `Lite-DFE` (`B1`: `DS-MSA + BDRF`), Thu hẹp Kênh Học sinh Kép `Slim-DSKD` (`B2`: đối chứng 3 hàm kích hoạt đầu chiếu `BN-only` vs. `BN+ReLU` vs. `BN+LeakyReLU`), và Mô hình Hợp nhất Nhẹ `Lite-DSKD` (`B3-0.75` và `B3-0.50`)**.
  - **Group 4 (Combined `FC-Lite-DSKD` — `C1-0.75` & `C1-0.50`):** Kết hợp cấu hình tốt nhất của Method A (`FC-DSKD`) và Method B (`Lite-DSKD`) để kiểm chứng tính bổ trợ lẫn nhau trên Core-5 Subset và toàn bộ 15 categories.

### 4.2. Chi tiết Thiết kế Từng Nhóm Thí nghiệm & Phân loại 3 Mức Chi phí (`Full-Retrain` / `Recalibration-Only` / `Inference-Only`)

#### Group 0: Sanity Checks & DSKD Architecture Verification (Đối chiếu Bảng 7, 8, 9 của Bài báo)
- `[FACT - Paper]` Trong Mục 4.3 (tr. 10–11), bài báo DSKD chứng minh tính hiệu quả của kiến trúc hai học sinh thông qua 3 bảng Ablation:
  1. **Bảng 7 (Model Architecture):** So sánh `DS` (`98.5 / 97.6 / 93.5`), `T-E` (`95.3 / 96.1 / 89.4`), `T-D` (`97.8 / 97.1 / 93.1`), và `E-D` (`93.2 / 94.6 / 87.7`).
  2. **Bảng 8 (Deep Feature Embedding — DFE):** Có DFE (`98.5 / 97.6 / 93.5`) so với Không có DFE (`75.6 / 89.6 / 76.5`).
  3. **Bảng 9 (Multi-scale Feature Maps Fusion):** So sánh suy luận chỉ dùng từng tầng đơn lẻ $M_3, M_2, M_1$ so với tổng hợp cả 3 tầng $M_{1-3}$.
- `[INTERPRETATION - Technical Audit]` **Phân tách rõ 3 loại chi phí chạy Ablation:**
  - **`Full-Retrain` (Train đủ 200 epochs):** `S0_eq3_norm`, `S0_input_norm`, `S0_split`, `B0-Clean`, `G0_TD`, `G0_noDFE`.
  - **`Inference-Only` (Tái sử dụng checkpoint `B0-Clean`, 0 epochs train, không cần `val/good`):** `G0_TE`, `G0_ED`, `G0_M1`, `G0_M2`, `G0_M3`.

#### Group 1: Baseline DSKD (`B0-Clean`)
- `[FACT - Paper]` Mô hình DSKD chuẩn gồm $T$ (ResNet18 đóng băng), $S_e$ (ResNet18 encoder), khối DFE (hợp nhất đa tỷ lệ + khối Residual bottleneck), và $S_d$ (Reversed ResNet18 decoder), trong đó đầu vào `DFE` bắt buộc ngắt đạo hàm `f.detach()` từ $S_e$ để đảm bảo $S_e$ và $S_d$ được tối ưu hóa độc lập tuyệt đối theo Mục 3.1 (tr. 3) và `Algorithm 1` dòng 16–17.

#### Group 2: DSKD + Method A (`FC-DSKD` — Sinh viên 1)
- **Các biến thể Ablation của Method A:**
  - **`A1a` (`Full-Retrain`):** DSKD + Contextual Patch Discrepancy ($\omega_{\text{ctx}}=0.5, \gamma=0$).
  - **`A1` (`Full-Retrain`):** DSKD + Contextual Patch + Spatial-Focal Weighting ($\omega_{\text{ctx}}=0.5, \gamma=1.0, w_{\max}=5.0$).
  - **`A2` (`Full-Retrain`):** DSKD + Asymmetric Stop-Grad Cross-Student Consistency ($\beta=0.15$, cô lập đạo hàm cả ở `DFE` và $\ell_{e-d}$).
  - **`A1+A2` (`Full-Retrain`):** Kết hợp toàn bộ cải tiến pha huấn luyện của Method A, giữ nguyên suy luận gốc.
  - **`A3-ZeroRetrain` (`Recalibration-Only` trên checkpoint `B0-Clean`, 0 epochs train):** Ước lượng $(\mu_{s,c}^k, \sigma_{s,c}^k)$ trên tập `val/good` của từng category $c$ (với sàn $\sigma_{\min}=10^{-3}$), đối chứng 3 quy tắc hợp nhất (`Linear` $\eta=0$, `Synergistic` $\eta=0.3$, `SoftMaxUnion`) và chấm điểm bằng phân số đuôi $\rho_{\text{tail}} = 0.001$ ($0.1\%$ diện tích ảnh = $65$ pixels trên lưới $256\times 256$).
  - **`A_full` (`Recalibration-Only` trên checkpoint `A1+A2`):** Mô hình **`FC-DSKD` đầy đủ** (`A1 + A2 + A3`), mở rộng ra toàn bộ 15 categories ở Stage 2.

#### Group 3: DSKD + Method B (`Lite-DSKD` — Sinh viên 2)
- **Các biến thể Ablation của Method B:**
  - **`B1` (`Lite-DFE`, `Full-Retrain`):** Thay khối `Dense-DFE` (`3.222M` params) bằng `DS-MSA + BDRF` (`0.409M` params tại $\alpha=1.0$).
  - **`B2` (`Slim-DSKD` + Projection Activation Ablation, `Full-Retrain`):** Thu hẹp độ rộng kênh $\alpha \in \{0.75, 0.50\}$ trên $S_e^{(\alpha)}, S_d^{(\alpha)}$ và đối chứng 3 hàm kích hoạt đầu chiếu $\mathcal{P}_{e,d}^k$: `BN-only` vs. `BN+ReLU` vs. `BN+LeakyReLU(0.01)` để phòng ngừa hiện tượng chết kênh (*dead channels*).
  - **`B3-0.75` & `B3-0.50` (`Full Method B — Lite-DSKD`, `Full-Retrain`):** Tích hợp `Lite-DFE` + `Slim-DSKD` ở $\alpha=0.75$ (`3.457M` trainable params, `-60.7%` vs `B0-Clean`) và $\alpha=0.50, \rho=1.0$ (`1.586M` trainable params, `-82.0%` vs `B0-Clean`, `-58.3%` FLOPs), mở rộng ra toàn bộ 15 categories ở Stage 2.

#### Group 4: Combined DSKD + Method A + Method B (`C1: FC-Lite-DSKD`)
- **`C1-0.75` & `C1-0.50` (`Full-Retrain` + `Recalibration`):** Tích hợp trực tiếp hàm mất mát `A1+A2` và suy luận hiệu chuẩn `A3` của Method A lên kiến trúc nhẹ `Lite-DSKD` ($\alpha=0.75$ và $\alpha=0.50$) của Method B trên Core-5 (Stage 1) và toàn bộ 15 categories (Stage 2).

### 4.3. Bảng Tổng hợp Ma trận Thí nghiệm (Master Experiment Matrix — Đã Kiểm Toán)

| Nhóm | Mã Run (`run_id`) | Cấu trúc / Mô tả Kỹ thuật | Thay đổi Kiến trúc (`Method B`) | Thay đổi Loss / Inference (`Method A`) | Phạm vi Category (Stage 1 $\rightarrow$ Stage 2) | Phân loại Chi phí | Trạng thái |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **G0** | `S0_eq3_norm` | Kiểm chứng `Eq. (3)`: `N1 (Unit-L2)` vs `N2 (Raw-L2+Cos)` vs `N3 (Inv-Sq L2)` | Vanilla DFE | Kiểm tra `Eq. 3–6` | 2 cats (`carpet`, `bottle`) | `Full-Retrain` | `PENDING` |
| **G0** | `S0_input_norm` | Kiểm chứng Mục 4.1.2: `P_imagenet` vs `P_img_l2` | Vanilla DFE | Vanilla ($\lambda=0.1$) | 2 cats (`carpet`, `bottle`) | `Full-Retrain` | `PENDING` |
| **G0** | `M0_bench` | Pilot Runtime Benchmark (đo wall-clock iter, checkpoint, CPU AUPRO) | Canonical DFE | Vanilla | 1 cat (`carpet`) | `Pilot-Bench` | `PENDING` |
| **G0** | `S0_split` | Kiểm chứng cỡ mẫu: `80/20 split` vs `100% train/good` | Vanilla DFE | Vanilla ($\lambda=0.1$) | 2 cats (`toothbrush`, `tile`) | `Full-Retrain` (200 ep) | `PENDING` |
| **G0** | `G0_TE`, `G0_ED`, `G0_M1..3` | Ablation Bảng 7 (`T-E`, `E-D`) & Bảng 9 ($M_1, M_2, M_3$) | Vanilla DFE | Chọn nhánh/tầng suy luận | Core-5 Subset (5 cats) | `Inference-Only` (0 train) | `PENDING` |
| **G0** | `G0_TD`, `G0_noDFE` | Ablation Bảng 7 (`T-D`) & Bảng 8 (`Without DFE`) | `T->DFE->Sd` / `No DFE` | Vanilla ($\lambda=0.1$) | Core-5 Subset (5 cats) | `Full-Retrain` (10 models) | `PENDING` |
| **G1** | `B0-Clean` | **Baseline DSKD Chuẩn hóa** (3-Stage ResNet18 + Dense-DFE) | Vanilla (`8.788M` trainable) | Vanilla ($\lambda=0.1$) | Core-5 (Stage 1) $\rightarrow$ **All 15 (Stage 2)** | `Full-Retrain` (15 models) | `PENDING` |
| **G2** | `A1a`, `A1`, `A2`, `A1+A2` | Ablation Pha Huấn luyện của **Method A (`FC-DSKD`)** | Vanilla (`+0.00M` params) | `Contextual` / `Focal` / `Asym-ED` | Core-5 Subset (5 cats) | `Full-Retrain` (20 models) | `PENDING` |
| **G2** | `A3-ZeroRetrain` | **Method A (`Only A3`)** chạy trực tiếp trên checkpoint `B0-Clean` | Vanilla (`+0.00M` params) | `Val Z-score` + `Linear/Synergistic/SoftMax` + $\rho_{\text{tail}}=0.001$ | Core-5 (Stage 1) $\rightarrow$ **All 15 (Stage 2)** | `Recalibration-Only` (0 train) | `PENDING` |
| **G2** | `A_full` (`FC-DSKD`) | **DSKD + Full Method A (`A1 + A2 + A3`)** | Vanilla (`+0.00M` params) | `Full FC-DSKD` | Core-5 (Stage 1) $\rightarrow$ **All 15 (Stage 2)** | `Recalibration-Only` trên `A1+A2` (+10 models Stage 2) | `PENDING` |
| **G3** | `B1` (`Lite-DFE`) | DSKD + Factorized `Lite-DFE` (`DS-MSA + BDRF`) | `Lite-DFE` (`5.975M` trainable) | Vanilla | Core-5 Subset (5 cats) | `Full-Retrain` (5 models) | `PENDING` |
| **G3** | `B2` (`Slim-DSKD`) | Thu hẹp kênh $\alpha$ + Đối chứng `BN` vs `BN+ReLU` vs `BN+LeakyReLU` | `Slim-DSKD` | Vanilla | Core-5 Subset (5 cats) | `Full-Retrain` (15 models) | `PENDING` |
| **G3** | `B3-0.75` & `B3-0.50` | **DSKD + Full Method B (`Lite-DSKD`, $\alpha \in \{0.75, 0.50\}$)** | `Lite-DFE + Slim-DSKD` (`3.457M` / `1.586M` trainable) | Vanilla | Core-5 (Stage 1) $\rightarrow$ **All 15 (Stage 2)** | `Full-Retrain` (30 models) | `PENDING` |
| **G4** | `C1-0.75` & `C1-0.50` | **Combined `FC-Lite-DSKD` (`Method A + Method B`)** | `Lite-DSKD-0.75 / 0.50` | `Full FC-DSKD` | Core-5 (Stage 1) $\rightarrow$ **All 15 (Stage 2)** | `Full-Retrain` + `Recalibrate` | `PENDING` |

---

## 5. Hệ thống Chỉ số Đánh giá Bắt buộc (Mandatory Evaluation Metrics Suite)

### 5.1. Nhóm 6 Chỉ số Độ chính xác Phát hiện & Định vị (Accuracy & Localization Metrics)
- `[FACT - Repo]` Toàn bộ 6 chỉ số độ chính xác dưới đây đã được cài đặt và kiểm thử đơn vị trong [`src/anomaly_detection/evaluation/metrics.py`](../../src/anomaly_detection/evaluation/metrics.py) và cấu hình tại [`configs/mvtec_ad.yaml`](../../configs/mvtec_ad.yaml#L29-L36):
  1. **`Image-AUROC` (`auroc_img`):** Diện tích dưới đường cong ROC ở mức ảnh, tính từ nhãn ảnh $y_m \in \{0, 1\}$ và điểm bất thường mức ảnh $\text{score}_m = \max_{i,j} \widetilde{M}(I_m)_{ij}$ ([`metrics.py`](../../src/anomaly_detection/evaluation/metrics.py#L174-L181)). Đây là chỉ số chính trong Bảng 1 của bài báo DSKD (`[FACT - Paper]`, tr. 7–8).
  2. **`Image-AP` (`ap_img`):** Độ chính xác trung bình (Average Precision — diện tích dưới đường cong Precision-Recall) ở mức ảnh ([`metrics.py`](../../src/anomaly_detection/evaluation/metrics.py#L182)). Bổ sung cho `Image-AUROC` để phản ánh độ nhạy với mất cân bằng lớp giữa số ảnh bình thường và ảnh lỗi trong tập `test/`.
  3. **`Pixel-AUROC` (`auroc_pix`):** Diện tích dưới đường cong ROC trên toàn bộ $N \times 256 \times 256$ pixel của tập kiểm thử ([`metrics.py`](../../src/anomaly_detection/evaluation/metrics.py#L183)). Đây là chỉ số thứ nhất trong Bảng 2 của bài báo DSKD (`[FACT - Paper]`, tr. 8).
  4. **`Pixel-AP` (`ap_pix`):** Average Precision ở mức pixel ([`metrics.py`](../../src/anomaly_detection/evaluation/metrics.py#L184)). Do tỷ lệ pixel thực sự bị lỗi trên MVTec AD thường rất nhỏ (chỉ chiếm $1\%\text{--}5\%$ diện tích ảnh), `Pixel-AUROC` dễ bị thổi phồng bởi lượng lớn True Negatives ở vùng nền; `Pixel-AP` đánh giá khắt khe hơn tỷ lệ dương tính giả (False Positives) trên pixel.
  5. **`AUPRO@0.30` (`aupro_30` — Chuẩn Bài báo Gốc):** Diện tích chuẩn hóa dưới đường cong Per-Region Overlap (PRO) tích phân trên miền tỷ lệ dương tính giả $\text{FPR} \in [0.0, 0.30]$ và chia chuẩn hóa cho `max_fpr = 0.30` ([`metrics.py`](../../src/anomaly_detection/evaluation/metrics.py#L32-L152)):
     $$\text{AUPRO@}0.30 = \frac{1}{0.30} \int_{0}^{0.30} \text{PRO}(\text{FPR}) \, d\text{FPR}$$
     Trong đó, mỗi thành phần liên thông (connected component qua `scipy.ndimage.label`) trên mặt nạ ground-truth được tính tỷ lệ giao cắt với trọng số ngang nhau bất kể diện tích vùng lỗi lớn hay nhỏ (`[FACT - Paper]`, Mục 4.1.3, tr. 7).
  6. **`AUPRO@0.05` (`aupro_05` — Định vị Nghiêm ngặt ở Mức FPR Thấp):** Cũng sử dụng hàm `compute_aupro(..., max_fpr=0.05)` ([`metrics.py`](../../src/anomaly_detection/evaluation/metrics.py#L236)). `[INTERPRETATION - Agent 5]` Vì bài báo DSKD thừa nhận điểm yếu nhận diện nhầm pixel bình thường xung quanh vùng lỗi thành bất thường (*surrounding false positives*), `AUPRO@0.05` (chỉ cho phép tối đa 5% pixel nền bị báo nhầm) là chỉ số phân biệt sắc bén nhất xem Method A và Method B có thực sự làm giảm viền dương tính giả quanh khuyết tật hay không.

### 5.2. Nhóm Chỉ số Độ phức tạp & Hiệu quả Tính toán (Complexity & Efficiency Profiling)
- `[FACT - Paper]` Tại Mục 4.1.3 (tr. 7) và Mục 4.2.4 (Bảng 6, tr. 10), bài báo đánh giá độ phức tạp của mô hình qua 4 đại lượng: `Parameters (M)` (`39.3M`), `FLOPs (G)` (`6.17G`), `Inference Time (s)` (`0.217s` trên CPU Intel i7 @ 2.3GHz), và `Memory (MB)` (`105 MB`).
- `[FACT - Repo]` & `[INTERPRETATION - Agent 5]` Module [`src/anomaly_detection/utils/profiling.py`](../../src/anomaly_detection/utils/profiling.py) hiện có các hàm `count_parameters()`, `measure_student_flops()`, và `profile_model_cost()`. Tuy nhiên, qua kiểm tra mã nguồn, Agent 5 ghi nhận một chi tiết kỹ thuật quan trọng cần lưu ý khi tích hợp DSKD:
  - Trong [`profiling.py`](../../src/anomaly_detection/utils/profiling.py#L28-L35) hiện tại, hàm `measure_student_flops()` khởi tạo các tensor giả lập `e1 (256ch), e2 (512ch), e3 (1024ch)` (tương ứng với backbone ResNet50/WideResNet50 của các nhánh nghiên cứu cũ) và chỉ đo `student(e1, e2, e3)`.
  - Đối với kiến trúc **DSKD trên ResNet18** (`[FACT - Paper]`, Mục 3.1 & Algorithm 2):
    - Các tầng `layer1, layer2, layer3` của ResNet18 có số kênh là **`(64, 128, 256)`** với kích thước `(64x64, 32x32, 16x16)`.
    - Trong pha suy luận (Algorithm 2), ảnh đầu vào $I^a \in \mathbb{R}^{1 \times 3 \times 256 \times 256}$ đi qua cả $T(I^a)$ và nhánh học sinh $S_e(I^a) \rightarrow \text{DFE}(\hat{F}_e) \rightarrow S_d(F_{\text{emb}})$.
  - Do đó, khi báo cáo nhóm chỉ số hiệu quả tính toán trong dự án này, chúng ta chuẩn hóa cách đo như sau:
  7. **`Params (M)` (Số lượng Tham số — Triệu tham số):** Báo cáo tách bạch:
     - `Total Params (M)`: Tổng số tham số của toàn bộ hệ thống $T + S_e + \text{DFE} + S_d + \mathcal{P}$ (để đối chiếu với `39.30M` Table 6 và mẫu chuẩn `12.180M` của `B0-Clean`).
     - `Trainable Student Params (M)`: Tổng tham số được huấn luyện của $S_e + \text{DFE} + S_d + \mathcal{P}$ (không tính Teacher $T$ đóng băng, `9.397M` trên `B0-Clean`).
     - `BN Buffers`: Số lượng tham số đệm running_mean / running_var của BatchNorm (`16,307` trên `B0-Clean`).
  8. **`FLOPs (G)` (Số phép toán dấu phẩy động — Tỷ phép toán trên 1 ảnh $256 \times 256$):** Đo bằng `torch.utils.flop_counter.FlopCounterMode` với `batch_size = 1`, báo cáo cả `Total Inference FLOPs (G)` (luồng đầy đủ $T + S_e + \text{DFE} + S_d$ để đối chiếu với mốc `6.17G` của Bảng 6) và `Student + DFE FLOPs (G)` (để làm nổi bật mức giảm FLOPs của Method A).
  9. **`Inference Latency (ms/img trên GPU & CPU)` & **`Peak GPU VRAM (MB)`**:**
     - Đo với `batch_size = 1`, `warmup = 10` vòng lặp, lấy trung bình và độ lệch chuẩn trên `iters = 30` vòng lặp (kèm `torch.cuda.synchronize()` khi đo trên GPU).
     - Báo cáo cả **Forward Pass Latency (ms/img)** và **End-to-End Inference Latency (ms/img)** (bao gồm bước tính bản đồ bất thường, nội suy `bilinear` lên $256\times 256$ và lọc Gaussian $\sigma=4.0$) trên cả **GPU (Kaggle Tesla T4 / P100)** lẫn **CPU (trên node Kaggle CPU)** để so sánh trực tiếp với mốc `217 ms` trên CPU của Bảng 6 trong bài báo.
     - Đo **`Peak GPU VRAM (MB)`** bằng `torch.cuda.max_memory_allocated() / (1024**2)` sau khi `reset_peak_memory_stats()`.

---

## 6. Quy tắc Báo cáo Trung thực (Negative Results & Unexecuted Runs Policy)

Để đảm bảo tính liêm chính học thuật cao nhất cho báo cáo nghiên cứu CS331, toàn bộ thành viên và các Agent trong nhóm tuân thủ nghiêm ngặt 4 quy tắc báo cáo sau:

1. **Chính sách Thí nghiệm Chưa Thực thi (`Unexecuted Runs Policy`):**
   - Mọi tổ hợp thí nghiệm trong Ma trận Thực nghiệm (Mục 4.3) chưa được chạy thực tế trên GPU và chưa xuất ra tệp log xác thực (`metrics.json` / `summary.csv`) phải được ghi rõ trạng thái là **`PENDING`** hoặc **`NOT EXECUTED`**.
   - Tuyệt đối không được điền số liệu dự đoán, số liệu nội suy, hay sao chép số liệu từ bài báo gốc vào cột kết quả thực nghiệm của nhóm.
2. **Tách biệt Minh bạch Giữa Số liệu Bài báo và Số liệu Tái lập (`Paper vs. Reproduced Separation`):**
   - Trong mọi bảng so sánh, số liệu trích từ bài báo `2402.00448v1.pdf` phải nằm ở cột riêng có nhãn `DSKD (Paper Reported)` và số liệu chạy thực tế từ mã nguồn của nhóm phải nằm ở cột `DSKD (Ours Reproduced - B0)`, kèm theo cột độ lệch $\Delta = \text{Ours} - \text{Paper}$.
3. **Báo cáo Trung thực Kết quả Âm tính (`Negative Results Reporting Policy`):**
   - Nếu kết quả tái lập `B0` thấp hơn số liệu bài báo công bố trên một số categories, hoặc nếu một biến thể cải tiến (`A1`, `A2`, `B1`, `B2`, `C1`) làm giảm hiệu năng so với Baseline `B0` (ví dụ: giảm `Params`/`FLOPs` nhưng làm tụt `Image-AUROC` hoặc `AUPRO@0.05`), kết quả đó **bắt buộc phải được giữ nguyên trong bảng Ablation**.
   - Một kết quả âm tính được phân tích nguyên nhân sâu sắc (chẳng hạn: nén nút cổ chai DFE quá mức ở `A1` làm mất thông tin tần số cao cần cho `screw`, hoặc chuẩn hóa theo `val/good` ở `B2` bị phương sai lớn khi tập `val/good` chỉ có 12 ảnh như `toothbrush`) có giá trị khoa học cao hơn nhiều so với việc chọn lọc kết quả có lợi (cherry-picking).
4. **Truy xuất Nguồn gốc Hiện vật (`Artifact Provenance`):**
   - Mỗi con số xuất hiện trong báo cáo cuối cùng phải truy ngược được về một tệp `results/<run_id>/per_category_metrics.json` chứa thông tin: mã commit Git (`git_sha`), mã băm tệp cấu hình (`config_sha256`), tên thiết bị GPU, thời gian chạy, và đường dẫn checkpoint đã dùng để đánh giá.
