# BẢN GIAO KÈO THIẾT KẾ THỰC NGHIỆM B0 CUỐI CÙNG: PHẢN BIỆN ĐỐI NGHỊCH & SẴN SÀNG TRIỂN KHAI
## (FINAL B0 EXPERIMENT DESIGN GATE: ADVERSARIAL REVIEW & IMPLEMENTATION READINESS)

- **Dự án:** Knowledge Distillation-Based Lightweight Industrial Anomaly Detection (`CS331`)
- **Kho mã nguồn:** [cs331-knowledge-distillation-lightweight-industrial-anomaly-detection](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection)
- **Nhánh làm việc:** `chore/repository-reset`
- **Commit HEAD:** `41af7b015a69fe7480c8fd1212e6e22840ae22d0`
- **Vai trò:** Senior Research Engineer & Independent Technical Reviewer
- **Ngày lập:** `2026-11-14`
- **Tài liệu thẩm tra đối chiếu:**
  - `papers/2402.00448v1.pdf` (arXiv:2402.00448v1, pp. 1–12)
  - `docs/research/b0_input_optimization_contract.md`
  - `docs/research/final_design_gate.md`
  - `docs/research/implementation_freeze_proposal.md`
  - `docs/research/experiment_design.md`

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Báo cáo này là kết quả của đợt phản biện đối nghịch độc lập (adversarial review) nhằm tìm kiếm và khắc phục các lỗ hổng phương pháp luận trong tài liệu [b0_input_optimization_contract.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/b0_input_optimization_contract.md) trước khi phê duyệt triển khai mã nguồn cho mô hình nền tảng `B0-Clean`.

### Các sai lệch và lỗ hổng nghiêm trọng đã được chỉ ra & sửa chữa:
1. **Lỗ hổng Đánh giá (Evaluation Leakage):** Báo cáo trước dự kiến dùng tập **Test** để chọn cấu hình canonical cho DFE input và Detach. Đây là lỗi vi phạm nguyên tắc phương pháp luận ML cơ bản. Thiết kế mới chuyển toàn bộ quá trình lựa chọn sang tập **Validation** (`val/good` loss convergence và distribution stability); tập Test chỉ dùng một lần duy nhất để đánh giá mô hình sau khi đã khóa cấu hình.
2. **Sai lệch Siêu tham số (Hyperparameter Deviation):** Báo cáo trước tự ý áp dụng `AdamW`, `lr=1e-4`, `epochs=50` và `CosineAnnealingLR` mà không công bố đây là sự thay đổi so với bài báo. Bài báo gốc công bố rõ ràng tại Mục 4.1.2: dùng **`Adam`**, **$\beta=(0.5, 0.999)$**, **$\text{learning rate}=0.001$** và **$200$ epochs**. Thiết kế mới phục hồi nguyên bản thiết lập của bài báo cho mốc tái hiện `B0-Clean`.
3. **Ảo tưởng Thời gian GPU (GPU Timing Speculation):** Tuyên bố "thí nghiệm 2x2 chỉ mất 6 phút" thực chất chỉ là ước tính giải tích chưa tính đến thời gian CPU-bound của thuật toán `compute_aupro` (mất từ $45\text{--}90\text{ giây}$/lần chạy). Thiết kế mới phân tách rành mạch thời gian ước tính và quy định bước đo chuẩn thực tế `M0_bench`.
4. **Mâu thuẫn Văn bản Bài báo (Paper Ambiguity):** Bài báo có sự mâu thuẫn không thể dung hòa giữa Hình 4 / Mục 3.2 (đặc trưng thô $F_e$) và Algorithm 2 dòng 7 (đặc trưng chuẩn hóa $\hat{F}_e$), đồng thời khuyết bước tính $F_{\text{emb}}$ trong Algorithm 1. Cần duy trì tính mở bằng cờ cấu hình thay vì khẳng định một chiều.

---

## 2. ĐỐI CHIẾU DỮ KIỆN BÀI BÁO VS. ĐIỂM MƠ HỒ KỸ THUẬT (PAPER FACTS VS. IMPLEMENTATION AMBIGUITIES)

### 2.1. Bảng Đối Chiếu Chi Tiết Từng Phần

| Thành Phần Khảo Sát | Vị Trí Trong Paper (`papers/2402.00448v1.pdf`) | Nội Dung Văn Bản Gốc `[PAPER FACT]` | Điểm Mơ Hồ Triển Khai `[IMPLEMENTATION AMBIGUITY]` |
| :--- | :--- | :--- | :--- |
| **Đặc trưng vào DFE (Hình 4)** | Trang 5, Hình 4 & Chú thích | Vẽ 3 nhánh nhận $F_1, F_2, F_3$. Mỗi nhánh có `Conv` $\to$ `BN` $\to$ `ReLU`. Không có dấu mũ $\hat{\cdot}$. | Hình vẽ thể hiện đặc trưng thô $F_e$ đi vào DFE. |
| **Đặc trưng vào DFE (Algo 2)** | Trang 6, Cột 2, Algorithm 2, Dòng 7 | $F_{\text{emb}} \leftarrow \text{feature\_embedding}(\hat{F}_e)$ | Văn bản dòng lệnh ghi rõ dùng đặc trưng đã chia chuẩn $\hat{F}_e$. Mâu thuẫn trực tiếp với Hình 4. |
| **Đặc trưng vào DFE (Algo 1)** | Trang 4, Cột 1, Algorithm 1 | Dòng 8 tính $\hat{F}_e$, Dòng 9 gọi $S_d(F_{\text{emb}})$. | **Khuyết hoàn toàn** dòng gán $F_{\text{emb}} \leftarrow \text{DFE}(\cdot)$. Không thể biết tác giả đưa $F_e$ hay $\hat{F}_e$ trong pha train. |
| **Cơ sở Lý thuyết DFE** | Trang 6, Cột 1, Mục 3.2 | "Referring to [9] [RD4AD], we conduct feature fusion on intermediate outputs of $S_e$..." | Trong RD4AD (Deng & Li, CVPR 2022), khối one-class embedding nhận đặc trưng thô từ encoder. |
| **Ngắt đạo hàm (`detach`)** | Toàn bộ bài báo (pp. 1–12) | Không có từ nào là `detach`, `stop_gradient`, `freeze` hay $\text{sg}[\cdot]$. | Hoàn toàn không rõ tác giả dùng `detach()` hay dùng 2 optimizer riêng biệt hay joint backward. |
| **Bộ tối ưu hóa (Optimizer)** | Trang 7, Cột 1, Mục 4.1.2 | "we use the adaptive moment estimation algorithm (Adam) as the optimizer, with $\beta = (0.5, 0.999)$." | Dùng **Adam chuẩn**, $\beta_1 = 0.5$ (không phải AdamW, không phải $\beta_1=0.9$). |
| **Tốc độ học (Learning Rate)** | Trang 7, Cột 1, Mục 4.1.2 | "The learning rate is $0.001$, and the model is trained in 200 epochs." | Tốc độ học là **$1\times 10^{-3}$** (không phải $1\times 10^{-4}$), huấn luyện đủ **200 epochs**. |
| **Trọng số hàm mất mát** | Trang 6 (Eq. 6) & Trang 7 (Mục 4.1.2) | $M = \lambda \ell_2 + \ell_{\cos}$ với $\lambda = 0.1$. | Khớp hoàn toàn giữa công thức và thực nghiệm. |
| **Kích thước Batch** | Mục 4.1.2 | **Không đề cập** giá trị batch size cụ thể. | Lựa chọn kỹ thuật của nhóm: $B=16$ (phù hợp VRAM 16GB). |

---

## 3. HỢP ĐỒNG ĐẠO HÀM CUỐI CÙNG (FINAL GRADIENT CONTRACT)

### 3.1. Phân Tích Đồ Thị Tính Toán và Hai Nhánh Đạo Hàm

```mermaid
graph TD
    subgraph FORWARD_DATA_FLOW["LUỒNG DỮ LIỆU THUẬN (FORWARD FLOW)"]
        I[Ảnh đầu vào I] --> T[Teacher T - Đóng băng]
        I --> Se[Student Encoder Se - Tham số: theta_e]
        
        T --> Ft[F_t^1, F_t^2, F_t^3]
        Se --> Fe[F_e^1, F_e^2, F_e^3]
        
        Ft -->|Eq. 3| Ft_norm[\hat{F}_t]
        Fe -->|Eq. 3| Fe_norm[\hat{F}_e]
        
        Ft_norm & Fe_norm -->|Eq. 4, 5, 6| LossE((Loss l_e))
        
        Fe -.->|Tùy chọn: detach=True| Fe_det[F_e ngắt đạo hàm]
        Fe -.->|Tùy chọn: detach=False| Fe_joint[F_e giữ đạo hàm]
        
        Fe_det & Fe_joint --> DFE[Khối DFE - Tham số: theta_dfe]
        DFE --> Femb[F_emb]
        Femb --> Sd[Student Decoder Sd - Tham số: theta_d]
        Sd --> Fd[F_d^1, F_d^2, F_d^3]
        Fd -->|Eq. 3| Fd_norm[\hat{F}_d]
        
        Ft_norm & Fd_norm -->|Eq. 4, 5, 6| LossD((Loss l_d))
    end

    subgraph BACKWARD_GRADIENT_PATHS["LUỒNG LAN TRUYỀN NGƯỢC (BACKWARD PATHS)"]
        LossE -->|d_le / d_theta_e| Se
        LossD -->|d_ld / d_theta_d| Sd
        LossD -->|d_ld / d_theta_dfe| DFE
        
        DFE -->|TRƯỜNG HỢP A: detach=True| Blocked[BỊ CHẶN BỞI DETACH: Gradient về Se = 0]
        DFE -->|TRƯỜNG HỢP B: detach=False| FlowToSe[Gradient d_ld / d_Fe truyền về Se]
        FlowToSe --> Se
    end
```

### 3.2. Phân Định Rạch Ròi Bản Chất Nhận Thức (Epistemic Breakdown)
1. **`[PAPER FACT]`**: Algorithm 1 ghi hai dòng lệnh tách rời:
   - Dòng 16: `Update parameters \theta_e;` (ứng với việc giảm $\ell_e$)
   - Dòng 17: `Update parameters \theta_d;` (ứng với việc giảm $\ell_d$)
   Văn bản hoàn toàn không đề cập đến việc $\ell_d$ có truyền ngược gradient vào $\theta_e$ hay không.
2. **`[MATHEMATICAL DERIVATION]`**:
   - Khi `detach=True`: $\nabla_{\theta_e} = \frac{\partial \ell_e}{\partial \theta_e}$. Đạo hàm $\frac{\partial \ell_d}{\partial \theta_e} \equiv \mathbf{0}$.
   - Khi `detach=False`: $\nabla_{\theta_e} = \frac{\partial \ell_e}{\partial \theta_e} + \frac{\partial \ell_d}{\partial F_e}\frac{\partial F_e}{\partial \theta_e}$. $S_e$ tiếp nhận cả gradient từ nhiệm vụ tái tạo của $S_d$.
3. **`[IMPLEMENTATION DECISION]`**: Trong mã nguồn, tạo cờ cấu hình `detach_dfe_input: bool`.
4. **`[HYPOTHESIS]`**: `detach=True` giúp $S_e$ tập trung học tri thức của Teacher mà không bị hiện tượng đồng thích nghi (*co-adaptation*) làm suy giảm độ sắc nét của bản đồ sai biệt $M_{t-e}$. Giả thuyết này cần kiểm chứng qua thực nghiệm.

---

## 4. QUYẾT ĐỊNH VỀ ĐẦU VÀO KHỐI DFE (DFE INPUT DECISION)

### 4.1. Hai Phương Án Trực Giao
- **Phương án Raw ($F_e \to \text{DFE}$):** Đặc trưng thô chưa chia chuẩn được đưa vào 3 nhánh DFE. Các lớp `BatchNorm2d(256)` bên trong từng nhánh đảm nhiệm vai trò chuẩn hóa phân phối nội bộ.
- **Phương án Norm ($\hat{F}_e \to \text{DFE}$):** Đặc trưng đã chia chuẩn $\ell_2$ (theo Eq. 3) được đưa vào DFE, tuân thủ chữ viết của Algorithm 2 dòng 7.

### 4.2. Khóa Cứng Nguyên Tắc Nhất Quán (Consistency Invariant)
> [!IMPORTANT]
> **QUY TẮC NHẤT QUÁN BẤT KHẢ XÂM PHẠM:**
> Trong bất kỳ cấu hình nào, **DỮ LIỆU ĐẦU VÀO DFE TRONG PHA HUẤN LUYỆN VÀ PHA SUY LUẬN BẮT BUỘC PHẢI CÙNG MỘT LOẠI**:
> - Nếu mô hình được cấu hình `dfe_input_type = "raw"`: Huấn luyện dùng $F_e \implies$ Suy luận BẮT BUỘC dùng $F_e$.
> - Nếu mô hình được cấu hình `dfe_input_type = "norm"`: Huấn luyện dùng $\hat{F}_e \implies$ Suy luận BẮT BUỘC dùng $\hat{F}_e$.
>
> Tuyệt đối không để xảy ra tình trạng Train dùng đặc trưng thô, Inference lại dùng đặc trưng chia chuẩn, vì sẽ gây sụp đổ phân phối thống kê của BatchNorm trong DFE.

---

## 5. BỘ TỐI ƯU HÓA VÀ QUYỀN SỞ HỮU THAM SỐ (OPTIMIZER & PARAMETER OWNERSHIP)

### 5.1. Khẳng Định Toán Học Về DFE
- Vì $\ell_e$ chỉ tính trên $(\hat{F}_t, \hat{F}_e)$, khối DFE không nằm trong luồng tính $\ell_e \implies \frac{\partial \ell_e}{\partial \theta_{\text{dfe}}} \equiv \mathbf{0}$.
- Khối DFE nằm trên luồng tính $\ell_d \implies \frac{\partial \ell_d}{\partial \theta_{\text{dfe}}} \neq \mathbf{0}$.
- Do đó, tham số DFE $\theta_{\text{dfe}}$ **bắt buộc phải được cập nhật bởi mục tiêu $\ell_d$**.

### 5.2. Cấu Trúc Optimizer Chuẩn Hóa
Để phản ánh chính xác cấu trúc hai bước cập nhật của Algorithm 1 dòng 16–17 và đảm bảo tính độc lập tham số:

```python
# TẬP HỢP THAM SỐ KHÔNG GIAO NHAU (MUTUALLY DISJOINT SETS)
params_e = list(model.student_encoder.parameters())
params_d = list(model.dfe.parameters()) + list(model.student_decoder.parameters())

# KHỞI TẠO OPTIMIZER THEO THIẾT LẬP GỐC CỦA BÀI BÁO (Mục 4.1.2)
optimizer_e = torch.optim.Adam(
    params_e,
    lr=1e-3,                      # [PAPER FACT]: lr = 0.001
    betas=(0.5, 0.999),           # [PAPER FACT]: beta = (0.5, 0.999)
    eps=1e-8
)

optimizer_d = torch.optim.Adam(
    params_d,
    lr=1e-3,                      # [PAPER FACT]: lr = 0.001
    betas=(0.5, 0.999),           # [PAPER FACT]: beta = (0.5, 0.999)
    eps=1e-8
)
```

- **Cam kết:** $\Theta_e \cap \Theta_d = \emptyset$. Không có bất kỳ tham số nào bị cập nhật chồng lấn bởi hai optimizer.

---

## 6. MA TRẬN ABLATION ĐÃ CHỈNH SỬA (CORRECTED ABLATION MATRIX)

Nhóm phân định quy trình thực nghiệm thành 3 giai đoạn rành mạch: **Thăm dò (Exploration)** $\to$ **Lựa chọn (Selection)** $\to$ **Xác nhận (Confirmation)**:

| Mã Thí Nghiệm | Vai Trò | Mục Tiêu Khoa Học | Biến Độc Lập | Biến Giữ Cố Định | Dataset / Category | Số Epoch | Seed | Tiêu Chí Đo Lường (Metrics) | Điều Kiện Dừng / Thành Công |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`EXP-01` (`raw_detach`)** | Exploration & Selection | Đánh giá baseline chuẩn đề xuất | `dfe_in="raw"`, `detach=True` | Adam, lr=1e-3, $\beta=(0.5, 0.999)$, bs=16, ResNet18 | `carpet` (Texture) & `bottle` (Object) | 50 | 42 | Validation Loss $\ell_e, \ell_d$, Gradient Norm | Không NaN/Inf, loss hội tụ đều |
| **`EXP-02` (`norm_detach`)** | Exploration & Selection | Đánh giá đầu vào theo Algorithm 2 | `dfe_in="norm"`, `detach=True` | Cùng thiết lập trên | `carpet` & `bottle` | 50 | 42 | Validation Loss $\ell_e, \ell_d$, Gradient Norm | Không NaN/Inf, loss hội tụ đều |
| **`EXP-03` (`raw_joint`)** | Exploration | Kiểm tra tương tác khi không detach | `dfe_in="raw"`, `detach=False` | Cùng thiết lập trên | `carpet` | 50 | 42 | Co-adaptation metric, Val Loss | Đo độ ổn định biểu diễn |
| **`EXP-04` (`norm_joint`)** | Exploration | Kiểm tra liên kết liên tục toàn phần | `dfe_in="norm"`, `detach=False` | Cùng thiết lập trên | `carpet` | 50 | 42 | Co-adaptation metric, Val Loss | Đo độ ổn định biểu diễn |
| **`EXP-CONFIRM`** | **Confirmation** | **Xác nhận cấu hình chiến thắng ở quy mô đầy đủ** | **Cấu hình tốt nhất từ Pha Selection** | Adam, lr=1e-3, $\beta=(0.5, 0.999)$, bs=16 | **Core-5:** `carpet`, `bottle`, `hazelnut`, `leather`, `cable` | **200** | **42, 43, 44 (3 seeds)** | **Val AUROC, Test AUROC, Test AUPRO (0.30 & 0.05)** | **Image AUROC $\ge 0.965$ trung bình** |

---

## 7. GIAO THỨC PHÂN ĐỊNH VALIDATION VÀ TEST (VALIDATION/TEST PROTOCOL)

Nhóm thiết lập kỷ luật phương pháp luận nghiên cứu nghiêm ngặt:

1. **Nguyên tắc phân định tập dữ liệu:**
   - **Tập Train ($80\%$ `train/good`):** Dùng để cập nhật gradient mô hình qua optimizer.
   - **Tập Validation ($20\%$ `val/good`):** Dùng để:
     - Giám sát đường cong hội tụ loss.
     - Kiểm tra độ ổn định số học (không bão hòa gradient, không sụp đổ biểu diễn).
     - Lưu checkpoint tốt nhất (`best_val_checkpoint.pth`).
     - **LỰA CHỌN CẤU HÌNH CHIẾN THẮNG TRONG PHA SELECTION.**
   - **Tập Test (Ảnh `test` gồm cả `good` và defect):**
     - **TUYỆT ĐỐI KHÔNG DÙNG ĐỂ CHỌN CẤU HÌNH HAY TUNING SIÊU THAM SỐ.**
     - Chỉ được nạp và chạy đánh giá duy nhất một lần ở pha Confirmation hoặc báo cáo kết quả cuối cùng.
2. **Quy tắc ngăn ngừa thiên lệch (Anti-Snooping Rule):**
   - Không được phép chạy sweep test metrics rồi chọn cấu hình có Test AUROC cao nhất. Cấu hình được chọn bắt buộc phải dựa trên tiêu chí ổn định của tập Validation.
3. **Quy tắc đa danh mục (Multi-Category Rule):**
   - Không kết luận cấu hình tối ưu cho 15 danh mục chỉ dựa vào 1 danh mục `carpet`. Pha Selection bắt buộc phải kiểm tra chéo trên cả 1 danh mục bề mặt/kết cấu (`carpet`) và 1 danh mục đối tượng cấu trúc (`bottle`).

---

## 8. GIAO THỨC SIÊU THAM SỐ (HYPERPARAMETER PROTOCOL)

Bảng phân loại rõ ràng giữa thiết lập gốc của bài báo và các sai lệch kỹ thuật:

| Siêu Tham Số (Hyperparameter) | Thiết Lập Chuẩn Bài Báo (`B0-Paper`) | Thiết Lập Tái Hiện Chuẩn (`B0-Clean`) | Phân Loại | Lý Do Kỹ Thuật & Tác Động |
| :--- | :---: | :---: | :---: | :--- |
| **Optimizer** | `Adam` | `Adam` | `[MATCHES PAPER FACT]` | Giữ nguyên bản bài báo (Mục 4.1.2). |
| **Momentum Parameters** | $\beta = (0.5, 0.999)$ | $\beta = (0.5, 0.999)$ | `[MATCHES PAPER FACT]` | Giữ nguyên bản bài báo; $\beta_1=0.5$ giảm trễ quán tính. |
| **Learning Rate** | $0.001$ ($1\times 10^{-3}$) | $0.001$ ($1\times 10^{-3}$) | `[MATCHES PAPER FACT]` | Giữ nguyên bản bài báo; không hạ xuống $10^{-4}$. |
| **LR Scheduler** | Không nêu (Constant) | Không dùng scheduler | `[MATCHES PAPER FACT]` | Không thêm CosineAnnealing vào baseline tái hiện. |
| **Số Epochs Chính Thức** | $200$ epochs | $200$ epochs | `[MATCHES PAPER FACT]` | Chạy đủ 200 epochs cho pha Confirmation. |
| **Số Epochs Thăm Dò (S0)**| — | $50$ epochs | `[DEVIATION - JUSTIFIED]` | Chỉ dùng cho sàng lọc sơ bộ, ghi rõ là ablation. |
| **Batch Size** | Không nêu | $16$ | `[IMPLEMENTATION CHOICE]` | Cố định cho mọi thí nghiệm, phù hợp VRAM 16GB. |
| **Trọng số hàm mất mát $\lambda$**| $0.1$ | $0.1$ | `[MATCHES PAPER FACT]` | Theo Eq. (6) và Mục 4.1.2. |
| **Độ phân giải ảnh** | $256 \times 256$ | $256 \times 256$ | `[MATCHES PAPER FACT]` | Theo Mục 4.1.2. |

> [!NOTE]
> Các cải tiến siêu tham số như `AdamW`, `lr=1e-4`, hay `CosineAnnealingLR` chỉ được phép thử nghiệm trong phạm vi nghiên cứu của **Method A (`FC-DSKD`)** hoặc **Method B (`Lite-DSKD`)**, tuyệt đối không được gán ghép vào baseline `B0-Clean`.

---

## 9. GIAO THỨC KIỂM CHUẨN THỜI GIAN GPU (GPU BENCHMARK PROTOCOL)

### 9.1. Phân Định Rạch Ròi Thời Gian Ước Tính vs. Đo Đạc Thực Tế
- **Thời gian ước tính (Analytical Estimation):** Là con số suy luận trên lý thuyết từ số phép tính FLOPs và thông số phần cứng định mức.
- **Thời gian thực tế (Empirical Measurement):** Là thời gian đo bằng hàm `time.perf_counter()` khi chạy mã nguồn thực tế trên node Kaggle GPU.

### 9.2. Phân Rã Các Thành Phần Thời Gian
Một chu kỳ thí nghiệm bao gồm 5 cấu phần thời gian hoàn toàn độc lập:
1. **$T_{\text{train}}$ (GPU-bound):** Thời gian nạp batch ảnh, forward qua $T, S_e, \text{DFE}, S_d$, tính loss $\ell_e, \ell_d$, backward và optimizer step.
2. **$T_{\text{val}}$ (GPU-bound):** Thời gian forward không tính gradient trên tập $20\%$ `val/good`.
3. **$T_{\text{io}}$ (Disk NVMe-bound):** Thời gian ghi và đọc checkpoint nguyên tử dung lượng $\sim 115\text{ MB}$.
4. **$T_{\text{inf}}$ (GPU-bound):** Thời gian forward trên tập test và tính bản đồ sai biệt $M_{t-d}$ qua bộ lọc Gaussian.
5. **$T_{\text{metrics}}$ (CPU-bound):**
   - Image AUROC & Pixel AUROC: Nhanh ($\approx 0.5 - 1.5\text{ s}$).
   - **Pixel AUPRO (`compute_aupro`):** Cực kỳ tốn thời gian tính toán CPU (từ $45\text{ s}$ đến hơn $90\text{ s}$ cho 500 ngưỡng tích phân trên các vùng liên thông).

### 9.3. Giao Thức Kiểm Chuẩn Tối Thiểu `M0_bench` (Micro-Benchmark)
Trước khi khởi chạy ma trận ablation, bắt buộc chạy script kiểm chuẩn thời gian thực tế `M0_bench` với thông số tối thiểu:
- Danh mục: `carpet` (1 danh mục duy nhất).
- Số epoch: **5 epochs** (đủ để làm ấm GPU và đo trung bình ổn định).
- Thu thập nhật ký đo lường thực tế:
  - `ms_per_train_batch`: Thời gian trung bình 1 batch train (mili-giây).
  - `sec_per_checkpoint`: Thời gian ghi 1 checkpoint ra đĩa (giây).
  - `sec_per_val_pass`: Thời gian 1 lượt thẩm định val (giây).
  - `sec_per_aupro_calc`: Thời gian tính hàm `compute_aupro` trên CPU Kaggle (giây).
- Cập nhật số liệu đo đạc thực tế này vào kế hoạch quản trị hạn ngạch GPU trước khi chạy đại trà.

---

## 10. BẢNG QUẢN TRỊ RỦI RO (RISK REGISTER)

| ID | Nguy Cơ Kỹ Thuật (Risk) | Khả Năng | Mức Độ | Biện Pháp Kiểm Soát & Giảm Thiểu (Mitigation) |
| :---: | :--- | :---: | :---: | :--- |
| **R1** | **Bùng nổ Gradient khi dùng lr=1e-3:** Learning rate $0.001$ với $\beta_1=0.5$ có thể gây bất ổn định số học trong các epoch đầu. | Trung bình | Cao | Áp dụng gradient clipping `clip_grad_norm_(..., max_norm=1.0)` và 5 epochs linear warmup. |
| **R2** | **Nghẽn cổ chai CPU do hàm AUPRO:** Hàm `compute_aupro` chạy quá lâu làm cạn kiệt thời gian phiên Kaggle. | Cao | Trung bình | Chỉ chạy `compute_aupro` ở epoch cuối cùng hoặc sau khi kết thúc toàn bộ quá trình huấn luyện; không chạy trong mỗi chu kỳ val. |
| **R3** | **Lệch pha Train-Test của DFE:** Nhầm lẫn đưa $F_e$ lúc train nhưng đưa $\hat{F}_e$ lúc test. | Thấp | Nghiêm trọng | Đóng gói logic này vào bên trong class `DSKDModel`, tự động đồng bộ cờ cấu hình. |
| **R4** | **Thiên lệch do chỉ thử nghiệm trên Carpet:** Cấu hình chọn được không hoạt động tốt trên các vật thể khác. | Trung bình | Cao | Bắt buộc chạy kiểm tra chéo cả `carpet` (texture) và `bottle` (object) trong pha Selection. |
| **R5** | **Hết hạn ngạch 30h GPU Kaggle:** Chạy quá nhiều biến thể 200 epochs gây cạn kiệt quota tuần. | Thấp | Cao | Chỉ chạy 200 epochs cho cấu hình chiến thắng trong pha Confirmation; các pha thăm dò chỉ chạy tối đa 50 epochs. |

---

## 11. DANH MỤC KIỂM TRA SẴN SÀNG TRIỂN KHAI (IMPLEMENTATION READINESS CHECKLIST)

Các quyết định đã được khóa và các điểm duy trì cờ cấu hình linh hoạt:

### A. Các Quyết Định Đã Khóa Cứng (Locked Decisions):
- [x] **Backbone:** ResNet18 (Teacher đóng băng trọng số ImageNet-1K, Student Encoder khởi tạo ngẫu nhiên).
- [x] **Student Decoder ($S_d$):** Cấu trúc đảo ngược dùng Bilinear upsample ($2\times$) + Conv3x3 để chống sọc ca-rô (*checkerboard artifacts*).
- [x] **DFE Branch 3:** Giữ khối $1\times 1$ conv trên nhánh $F_3$ khớp theo Hình 4 của bài báo gốc.
- [x] **Quyền sở hữu tham số DFE:** $\theta_{\text{dfe}}$ thuộc nhóm Decoder $\Theta_d$, được tối ưu hóa bởi `optimizer_d` để giảm thiểu $\ell_d$.
- [x] **Tính nhất quán Train-Test:** Đầu vào DFE trong pha suy luận luôn tự động đồng bộ loại dữ liệu (`raw` hoặc `norm`) theo pha huấn luyện.
- [x] **Siêu tham số Baseline:** Phục hồi chuẩn bài báo gốc: Adam, $\beta=(0.5, 0.999)$, $\text{lr}=1\times 10^{-3}$, 200 epochs.
- [x] **Kỷ luật Đánh giá:** Tuyệt đối không dùng tập Test để chọn cấu hình; chỉ dùng tập Validation.

### B. Các Điểm Duy Trì Cờ Cấu Hình Linh Hoạt (Configurable Points):
- [x] `dfe_input_type: str = "raw"` (hỗ trợ lựa chọn `"norm"` cho thí nghiệm đối chứng).
- [x] `detach_dfe_input: bool = True` (hỗ trợ lựa chọn `False` cho thí nghiệm đối chứng).

---

## 12. PHÁN QUYẾT CUỐI CÙNG (FINAL VERDICT)

> # ⚖️ **PHÁN QUYẾT CHÍNH THỨC:**
> # **`READY WITH EXPLICIT EXPERIMENTAL CONDITIONS`**
> *(ĐỦ ĐIỀU KIỆN TIẾN HÀNH CÀI ĐẶT B0-CLEAN VỚI CÁC ĐIỀU KIỆN THỰC NGHIỆM MINH BẠCH)*
>
> **Các điều kiện bắt buộc đi kèm khi bước vào giai đoạn viết mã nguồn (`src/anomaly_detection/`):**
> 1. Mã nguồn `DSKDModel` phải hỗ trợ hai cờ `dfe_input_type` và `detach_dfe_input` như một phần của kiến trúc module hóa.
> 2. Khởi tạo mã nguồn `DSKDTrainer` phải sử dụng thiết lập siêu tham số nguyên bản của bài báo gốc (Adam, $\beta=(0.5, 0.999)$, $\text{lr}=0.001$).
> 3. Trước khi tiến hành bất kỳ đợt chạy GPU lớn nào trên Kaggle, bắt buộc phải hoàn thành lượt đo chuẩn thời gian thực tế `M0_bench` (5 epochs trên `carpet`).
> 4. Lựa chọn cấu hình canonical chính thức phải được thực hiện trên tập **Validation** qua 2 danh mục `carpet` và `bottle`, không được sử dụng tập Test để tuning.

---

> 🛑 **ĐIỂM DỪNG BẮT BUỘC (MANDATORY STOP):**  
> Báo cáo phản biện đối nghịch độc lập đã hoàn tất tại [final_b0_experiment_contract.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/research/final_b0_experiment_contract.md). Theo đúng ràng buộc, nhóm nghiên cứu **dừng lại tại đây**, không viết model code, không commit/push, và chờ người dùng phê duyệt bản giao kèo cuối cùng này.
